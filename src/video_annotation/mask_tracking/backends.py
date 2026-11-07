"""Lazy native SAM adapters; no Torch/native imports until model loading.

Inputs use displayed native-pixel XY and XYWH. Propagation consumes a prepared
zero-based JPEG directory, never independently decodes or rotates a source video.
Models persist across jobs; per-image/per-video inference state never does.
"""
from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import gc
import hashlib
from importlib import metadata as import_metadata, util as import_util
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import traceback
import weakref

import numpy as np

from .memory import MemoryGuard, MemoryLimitError, is_allocation_error
from ..paths import models_root


NAMES = ("sam21", "sam31")
DEFAULT_CHECKPOINTS = {
    "sam21": models_root() / "checkpoints/sam2.1/sam2.1_hiera_large.pt",
    "sam31": models_root() / "checkpoints/sam3.1/sam3.1_multiplex.pt",
}
SAM21_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class MaskCandidate:
    object_id: int
    mask: np.ndarray
    score: float | None = None


@dataclass
class MaskResult:
    frame_index: int
    masks: dict[int, np.ndarray]
    scores: dict[int, float | None] = field(default_factory=dict)
    candidates: list[MaskCandidate] = field(default_factory=list)


def _cancel(cancel):
    if cancel is not None and (cancel.is_set() if hasattr(cancel, "is_set") else cancel()):
        raise CancelledError("SAM operation cancelled; native state was closed")


def _progress(callback, phase, **values):
    if callback is not None:
        callback({"phase": phase, **values})


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(getattr(value,"dtype",None)) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _frame_image(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.shape[0] or not image.shape[1]:
        raise ValueError("frame must be a nonempty H x W x 3 uint8 BGR image")
    return image


def _binary(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError(f"mask must be binary with displayed frame shape {shape}")
    return mask.astype(bool, copy=True)


def validate_prompt(prompt, shape, name):
    """Validate/copy one ROI prompt without importing a model runtime."""
    if not isinstance(prompt, dict):
        raise ValueError("prompt must be a dictionary")
    unknown = set(prompt)-{"object_id", "points_xy", "point_labels", "bbox_xywh", "text", "mask", "multimask_output"}
    if unknown:
        raise ValueError(f"unknown prompt fields: {', '.join(sorted(unknown))}")
    h, w = shape
    result = {"object_id": _integer(prompt.get("object_id", 1), "object_id")}
    if prompt.get("mask") is not None:
        if name == "sam31":
            raise NotImplementedError("SAM3.1 multiplex does not support exact binary-mask injection; use clicks/text/box inference or SAM2.1")
        if any(prompt.get(key) is not None for key in ("points_xy", "point_labels", "bbox_xywh", "text")):
            raise ValueError("an exact mask cannot be combined with other prompts")
        result["mask"] = _binary(prompt["mask"], shape)
        return result
    points = prompt.get("points_xy")
    if points is not None:
        points = np.asarray(points, np.float32)
        labels = np.asarray(prompt.get("point_labels"))
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                or labels.shape != (len(points),) or not np.isin(labels, (0, 1)).all()):
            raise ValueError("points_xy needs finite N x 2 coordinates and matching 0/1 point_labels")
        inside = (points[:,0] >= 0) & (points[:,0] <= w-1) & (points[:,1] >= 0) & (points[:,1] <= h-1)
        if not inside.all():
            bad = int(np.flatnonzero(~inside)[0])
            raise ValueError(f"point {bad} {points[bad].tolist()} lies outside displayed frame {w}x{h}")
        if not np.any(labels == 1):
            raise ValueError("at least one foreground click is required")
        result.update(points_xy=points.copy(), point_labels=labels.astype(np.int32))
    elif prompt.get("point_labels") is not None:
        raise ValueError("point_labels requires points_xy")
    if prompt.get("bbox_xywh") is not None:
        box = np.asarray(prompt["bbox_xywh"], np.float32)
        if (box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0
                or min(box[:2]) < 0 or box[0]+box[2] > w or box[1]+box[3] > h):
            raise ValueError(f"bbox_xywh must lie inside displayed frame {w}x{h}")
        result["bbox_xywh"] = box.copy()
    if prompt.get("text") is not None:
        if name == "sam21":
            raise NotImplementedError("SAM2.1 accepts geometric prompts, not text")
        if not isinstance(prompt["text"], str) or not prompt["text"].strip():
            raise ValueError("text prompt must be nonempty")
        result["text"] = prompt["text"].strip()
    if name == "sam31" and "points_xy" in result and ("text" in result or "bbox_xywh" in result):
        raise ValueError("SAM3.1 instance clicks cannot be combined with semantic text/box prompts")
    if not any(key in result for key in ("points_xy", "bbox_xywh", "text")):
        raise ValueError("provide foreground clicks, a box, text or an exact SAM2.1 mask")
    if "multimask_output" in prompt:
        if not isinstance(prompt["multimask_output"], bool):
            raise ValueError("multimask_output must be a boolean")
        result["multimask_output"] = prompt["multimask_output"]
    return result


def _prepared_directory(resource_path):
    from PIL import Image
    folder = Path(resource_path).resolve()
    if not folder.is_dir():
        raise ValueError("tracking requires a prepared displayed-orientation JPEG directory")
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    if not files or any(not p.stem.isdecimal() for p in files):
        raise ValueError("prepared JPEG files must have consecutive numeric names starting at zero")
    files.sort(key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError("prepared JPEG frame IDs must be consecutive from zero without duplicates")
    with Image.open(files[0]) as image:
        shape = (image.height, image.width)
    return folder, len(files), shape


def _configure_state_factory(model, state_device):
    """Scope native offload policy to this loaded inner tracker instance.

    The outer SAM3.1 session does not accept state offload, while every actual
    inner tracker does. Preserve its call signature and all return values. A
    weak bound-method reference avoids retaining the released model in a cycle.
    """
    original = model.init_state
    signature = inspect.signature(original)
    if "offload_state_to_cpu" not in signature.parameters:
        raise RuntimeError("installed SAM3.1 inner tracker does not support state offloading")
    original_ref = weakref.WeakMethod(original)
    offload = state_device == "cpu"

    def init_state(*args, **kwargs):
        factory = original_ref()
        if factory is None:
            raise RuntimeError("native tracker model has been released")
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments["offload_state_to_cpu"] = offload
        return factory(*bound.args, **bound.kwargs)

    init_state.__signature__ = signature
    init_state.__name__ = "init_state"
    model.init_state = init_state


class Backend:
    """One reusable native model; serialize preview/track calls per instance.

    max_frames is a strict TOTAL cap on unique yielded local frame IDs, including
    the reference. With both directions, native forward order precedes backward.
    Call close() on an abandoned track generator to guarantee immediate cleanup.
    Native frame/model loads are synchronous; cancellation is checked afterward.
    """
    def __init__(self, name="sam21", checkpoint=None, *, device="cuda", precision=None,
                 model_config=None, sam21_config=SAM21_CONFIG, state_device=None,
                 ram_reserve_gb=2., vram_reserve_gb=1.):
        if name not in NAMES:
            raise ValueError(f"unknown SAM backend {name!r}; choose {NAMES}")
        if device != "cpu" and not str(device).startswith("cuda"):
            raise ValueError("SAM device must be cpu or cuda")
        if state_device is None:
            state_device = "cpu" if name=="sam31" or device=="cpu" else "cuda"
        if state_device not in ("cpu","cuda") or (device=="cpu" and state_device!="cpu"):
            raise ValueError("state_device must be cpu or cuda; CPU model requires CPU state")
        precision = precision or ("float32" if device == "cpu" else "bfloat16")
        if precision not in ("float32", "float16", "bfloat16") or (device == "cpu" and precision != "float32"):
            raise ValueError("supported precision is CUDA float16/bfloat16/float32 or CPU float32")
        if name == "sam31" and (device != "cuda" or precision != "bfloat16"):
            raise ValueError("installed native SAM3.1 builder requires default CUDA and bfloat16")
        if name == "sam31" and (model_config is not None or sam21_config != SAM21_CONFIG):
            raise ValueError("model_config is a SAM2.1-only option; SAM3.1 uses its native multiplex architecture")
        if not isinstance(model_config or sam21_config,str) or not (model_config or sam21_config).strip():
            raise ValueError("SAM2.1 model_config must be a nonempty configuration name")
        self.name, self.device, self.precision = name, device, precision
        self.state_device = state_device
        self.memory = MemoryGuard(ram_reserve_gb=ram_reserve_gb, vram_reserve_gb=vram_reserve_gb)
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINTS[name]).resolve()
        self.sam21_config = model_config or sam21_config
        self.model_config = self.sam21_config if name=="sam21" else "sam3.1_multiplex"
        self._predictor = self._image_predictor = self._torch = None
        self._checkpoint_stat = self._provenance = None
        self._loaded_source_identity = None
        self._lock = threading.Lock()
        self.load_count = 0

    @property
    def capabilities(self):
        return {"points": True, "box": True, "text": self.name == "sam31",
                "exact_mask": self.name == "sam21", "both_directions": True,
                "multimask_candidates": self.name == "sam21",
                "supported_precisions": ["bfloat16"] if self.name=="sam31" else ["bfloat16","float16","float32"],
                "coordinate_space": "displayed native pixels XY/XYWH",
                "video_memory": "all prepared frames may remain in CPU RAM"}

    @property
    def supports(self):
        return self.capabilities

    @property
    def loaded(self):
        return self._predictor is not None

    def _stat(self):
        stat = self.checkpoint.stat()
        return stat.st_size, stat.st_mtime_ns

    def _source_identity(self):
        package = self._source_package()
        repo = package.parent
        if (repo/".git").exists():
            try:
                revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=False, timeout=10)
                diff = subprocess.run(["git", "-C", str(repo), "diff", "--binary", "HEAD", "--"],
                    capture_output=True, check=False, timeout=10)
                if not revision.returncode and not diff.returncode:
                    return {"source_revision": revision.stdout.strip(),
                            "source_tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
                            "source_tracked_diff_bytes": len(diff.stdout), "source_location": str(package)}
            except (OSError, subprocess.TimeoutExpired):
                pass
        # Wheel/source-package installs need not have a Git checkout. Hash actual
        # Python/config files so reuse still detects local source modifications.
        version = None
        for distribution in (("SAM-2", "sam2") if self.name == "sam21" else ("sam3",)):
            try:
                version = import_metadata.version(distribution)
                break
            except import_metadata.PackageNotFoundError:
                pass
        digest, count = hashlib.sha256(), 0
        for path in sorted(package.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json"):
                digest.update(path.relative_to(package).as_posix().encode()+b"\0")
                digest.update(path.read_bytes())
                count += 1
        if not count:
            raise RuntimeError(f"cannot identify installed {self.name} source package: {package}")
        return {"source_package": package.name, "source_version": version,
                "source_location": str(package), "source_files_sha256": digest.hexdigest(),
                "source_file_count": count, "source_hash_scope": "Python/YAML/JSON package files"}

    def _source_package(self):
        """Resolve an explicit models-root checkout, or the installed package."""
        name = "sam2" if self.name == "sam21" else "sam3"
        selected = models_root()/name/name
        if (selected/"__init__.py").is_file():
            return selected.resolve()
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "__file__", None):
            return Path(loaded.__file__).resolve().parent
        try:
            spec = import_util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            return Path(spec.origin).resolve().parent
        raise ModuleNotFoundError(f"{name} is unavailable; install it or put its checkout under VIDEO_ANNOTATION_MODELS/{name}")

    def _activate_source(self):
        """Allow an existing checkout without importing SAM during GUI startup."""
        package = self._source_package()
        loaded = sys.modules.get(package.name)
        if loaded is not None and getattr(loaded, "__file__", None) and Path(loaded.__file__).resolve().parent != package:
            raise RuntimeError(f"{package.name} was already imported from another location; restart after selecting VIDEO_ANNOTATION_MODELS")
        directory = str(package.parent)
        if directory not in sys.path:
            sys.path.insert(0, directory)

    @contextmanager
    def _operation(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("SAM backend is already processing; wait for it to finish")
        try:
            yield
        except BaseException as error:
            if isinstance(error, MemoryLimitError) or is_allocation_error(error, self._torch):
                message = str(error) if isinstance(error, MemoryLimitError) else (
                    "Native SAM ran out of memory despite admission checks. Use --objects-per-pass 1; "
                    "for GPU pressure choose --state-device cpu, for RAM pressure choose --state-device cuda "
                    "or free host memory. Model/session memory was released; saved references and completed results are retained.")
                traceback.clear_frames(error.__traceback__)
                error.__traceback__ = None
                self._drop_model()
                raise MemoryLimitError(message) from None
            raise
        finally:
            self._lock.release()

    def _check_memory(self, stage, *, required_ram=0, required_vram=0):
        return self.memory.check(stage, torch=self._torch, device=self.device,
                                 required_ram=required_ram, required_vram=required_vram)

    def preflight(self, frame_count):
        """Reject an impossible preload before preparing JPEGs or loading weights."""
        frame_count = _integer(frame_count, "frame_count", 1)
        # Pinned JPEG loader: float16 [T,3,1008,1008]. History/peaks are additional.
        required = frame_count*3*1008*1008*2 if self.name == "sam31" else 0
        with self._operation():
            # Query the driver before expensive JPEG preparation; this imports
            # Torch, never model code or weights.
            if self._torch is None and self.device != "cpu":
                import torch
                self._torch = torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable; use the installed SAM runtime on the GPU machine")
            model_budget = self.checkpoint.stat().st_size if not self.loaded else 0
            return self._check_memory("frame/model admission", required_ram=required, required_vram=model_budget)

    def _load(self, cancel=None, progress=None):
        _cancel(cancel)
        if self.loaded:
            if self._checkpoint_stat != self._stat():
                raise RuntimeError("checkpoint changed after model loading; restart this backend session")
            if self._loaded_source_identity != self._source_identity():
                raise RuntimeError("native source changed after model loading; restart this backend session")
            self._check_memory("model reuse")
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"installed {self.name} checkpoint not found: {self.checkpoint}")
        self._activate_source()
        _progress(progress, "loading_model", backend=self.name)
        import torch
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the installed SAM runtime on the GPU machine")
        self._torch = torch
        self._check_memory("model loading", required_vram=self.checkpoint.stat().st_size)
        before = self._stat()
        source_identity = self._source_identity()
        if self.name == "sam21":
            from sam2.build_sam import build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = build_sam2_video_predictor(self.sam21_config, str(self.checkpoint), device=self.device)
            predictor.fill_hole_area = 0
            predictor.use_mask_input_as_output_without_sam = True
            image_predictor = SAM2ImagePredictor(predictor)
        else:
            from sam3.model_builder import build_sam3_predictor
            predictor = build_sam3_predictor(version="sam3.1", checkpoint_path=str(self.checkpoint),
                use_fa3=False, use_rope_real=False, compile=False, async_loading_frames=False)
            # The pinned tracker wrapper and outer predictor each enter a
            # process-thread autocast context permanently. Balance both in
            # reverse construction order on this loading thread now;
            # later operations own explicit contexts and close() may run on a
            # different GUI worker thread.
            tracker_wrapper = getattr(predictor.model,"tracker",None)
            for owner in (predictor,tracker_wrapper):
                constructor_context = getattr(owner,"bf16_context",None)
                if constructor_context is not None:
                    constructor_context.__exit__(None,None,None)
                    owner.bf16_context = None
            predictor.model.batched_grounding_batch_size = 1
            native_tracker = getattr(tracker_wrapper,"model",None)
            if native_tracker is None:
                raise RuntimeError("installed SAM3.1 predictor has no inner tracker model")
            _configure_state_factory(native_tracker,self.state_device)
            image_predictor = None
        if self._stat() != before:
            raise RuntimeError("checkpoint changed while loading; restart the backend")
        if self._source_identity() != source_identity:
            raise RuntimeError("native source changed while loading; restart the backend")
        self._torch, self._predictor, self._image_predictor = torch, predictor, image_predictor
        self._checkpoint_stat = before
        self._loaded_source_identity = source_identity
        self.load_count += 1
        _cancel(cancel)
        _progress(progress, "model_ready", backend=self.name)

    @contextmanager
    def _inference(self):
        dtype = getattr(self._torch,self.precision) if self.precision != "float32" else None
        autocast = self._torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
        with self._torch.inference_mode(), autocast:
            yield

    def provenance(self):
        """Weight identity is hashed once per unchanged checkpoint, without Torch."""
        stat = self._stat()
        source_identity = self._source_identity()
        if self.loaded and self._loaded_source_identity != source_identity:
            raise RuntimeError("native source changed after model loading; restart this backend session")
        if self._provenance is None or self._provenance[0] != stat:
            if self.loaded and stat != self._checkpoint_stat:
                raise RuntimeError("checkpoint changed after model loading; restart this backend session")
            digest = hashlib.sha256()
            with self.checkpoint.open("rb") as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b""):
                    digest.update(block)
            if self._stat() != stat:
                raise RuntimeError("checkpoint changed while hashing")
            self._provenance = stat, {"backend": self.name, "checkpoint": self.checkpoint.name,
                "checkpoint_sha256": digest.hexdigest(), "checkpoint_bytes": stat[0],
                "device": self.device, "precision": self.precision,
                "state_device": self.state_device, "state_policy": "native offload without trimming or deleting history",
                "model_config": self.sam21_config if self.name == "sam21" else "sam3.1_multiplex",
                "use_fa3": False if self.name == "sam31" else None, "compile": False,
                "seed_policy": "exact supplied SAM2 mask at reference; other frames native predictions" if self.name == "sam21" else "native prompt inference; binary mask injection unsupported"}
        return {**self._provenance[1], **source_identity}

    def environment(self):
        result = {"backend": self.name, "device": self.device, "precision": self.precision,
                  "state_device": self.state_device,
                  "loaded": self.loaded, "load_count": self.load_count,
                  "memory": self.memory.last.copy()}
        if self._torch is not None:
            result["torch"] = self._torch.__version__
        return result

    def _sam31_prompt(self, session, frame_index, prompt, shape):
        h,w = shape
        request = {"type": "add_prompt", "session_id": session, "frame_index": frame_index}
        if "points_xy" in prompt:
            # rel_coordinates=False means model-canvas pixels (1008 square),
            # not original frame pixels. Normalize displayed XY explicitly.
            request.update(obj_id=prompt["object_id"], points=(prompt["points_xy"]/[w,h]).tolist(),
                point_labels=prompt["point_labels"].tolist(), rel_coordinates=True)
        else:
            if "text" in prompt:
                request["text"] = prompt["text"]
            if "bbox_xywh" in prompt:
                request.update(bounding_boxes=[(prompt["bbox_xywh"] / [w,h,w,h]).tolist()], bounding_box_labels=[1])
        return self._predictor.handle_request(request)

    def _sam31_result(self, response, shape):
        outputs = response["outputs"]
        ids = _array(outputs["out_obj_ids"]).reshape(-1)
        if len(set(ids.tolist())) != len(ids) or any(not np.isfinite(i) or int(i) != i or i < 0 for i in ids):
            raise ValueError("SAM3.1 returned duplicate or invalid object IDs")
        arrays = _array(outputs.get("out_binary_masks", np.empty((0,*shape), bool)))
        if arrays.shape != (len(ids), *shape):
            raise ValueError(f"SAM3.1 returned masks {arrays.shape}, expected {(len(ids),*shape)}")
        probabilities = outputs.get("out_probs")
        if probabilities is not None:
            probabilities = _array(probabilities).reshape(-1)
            if len(probabilities) != len(ids) or not np.isfinite(probabilities).all():
                raise ValueError("SAM3.1 scores disagree with object IDs")
        masks = {int(i): _binary(mask,shape) for i,mask in zip(ids,arrays)}
        scores = {int(i): float(probabilities[j]) if probabilities is not None else None for j,i in enumerate(ids)}
        return MaskResult(_integer(response["frame_index"], "native frame index"), masks, scores)

    def _sam21_result(self, index, ids, logits, shape, prompt):
        ids = [_integer(value, "native object_id") for value in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("SAM2.1 returned duplicate object IDs")
        arrays = _array(logits)
        if arrays.shape == (len(ids),1,*shape):
            arrays = arrays[:,0]
        if arrays.shape != (len(ids),*shape) or not np.isfinite(arrays).all():
            raise ValueError("SAM2.1 logits disagree with displayed frame geometry/object IDs")
        masks = {identity: (mask > 0).copy() for identity,mask in zip(ids,arrays)}
        if "mask" in prompt:
            masks[prompt["object_id"]] = prompt["mask"].copy()
        return MaskResult(_integer(index,"native frame index"), masks, {i:None for i in masks})

    def preview(self, frame_bgr, prompt, *, cancel=None, progress=None):
        frame = _frame_image(frame_bgr)
        shape = frame.shape[:2]
        prompt = validate_prompt(prompt,shape,self.name)
        _cancel(cancel)
        if "mask" in prompt:
            return MaskResult(0,{prompt["object_id"]:prompt["mask"].copy()},{prompt["object_id"]:None})
        with self._operation():
            self._load(cancel,progress)
            self._check_memory("segmentation preview")
            _progress(progress,"preview",backend=self.name)
            with self._inference():
                if self.name == "sam21":
                    try:
                        self._image_predictor.set_image(np.ascontiguousarray(frame[...,::-1]))
                        options = {}
                        if "points_xy" in prompt:
                            options.update(point_coords=prompt["points_xy"],point_labels=prompt["point_labels"])
                        if "bbox_xywh" in prompt:
                            box = prompt["bbox_xywh"].copy(); box[2:] += box[:2]
                            options["box"] = box
                        multiple = prompt.get("multimask_output",len(prompt.get("points_xy",[])) <= 1)
                        masks,scores,_ = self._image_predictor.predict(**options,multimask_output=multiple)
                        masks,scores = _array(masks),_array(scores).reshape(-1)
                        if len(masks) != len(scores) or not len(masks) or not np.isfinite(scores).all():
                            raise ValueError("SAM2.1 returned invalid preview candidates")
                        candidates = [MaskCandidate(prompt["object_id"],_binary(mask,shape),float(score)) for mask,score in zip(masks,scores)]
                        def order(candidate):
                            agreement = 0
                            if "points_xy" in prompt:
                                points = np.rint(prompt["points_xy"]).astype(int)
                                agreement = int(np.sum(candidate.mask[points[:,1],points[:,0]] == prompt["point_labels"]))
                            return agreement,candidate.score
                        candidates.sort(key=order,reverse=True)
                        chosen = candidates[0]
                        result = MaskResult(0,{chosen.object_id:chosen.mask.copy()},{chosen.object_id:chosen.score},candidates)
                    finally:
                        self._image_predictor.reset_predictor()
                else:
                    from PIL import Image
                    with tempfile.TemporaryDirectory(prefix="sam-preview-") as directory:
                        image_path = Path(directory)/"frame.png"
                        Image.fromarray(frame[...,::-1]).save(image_path)
                        session = None
                        try:
                            session = self._predictor.handle_request({"type":"start_session","resource_path":str(image_path),"offload_video_to_cpu":True})["session_id"]
                            _cancel(cancel)
                            result = self._sam31_result(self._sam31_prompt(session,0,prompt,shape),shape)
                        finally:
                            if session is not None:
                                self._predictor.handle_request({"type":"close_session","session_id":session})
            _cancel(cancel)
            return result

    def track(self, resource_path, reference_frame, prompt, max_frames=None, direction="forward", *, cancel=None, progress=None):
        folder,count,shape = _prepared_directory(resource_path)
        reference_frame = _integer(reference_frame,"reference_frame")
        if reference_frame >= count:
            raise ValueError("reference_frame is outside the prepared JPEG window")
        if direction not in ("forward","both"):
            raise ValueError("direction must be forward or both")
        if max_frames is not None:
            max_frames = _integer(max_frames,"max_frames",1)
        multiple = isinstance(prompt, (list, tuple))
        seeds = []
        if multiple:
            if self.name != "sam31" or not prompt:
                raise ValueError("Joint tracking needs SAM3.1 and at least one per-object prompt")
            for item in prompt:
                item = dict(item)
                index = _integer(item.pop("frame_index", reference_frame), "prompt frame_index")
                checked = validate_prompt(item, shape, self.name)
                if index >= count or "points_xy" not in checked:
                    raise ValueError("Joint SAM3.1 references need instance clicks inside the prepared window; select preview masks for text/box references")
                seeds.append((index, checked))
            ids = [item["object_id"] for _, item in seeds]
            if len(set(ids)) != len(ids):
                raise ValueError("Joint references require distinct object IDs")
        else:
            prompt = validate_prompt(prompt,shape,self.name)
        self.preflight(count)
        with self._operation():
            self._load(cancel,progress)
            self._check_memory("native frame loading", required_ram=count*3*1008*1008*2 if self.name == "sam31" else 0)
            state = session = stream = None
            emitted = set()
            _progress(progress,"loading_frames",backend=self.name,total=count)
            try:
                with self._inference():
                    if self.name == "sam21":
                        state_options = {"offload_state_to_cpu":True} if self.state_device=="cpu" else {}
                        state = self._predictor.init_state(str(folder),offload_video_to_cpu=True,async_loading_frames=False,**state_options)
                        _cancel(cancel)
                        if (state["video_height"],state["video_width"]) != shape or state["num_frames"] != count:
                            raise ValueError("SAM2.1 loaded geometry/frame count differs from prepared JPEG window")
                        if "mask" in prompt:
                            seed = self._predictor.add_new_mask(state,frame_idx=reference_frame,obj_id=prompt["object_id"],mask=prompt["mask"])
                        else:
                            options = {}
                            if "points_xy" in prompt:
                                options.update(points=prompt["points_xy"],labels=prompt["point_labels"])
                            if "bbox_xywh" in prompt:
                                box = prompt["bbox_xywh"].copy(); box[2:] += box[:2]
                                options["box"] = box
                            seed = self._predictor.add_new_points_or_box(state,frame_idx=reference_frame,obj_id=prompt["object_id"],**options)
                        first = self._sam21_result(*seed,shape,prompt)
                    else:
                        session = self._predictor.handle_request({"type":"start_session","resource_path":str(folder),"offload_video_to_cpu":True})["session_id"]
                        _cancel(cancel)
                        if multiple:
                            for index, item in seeds:
                                _cancel(cancel)
                                self._check_memory(f"initializing object {item['object_id']}")
                                seed = self._sam31_result(self._sam31_prompt(session, index, item, shape), shape)
                                if item["object_id"] not in seed.masks or not seed.masks[item["object_id"]].any():
                                    raise ValueError(f"No mask found for joint object {item['object_id']}; refine its saved reference")
                            first = None  # Propagation returns all initialized objects together.
                        else:
                            first = self._sam31_result(self._sam31_prompt(session,reference_frame,prompt,shape),shape)
                    if first is not None and first.frame_index != reference_frame:
                        raise ValueError("native seed output frame differs from requested reference_frame")
                    _cancel(cancel)
                    if first is not None:
                        emitted.add(first.frame_index)
                        _progress(progress,"tracking",current=1,total=count,frame_index=first.frame_index)
                        yield first
                        if max_frames == 1 or count == 1:
                            return
                    for reverse in ([False,True] if direction=="both" and reference_frame>0 else [False]):
                        _cancel(cancel)
                        # SAM2 counts an inclusive offset. The pinned SAM3.1
                        # tracker and detector disagree on inclusive/exclusive
                        # limits, producing an empty final detector chunk. Its
                        # resource window is already bounded; omit that native
                        # limit and enforce our strict yielded-frame cap below.
                        limit = {} if max_frames is None or self.name=="sam31" else {"max_frame_num_to_track":max_frames-1}
                        if self.name == "sam21":
                            stream = self._predictor.propagate_in_video(state,start_frame_idx=reference_frame,reverse=reverse,**limit)
                        else:
                            stream = self._predictor.handle_stream_request({"type":"propagate_in_video","session_id":session,
                                "start_frame_index":reference_frame,"propagation_direction":"backward" if reverse else "forward",
                                "evict_cached_frame_outputs":True,**limit})
                        try:
                            for response in self._guarded_stream(stream):
                                _cancel(cancel)
                                if self.name == "sam21":
                                    index,ids,logits = response
                                    # Override only the annotated seed; subsequent
                                    # native predictions never inherit its mask.
                                    result = self._sam21_result(index,ids,logits,shape,prompt if index==reference_frame else {})
                                else:
                                    result = self._sam31_result(response,shape)
                                if result.frame_index >= count:
                                    raise ValueError("native frame ID lies outside the prepared JPEG window")
                                if result.frame_index in emitted:
                                    continue
                                emitted.add(result.frame_index)
                                _progress(progress,"tracking",current=len(emitted),total=count,frame_index=result.frame_index)
                                yield result
                                if len(emitted) == count:
                                    return
                                if max_frames is not None and len(emitted)>=max_frames:
                                    return
                        finally:
                            if hasattr(stream,"close"):
                                stream.close()
                            stream = None
            finally:
                if state is not None:
                    self._predictor.reset_state(state)
                if session is not None:
                    self._predictor.handle_request({"type":"close_session","session_id":session})

    def _guarded_stream(self, stream):
        while True:
            self._check_memory("next tracking frame")
            try:
                response = next(stream)
            except StopIteration:
                return
            yield response

    def _drop_model(self):
        torch = self._torch
        self._image_predictor = self._predictor = self._torch = None
        self._checkpoint_stat = self._loaded_source_identity = None
        gc.collect()
        if torch is not None and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

    def close(self):
        with self._operation():
            self._drop_model()
