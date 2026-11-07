"""Stream native mask exports through the existing portable H.264 writer."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from video_annotation.common.saved_masks import SavedMasks, _ids
from video_annotation.rgb_tracking.tapnextpp import contained_path, video_frames, write_json
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation
from video_annotation.rgb_tracking.workflow import archive_files, check_cancel
from video_annotation.common.video import H264Writer
from video_annotation.common.video import probe_video


def draw_masks(frame_bgr, masks_by_id, source_frame=None):
    result = frame_bgr.copy()
    colors = ((0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 80, 255), (80, 150, 255))
    for identity, mask in sorted(masks_by_id.items(), key=lambda item: str(item[0])):
        mask = np.asarray(mask, bool)
        if mask.shape != result.shape[:2]:
            raise ValueError("Mask dimensions differ from the displayed video")
        if not mask.any():
            continue
        position = int(identity) if str(identity).isdigit() else int.from_bytes(hashlib.sha256(str(identity).encode()).digest()[:4], "big")
        color = np.asarray(colors[position % len(colors)], np.uint8)
        result[mask] = np.rint(result[mask]*.6 + color*.4).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color.tolist(), 2, cv2.LINE_AA)
        y, x = np.argwhere(mask)[0]
        for thickness, ink in ((3, (0, 0, 0)), (1, tuple(map(int, color)))):
            cv2.putText(result, str(identity), (int(x), max(15, int(y)-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, ink, thickness, cv2.LINE_AA)
    if source_frame is not None:
        for thickness, ink in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(result, f"Source frame {source_frame}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, ink, thickness, cv2.LINE_AA)
    return result


def load_frame_masks(folder, metadata, index, reader=None):
    if metadata["native_format"] == "sam2":
        reader = reader or SavedMasks(folder, format="sam2")
        mask = reader.load(index)
        return {} if mask is None else {str(metadata["roi_id"]): np.asarray(mask, bool)}
    path = Path(folder) / f"{index:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["frame_index"]) != index:
            raise ValueError("Native frame IDs disagree with mask filenames")
        ids = _ids(archive["out_obj_ids"])
        masks = archive["out_binary_masks"]
        if masks.ndim != 3 or len(masks) != len(ids):
            raise ValueError("SAM3 masks must have shape N,H,W")
        return {str(identity): mask.astype(bool) for identity, mask in zip(ids, masks)}


def render_masks(tracks, output, *, fps=None, max_width=1280, force=False, progress=None, cancel=None):
    folders = [contained_path(path, "data") for path in (tracks if isinstance(tracks, (list, tuple)) else [tracks])]
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("Choose one or more distinct mask export directories")
    output = contained_path(output, "outputs")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Mask overlays must use a portable .mp4 file")
    from .workflow import export_intact
    exports, video, source = [], None, None
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text())
        if not export_intact(folder, metadata):
            raise ValueError("Mask export is incomplete or changed; run process to rebuild before rendering")
        current_video = contained_path(folder / metadata["source"], "data")
        if video is None:
            video, source = current_video, probe_video(current_video)[0]
        elif current_video != video:
            raise ValueError("Combined mask exports must belong to the same source video")
        if (metadata.get("coordinate_space") != DISPLAY_COORDINATE_SPACE or
                metadata["image_size"] != [source["width"], source["height"]] or
                metadata["display_rotation_degrees"] != display_rotation(source) or
                metadata["source_file_stat"] != [video.stat().st_size, video.stat().st_mtime_ns]):
            raise ValueError("Tracked source geometry/file changed; regenerate from the original video")
        indices = metadata["frames"]
        if not indices or indices != list(range(indices[0], indices[-1]+1)):
            raise ValueError("Mask overlay needs a consecutive source-frame interval")
        reader = SavedMasks(folder, format="sam2") if metadata["native_format"] == "sam2" else None
        exports.append((folder, metadata, reader))
    fps = source["fps"] if fps is None else fps
    if not np.isfinite(fps) or fps <= 0 or max_width < 1:
        raise ValueError("Overlay FPS/width must be positive")
    digest = hashlib.sha256(json.dumps({"metadata": [meta for _, meta, _ in exports], "fps": fps, "width": max_width, "renderer": 2}, sort_keys=True).encode()).hexdigest()
    sidecar = output.with_suffix(".json")
    if not force and output.is_file() and sidecar.is_file():
        existing = json.loads(sidecar.read_text())
        if existing.get("input_signature") == digest and existing.get("output_bytes") == output.stat().st_size:
            return {"output": str(output), "reused": True}
    first = min(meta["frames"][0] for _, meta, _ in exports)
    last = max(meta["frames"][-1] for _, meta, _ in exports)
    indices = range(first, last+1)
    w, h = source["width"], source["height"]
    ratio = min(1., max_width/w)
    dw, dh = max(1, round(w*ratio)), max(1, round(h*ratio))
    size = (dw+dw%2, dh+dh%2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.stem}.{uuid4().hex}.mp4")
    writer, count = None, 0
    try:
        writer = H264Writer(temp, size, fps)
        for offset, image in enumerate(video_frames(video, indices[0], len(indices), source=source)):
            check_cancel(cancel)
            index = indices[offset]
            masks = {}
            for folder, metadata, reader in exports:
                if metadata["frames"][0] <= index <= metadata["frames"][-1]:
                    for identity, mask in load_frame_masks(folder, metadata, index, reader).items():
                        label = f"{metadata['roi_id']}:{identity}" if len(exports) > 1 else identity
                        if label in masks:
                            raise ValueError(f"Duplicate combined object label {label}; choose distinct ROI exports")
                        masks[label] = mask
            image = draw_masks(image, masks, index)
            image = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA) if (dw, dh) != (w, h) else image
            image = cv2.copyMakeBorder(image, 0, size[1]-dh, 0, size[0]-dw, cv2.BORDER_CONSTANT)
            writer.write(image)
            count += 1
            if progress and (count % 30 == 0 or count == len(indices)):
                progress(f"Rendering {video.name}: {count}/{len(indices)} frames")
        if count != len(indices):
            raise ValueError("Source ended before the mask interval; previous overlay retained")
        writer.close()
        check_cancel(cancel)
        archive_files([output, sidecar], video.stem, category="rgb_masks")
        temp.replace(output)
        result = {"output": str(output), "frames": count, "source_frame_start": indices[0], "fps": fps,
                  "width": size[0], "height": size[1], "codec": "h264", "pixel_format": "yuv420p", "faststart": True,
                  "output_bytes": output.stat().st_size, "input_signature": digest,
                  "mask_exports": [str(folder) for folder, _, _ in exports],
                  "object_labels": "ROI:object" if len(exports) > 1 else "object",
                  "timing": "CFR sanity preview; FPS changes viewing speed only, source frame IDs are retained", "reused": False}
        write_json(sidecar, result)
        return result
    finally:
        if writer:
            writer.close()
        temp.unlink(missing_ok=True)


def render_folder(source_dir, *, tracker="sam31", output_dir=None, fps=None,
                  max_width=1280, force=False, progress=None):
    """Combine completed ROI exports per source video, without model loading."""
    from video_annotation.rgb_tracking.tapnextpp import ROOT
    from .workflow import artifact_paths
    source_dir = contained_path(source_dir, "data")
    if not source_dir.is_dir():
        raise ValueError(f"Source folder does not exist: {source_dir}")
    output_root = ROOT/"outputs/videos/rgb_masks"
    output_dir = contained_path(output_dir or output_root, "outputs")
    if tracker not in ("sam21", "sam31"):
        raise ValueError("Choose sam21 or sam31")
    groups = {}
    for path in sorted((ROOT/"data/pseudo-ground-truths").rglob("metadata.json")):
        if (path.parent.name != "track" or path.parent.parent.parent.name != tracker
                or any(part.startswith(".") for part in path.relative_to(ROOT/"data").parts)):
            continue
        metadata = json.loads(path.read_text())
        video = contained_path(path.parent/metadata["source"], "data")
        if video.is_relative_to(source_dir):
            groups.setdefault(video, []).append(path.parent)
    results = []
    for video, tracks in sorted(groups.items()):
        output = artifact_paths(video, tracker, "all_rois")["overlay"].with_name("all_rois.mp4")
        output = output_dir/output.relative_to(output_root)
        if progress:
            progress(f"Combined overlay: {video.name} ({len(tracks)} saved exports)")
        results.append(render_masks(tracks, output, fps=fps, max_width=max_width,
                                    force=force, progress=progress))
    if progress and not results:
        progress(f"No completed {tracker} masks found for {source_dir}")
    return results
