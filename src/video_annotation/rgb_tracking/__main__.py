"""Label RGB points, choose a tracker, and render portable MP4 previews."""
from __future__ import annotations

from ..paths import configure_cli
if __name__ == "__main__":
    import sys
    sys.argv[1:] = configure_cli()

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .tapnextpp import (DEFAULT_REPO, Tracker, chunks, contained_path,
                       track_paths, video_frames, write_json)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog="Locations: --workspace DIR (default: current directory); --models-root DIR (optional model checkouts/weights). Relative file arguments use the current directory.")
    sub = p.add_subparsers(dest="command", required=True)
    descriptions = {
        "check": "check the TAPNext++ model and runtime",
        "track": "track explicit points in one video",
        "benchmark": "measure temporal batching throughput",
        "annotate": "pick and save points across a video folder; optionally track and render",
        "process": "track saved point references across a folder without the GUI",
    }
    from .backends import BACKENDS, DEFAULT_BACKEND
    for name, description in descriptions.items():
        s = sub.add_parser(name, help=description, description=description)
        tap_only = name in ("check", "benchmark")
        if not tap_only:
            s.add_argument("--reacquire-seconds", type=float,
                           help="allow tracker recovery within this many source-video seconds (default 0.5; 0 ends immediately)")
            s.add_argument("--tracker", choices=BACKENDS, default=DEFAULT_BACKEND,
                           help="point tracking algorithm (default: %(default)s)")
            s.add_argument("--tracker-options", type=Path,
                           help="JSON object of algorithm-specific options; unknown keys fail")
            s.add_argument("--lk-window", type=positive, help="KLT window side length (odd pixels)")
            s.add_argument("--fb-threshold", type=float, help="KLT forward/backward limit in pixels")
            s.add_argument("--patch-radius", type=positive, help="local support/template radius in pixels")
            s.add_argument("--reanchor-interval", type=positive, help="klt-template registration period in frames")
            s.add_argument("--motion-model", choices=("similarity", "affine", "homography", "adaptive"),
                           help="local geometric model for klt-affine/klt-template (default: adaptive)")
            s.add_argument("--keyframe-interval", type=nonnegative,
                           help="klt-template validated keyframe period; 0 disables (default)")
            s.add_argument('--motion-gate', action=argparse.BooleanOptionalAction, default=None,
                           help='enable/disable shared temporal motion acceptance for KLT')
            s.add_argument('--motion-sigma', type=float, help='temporal innovation limit in uncertainty standard deviations')
            s.add_argument('--motion-noise-px', type=float, help='motion uncertainty floor in source pixels')
            s.add_argument('--motion-memory-seconds', type=float, help='accepted motion memory in source-video seconds')
            s.add_argument('--recovery-enabled', action=argparse.BooleanOptionalAction, default=None,
                           help='allow expensive patch recovery search (default: disabled)')
            s.add_argument('--stop-on-occlusion', action=argparse.BooleanOptionalAction, default=None,
                           help='end a patch track on definite occlusion (default: enabled)')
            s.add_argument('--min-foreground-fraction', type=float,
                           help='minimum visible foreground fraction for occlusion acceptance (default: 0.5)')
            s.add_argument('--center-smoothing', action=argparse.BooleanOptionalAction, default=None,
                           help='smooth small center jitter (default: enabled)')
            s.add_argument('--center-smoothing-px', type=float,
                           help='source-pixel center-smoothing range (default: 2)')
        s.add_argument("--repo", type=Path, default=DEFAULT_REPO if tap_only else None,
                       help="TAPNext++ checkout")
        s.add_argument("--checkpoint", type=Path, help="TAPNext++ checkpoint")
        s.add_argument("--resolution", type=int, choices=(256, 512), default=512 if tap_only else None,
                       help="TAPNext++ image input side (default: 512)")
        s.add_argument("--device", default="cuda" if tap_only else None, help="TAPNext++ device (default: cuda)")
        s.add_argument("--precision", choices=("auto", "none", "float16", "bfloat16"),
                       default="auto" if tap_only else None, help="TAPNext++ precision")
        s.add_argument("--compile", action="store_true", default=False if tap_only else None,
                       help="experimental TAPNext++ compilation; included in timings")
        s.add_argument("--threads", type=positive, default=8 if tap_only else None,
                       help="CPU worker threads (KLT default: 1; TAPNext++ default: 8)")
    track = sub.choices["track"]
    track.add_argument("--video", type=Path, required=True)
    queries = track.add_mutually_exclusive_group(required=True)
    queries.add_argument("--query", type=float, nargs=2, action="append", metavar=("X", "Y"))
    queries.add_argument("--queries", type=Path, help='reference CSV with JSON sidecar, or JSON: {"frame":0,"points_xy":[[x,y],...]}')
    track.add_argument("--frame", type=int, default=0, help="zero-based query frame; JSON frame takes precedence")
    track.add_argument("--batch-size", type=positive,
                       help="consecutive frames per call; KLT default 1, TAP default 1 at 512 / 32 at 256")
    track.add_argument("--max-frames", type=positive)
    track.add_argument("--output", type=Path, required=True, help="point CSV, under data/")
    track.add_argument("--profile", type=Path, help="optional timing JSON, under outputs/")
    bench = sub.choices["benchmark"]
    bench.add_argument("--video", type=Path, help="otherwise use a reproducible moving synthetic texture")
    bench.add_argument("--frames", type=positive, default=33)
    bench.add_argument("--points", type=positive, default=256)
    bench.add_argument("--batch-sizes", type=positive, nargs="+", default=[1, 8, 32])
    bench.add_argument("--output", type=Path, required=True, help="timing/correctness JSON, under outputs/")
    for name in ("annotate", "process"):
        command = sub.choices[name]
        command.add_argument("--source-dir", type=Path, help="RGB folder; may be supplied by --config")
        command.add_argument("--config", type=Path, help="saved GUI session settings; explicit flags override them")
        command.add_argument("--no-recursive", action="store_true")
        command.add_argument("--batch-size", type=positive)
        command.add_argument("--max-frames", type=positive)
        command.add_argument("--overlay", action="store_true", help="save a portable MP4 sanity preview; enables tracking in annotate")
        command.add_argument("--overlay-fps", type=float, help="preview FPS only; defaults to source metadata")
        command.add_argument("--max-width", type=positive, default=1280, help="maximum overlay width")
    sub.choices["annotate"].add_argument("--frame", type=int, default=0, help="default query frame for unlabeled videos")
    sub.choices["annotate"].add_argument("--track", action="store_true", help="track after saving each video's points")
    sub.choices["process"].add_argument("--force", action="store_true", help="recompute matching tracks; prior results preserved")
    sub.choices["process"].add_argument("--render-only", action="store_true", help="render existing predictions across the folder, without tracking")
    sub.choices["process"].add_argument("--clean-existing", action="store_true",
                                        help="with --render-only, first clean original and corrected CSVs in place; keep backups")
    sub.choices["process"].add_argument("--corrected-name", default="corrected/points.csv",
                                        help="prefer these saved corrections relative to each prediction folder when rendering")
    render = sub.add_parser("render", help="render an existing point CSV without loading the model")
    render.add_argument("--tracks", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--fps", type=float, help="preview FPS, not physical playback/label timing")
    render.add_argument("--max-width", type=positive, default=1280)
    render.add_argument("--force", action="store_true")
    render.add_argument("--reacquire-seconds", type=float, help="override stored recovery window in source-video seconds")
    for name in ('render-named', 'export-ready'):
        command = sub.add_parser(name, help='render or export uniformly named corrected tracks across a master folder')
        command.add_argument('--master', type=Path, default=Path('data/pseudo-ground-truths/point_tracks'))
        command.add_argument('--name', default='corrected/points.csv', help='CSV path relative to each prediction folder')
        command.add_argument('--output-dir', type=Path, required=name == 'export-ready')
    sub.choices['render-named'].add_argument('--fps', type=float)
    sub.choices['render-named'].add_argument('--max-width', type=positive, default=1280)
    sub.choices['render-named'].add_argument('--reacquire-seconds', type=float)
    clean = sub.add_parser("clean", help="clean existing original/corrected point CSVs in place, without tracking or rendering")
    clean.add_argument("--master", type=Path, required=True,
                       help="point prediction folder to search recursively; includes saved corrections")
    clean.add_argument("--reacquire-seconds", type=float, default=.5,
                       help="recovery window in source-video seconds (default 0.5; 0 ends immediately)")
    clean.add_argument("--dry-run", action="store_true", help="report changes without writing files")
    reindex = sub.add_parser('reindex',help='compact saved point IDs, including linked corrections/poses, with backups')
    reindex.add_argument('--data-dir',type=Path,help='workspace data directory; default data/')
    reindex.add_argument('--dry-run',action='store_true')
    from ..pose.methods import METHODS, DEFAULT_METHOD
    refine = sub.add_parser("refine", help="refine saved point patches with selectable correlation/geometric methods")
    refine.add_argument("--tracks", type=Path, required=True, help="original or corrected point CSV")
    refine.add_argument('--render-only', action='store_true',
                        help='render poses already stored in the unified CSV; run no fitting')
    refine.add_argument('--interpolate-existing', action='store_true',
                        help='fill short pose gaps for --render-only, using gap settings')
    refine.add_argument('--fill-missing', action=argparse.BooleanOptionalAction, default=True,
                        help='keep existing valid poses and fit missing ones; --no-fill-missing refines them again')
    refine.add_argument("--pose-method", choices=METHODS, default=DEFAULT_METHOD,
                        help="pose algorithm; default shared patch-hybrid tracker")
    refine.add_argument("--output-dir", type=Path, help="data-local pose folder; default beside the input tracks")
    refine.add_argument("--point-ids", type=int, nargs="+", help="stable point IDs; defaults to all")
    refine.add_argument("--patch-radius", type=float, help="override saved patch radii; pixels")
    refine.add_argument("--max-frames", type=positive)
    refine.add_argument("--patch-size", type=positive, default=128,
                        help="resampling grid size; does not change saved per-point support radii/masks")
    refine.add_argument("--ecc-iters", type=positive, default=45, help="ECC iteration limit")
    refine.add_argument("--ic-iters", type=positive, default=20, help="inverse-compositional iteration limit")
    refine.add_argument("--ic-stride", type=positive, default=2, help="inverse-compositional pixel sampling stride")
    refine.add_argument("--min-gain", type=float, default=.0025, help="minimum accepted geometric correlation gain")
    refine.add_argument("--klt-max-points", type=positive, default=200, help="KLT maximum features, at least 8")
    refine.add_argument("--klt-min-distance", type=float, default=4., help="KLT feature spacing in source pixels, at least 1")
    refine.add_argument("--klt-window", type=positive, default=21, help="KLT optical-flow window in pixels, odd and at least 3")
    refine.add_argument("--no-recent-template", dest="use_recent", action="store_false", default=True,
                        help="freeze original appearance, including correlation seeds")
    refine.add_argument("--geometric-min-score", type=float, default=.70,
                        help="geometric acceptance NCC threshold, separate from the correlation seed score")
    refine.add_argument("--original-score-slack", type=float, default=.03,
                        help="geometric recent-template score allowance relative to the original")
    refine.add_argument("--template-min-overlap", type=float, default=.6,
                        help="geometric template minimum foreground overlap")
    refine.add_argument("--template-update-score", type=float, default=.75,
                        help="geometric template minimum NCC for accepting an update")
    refine.add_argument("--original-check-interval", type=positive, default=10,
                        help="geometric original-template check interval in source frames")
    refine.add_argument("--center-search", action="store_true", help="enable bounded center recovery for noisy input tracks")
    refine.add_argument("--no-center-search", dest="center_search", action="store_false", default=False,
                        help="disable bounded center search")
    refine.add_argument("--search-radius", type=float, default=64., help="center search hard cap in source pixels")
    refine.add_argument("--search-trigger", type=float, default=.75, help="center search trigger NCC")
    refine.add_argument("--search-top-k", type=positive, default=3, help="center search candidate count")
    refine.add_argument('--search-shapes', type=positive, default=81,
                        help='joint shape-bank budget per reference/seed (1–729; 81 default, 729 full projective bank)')
    refine.add_argument("--center-trust", type=float, default=.5, help="center search prediction trust in [0,1]")
    refine.add_argument("--search-interval", type=positive, default=5, help="center search interval in source frames")
    refine.add_argument('--motion-gate', action=argparse.BooleanOptionalAction, default=True,
                        help='reject temporally inconsistent pose candidates (default enabled)')
    refine.add_argument('--motion-sigma', type=float, default=4., help='motion acceptance limit in uncertainty standard deviations')
    refine.add_argument('--motion-noise-px', type=float, default=1., help='motion uncertainty floor in source pixels')
    refine.add_argument('--motion-memory-seconds', type=float, default=.1, help='accepted motion memory in source-video seconds')
    refine.add_argument('--recovery-enabled', action=argparse.BooleanOptionalAction, default=False,
                        help='allow expensive patch recovery search during refinement (default: disabled)')
    refine.add_argument("--min-score", type=float, default=.75)
    refine.add_argument("--full-search", action="store_true", help="disable the cheaper guarded local path")
    refine.add_argument("--reference-interval", type=positive, default=10)
    refine.add_argument("--iterations", type=positive, default=8)
    refine.add_argument("--fast-iterations", type=positive, default=8)
    refine.add_argument("--fast-samples", type=positive, default=2500)
    refine.add_argument("--scale-limit", type=float, default=1.6)
    refine.add_argument("--center-limit", type=float, default=.3)
    refine.add_argument("--rotation-smoothness", type=float, default=.5)
    refine.add_argument("--scale-smoothness", type=float, default=2.)
    refine.add_argument("--tracker-center-weight", type=float, default=.9)
    refine.add_argument("--registration-center-weight", type=float, default=.1)
    anchor = refine.add_argument_group('Original fusion options (--pose-method anchored only)')
    anchor.add_argument("--anchor-preference", type=float, default=.8,
                        help="original-reference weight when equally scored fits agree (default 0.8)")
    anchor.add_argument("--anchor-score-delta", type=float, default=.05,
                        help="fit-score advantage multiplying its fusion odds by four (default 0.05)")
    anchor.add_argument("--anchor-angle-tolerance", type=float, default=10.,
                        help="maximum angle disagreement for fusing fits, degrees")
    anchor.add_argument("--anchor-scale-tolerance", type=float, default=.1,
                        help="fractional scale disagreement for fusing fits (default 0.1)")
    anchor.add_argument("--recovery-scale-min", type=float, default=.25,
                        help="minimum absolute scale during original-reference recovery")
    anchor.add_argument("--recovery-scale-max", type=float, default=4.,
                        help="maximum absolute scale during original-reference recovery")
    anchor.add_argument("--recovery-angle-step", type=float, default=15.,
                        help="coarse original-reference recovery angle step, degrees")
    anchor.add_argument("--recovery-scale-step", type=float, default=1.25,
                        help="coarse original-reference recovery multiplicative scale step")
    refine.add_argument("--gap-seconds", type=float, default=.1,
                        help="maximum bracketed gap in source-video seconds; no extrapolation; 0 disables")
    refine.add_argument("--gap-neighbors", type=nonnegative, default=2,
                        help="raw accepted samples per side of a filled gap, at least 1; 0 disables")
    refine.add_argument("--opencv-threads", type=positive, default=1)
    refine.add_argument("--overlay", action="store_true", help="render a combined bounding-circle/rotation MP4")
    refine.add_argument("--overlay-fps", type=float, help="preview speed only; source frame IDs/timestamps are preserved")
    refine.add_argument("--max-width", type=positive, default=1280)
    refine.add_argument("--force", action="store_true", help="recompute matching results")
    from .render_options import RENDER_PRESETS
    for name in ('render', 'render-named', 'annotate', 'process', 'refine'):
        command = sub.choices[name]
        command.add_argument('--render-max-frames', type=positive, help='preview limit only; does not shorten tracks')
        command.add_argument('--crf', type=int, help='H.264 quality 0–51; lower is higher quality (default 18)')
        command.add_argument('--preset', choices=RENDER_PRESETS)
        if name != 'refine':
            command.add_argument('--trails', action=argparse.BooleanOptionalAction, default=None)
            command.add_argument('--trail-length', type=positive)
    sub.add_parser("migrate", help="move legacy point tracks/overlays into source-matching folders without inference")
    return p


def tracking_configuration(args):
    """Resolve model-specific options before opening videos or writing artifacts."""
    from .backends import validate_backend_options
    options = dict(getattr(args, '_session_tracker_options', {}))
    if args.tracker_options:
        supplied = json.loads(args.tracker_options.read_text())
        if not isinstance(supplied, dict):
            raise ValueError("--tracker-options must contain a JSON object")
        options.update(supplied)
    overrides = {"win_size": args.lk_window, "fb_threshold": args.fb_threshold,
                 "patch_radius": args.patch_radius, "reanchor_interval": args.reanchor_interval,
                 "motion_model": args.motion_model, "keyframe_interval": args.keyframe_interval}
    overrides.update({name:getattr(args,name,None) for name in
                     ('motion_gate','motion_sigma','motion_noise_px','motion_memory_seconds',
                      'recovery_enabled','stop_on_occlusion','min_foreground_fraction',
                      'center_smoothing','center_smoothing_px')})
    options.update({key: value for key, value in overrides.items() if value is not None})
    tap_flags = ("repo", "checkpoint", "resolution", "device", "precision", "compile")
    explicit = ["--" + key.replace("_", "-") for key in tap_flags if getattr(args, key) is not None]
    if args.tracker != "tapnextpp" and explicit:
        raise ValueError(f"{' '.join(explicit)} only apply to --tracker tapnextpp")
    tap = dict(repo=args.repo or DEFAULT_REPO, checkpoint=args.checkpoint,
               resolution=args.resolution or 512, device=args.device or "cuda",
               precision=args.precision or "auto", compile_model=bool(args.compile), threads=args.threads or 8)
    if args.tracker == "tapnextpp":
        canonical = validate_backend_options(args.tracker, options)
    else:
        from .klt import available_options
        inapplicable = sorted(set(options) - set(available_options(args.tracker)))
        if inapplicable:
            raise ValueError(f"{', '.join(inapplicable)} not applicable to {args.tracker}")
        if args.threads is not None:
            options["threads"] = args.threads
        else:
            options.setdefault("threads", 8)
        canonical = validate_backend_options(args.tracker, options)
    return canonical, tap


def resolve_session_args(args, argv):
    if args.command not in ('annotate', 'process'):
        return
    if args.config:
        from .session import load_settings
        value = load_settings(args.config)
        explicit = {arg.split('=', 1)[0] for arg in argv if arg.startswith('--')}
        if '--tracker' not in explicit:
            args.tracker = value['tracker']
        args._session_tracker_options = (value['tracker_options']
                                        if args.tracker == value['tracker'] else {})
        for key in ('source_dir', 'batch_size', 'max_frames', 'reacquire_seconds', 'overlay_fps', 'max_width', 'frame'):
            if hasattr(args, key) and '--'+key.replace('_', '-') not in explicit:
                setattr(args, key, value[key])
        if '--no-recursive' not in explicit:
            args.no_recursive = not value['recursive']
        for key in ('mask_color', 'mask_alpha', 'mask_visible'):
            setattr(args, key, value[key])
        args._render_options = value['render_options']
        if args.tracker == 'tapnextpp':
            for key, setting in value['tap_options'].items():
                flag = '--compile' if key == 'compile_model' else '--'+key.replace('_', '-')
                if flag not in explicit:
                    setattr(args, 'compile' if key == 'compile_model' else key, setting)
    if not args.source_dir:
        raise ValueError('Provide --source-dir or load a --config containing source_dir')


def rendering_configuration(args):
    from .render_options import normalize_render_options
    value = dict(getattr(args, '_render_options', {}))
    for key in ('crf', 'preset', 'trails', 'trail_length', 'max_frames'):
        field = 'render_max_frames' if key == 'max_frames' else key
        setting = getattr(args, field, None)
        if setting is not None:
            value[key] = setting
    return normalize_render_options(value)


def load_tracker(args):
    begin = time.perf_counter()
    if hasattr(args, "tracker"):
        from .backends import make_tracker
        options, tap = tracking_configuration(args)
        tracker = make_tracker(args.tracker, options, **tap) if args.tracker == "tapnextpp" else make_tracker(args.tracker, options)
    else:
        tracker = Tracker(repo=args.repo, checkpoint=args.checkpoint, resolution=args.resolution,
                          device=args.device, precision=args.precision, compile_model=args.compile,
                          threads=args.threads)
    tracker.synchronize()
    return tracker, time.perf_counter() - begin


def track(args):
    from .workflow import export_tracks
    from video_annotation.common.video import probe_video
    from .annotations import validate_reference
    from .video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation
    setup_begin = time.perf_counter()
    tracking_configuration(args)
    batch_size = args.batch_size or (32 if args.tracker == "tapnextpp" and args.resolution == 256 else 1)
    output, video = track_paths(args.output, args.video, args.queries)
    profile = contained_path(args.profile, "outputs") if args.profile else None
    points, frame = args.query, args.frame
    query = {"frame": frame, "points_xy": points}
    if args.queries:
        from .annotations import read_reference_file
        query = read_reference_file(args.queries)
        points, frame = query["points_xy"], query.get("frame", frame)
        query = {**query, "frame": frame}
    source = probe_video(video)[0]
    # Literal queries and the historical minimal CLI query JSON use displayed
    # coordinates. Source-adjacent saved references carry dimensions, allowing
    # safe recovery of old encoded-raster annotations instead of guessing.
    if not args.queries or ("width" not in query and "height" not in query and "coordinate_space" not in query):
        query.update(coordinate_space=DISPLAY_COORDINATE_SPACE,
                     display_rotation_degrees=display_rotation(source))
    reference = validate_reference(query, video, source["width"], source["height"], source=source)
    points, frame = reference["points_xy"], reference["frame"]
    if frame >= source["number_of_frames"]:
        raise ValueError("query frame is outside the video")
    tracker, load_seconds = load_tracker(args)
    provenance_begin = time.perf_counter()
    provenance = tracker.provenance()
    provenance_seconds = time.perf_counter() - provenance_begin
    setup_seconds = time.perf_counter() - setup_begin
    from .regions import load_patch_regions
    patch_regions = load_patch_regions(reference, video, tracker.config.patch_radius) if args.tracker != "tapnextpp" and "patches" in reference else None
    result = export_tracks(video, points, frame, output, tracker, batch_size=batch_size,
                           max_frames=args.max_frames, provenance=provenance, source=source, patch_regions=patch_regions,
                           point_ids=reference.get("point_ids"),
                           reacquire_seconds=.5 if args.reacquire_seconds is None else args.reacquire_seconds)
    result.update(load_seconds=load_seconds, provenance_seconds=provenance_seconds,
                  setup_seconds=setup_seconds)
    if profile:
        write_json(profile, result)
    print(json.dumps(result, indent=2))


def folder_command(args):
    from .annotations import PointController, discover_videos, reference_path
    from .workflow import Processor
    options, tap = tracking_configuration(args)
    skipped = []
    videos = discover_videos(args.source_dir, not args.no_recursive, skipped)
    for video, reason in skipped:
        print(f"Skipped {video}: {reason}", flush=True)
    if not videos:
        raise ValueError("no valid source videos found")
    print(f"Found {len(videos)} source videos. Saved point references are reloaded automatically.", flush=True)
    processor = Processor(backend=args.tracker, backend_options=options, **tap,
                          batch_size=args.batch_size, max_frames=args.max_frames, source_root=args.source_dir,
                          reacquire_seconds=.5 if args.reacquire_seconds is None else args.reacquire_seconds)
    if args.command == "annotate":
        from .labeler import PointLabeler
        PointLabeler(PointController(videos, args.frame, default_radius=options.get("patch_radius",32),
                                    source_dir=args.source_dir, recursive=not args.no_recursive), processor,
                     track=args.track or args.overlay, overlay=args.overlay,
                     overlay_fps=args.overlay_fps, max_width=args.max_width,
                     mask_color=getattr(args, 'mask_color', '#45b8e8'),
                     mask_alpha=getattr(args, 'mask_alpha', .3),
                     mask_visible=getattr(args, 'mask_visible', True),
                     render_options=rendering_configuration(args)).run()
    else:
        if args.clean_existing:
            if not args.render_only:
                raise ValueError("--clean-existing requires --render-only; use clean --master DIR for data-only cleanup")
            from .cleanup_tracks import cleanup_saved_tracks
            paths = []
            for video in videos:
                original = processor.paths_for(video)["tracks"]
                for candidate in saved_track_files(original.parent):
                    metadata = json.loads(candidate.with_suffix(".json").read_text())
                    base = (metadata.get("manual_edits") or {}).get("base_tracks")
                    if candidate == original or (base and (candidate.parent/base).resolve() == original):
                        paths.append(candidate)
            cleaned = cleanup_saved_tracks(paths, reacquire_seconds=processor.reacquire_seconds)
            print(json.dumps({"cleanup": cleaned}), flush=True)
        result = processor.process_all(videos,
            mode='render' if args.render_only else 'both' if args.overlay else 'track',
            force=args.force, overlay_fps=args.overlay_fps, max_width=args.max_width,
            progress=lambda text: print(text, flush=True), render_options=rendering_configuration(args),
            corrected_name=args.corrected_name)
        print(json.dumps(result), flush=True)


def saved_track_files(master):
    """Find point predictions, excluding references, diagnostics and backups."""
    master = contained_path(master, "data")
    if not master.is_dir():
        raise ValueError(f"Prediction folder does not exist: {master}")
    paths = []
    for path in sorted(master.rglob("*.csv")):
        metadata = path.with_suffix(".json")
        if metadata.is_file() and json.loads(metadata.read_text()).get("kind") == "rgb_point_tracks":
            paths.append(path)
    return paths


def refine_command(args):
    if args.render_only:
        from ..pose.workflow import render_existing_point_poses
        result = render_existing_point_poses(args.tracks, fps=args.overlay_fps, max_width=args.max_width,
            render_options=rendering_configuration(args), force=args.force,
            gap_seconds=args.gap_seconds if args.interpolate_existing else 0.,
            gap_neighbors=args.gap_neighbors if args.interpolate_existing else 0,
            point_ids=args.point_ids,max_frames=args.max_frames,
            progress=lambda value: print(value, flush=True))
        print(json.dumps(result, indent=2))
        return
    from ..pose.registration import Config
    from ..pose.workflow import refine_point_tracks
    config = Config(support="circle", radius_blend=0., tracker_scale_weight=0., registration_scale_weight=1.,
        **{name:getattr(args, name) for name in ("patch_size", "min_score", "reference_interval", "iterations",
            "fast_iterations", "fast_samples", "scale_limit", "center_limit", "rotation_smoothness",
            "scale_smoothness", "tracker_center_weight", "registration_center_weight", "anchor_preference",
            "anchor_score_delta", "anchor_angle_tolerance", "anchor_scale_tolerance", "recovery_scale_min",
            "recovery_scale_max", "recovery_angle_step", "recovery_scale_step")
            if args.pose_method == 'anchored' or not name.startswith(('anchor_', 'recovery_'))}, fast=not args.full_search)
    result = refine_point_tracks(args.tracks, output_dir=args.output_dir, point_ids=args.point_ids,
        config=config, radius=args.patch_radius, max_frames=args.max_frames, overlay=args.overlay,
        overlay_fps=args.overlay_fps, max_width=args.max_width, force=args.force,
        render_options=rendering_configuration(args), opencv_threads=args.opencv_threads,
        gap_seconds=args.gap_seconds, gap_neighbors=args.gap_neighbors, fill_missing=args.fill_missing,
        method=args.pose_method, method_options={name:getattr(args, name) for name in
            ('ecc_iters','ic_iters','ic_stride','min_gain','klt_max_points','klt_min_distance','klt_window',
             'use_recent','geometric_min_score','original_score_slack','template_min_overlap',
             'template_update_score','original_check_interval','center_search','search_radius',
             'search_trigger','search_top_k','search_shapes','center_trust','search_interval',
             'motion_gate','motion_sigma','motion_noise_px','motion_memory_seconds','recovery_enabled')},
        progress=lambda value: print(value, flush=True))
    print(json.dumps(result, indent=2))


def benchmark(args):
    output = contained_path(args.output, "outputs")
    begin = time.perf_counter()
    if args.video:
        frames = list(video_frames(args.video, limit=args.frames))
    else:
        texture = np.random.default_rng(20261031).integers(0, 256, (360, 640, 3), np.uint8)
        frames = [np.roll(texture, i, axis=1) for i in range(args.frames)]
    preparation = time.perf_counter() - begin
    if not frames:
        raise ValueError("no benchmark frames")
    h, w = frames[0].shape[:2]
    side = int(np.ceil(np.sqrt(args.points)))
    x, y = np.meshgrid(np.linspace(w * .1, w * .9, side), np.linspace(h * .1, h * .9, side))
    points = np.column_stack([x.ravel(), y.ravel()])[:args.points]
    tracker, load_seconds = load_tracker(args)
    begin = time.perf_counter()
    tracker.infer(frames[:1], points)
    warmup = time.perf_counter() - begin
    reference = None
    rows = []
    def save():
        write_json(output, {**tracker.environment(), "source": str(args.video) if args.video else "moving seeded RGB texture",
                           "point_count": args.points, "load_seconds": load_seconds, "initial_warmup_seconds": warmup,
                           "decode_or_generation_seconds": preparation, "results": rows,
                           "compiled": args.compile,
                           "measurement": "one measured pass per size, after initial and per-size initial/recurrent warmups; wall includes preprocessing/inference/CPU transfer, excludes disk export; decode measured separately; any remaining compilation is included"})
    # Always obtain the frame-by-frame reference even if only larger sizes given.
    for size in dict.fromkeys([1, *args.batch_sizes]):
        tracker.reset()
        if tracker.device.type == "cuda":
            tracker.torch.cuda.empty_cache()
            tracker.torch.cuda.reset_peak_memory_stats(tracker.device)
        positions, visibility, times = [], [], dict(preprocessing_seconds=0., inference_seconds=0., transfer_seconds=0.)
        begin = time.perf_counter()
        try:
            warm_frames = (frames * (2 * size // len(frames) + 1))[:2 * size]
            for batch in chunks(warm_frames, size):
                tracker.infer(batch, points if tracker.frames_seen == 0 else None)
            shape_warmup = time.perf_counter() - begin
            tracker.reset()
            begin = time.perf_counter()
            for batch in chunks(frames, size):
                xy, logits, timing = tracker.infer(batch, points if tracker.frames_seen == 0 else None)
                positions.append(xy)
                visibility.append(logits)
                for key in times:
                    times[key] += timing[key]
        except tracker.torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": size, "status": "out_of_memory"})
            tracker.reset()
            tracker.torch.cuda.empty_cache()
            save()
            print(json.dumps(rows[-1]), flush=True)
            continue
        elapsed = time.perf_counter() - begin
        xy, logits = np.concatenate(positions), np.concatenate(visibility)
        if reference is None:
            reference = (xy, logits)
        error = np.linalg.norm(xy - reference[0], axis=-1)
        row = {"batch_size": size, "status": "ok", "frames": len(xy), **times,
               "shape_warmup_seconds": shape_warmup,
               "wall_seconds": elapsed, "wall_fps": len(xy) / elapsed,
               "inference_fps": len(xy) / times["inference_seconds"],
               "max_xy_difference_px": float(error.max()),
               "mean_xy_difference_px": float(error.mean()),
               "max_visibility_logit_difference": float(np.abs(logits - reference[1]).max()),
               "visible_disagreements": int(np.count_nonzero((logits > 0) != (reference[1] > 0))),
               "final_state_step": tracker.state.step}
        if tracker.device.type == "cuda":
            row["peak_allocated_bytes"] = tracker.torch.cuda.max_memory_allocated(tracker.device)
        rows.append(row)
        save()
        print(json.dumps(row), flush=True)


def main(argv=None):
    p = parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(argv)
    try:
        resolve_session_args(args, argv)
        if args.command == "track":
            track(args)
        elif args.command in ("annotate", "process"):
            folder_command(args)
        elif args.command == "render":
            from .render import render_tracks
            print(json.dumps(render_tracks(args.tracks, args.output, fps=args.fps,
                                           max_width=args.max_width, force=args.force,
                                           render_options=rendering_configuration(args),
                                           reacquire_seconds=args.reacquire_seconds), indent=2))
        elif args.command == 'reindex':
            from .reindex import reindex_saved_points
            print(json.dumps(reindex_saved_points(args.data_dir,dry_run=args.dry_run),indent=2))
        elif args.command == "clean":
            from .cleanup_tracks import cleanup_saved_tracks
            print(json.dumps(cleanup_saved_tracks(saved_track_files(args.master),
                reacquire_seconds=args.reacquire_seconds, dry_run=args.dry_run), indent=2))
        elif args.command == "refine":
            refine_command(args)
        elif args.command in ('render-named', 'export-ready'):
            from .publication import render_named_tracks, export_ready_tracks
            if args.command == 'render-named':
                result = render_named_tracks(args.master, args.name, args.output_dir, fps=args.fps,
                    max_width=args.max_width, render_options=rendering_configuration(args), progress=print,
                    reacquire_seconds=args.reacquire_seconds)
            else:
                result = export_ready_tracks(args.master, args.name, args.output_dir, progress=print)
            print(json.dumps(result, indent=2))
        elif args.command == "benchmark":
            benchmark(args)
        elif args.command == "migrate":
            from .storage import migrate_legacy_exports
            print(json.dumps({"moved": migrate_legacy_exports()}, indent=2))
        else:
            tracker, elapsed = load_tracker(args)
            print(json.dumps({**tracker.environment(), "load_seconds": elapsed}, indent=2))
    except Exception as error:
        p.exit(1, f"Point tracking failed: {error}\n")


if __name__ == "__main__":
    main()
