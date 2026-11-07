"""Label object-mask prompts, track saved ROIs with SAM, and render MP4 previews."""
from ..paths import configure_cli
if __name__ == "__main__":
    import sys
    sys.argv[1:] = configure_cli()

import argparse
import json
import math
from pathlib import Path


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def parser():
    root = argparse.ArgumentParser(description=__doc__, epilog="Locations: --workspace DIR (default: current directory); --models-root DIR (optional model checkouts/weights). Relative file arguments use the current directory.")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (("annotate", "label a video folder, preview segmentation and optionally track"),
                            ("process", "track all saved ROI references without the GUI")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--source-dir", type=Path, required=True)
        command.add_argument("--tracker", choices=("sam21", "sam31"), default="sam21")
        command.add_argument("--tracker-options", type=Path, help="JSON options object; unknown/unsupported settings fail")
        command.add_argument("--initialization", choices=("infer", "exact"), help="infer object outline (default); exact drawn region requires SAM2.1")
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--model-config", help="SAM2.1 model config override")
        command.add_argument("--device", help="default cuda; SAM3.1 currently requires CUDA")
        command.add_argument("--precision", choices=("bfloat16", "float16", "float32"), help="SAM3.1 currently supports bfloat16 only")
        command.add_argument("--state-device", choices=("cpu", "cuda"), help="history storage; SAM3.1 defaults to cpu, SAM2.1 CUDA defaults to cuda")
        command.add_argument("--objects-per-pass", type=nonnegative, help="SAM3.1 ROI limit per full-video pass; default one ROI for 256+ frames, all for shorter clips; 0 forces all")
        command.add_argument("--ram-reserve-gb", type=nonnegative_float, help="stop below this available RAM in GiB (default 2; 0 disables)")
        command.add_argument("--vram-reserve-gb", type=nonnegative_float, help="stop below this free VRAM in GiB (default 1; 0 disables)")
        cleanup = command.add_mutually_exclusive_group()
        cleanup.add_argument("--cleanup-area", type=nonnegative, help="remove foreground components up to this source-pixel area; preserve holes (default 8; 0 disables)")
        cleanup.add_argument("--no-cleanup", dest="cleanup_area", action="store_const", const=0, help="disable small foreground-component cleanup")
        command.add_argument("--max-frames", type=positive, help="bounded trial; limits prepared source frames")
        command.add_argument("--direction", choices=("forward", "both"), default="forward")
        command.add_argument("--separate", action="store_true", help="run each ROI separately; default SAM3.1 tracks selected saved regions together")
        command.add_argument("--no-recursive", action="store_true")
        command.add_argument("--overlay", action="store_true", help="write H.264 MP4; enables tracking after Save & next")
        command.add_argument("--overlay-fps", type=positive_float, help="preview viewing FPS; does not alter source labels")
        command.add_argument("--max-width", type=positive, default=1280)
    annotate = commands.choices["annotate"]
    annotate.add_argument("--frame", type=nonnegative, default=0, help="zero-based default frame for unsaved ROIs")
    annotate.add_argument("--track", action="store_true", help="process all saved ROIs before advancing")
    process = commands.choices["process"]
    process.add_argument("--roi", action="append", help="only these ROI IDs (repeatable); default all saved ROIs")
    process.add_argument("--force", action="store_true", help="recompute and preserve prior results in history")
    process.add_argument("--skip-existing", action="store_true", help="reuse complete predictions with matching source, reference and frame window despite changed options/code")
    process.add_argument("--rerender-overlays", action="store_true", help="replace MP4 overlays, including for reused predictions; implies --overlay")
    cleanup = commands.add_parser("cleanup", help="clean completed mask predictions and update geometry without loading SAM")
    cleanup.add_argument("--source-dir", type=Path, required=True)
    cleanup.add_argument("--cleanup-area", type=nonnegative, default=8, help="maximum foreground-component area in source pixels (default 8; 0 disables)")
    render = commands.add_parser("render", help="render saved masks without loading SAM")
    render_source = render.add_mutually_exclusive_group(required=True)
    render_source.add_argument("--tracks", type=Path, nargs="+", help="one or more completed mask track directories for the same video")
    render_source.add_argument("--source-dir", type=Path, help="RGB source folder; recursively combine saved ROIs into one overlay per video")
    render.add_argument("--tracker", choices=("sam21", "sam31"), default="sam31", help="tracker for folder mode (default sam31)")
    render.add_argument("--output", type=Path, help="portable MP4 for --tracks, under outputs/")
    render.add_argument("--output-dir", type=Path, help="folder-mode output root (default outputs/videos/rgb_masks)")
    render.add_argument("--fps", type=positive_float)
    render.add_argument("--max-width", type=positive, default=1280)
    render.add_argument("--force", action="store_true")
    return root


def tracker_options(args):
    options = {}
    if args.tracker_options:
        options = json.loads(args.tracker_options.read_text())
        if not isinstance(options, dict):
            raise ValueError("--tracker-options must contain a JSON object")
    for field in ("initialization", "checkpoint", "model_config", "device", "precision", "state_device", "objects_per_pass", "ram_reserve_gb", "vram_reserve_gb", "cleanup_area"):
        value = getattr(args, field)
        if value is not None:
            options[field] = str(value) if isinstance(value, Path) else value
    return options


def folder_command(args):
    from .annotations import MaskController, discover_videos, saved_references
    from .workflow import Processor
    processor = Processor(backend=args.tracker, options=tracker_options(args),
                          max_frames=args.max_frames, direction=args.direction, source_root=args.source_dir)
    skipped = []
    videos = discover_videos(args.source_dir, not args.no_recursive, skipped)
    for video, reason in skipped:
        print(f"Skipped {video}: {reason}", flush=True)
    if not videos:
        raise ValueError("No valid source videos found")
    print(f"Found {len(videos)} source videos. Saved ROIs reload automatically.", flush=True)
    if args.command == "annotate":
        from .labeler import MaskLabeler
        MaskLabeler(MaskController(videos, args.frame, source_root=args.source_dir), processor,
                    track=args.track or args.overlay, overlay=args.overlay,
                    overlay_fps=args.overlay_fps, max_width=args.max_width, joint=not args.separate).run()
    else:
        for video in videos:
            references = saved_references(video, source_root=args.source_dir)
            selected = list(dict.fromkeys(args.roi or references))
            if not selected:
                print(f"Unlabeled; skipped: {video}", flush=True)
            valid = []
            for identity in selected:
                if identity not in references:
                    print(f"ROI {identity} not saved; skipped: {video}", flush=True)
                    continue
                valid.append(identity)
            if args.tracker == "sam31" and not args.separate and len(valid) > 1:
                result = processor.process_many(video,valid,overlay=args.overlay or args.rerender_overlays,
                    overlay_fps=args.overlay_fps,max_width=args.max_width,force=args.force,
                    skip_existing=args.skip_existing,force_overlay=args.rerender_overlays,
                    progress=lambda message:print(message,flush=True))
                print(json.dumps(result,default=str),flush=True)
                continue
            for identity in valid:
                result = processor.process(video, roi_id=identity, overlay=args.overlay or args.rerender_overlays,
                    overlay_fps=args.overlay_fps, max_width=args.max_width, force=args.force,
                    skip_existing=args.skip_existing, force_overlay=args.rerender_overlays,
                    progress=lambda message: print(message, flush=True))
                print(json.dumps(result, default=str), flush=True)


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "cleanup":
            from .cleanup import cleanup_existing
            result = cleanup_existing(args.source_dir, area=args.cleanup_area, progress=print)
            if result is not None:
                print(json.dumps(result, default=str, indent=2))
        elif args.command == "render":
            from .render import render_masks, render_folder
            if args.source_dir:
                if args.output:
                    raise ValueError("Use --output-dir with --source-dir; filenames are automatic")
                result = render_folder(args.source_dir, tracker=args.tracker, output_dir=args.output_dir,
                                       fps=args.fps, max_width=args.max_width, force=args.force,
                                       progress=lambda message: print(message, flush=True))
            else:
                if not args.output or args.output_dir:
                    raise ValueError("Use --output FILE.mp4 with --tracks")
                result = render_masks(args.tracks, args.output, fps=args.fps,
                                      max_width=args.max_width, force=args.force)
            print(json.dumps(result, default=str, indent=2))
        else:
            folder_command(args)
    except Exception as error:
        root.exit(1, f"Mask tracking failed: {error}\n")


if __name__ == "__main__":
    main()
