"""Explicit workspace and optional model locations, independent of installation."""
import os
from pathlib import Path
import sys


def workspace_root():
    return Path(os.environ.get("VIDEO_ANNOTATION_WORKSPACE", Path.cwd())).expanduser().resolve()


def models_root():
    configured = os.environ.get("VIDEO_ANNOTATION_MODELS")
    if configured:
        return Path(configured).expanduser().resolve()
    existing = workspace_root() / "src/rgb_trackers"
    if existing.is_dir():
        return existing
    return Path.home() / ".cache/video-annotation/models"


def configure_cli(argv=None):
    """Consume location flags before modules bind workspace defaults.

    Flags can appear before or after the command. Other arguments are returned
    unchanged for the existing point/mask parsers. Relative argument paths stay
    relative to the caller's working directory, like ordinary command-line tools.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    remaining = []
    keys = {"--workspace": "VIDEO_ANNOTATION_WORKSPACE", "--models-root": "VIDEO_ANNOTATION_MODELS"}
    index = 0
    while index < len(args):
        arg = args[index]
        flag, sep, value = arg.partition("=")
        if flag not in keys:
            remaining.append(arg)
            index += 1
            continue
        if not sep:
            index += 1
            if index == len(args) or args[index].startswith("--"):
                raise SystemExit(f"{flag} requires a directory")
            value = args[index]
        if not value:
            raise SystemExit(f"{flag} requires a directory")
        os.environ[keys[flag]] = str(Path(value).expanduser().resolve())
        index += 1
    return remaining
