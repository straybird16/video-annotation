"""Standalone command entry points; model libraries load only when requested."""
import sys

from .paths import configure_cli


def main(argv=None):
    args = configure_cli(argv)
    if args and args[0] == "masks":
        return masks(args[1:])
    if args and args[0] == "points":
        args = args[1:]
    from .rgb_tracking.__main__ import main as run
    return run(args)


def masks(argv=None):
    args = configure_cli(argv)
    from .mask_tracking.__main__ import main as run
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
