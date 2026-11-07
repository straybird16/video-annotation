"""Lazy point-tracker selection; classical CPU tracking never imports Torch."""
BACKENDS = ("klt", "klt-affine", "klt-template", "tapnextpp")
DEFAULT_BACKEND = "klt-template"


def validate_backend_options(name, options=None):
    if name not in BACKENDS:
        raise ValueError(f"unknown point tracker {name!r}; choose one of {', '.join(BACKENDS)}")
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("tracker options must be a JSON object")
    if name == "tapnextpp":
        if options:
            raise ValueError("TAPNext++ uses --resolution, --device, --precision and other TAP flags; --tracker-options configures KLT")
        return {}
    from .klt import validate_options
    try:
        return validate_options(name, options)
    except TypeError as error:
        raise ValueError(f"invalid {name} settings: {error}") from error


def make_tracker(name, options=None, **tap_options):
    options = validate_backend_options(name, options)
    if name == "tapnextpp":
        from .tapnextpp import Tracker
        return Tracker(**tap_options)
    if tap_options:
        raise ValueError("TAPNext++ runtime options cannot configure a KLT tracker")
    from .klt import KLTTracker
    return KLTTracker(algorithm=name, **options)
