"""Portable point-GUI settings and short parameter help; no model imports."""
import json
import math
import os
from pathlib import Path
import re

from .tapnextpp import ROOT, contained_path

PARAMETER_GUIDE = Path(__file__).resolve().parents[1]/'guides/KLT_PARAMETERS.md'

PARAMETER_HELP = {
    'tracking_policy': 'Combined (default) adds full sparse-RANSAC proposals, soft acceleration-continuity regularization and separate template admission. Accuracy fits the immutable original patch densely every frame, with conditional homography and accepted recent fallback. Fast retains the earlier sparse-KLT/correlation cascade. Only applies to correlation-mode klt-template.',
    'combined_motion_weight': 'Maximum soft center/shape prior cost in normalized correlation units; default 0.03, matching correlation_margin. Larger image improvements win. Zero disables regularization.',
    'template_stable_frames': 'Consecutive accepted, motion-consistent full-pose fits before recent-template admission in combined mode; default 3. Center-only fallbacks never refresh appearance.',
    'output_smoothing_seconds': 'Optional degree-2 Savitzky-Golay center filter after tracking/lifetime cleanup. Source seconds; 0 disables (default). Keeps raw_x/raw_y, gaps, endings and anchors; translates complete H.',
    'dense_iterations': 'Maximum inverse-compositional iterations per dense local fit, from 1 to 100; default 12. More iterations may improve convergence but cost more time.',
    'dense_original_threshold': 'Masked NCC below this tries recent appearance and an independent original-shape seed, in (0,1]; default 0.93. Scores at or above it count as strong original support; min_patch_ncc remains the acceptance floor.',
    'dense_recent_threshold': 'Minimum masked NCC for an accepted recent-template dense fallback, in (0,1]; default 0.94. Recent appearance is updated only after accepted geometry.',
    'dense_homography_trigger': 'Try a supported homography when dense affine NCC is below this value, in (0,1]; default 0.965. A larger value tries perspective fitting more often.',
    'recovery_enabled': 'Allow expensive shape/center recovery after a local fit fails. Default off keeps the fast local path. Definite occlusion never triggers a recovery search.',
    'stop_on_occlusion': 'End a patch track when foreground evidence indicates definite occlusion; default on. This is distinct from ordinary uncertain fits and the source-time reacquisition grace.',
    'min_foreground_fraction': 'Minimum fraction of original foreground evidence required by the occlusion check, in (0,1]; default 0.5. Lower values permit more partial foreground.',
    'center_smoothing': 'Fast-policy only: reduce small accepted center jitter using source-pixel motion uncertainty. Accuracy retains the dense subpixel optimum and uses the motion prior only as a fit seed.',
    'center_smoothing_px': 'Source-pixel range used to smooth small center jitter; default 2 pixels. Larger values smooth more, independently of overlay resolution or FPS.',
    'reacquire_seconds': 'Maximum source-video seconds after first loss for native tracker recovery. Default 0.5; zero ends immediately. Lost points are hidden and unrecovered tails are omitted from new predictions.',
    'motion_model': 'Patch geometry: adaptive (default) upgrades only with evidence; fixed similarity, affine or homography are also available.',
    'motion_gate': 'Reject candidate pose jumps that disagree with recent accepted source-time motion and uncertainty. Rejected candidates cannot update the template or motion history.',
    'motion_sigma': 'Temporal innovation limit in uncertainty standard deviations; default 4. Larger values tolerate larger motion surprises.',
    'motion_noise_px': 'Source-pixel uncertainty floor for the motion gate; default 1 pixel. Increase for noisier measurements.',
    'motion_memory_seconds': 'Source-time adaptation and prediction memory; default 0.1 second. Independent of overlay FPS and lost-point lifetime.',
    'center_trust': 'Soft supplied/predicted-center prior in [0,1] when ranking image candidates. Larger values favor candidates nearer that center; accepted geometry is never blended afterward.',
    'use_recent_template': 'Use accepted recent patch appearance alongside the immutable original. Disable to freeze original appearance; rejected matches never refresh it.',
    'template_update_score': 'Minimum foreground correlation before an otherwise valid accepted fit may refresh recent appearance from raw RGB; default 0.75. Rejected fits never update it.',
    'correlation_top_k': 'Number of distinct high-scoring correlation peaks retained for final candidate validation; default 3. Larger values can recover alternatives but cost more.',
    'correlation_max_shapes': 'Maximum shape hypotheses in the joint correlation-map recovery bank; default 81. Larger budgets add affine/projective candidates and cost more; separate from the legacy seed budget.',
    'search_shapes': 'Joint pose-search shape budget per reference/seed, from 1 to 729. Start with 9 similarity shapes, expand to the default 81 affine shapes, or allow the optional full 729 projective bank. Only bounded top-K peaks receive fine fitting.',
    'adaptive_max_model': 'Most flexible model adaptive fitting may use. Affine is more conservative than homography.',
    'win_size': 'Odd LK window width in source pixels (at least 5). Larger windows need a wider textured mask interior.',
    'max_level': 'Maximum optical-flow pyramid level; flow windows may use image context beyond the selected foreground support.',
    'fb_threshold': 'Maximum forward/backward LK disagreement in source pixels. Lower rejects more uncertain matches.',
    'min_eigenvalue': 'Minimum LK corner strength. Increase to reject weak or flat features.',
    'max_lk_error': 'Maximum OpenCV LK photometric error (8-bit image units). Lower rejects poorer local matches.',
    'patch_radius': 'Default radius in source pixels for newly labeled points; existing per-point radii override it.',
    'min_patch_overlap': 'Minimum fraction of supported patch pixels inside the image after motion; range (0, 1].',
    'min_visible_fraction': 'Minimum fraction of real reference support shared with the current image; default 0.2 permits partial patches at edges/corners.',
    'min_patch_ncc': 'Minimum consecutive-patch normalized correlation, from -1 to 1. Higher is stricter.',
    'min_patch_std': 'Minimum grayscale standard deviation on foreground, with intensities scaled to [0, 1]. Rejects flat patches.',
    'max_displacement': 'Maximum accepted point displacement per frame, in source pixels.',
    'max_features': 'Maximum support corners per patch. More features cost more but can stabilize geometric fitting.',
    'feature_quality': 'Shi–Tomasi corner quality relative to the strongest corner, in (0, 1].',
    'min_feature_distance': 'Minimum spacing between support corners in source pixels.',
    'min_inliers': 'Minimum RANSAC inliers for a patch motion fit; at least 3.',
    'min_inlier_ratio': 'Minimum fraction of matched features consistent with fitted motion.',
    'min_support_coverage': 'Minimum convex-hull area of inlier features divided by the square patch area.',
    'ransac_threshold': 'RANSAC inlier residual threshold in source pixels.',
    'feature_refresh_interval': 'Frames between support-feature refreshes after accepted motion; point IDs never change.',
    'min_scale_step': 'Minimum accepted local linear scale multiplier per frame.',
    'max_scale_step': 'Maximum accepted local linear scale multiplier per frame.',
    'max_rotation_step': 'Maximum absolute local rotation change per frame, in degrees.',
    'max_anisotropy_step': 'Maximum singular-value ratio per update; 1 means equal scale on both axes.',
    'max_shear_step': 'Maximum dimensionless local/domain shear per update.',
    'max_perspective_step': 'Maximum dimensionless projective deformation across one patch per update.',
    'max_corner_extra_step': 'Maximum non-similarity corner displacement per update, divided by patch radius.',
    'min_scale_total': 'Minimum accumulated local linear scale relative to the initial reference.',
    'max_scale_total': 'Maximum accumulated local linear scale relative to the initial reference.',
    'max_anisotropy_total': 'Maximum accumulated singular-value ratio over the original patch.',
    'max_shear_total': 'Maximum accumulated dimensionless shear over the original patch.',
    'max_perspective_total': 'Maximum accumulated dimensionless projective deformation over the original patch.',
    'max_corner_extra_total': 'Maximum accumulated non-similarity corner displacement divided by reference radius.',
    'min_model_features': 'Minimum supporting features before fitting affine motion.',
    'min_projective_features': 'Minimum supporting features before fitting homography motion.',
    'adaptive_min_improvement': 'Required fractional validation-residual improvement before using a more flexible model.',
    'adaptive_min_improvement_px': 'Required absolute validation-residual improvement, in source pixels.',
    'adaptive_complexity_penalty': 'Additional validation penalty per model complexity level, in pixel residual units.',
    'adaptive_registration_min_ncc': 'Original-registration NCC below this triggers one supported higher-model attempt.',
    'adaptive_registration_min_gain': 'NCC improvement required to replace an accepted simpler registration.',
    'template_mode': 'Correlation supports combined, dense accuracy and sparse fast policies. Legacy ECC keeps the former periodic iterative registration route.',
    'correlation_trigger': 'Correlation below this score triggers bounded nearby motion/center search. Higher values search more often.',
    'correlation_margin': 'Required normalized-correlation advantage over a distinct recovery candidate. Larger rejects ambiguous matches.',
    'correlation_search_radius': 'Maximum nearby center search radius in source pixels during recovery; max_displacement remains an acceptance limit.',
    'correlation_search_step': 'Coarse recovery center spacing in source pixels; smaller spacing evaluates more locations.',
    'correlation_pose_step': 'Dimensionless coarse scale/shear/perspective candidate step, bounded by the selected motion model and geometric limits.',
    'correlation_angle_step': 'Coarse recovery rotation candidate spacing in degrees.',
    'correlation_max_candidates': 'Maximum bounded coarse pose proposals per recovery; larger budgets cost more compute.',
    'correlation_refine_iterations': 'Maximum fine alignment iterations for the best recovery proposal, from 1 to 30; default 8.',
    'reanchor_interval': 'Frames between periodic template registration attempts. Lower costs more compute.',
    'registration_min_ncc': 'Minimum reference-to-current normalized correlation to accept a template registration.',
    'registration_min_gain': 'Required NCC improvement over the prealigned patch; 0 permits equal scores within tolerance.',
    'registration_min_overlap': 'Minimum fraction of supported registration/reference pixels inside the image.',
    'registration_max_translation': 'Maximum native-pixel displacement caused by a registration correction.',
    'registration_max_rotation': 'Maximum rotation of a registration correction, in degrees.',
    'registration_max_scale': 'Maximum correction scale multiplier; its reciprocal is the minimum.',
    'registration_max_anisotropy': 'Allowed singular-value ratio minus 1 when converting ECC affine output to similarity.',
    'registration_min_observability': 'Minimum reference orientation observability for ECC; low means weak rotation evidence.',
    'registration_iterations': 'Maximum ECC iterations per registration attempt.',
    'registration_epsilon': 'ECC convergence tolerance. Smaller can take more iterations.',
    'keyframe_interval': 'Minimum frames between candidate keyframes; 0 disables the bank. Promotion needs original-reference confirmation.',
    'max_keyframes': 'Maximum stored references per point, including the immutable original; 1 keeps only the original.',
    'keyframe_min_ncc': 'Minimum successful original-registration NCC before promoting a keyframe.',
    'keyframe_min_inlier_ratio': 'Minimum local inlier ratio before promoting a keyframe.',
    'threads': 'OpenCV CPU worker threads for KLT; Torch CPU threads for TAPNext++.',
    'repo': 'TAPNext++ source checkout. Leave blank to use the installed default.',
    'checkpoint': 'TAPNext++ checkpoint file. Leave blank for the selected resolution default.',
    'resolution': 'TAPNext++ model input side: 256 or 512 pixels; independent of source coordinates.',
    'device': 'TAPNext++ Torch device, such as cuda or cpu.',
    'precision': 'TAPNext++ arithmetic: auto, none, float16 or bfloat16.',
    'compile_model': 'Experimental TAPNext++ compilation; startup and memory costs can increase.',
    'batch_size': 'Consecutive frames per inference call. Blank uses the backend default; does not change source FPS.',
    'max_frames': 'Maximum frames to track after the reference. Blank tracks through the end.',
    'overlay_fps': 'Overlay viewing FPS only; does not change tracking, frame IDs or alignment. Blank uses source FPS.',
    'max_width': 'Maximum overlay width in pixels. The source coordinates and stored tracks retain full resolution.',
}


def parameter_help(name):
    return PARAMETER_HELP.get(name, name.replace('_', ' ').capitalize())


def _portable(path):
    path = Path(path).expanduser().resolve()
    return os.path.relpath(path, ROOT) if path.is_relative_to(ROOT) else str(path)


def _normalize(value):
    from .workflow import Processor
    from .backends import DEFAULT_BACKEND
    if not isinstance(value, dict):
        raise ValueError('Session config must be a JSON object')
    keys = {'kind', 'schema_version', 'source_dir', 'recursive', 'frame', 'tracker',
            'tracker_options', 'tap_options', 'batch_size', 'max_frames', 'overlay_fps', 'max_width',
            'mask_color', 'mask_alpha', 'mask_visible', 'render_options', 'reacquire_seconds'}
    if set(value)-keys:
        raise ValueError(f'Unknown session setting: {", ".join(sorted(set(value)-keys))}')
    if value.get('kind') != 'rgb_point_tracking_session' or value.get('schema_version') != 1:
        raise ValueError('Expected an exported point session config (schema 1); use --tracker-options for a plain KLT options JSON')
    frame, recursive, width, fps = (value.get('frame', 0), value.get('recursive', True),
                                    value.get('max_width', 1280), value.get('overlay_fps'))
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError('Session frame must be a nonnegative integer')
    if not isinstance(recursive, bool) or isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError('Session recursive must be boolean and max_width a positive integer')
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0):
        raise ValueError('Session overlay_fps must be positive or null')
    color, alpha, visible = value.get('mask_color', '#45b8e8'), value.get('mask_alpha', .3), value.get('mask_visible', True)
    if not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
        raise ValueError('Mask color must be #RRGGBB')
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 <= alpha <= 1 or not isinstance(visible, bool)):
        raise ValueError('Mask opacity must be from 0 to 1 and mask_visible must be boolean')
    candidate = Processor(backend=value.get('tracker', DEFAULT_BACKEND), backend_options=value.get('tracker_options'),
                          reacquire_seconds=value.get('reacquire_seconds', .5))
    tap = dict(value.get('tap_options') or {})
    for key in ('repo', 'checkpoint'):
        if tap.get(key):
            tap[key] = ROOT/Path(tap[key]).expanduser()
    candidate.set_runtime_options(tap, batch_size=value.get('batch_size'), max_frames=value.get('max_frames'))
    source = value.get('source_dir')
    if source is not None:
        source = _portable(contained_path(ROOT/Path(source).expanduser(), 'data'))
    return settings_snapshot(candidate, source_dir=source, recursive=recursive,
                             frame=frame, overlay_fps=fps, max_width=width,
                             mask_color=color, mask_alpha=alpha, mask_visible=visible,
                             render_options=value.get('render_options'))


def settings_snapshot(processor, *, overlay_fps=None, max_width=1280,
                      source_dir=None, recursive=True, frame=0,
                      mask_color='#45b8e8', mask_alpha=.3, mask_visible=True,
                      render_options=None):
    from .render_options import normalize_render_options
    options = dict(processor.backend_options)
    if processor.backend != 'tapnextpp':
        from .klt import available_options
        options = {k: v for k, v in options.items() if k in available_options(processor.backend)}
    tap = {key: _portable(value) if key in ('repo', 'checkpoint') and value else value
           for key, value in processor.options.items()}
    return {'kind': 'rgb_point_tracking_session', 'schema_version': 1,
            'source_dir': _portable(ROOT/Path(source_dir)) if source_dir is not None else None,
            'recursive': recursive, 'frame': frame, 'tracker': processor.backend,
            'tracker_options': options, 'tap_options': tap,
            'batch_size': processor._requested_batch_size, 'max_frames': processor.max_frames,
            'reacquire_seconds': getattr(processor, 'reacquire_seconds', .5),
            'overlay_fps': overlay_fps, 'max_width': max_width,
            'mask_color': mask_color, 'mask_alpha': mask_alpha, 'mask_visible': mask_visible,
            'render_options': normalize_render_options(render_options)}


def save_settings(path, value):
    value = _normalize(value)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_settings(path):
    return _normalize(json.loads(Path(path).expanduser().read_text()))


def apply_settings(processor, value):
    """Prevalidate the entire document without loading a tracker, then apply it."""
    value = _normalize(value)
    tap = dict(value['tap_options'])
    for key in ('repo', 'checkpoint'):
        if tap.get(key):
            tap[key] = ROOT/Path(tap[key])
    processor.set_runtime_options({'repo': None, 'checkpoint': None, **tap},
                                  batch_size=value['batch_size'], max_frames=value['max_frames'],
                                  reacquire_seconds=value['reacquire_seconds'])
    processor.set_backend(value['tracker'], value['tracker_options'])
    return value
