"""Small, model-free smoke checks for the shared-tool point patch modal."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking.patch_editor import (
    PatchSession, PatchViewportTransform, PointPatchEditor, _FrameProcessor,
    _PatchController, edit_patch_mask, point_circle_mask, smallest_mask_circle,
)
from video_annotation.rgb_tracking.workflow import Cancelled


class PointPatchEditorSmokeTests(unittest.TestCase):
    def test_missing_sam_stays_manual_and_one_brush_handles_both_buttons(self):
        with patch('video_annotation.rgb_tracking.patch_editor.importlib.util.find_spec', return_value=None):
            session = PatchSession()
        self.assertFalse(session.sam_available)
        self.assertIn('Python environment', session.unavailable_reason)
        self.assertIsNone(session._processor)

        class Value:
            def __init__(self, value): self.value = value
            def get(self): return self.value
            def set(self, value): self.value = value

        editor = PointPatchEditor.__new__(PointPatchEditor)
        editor.controller = _PatchController(np.zeros((40, 70, 3), np.uint8), (40, 20), 8, None)
        editor.tool_var, editor.brush_var, editor.label_var = Value('Brush'), Value('1'), Value(1)
        editor.transform = PatchViewportTransform(70, 40, 700, 400, (40, 20), 10)
        editor.brush_mask = editor.brush_last = editor.drag_start = editor.drag_end = None
        editor.region_cache_ready = False
        editor.preview_masks = {}
        editor.preview_info = SimpleNamespace(configure=lambda **kwargs: None)
        editor.sync_motion_controls = editor.redraw = lambda: None
        xy = editor.transform.to_canvas(40, 20)
        event = SimpleNamespace(x=xy[0], y=xy[1])
        editor.right_press(event)
        editor.release(event)
        self.assertFalse(editor.controller.region_mask()[20, 40])
        editor.press(event)
        editor.release(event)
        self.assertTrue(editor.controller.region_mask()[20, 40])
        self.assertEqual(editor.tool_var.get(), 'Brush')
        editor.tool_var.set('Clicks')
        editor.press(event)
        editor.right_press(event)
        self.assertEqual(editor.controller.prompt['point_labels'], [1, 0])
        session.close()

    def test_session_reuses_one_native_model_across_source_frames(self):
        session = PatchSession()
        first = np.zeros((40, 70, 3), np.uint8)
        processor = session.bind_image(first)
        seen, constructed = [], []

        class NativePreview:
            def preview(self, frame, prompt, **kwargs):
                seen.append(frame)
                return SimpleNamespace(masks={7: np.ones(frame.shape[:2], bool)})
            def close(self): seen.append('closed')

        def factory(*args, **kwargs):
            constructed.append(True)
            return NativePreview()

        processor.factory = factory
        processor._identity = lambda: ['test-only', None]
        processor.options['cleanup_area'] = 0
        for frame in (first, np.zeros((50, 90, 3), np.uint8)):
            self.assertIs(session.bind_image(frame), processor)
            controller = _PatchController(frame, (40, 20), 8, None)
            processor.preview(None, controller.current_reference())
            self.assertIs(seen[-1], frame)
        self.assertEqual(len(constructed), 1)
        self.assertEqual(processor.options['cleanup_area'], 0)
        session.close()
        session.close()
        self.assertEqual(sum(isinstance(item, str) and item == 'closed' for item in seen), 1)

    def test_session_close_during_preview_defers_release_once(self):
        session = PatchSession()
        frame = np.zeros((40, 70, 3), np.uint8)
        processor = session.bind_image(frame)
        released = []

        class NativePreview:
            def preview(self, image, prompt, **kwargs):
                session.close()
                self.assert_not_released = not released
                return SimpleNamespace(masks={7: np.ones(image.shape[:2], bool)})
            def close(self): released.append(True)

        native = NativePreview()
        processor.factory = lambda *args, **kwargs: native
        processor._identity = lambda: ['test-only', None]
        with self.assertRaises(Cancelled):
            processor.preview(None, _PatchController(frame, (40, 20), 8, None).current_reference())
        self.assertTrue(native.assert_not_released)
        self.assertEqual(released, [True])
        self.assertIsNone(processor.tracker)
        session.close()
        self.assertEqual(released, [True])

    def test_zoomed_crop_maps_back_to_source_pixels(self):
        for zoom in (.3, 1., 16.):
            transform = PatchViewportTransform(3840, 2160, 960, 650, (2800, 1400), zoom)
            point = [transform.x0+transform.width/2, transform.y0+transform.height/2]
            np.testing.assert_allclose(transform.to_source(*transform.to_canvas(*point)), point)
            self.assertIsNone(transform.to_source(-1, -1))
            if zoom == 16:
                self.assertLess(transform.width, 100)

    def test_manual_circle_support_and_shared_undo(self):
        image = np.zeros((40, 70, 3), np.uint8)
        controller = _PatchController(image, (40.5, 20.5), 8, None)
        circle = point_circle_mask(image.shape[:2], (40.5, 20.5), 8)
        np.testing.assert_array_equal(controller.region_mask(), circle)
        cut = circle.copy()
        cut[20:23, 40:43] = False
        controller.edit({"mask": cut})
        controller.undo()
        np.testing.assert_array_equal(controller.region_mask(), circle)
        controller.edit({"mask": np.ones(image.shape[:2], bool)})
        self.assertTrue(controller.region_mask().all())  # Not clipped until Apply.

    def test_enclosing_circle_includes_foreground_pixel_corners(self):
        mask = np.zeros((80, 100), bool)
        mask[10:21, 55:78] = True
        center, radius = smallest_mask_circle(mask)
        np.testing.assert_allclose(center, (66., 15.), atol=1e-5)
        self.assertEqual(radius, 13)
        corners = np.array([[54.5, 9.5], [77.5, 20.5]])
        self.assertTrue(np.all(np.linalg.norm(corners-center, axis=1) <= radius))
        mask[:] = False
        mask[0, 0] = True
        center, radius = smallest_mask_circle(mask)
        np.testing.assert_allclose(center, (0., 0.), atol=1e-5)
        self.assertEqual(radius, 4)
        with self.assertRaisesRegex(ValueError, "nonempty mask"):
            smallest_mask_circle(np.zeros_like(mask))

    def test_adopted_preview_keeps_outside_pixels_for_fit_and_brush_revision(self):
        editor = PointPatchEditor.__new__(PointPatchEditor)
        image = np.zeros((80, 100, 3), np.uint8)
        editor.controller = _PatchController(image, (20., 20.), 8, None)
        mask = np.zeros(image.shape[:2], bool)
        mask[10:21, 55:78] = True  # Entirely outside the old circle.
        editor.preview_masks, editor.preview_ids = {7: mask.copy()}, [7]
        editor.preview_list = SimpleNamespace(curselection=lambda: [0])
        editor.tool_var = SimpleNamespace(set=lambda value: None)
        editor.status = SimpleNamespace(set=lambda value: None)
        editor.edit = editor.controller.edit
        editor._prompt_ready = editor._update_patch_title = editor.center_patch = lambda: None
        editor.use_preview()
        np.testing.assert_array_equal(editor.controller.region_mask(), mask)
        edited = mask.copy()
        edited[16, 77] = False
        editor.controller.edit({"mask": edited})
        editor.preview_masks = {}
        editor.fit_mask_circle()
        self.assertGreater(editor.point_xy[0], 50)
        self.assertTrue(editor.controller.circle[edited].all())
        editor._finish_close = lambda: None
        editor.save()
        np.testing.assert_array_equal(editor.result, edited)

    def test_geometry_return_is_opt_in_and_default_apply_still_clips(self):
        editor = PointPatchEditor.__new__(PointPatchEditor)
        image = np.zeros((40, 70, 3), np.uint8)
        editor.point_xy, editor.radius = (40.5, 20.5), 8
        editor.controller = _PatchController(image, editor.point_xy, editor.radius, np.ones(image.shape[:2], bool))
        editor.preview_masks = {}
        editor._prompt_ready = editor._finish_close = lambda: None
        editor.save()
        np.testing.assert_array_equal(editor.result, editor.controller.circle)
        editor.root = SimpleNamespace(grab_set=lambda: None)
        parent = SimpleNamespace(wait_window=lambda window: None)
        with patch('video_annotation.rgb_tracking.patch_editor.PointPatchEditor', return_value=editor):
            self.assertIs(edit_patch_mask(parent, image, editor.point_xy, editor.radius), editor.result)
            result = edit_patch_mask(parent, image, editor.point_xy, editor.radius, return_geometry=True)
        self.assertIs(result.mask, editor.result)
        self.assertEqual(result.point_xy, editor.point_xy)
        self.assertEqual(result.radius, 8)

    def test_preview_uses_full_frame_shared_cleanup_and_lazy_model(self):
        image = np.zeros((40, 70, 3), np.uint8)
        controller = _PatchController(image, (40, 20), 8, None)
        controller.edit({"points_xy": [[40., 20.]], "point_labels": [1]})
        processor = _FrameProcessor(image)
        self.assertIsNone(processor.tracker)
        result_mask = np.zeros(image.shape[:2], bool)
        result_mask[15:25, 35:45] = True
        result_mask[1, 1] = True
        seen = []

        class NativePreview:
            def preview(self, frame, prompt, **kwargs):
                seen.append((frame, prompt))
                return SimpleNamespace(masks={7: result_mask.copy()})

            def close(self):
                seen.append("closed")

        processor.factory = lambda *args, **kwargs: NativePreview()
        processor._identity = lambda: ["test-only", None]
        result = processor.preview(None, controller.current_reference())
        self.assertIs(seen[0][0], image)
        self.assertEqual(seen[0][1]["points_xy"], [[40., 20.]])
        self.assertEqual(result.masks[7].shape, image.shape[:2])
        self.assertFalse(result.masks[7][1, 1])
        self.assertTrue(result.masks[7][20, 40])
        processor.close()
        self.assertEqual(seen[-1], "closed")
        self.assertIsNone(processor.tracker)


if __name__ == "__main__":
    unittest.main()
