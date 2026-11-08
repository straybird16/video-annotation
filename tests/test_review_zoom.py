"""Small source-coordinate checks for shared point/mask review zoom and pan."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_annotation.rgb_tracking.review_gui import TrackReview
from video_annotation.rgb_tracking.viewport import CanvasNavigation
from video_annotation.mask_tracking.review_gui import MaskTrackReview


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def canvas_view(view):
    view.busy = view.closed = False
    view.owner = SimpleNamespace(controller=SimpleNamespace(busy=False))
    view.canvas = Mock()
    view.canvas.winfo_width.return_value = 640
    view.canvas.winfo_height.return_value = 360
    view.cache = SimpleNamespace(get=lambda frame: np.zeros((720, 1280, 3), np.uint8))
    view.samples = []
    def fromarray(array):
        view.samples.append(array.copy())
        return SimpleNamespace(resize=lambda *args: object())
    view.Image = SimpleNamespace(fromarray=fromarray, Resampling=SimpleNamespace(BILINEAR=0))
    view.ImageTk = SimpleNamespace(PhotoImage=lambda *args, **kwargs: object())
    view.navigation = CanvasNavigation(view)


class ReviewZoomTests(unittest.TestCase):
    def test_shared_navigation_keeps_cursor_anchor_pans_fits_and_honors_busy(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        view = SimpleNamespace(model=object(), busy=False, closed=False, canvas=Mock(), drag_xy=[10, 20])
        view.canvas.winfo_width.return_value = 640
        view.canvas.winfo_height.return_value = 360
        navigation = CanvasNavigation(view)
        view.redraw = lambda: setattr(view, 'transform', navigation.transform(image, 'one'))
        view.redraw()
        source = [850., 300.]
        cursor = view.transform.to_canvas(*source)
        navigation.zoom(4, SimpleNamespace(x=cursor[0], y=cursor[1]))
        np.testing.assert_allclose(view.transform.to_source(*cursor), source, atol=1)
        self.assertLess(view.transform.width, 1280)
        center = navigation.viewport.center.copy()
        navigation.start_pan(SimpleNamespace(x=300, y=180))
        self.assertIsNone(view.drag_xy)
        navigation.pan(SimpleNamespace(x=380, y=180))
        self.assertLess(navigation.viewport.center[0], center[0])
        navigation.end_pan()
        self.assertIsNone(navigation.pan_start)
        view.busy = True
        previous = navigation.viewport.zoom
        navigation.zoom(2)
        self.assertEqual(navigation.viewport.zoom, previous)
        view.busy = False
        navigation.fit()
        self.assertEqual(navigation.viewport.zoom, 1.)
        self.assertEqual((view.transform.width, view.transform.height), (1280, 720))

    def test_point_review_crops_image_and_keeps_trails_and_drag_in_source_pixels(self):
        view = TrackReview.__new__(TrackReview)
        xy = np.array([[[880., 500.]], [[900., 500.]]])
        valid = np.ones((2, 1), bool)
        view.model = SimpleNamespace(width=1280, height=720, video=Path('one.mp4'), frames=np.arange(2),
            point_ids=[7], original_xy=xy.copy(), original_visible=valid)
        view.preview_data = xy.copy(), valid, np.full((2, 1), 'original')
        view.frame_index, view.drag_xy = 1, None
        view.point_var, view.trail_var = Value('7'), Value('30')
        view.only_selected, view.show_trail = Value(False), Value(True)
        canvas_view(view)
        view.redraw()
        view.navigation.viewport.zoom = 4
        view.navigation.viewport.center = np.array([900., 500.])
        view.canvas.reset_mock()
        view.redraw()
        self.assertEqual(view.samples[-1].shape, (180, 320, 3))
        expected = (*view.transform.to_canvas(880., 500.), *view.transform.to_canvas(900., 500.))
        np.testing.assert_allclose(view.canvas.create_line.call_args.args, expected)
        view.require_model = lambda: view.model
        view.set_anchor = Mock()
        target = [902., 504.]
        event = SimpleNamespace(x=view.transform.to_canvas(*target)[0], y=view.transform.to_canvas(*target)[1])
        view.press(event)
        view.release(event)
        np.testing.assert_allclose(view.set_anchor.call_args.args[0], target)
        np.testing.assert_array_equal(view.model.original_xy, xy)

    def test_mask_review_crops_overlays_but_editor_receives_full_native_frame(self):
        view = MaskTrackReview.__new__(MaskTrackReview)
        mask = np.zeros((720, 1280), bool); mask[490:510, 890:910] = True
        view.model = SimpleNamespace(width=1280, height=720, video=Path('one.mp4'), frames=np.array([0]),
            anchors={}, masks=Mock(return_value={'1':mask}), set_anchor=Mock())
        view.frame_index = 0
        view.object_var, view.show_others = Value('1'), Value(True)
        view.anchor_frames, view._anchor_labels, view._anchor_list_identity = [], [], None
        view.anchors = Mock(); view.anchors.curselection.return_value = ()
        view.patch_session, view.owns_session = object(), False
        view.set_status = Mock()
        view.require_model = lambda: view.model
        canvas_view(view)
        with patch('video_annotation.mask_tracking.review_gui.draw_masks', side_effect=lambda frame, masks, index: frame.copy()) as draw:
            view.redraw()
            view.navigation.viewport.zoom = 4
            view.navigation.viewport.center = np.array([900., 500.])
            view.redraw()
            self.assertEqual(draw.call_args.args[0].shape, (180, 320, 3))
            np.testing.assert_array_equal(draw.call_args.args[1]['1'], mask[view.transform.slices])
            with patch('video_annotation.rgb_tracking.patch_editor.edit_patch_mask', return_value=mask.copy()) as editor:
                view.edit_mask()
            self.assertEqual(editor.call_args.args[1].shape, (720, 1280, 3))
            self.assertEqual(editor.call_args.args[4].shape, (720, 1280))
            np.testing.assert_array_equal(view.model.set_anchor.call_args.args[2], mask)


if __name__ == '__main__':
    unittest.main()
