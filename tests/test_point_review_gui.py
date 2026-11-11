"""Focused model-free checks for review loss display and source-time policy."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch,PropertyMock

import numpy as np

from video_annotation.rgb_tracking.review_gui import TrackReview


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class PointReviewLossTests(unittest.TestCase):
    def test_render_unchanged_predictions_keeps_loaded_pose_file_without_manual_save(self):
        from pathlib import Path
        from video_annotation.rgb_tracking.tapnextpp import ROOT
        review = TrackReview.__new__(TrackReview)
        tracks = ROOT/'data/pseudo-ground-truths/point_tracks/example/points.csv'
        review.model = SimpleNamespace(input_tracks=tracks, output=None)
        review.require_model = lambda: review.model
        review.last_saved_path, review.show_pose = tracks, Value(True)
        review.owner = SimpleNamespace(render_options={'pose_alpha':.4}, overlay_fps=120, max_width=1280)
        review.save = Mock(side_effect=AssertionError('rendering must not discard saved poses'))
        review.launch_output, review.events = Mock(), Mock()
        review.cancel = SimpleNamespace(is_set=lambda: False)
        with patch.object(TrackReview, 'dirty', new_callable=PropertyMock, return_value=False), \
             patch('video_annotation.rgb_tracking.render.render_tracks') as render:
            review.render()
            review.launch_output.call_args.args[0]()
        self.assertEqual(render.call_args.args[0], Path(tracks))
        self.assertTrue(render.call_args.kwargs['render_options']['show_pose'])
        review.save.assert_not_called()

    def test_render_snapshot_uses_review_pose_switch_without_mutating_global_settings(self):
        review = TrackReview.__new__(TrackReview)
        review.owner = SimpleNamespace(render_options={'pose_alpha':.4, 'show_pose':False})
        review.show_pose = Value(True)
        self.assertEqual(review.render_snapshot(), {'pose_alpha':.4, 'show_pose':True})
        self.assertFalse(review.owner.render_options['show_pose'])
        review.show_pose.set(False)
        self.assertFalse(review.render_snapshot()['show_pose'])

    def review_canvas(self, visible, original_visible=None):
        review = TrackReview.__new__(TrackReview)
        count = len(visible)
        xy = np.array([[[20.+frame, 20.]] for frame in range(count)])
        original = np.ones((count, 1), bool) if original_visible is None else np.asarray(original_visible, bool)[:, None]
        review.model = SimpleNamespace(width=100, height=80, frames=np.arange(count), point_ids=[7],
            original_xy=xy.copy(), original_visible=np.ones_like(original), original_lifecycle_visible=original)
        review.preview_data = xy, np.asarray(visible, bool)[:, None], np.full((count, 1), 'manual')
        review.cache = SimpleNamespace(get=lambda frame: np.zeros((80, 100, 3), np.uint8))
        review.frame_index, review.closed, review.drag_xy = count-1, False, None
        review.point_var, review.trail_var = Value('7'), Value('30')
        review.only_selected, review.show_trail = Value(False), Value(True)
        review.canvas = Mock()
        review.canvas.winfo_width.return_value = 100
        review.canvas.winfo_height.return_value = 80
        review.Image = SimpleNamespace(fromarray=lambda array: SimpleNamespace(resize=lambda *args: object()),
                                       Resampling=SimpleNamespace(BILINEAR=0))
        review.ImageTk = SimpleNamespace(PhotoImage=lambda *args, **kwargs: object())
        return review

    def test_ended_points_hide_native_ghost_markers_labels_and_trails(self):
        review = self.review_canvas([True, True, False, False])
        review.redraw()
        review.canvas.create_oval.assert_not_called()
        review.canvas.create_line.assert_not_called()
        self.assertEqual(review.canvas.create_text.call_count, 1)  # Top status only.

    def test_recovered_point_trails_start_after_latest_loss_and_native_stays_ended(self):
        review = self.review_canvas([True, True, False, True, True], [True, True, False, False, False])
        review.redraw()
        self.assertEqual(review.canvas.create_oval.call_count, 1)  # Amber manual recovery only.
        self.assertEqual(review.canvas.create_line.call_count, 1)  # Frames 3 -> 4 only.
        line = review.canvas.create_line.call_args.args
        np.testing.assert_allclose(line, [23., 20., 24., 20.])

    def test_hidden_point_can_preview_a_new_recovery_anchor(self):
        review = self.review_canvas([True, False, False])
        review.drag_xy = [40., 35.]
        review.redraw()
        self.assertEqual(review.canvas.create_oval.call_count, 1)
        review.canvas.create_line.assert_not_called()
        self.assertEqual(review.canvas.create_text.call_count, 2)

    def test_grace_applies_to_model_and_future_runs_without_overlay_fps(self):
        review = TrackReview.__new__(TrackReview)
        model = SimpleNamespace(reacquire_seconds=.5, dirty=False, tail='keep', join_frame=None)
        model.set_reacquire_seconds = Mock(side_effect=lambda value: setattr(model, 'reacquire_seconds', value))
        review.model, review.require_model = model, lambda: model
        review.reacquire_var, review.status = Value('0.25'), Value('')
        review.owner = SimpleNamespace(processor=SimpleNamespace(reacquire_seconds=.5), overlay_fps=720)
        review.refresh_preview = Mock()
        review.tail_options = lambda: ('keep', None)
        self.assertTrue(review.dirty)
        review.apply_reacquisition_policy()
        model.set_reacquire_seconds.assert_called_once_with(.25)
        self.assertEqual(review.owner.processor.reacquire_seconds, .25)
        self.assertFalse(review.dirty)
        review.refresh_preview.assert_called_once_with()
        for value in ('bad', '-1', 'nan', 'inf'):
            review.reacquire_var.set(value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite nonnegative'):
                review.apply_reacquisition_policy()

    def test_optional_pose_uses_shared_renderer_keeps_centers_and_source_cache(self):
        review=self.review_canvas([True,True])
        image=np.zeros((80,100,3),np.uint8)
        review.cache.get=lambda frame:image
        row=dict(x=21.,y=20.,visible=1,valid=1,homography_valid=1)
        review.pose_data=dict(supports={7:dict(radius=12)},rows_by_frame={1:{7:row}})
        review.show_pose=Value(True)
        review.owner=SimpleNamespace(render_options={'pose_alpha':.7})
        with patch('video_annotation.pose.overlay.draw_point_pose_rows',create=True) as draw:
            draw.side_effect=lambda frame,*args,**kwargs:frame.fill(255)
            review.redraw()
        self.assertEqual(review.canvas.create_oval.call_count,2)  # Native + manual centers stay present.
        self.assertEqual(draw.call_args.args[1],{7:row})
        self.assertFalse(np.any(image))
        self.assertFalse(draw.call_args.kwargs['render_options']['pose_labels'])
        review.preview_data[0][1,0]=[30.,20.]
        self.assertEqual(review.pose_preview_rows(1),{})  # A moved manual center cannot inherit its old H.
        review.preview_data[0][1,0]=[21.,20.]
        review.drag_xy=[23.,20.]
        self.assertEqual(review.pose_preview_rows(1),{})


if __name__ == '__main__':
    unittest.main()
