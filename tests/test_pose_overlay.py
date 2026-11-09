"""Small source/output coordinate and readable-label checks, without video/models."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose import overlay


def row(**changes):
    return dict(valid=1, rotation_valid=1, rotation_degrees=30., circle_x=960., circle_y=540.,
                circle_radius=120., x=960., y=540., scale=1.2, **changes)


class PoseOverlayTests(unittest.TestCase):
    def test_source_geometry_scales_before_drawing_but_text_stays_full_size(self):
        image = np.full((180, 320, 3), 100, np.uint8)
        with patch.object(overlay, 'draw_tracks', wraps=overlay.draw_tracks) as draw, \
             patch.object(overlay.cv2, 'putText', wraps=cv2.putText) as text:
            result = overlay.draw_pose_overlay(image, {1:row()}, 42, source_size=(1920,1080))
        self.assertIs(result, image)
        geometry = draw.call_args.args[1][0]
        np.testing.assert_allclose(geometry['outline'].mean(axis=0), [160,90], atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(geometry['outline'][0]-[160,90]), 20.)
        self.assertTrue(all(call.args[4] == .7 for call in text.call_args_list))
        self.assertTrue(all(call.kwargs['alpha'] == 1. for call in draw.call_args_list))
        self.assertGreater(np.count_nonzero(np.all(image == 255, axis=2)), 30)
        self.assertGreater(np.count_nonzero(np.all(image == 18, axis=2)), 100)

    def test_edge_labels_clamp_and_invalid_rotation_still_draws_circle(self):
        image = np.full((180, 320, 3), 160, np.uint8)
        edge = row(); edge.update(x=1919., y=1079., circle_x=1919., circle_y=1079., rotation_valid=0,
                                  rotation_reason='low_texture', center_scale_interpolated=1)
        with patch.object(overlay.cv2, 'rectangle', wraps=cv2.rectangle) as boxes:
            overlay.draw_pose_overlay(image, {1:edge, 2:{'valid':0}}, 120, source_size=(1920,1080))
        self.assertGreaterEqual(len(boxes.call_args_list), 2)
        for call in boxes.call_args_list:
            for x, y in call.args[1:3]:
                self.assertTrue(0 <= x < 320 and 0 <= y < 180)
        with self.assertRaises(ValueError):
            overlay.draw_pose_overlay(image, {}, 0, source_size=(0,1080))


if __name__ == '__main__':
    unittest.main()
