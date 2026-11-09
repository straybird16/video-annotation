"""Small source/output coordinate and readable-label checks, without video/models."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose import overlay


def row(**changes):
    value=dict(valid=1, rotation_valid=1, rotation_degrees=30., circle_x=960., circle_y=540.,
               circle_radius=120., x=960., y=540., scale=1.2)
    value.update(changes)
    return value


class PoseOverlayTests(unittest.TestCase):
    def test_source_geometry_scales_before_drawing_but_text_stays_full_size(self):
        image = np.full((180, 320, 3), 100, np.uint8)
        with patch.object(overlay, 'draw_tracks', wraps=overlay.draw_tracks) as draw, \
             patch.object(overlay.cv2, 'putText', wraps=cv2.putText) as text, \
             patch.object(overlay.cv2, 'rectangle', wraps=cv2.rectangle) as boxes:
            result = overlay.draw_pose_overlay(image, {1:row()}, 42, source_size=(1920,1080))
        self.assertIs(result, image)
        geometry = draw.call_args.args[1][0]
        np.testing.assert_allclose(geometry['outline'].mean(axis=0), [160,90], atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(geometry['outline'][0]-[160,90]), 20.)
        self.assertTrue(all(call.args[4] == .4 for call in text.call_args_list))
        self.assertTrue(all(call.kwargs['alpha'] == 1. for call in draw.call_args_list))
        self.assertTrue(all(call.kwargs['width'] == 1 and call.kwargs['halo_width'] == 0 for call in draw.call_args_list))
        self.assertTrue(all(call.args[6] in (1, 2) for call in text.call_args_list))
        self.assertTrue(any(call.args[5] == overlay.COLORS[1] and call.args[6] == 1 for call in text.call_args_list))
        boxes.assert_not_called()

    def test_edge_labels_clamp_and_invalid_rotation_still_draws_circle(self):
        image = np.full((180, 320, 3), 160, np.uint8)
        edge = row(); edge.update(x=1919., y=1079., circle_x=1919., circle_y=1079., rotation_valid=0,
                                  rotation_reason='low_texture', center_scale_interpolated=1)
        with patch.object(overlay.cv2, 'putText', wraps=cv2.putText) as text:
            overlay.draw_pose_overlay(image, {1:edge, 2:{'valid':0}}, 120, source_size=(1920,1080))
        self.assertGreaterEqual(len(text.call_args_list), 2)
        for call in text.call_args_list:
            x, y = call.args[2]
            width, height = cv2.getTextSize(call.args[1], call.args[3], call.args[4], 1)[0]
            self.assertTrue(0 <= x and x+width <= 320)
            self.assertTrue(height <= y < 180)
        with self.assertRaises(ValueError):
            overlay.draw_pose_overlay(image, {}, 0, source_size=(0,1080))

    def test_default_thin_style_blends_all_markers_once_without_halo_or_background(self):
        base=np.full((180,320,3),100,np.uint8)
        poses={1:row(),2:row(circle_x=480.,circle_y=540.,x=480.,y=540.)}
        opaque=base.copy()
        overlay.draw_pose_overlay(opaque,poses,1,source_size=(1920,1080),
                                  render_options={'pose_labels':False,'pose_alpha':1.})
        translucent=base.copy()
        with patch.object(overlay.cv2,'addWeighted',wraps=cv2.addWeighted) as blend, \
             patch.object(overlay.cv2,'polylines',wraps=cv2.polylines) as lines, \
             patch.object(overlay.cv2,'putText',wraps=cv2.putText) as text:
            overlay.draw_pose_overlay(translucent,poses,1,source_size=(1920,1080),
                                      render_options={'pose_labels':False})
        self.assertEqual(blend.call_count,1)
        text.assert_not_called()
        self.assertTrue(all(call.args[4]==1 for call in lines.call_args_list))
        self.assertTrue(all(call.args[3]!=(0,0,0) for call in lines.call_args_list))
        np.testing.assert_array_equal(translucent,cv2.addWeighted(opaque,.7,base,.3,0))
        # Pixels outside geometry are not filled by a mask or background panel.
        np.testing.assert_array_equal(translucent[:20,:20],base[:20,:20])

    def test_configurable_labels_font_and_halo_preserve_shared_renderer_default(self):
        from video_annotation.common.pose_rendering import circle_geometry,draw_tracks
        image=np.full((180,320,3),100,np.uint8)
        with patch.object(overlay.cv2,'putText',wraps=cv2.putText) as text, \
             patch.object(overlay,'draw_tracks',wraps=draw_tracks) as draw:
            overlay.draw_pose_overlay(image,{1:row()},1,source_size=(1920,1080),
                render_options={'pose_line_width':2,'pose_alpha':1.,'pose_halo':True,'pose_font_scale':.7})
        self.assertTrue(all(call.args[4]==.7 for call in text.call_args_list))
        self.assertTrue(any(call.args[6]==2 and call.args[5]==(0,0,0) for call in text.call_args_list))
        self.assertEqual(draw.call_args.kwargs['halo_width'],2)
        geometry=circle_geometry([160,90],20,-60)
        default=np.full_like(image,100);explicit=default.copy()
        draw_tracks(default,[geometry],[overlay.COLORS[1]])
        draw_tracks(explicit,[geometry],[overlay.COLORS[1]],width=2,alpha=.75,halo_width=2)
        np.testing.assert_array_equal(default,explicit)


if __name__ == '__main__':
    unittest.main()
