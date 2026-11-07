"""Small final-sweep checks: component provenance, source visibility and motion."""
from copy import deepcopy
import math
import unittest

from video_annotation.pose.gaps import interpolate_pose_gaps, validate_gap_options


class PoseGapTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for frame in range(9):
            scale=math.exp(.01*frame);angle=170+3*frame
            rows.append(dict(frame=frame,point_visible=1,valid=1,rotation_valid=1,
                x=100+frame,y=100+.2*frame,scale=scale,similarity_scale=scale,
                circle_radius=30*scale,circle_x=100+frame,circle_y=100+.2*frame,
                rotation_degrees=(angle+180)%360-180,rotation_unwrapped_degrees=angle,
                score=.95,registration_score=.95,rotation_inlier_count=20,
                bbox_x=80+frame,bbox_y=80+.2*frame,bbox_w=40*scale,bbox_h=40*scale,bbox_scale=scale))
        return rows

    def test_bracketed_xy_logscale_rotation_and_measured_provenance(self):
        rows=self.rows()
        for row in rows[3:5]:
            row.update(valid=0,rotation_valid=0,x='',y='',scale='',similarity_scale='',reason='low_score')
        saved=deepcopy(rows)
        processed,summary=interpolate_pose_gaps(rows,fps=100,radius=30)
        self.assertEqual(rows,saved)
        self.assertEqual((summary['center_scale_interpolated'],summary['rotation_interpolated']),(2,2))
        for frame in (3,4):
            row=processed[frame]
            self.assertEqual((row['valid'],row['rotation_valid'],row['fit_valid'],row['rotation_fit_valid']),(1,1,0,0))
            self.assertEqual((row['center_scale_interpolated'],row['rotation_interpolated']),(1,1))
            self.assertAlmostEqual(row['x'],100+frame)
            self.assertAlmostEqual(row['scale'],math.exp(.01*frame))
            self.assertAlmostEqual(row['rotation_unwrapped_degrees'],170+3*frame)
            self.assertEqual(row['score'],'')
            self.assertEqual(row['rotation_inlier_count'],'')

    def test_no_extrapolation_hidden_points_long_gaps_or_motion_discontinuity(self):
        for mode in ('edge','hidden','long','discontinuous'):
            with self.subTest(mode=mode):
                rows=self.rows()
                for row in rows[3:5]: row.update(valid=0,rotation_valid=0)
                options={}
                if mode=='edge':
                    for row in rows[:4]:row.update(valid=0,rotation_valid=0)
                elif mode=='hidden':rows[3]['point_visible']=0
                elif mode=='long':options['gap_seconds']=.005
                else:
                    for row in rows[5:]:
                        row['x']+=100;row['scale']*=2;row['similarity_scale']*=2
                        row['rotation_unwrapped_degrees']+=70
                processed,summary=interpolate_pose_gaps(rows,fps=100,radius=30,**options)
                self.assertEqual(summary['center_scale_interpolated'],0)
                self.assertEqual(summary['rotation_interpolated'],0)
                self.assertFalse(processed[3]['valid'])

    def test_rotation_only_fill_and_off_switch(self):
        rows=self.rows()
        for row in rows[3:5]:row['rotation_valid']=0
        processed,summary=interpolate_pose_gaps(rows,fps=100,radius=30)
        self.assertEqual(summary['center_scale_interpolated'],0)
        self.assertEqual(summary['rotation_interpolated'],2)
        self.assertEqual(processed[3]['fit_valid'],1)
        self.assertEqual(processed[3]['rotation_fit_valid'],0)
        self.assertEqual(processed[3]['x'],rows[3]['x'])
        _,summary=interpolate_pose_gaps(rows,fps=100,radius=30,gap_seconds=0)
        self.assertEqual(summary['rotation_interpolated'],0)
        with self.assertRaises(ValueError):validate_gap_options(.1,1)


if __name__=='__main__':unittest.main()
