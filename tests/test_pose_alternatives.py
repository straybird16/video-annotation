"""Selectable methods keep annotated support, source geometry and separate caches."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose.methods import METHODS, DEFAULT_METHOD
from video_annotation.pose.workflow import refine_point_tracks
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


class AlternativeWorkflowTests(unittest.TestCase):
    def test_all_methods_source_warps_masked_offset_and_distinct_manifests(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as temporary:
            root=Path(temporary);video=root/'source.mp4';video.write_bytes(b'identity')
            rng=np.random.default_rng(44)
            original=cv2.GaussianBlur(rng.integers(0,256,(128,128),np.uint8),(0,0),.6)
            support=np.zeros((128,128),np.uint8)
            support[40:87,45:92]=255
            cv2.imwrite(str(root/'support.png'),support)
            tracks=root/'points.csv'
            tracks.write_text('frame,point_id,x,y,visible,quality_score\n0,0,64,64,1,1\n1,0,65,64,1,1\n2,0,66,64,1,1\n')
            value=dict(kind='rgb_point_tracks',tracker='tapnextpp',coordinate_space=DISPLAY_COORDINATE_SPACE,
                source_video='source.mp4',source_size_px=[128,128],source_metadata_fps=100.,
                source_file_stat=[video.stat().st_size,video.stat().st_mtime_ns],display_rotation_degrees=0,
                frames=3,query_frame=0,point_ids=[0],query_points_xy=[[64,64]],query_patches=[{'radius':28,'mask':'support.png'}])
            tracks.with_suffix('.json').write_text(json.dumps(value))
            source=dict(width=128,height=128,fps=100.,number_of_frames=3,display_rotation_degrees=0)
            images=[cv2.cvtColor(cv2.warpAffine(original,np.array([[1.,0.,f],[0.,1.,0.]]),(128,128)),cv2.COLOR_GRAY2BGR) for f in range(3)]
            outputs=[]
            for method in METHODS:
                with self.subTest(method=method), patch('video_annotation.pose.workflow.probe_video',return_value=(source,{})), \
                     patch('video_annotation.pose.workflow.video_frames',side_effect=lambda *a,**k:iter(images)):
                    result=refine_point_tracks(tracks,method=method,overlay=False,gap_seconds=0,
                        config={'patch_size':64,'angles':128,'radii':32,'min_score':.3},
                        method_options={'ecc_iters':12,'ic_iters':10})
                outputs.append(result['output'])
                self.assertEqual(result['method'],method)
                pose=Path(result['poses']['0'])
                metadata=json.loads(pose.with_suffix('.json').read_text())
                self.assertEqual(metadata['reference_radius_px'],28.)
                self.assertEqual(metadata['registration_method'],method)
                rows=list(csv.DictReader(io.StringIO(pose.read_text())))
                for row in rows:
                    if row['homography_valid']!='1':continue
                    H=np.array([float(row[f'h{r}{c}']) for r in range(3) for c in range(3)]).reshape(3,3)
                    mapped=H@np.array([64.,64.,1.]);mapped=mapped[:2]/mapped[2]
                    np.testing.assert_allclose([float(row['x']),float(row['y'])],mapped,atol=.02)
                self.assertTrue(any(row['valid']=='1' for row in rows))
                self.assertNotIn('outputs/',pose.with_suffix('.json').read_text())
                with patch('video_annotation.pose.workflow.probe_video',return_value=(source,{})):
                    self.assertTrue(refine_point_tracks(tracks,method=method,overlay=False,gap_seconds=0,
                        config={'patch_size':64,'angles':128,'radii':32,'min_score':.3},
                        method_options={'ecc_iters':12,'ic_iters':10})['reused'])
            self.assertEqual(len(set(outputs)),len(METHODS))
            self.assertEqual(DEFAULT_METHOD,'similarity_ic_homography')


    def test_geometric_gate_uses_query_center_not_off_center_mask_bbox(self):
        from video_annotation.pose.methods import create_estimator
        from video_annotation.pose.registration import Config
        class Seed:
            def __init__(self,*args,**kwargs):
                self.center=np.array([20.,40.]);self.initial_box=np.array([10.,30.,20.,20.])
            def estimate(self,*args,**kwargs):
                return dict(valid=1,x=20.,y=40.,scale=1.,similarity_scale=1.,
                    rotation_degrees=0.,rotation_unwrapped_degrees=0.,rotation_valid=1)
        with patch('video_annotation.pose.geometric.DenseReference') as dense:
            dense.return_value.estimate.return_value=(np.eye(3),dict(score_after=.9,score_gain=.1,runtime=0.,used=1))
            est=create_estimator(np.zeros((100,100),np.uint8),[18.,18.,64.,64.],Config(),
                                 method=DEFAULT_METHOD,base_factory=Seed,options={'use_recent':False})
            row=est.estimate(np.zeros((100,100),np.uint8),[18.,18.,64.,64.],1)
        self.assertEqual(row['valid'],1)
        self.assertEqual(row['candidate_used'],1)
        self.assertEqual(row['x'],20.) # Registration center; workflow maps the selected query separately.


if __name__=='__main__':unittest.main()
