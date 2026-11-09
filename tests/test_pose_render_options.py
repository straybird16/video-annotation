"""Small pose appearance validation and render-only cache replay checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.pose import workflow
from video_annotation.rgb_tracking.render_options import normalize_render_options
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


class PoseRenderOptionsTests(unittest.TestCase):
    def test_pose_defaults_leave_generic_render_values_unchanged(self):
        options=normalize_render_options()
        for key,value in dict(crf=18,preset='fast',max_frames=None,trails=False,trail_length=30,
                              pose_line_width=1,pose_alpha=.7,pose_halo=False,
                              pose_labels=True,pose_font_scale=.4).items():
            self.assertEqual(options[key],value)
        self.assertEqual(normalize_render_options({'pose_alpha':1})['pose_alpha'],1.)
        for values in ({'pose_line_width':0},{'pose_line_width':1.5},{'pose_line_width':True},
                       {'pose_alpha':0},{'pose_alpha':1.1},{'pose_alpha':float('nan')},
                       {'pose_alpha':True},{'pose_halo':1},{'pose_labels':'false'},
                       {'pose_font_scale':0},{'pose_font_scale':3.1},
                       {'pose_font_scale':float('inf')},{'pose_font_scale':False}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                normalize_render_options(values)

    def test_appearance_changes_replay_cached_poses_and_reach_inline_and_saved_draws(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as data, \
             tempfile.TemporaryDirectory(dir=ROOT/'outputs') as output:
            folder=Path(data);video=folder/'source.mp4';video.write_bytes(b'fixture')
            tracks=folder/'points.csv'
            tracks.write_text('frame,point_id,x,y,visible,quality_score\n0,0,32,32,1,1\n1,0,32,32,1,1\n')
            metadata=dict(kind='rgb_point_tracks',tracker='klt-template',coordinate_space=DISPLAY_COORDINATE_SPACE,
                source_video='source.mp4',source_size_px=[64,64],source_metadata_fps=100.,
                source_file_stat=[video.stat().st_size,video.stat().st_mtime_ns],display_rotation_degrees=0,
                frames=2,query_frame=0,point_ids=[0],query_points_xy=[[32,32]],query_patches=[{'radius':20}])
            tracks.with_suffix('.json').write_text(json.dumps(metadata))
            source=dict(width=64,height=64,fps=100.,number_of_frames=2,display_rotation_degrees=0)
            class Estimator:
                def __init__(self,gray,box,config,mask=None):self.center=np.array([32.,32.])
                def initial_result(self):return self.result()
                def estimate(self,*args,**kwargs):return self.result()
                def result(self):
                    return dict(valid=1,x=32.,y=32.,scale=1.,similarity_scale=1.,
                        rotation_degrees=0.,rotation_unwrapped_degrees=0.,rotation_valid=1,template_updated=0)
            class Writer:
                def __init__(self,path,*args,**kwargs):Path(path).write_bytes(b'overlay')
                def write(self,frame):pass
                def close(self):pass
            overlay=Path(output)/'pose.mp4'
            def decode(*args,**kwargs):return iter([np.full((64,64,3),100,np.uint8) for _ in range(2)])
            common=dict(method='similarity',method_options={'center_search':False},overlay_path=overlay,gap_seconds=0.)
            with patch.object(workflow,'probe_video',return_value=(source,{})), \
                 patch.object(workflow,'video_frames',side_effect=decode), \
                 patch.object(workflow,'H264Writer',Writer), \
                 patch.object(workflow,'PoseEstimator',Estimator), \
                 patch.object(workflow,'draw_pose_overlay',wraps=workflow.draw_pose_overlay) as draw:
                first=workflow.refine_point_tracks(tracks,render_options={'pose_alpha':.6},**common)
                self.assertEqual(draw.call_count,2)
                self.assertTrue(all(call.kwargs['render_options']['pose_alpha']==.6 for call in draw.call_args_list))
            pose=Path(first['poses']['0']);saved=pose.read_bytes()
            manifest=Path(first['output'])/'refinement.json'
            signature=json.loads(manifest.read_text())['input_signature']
            with patch.object(workflow,'probe_video',return_value=(source,{})), \
                 patch.object(workflow,'video_frames',side_effect=decode), \
                 patch.object(workflow,'H264Writer',Writer), \
                 patch.object(workflow,'create_estimator',side_effect=AssertionError('appearance changes must not refit')), \
                 patch.object(workflow,'draw_pose_overlay',wraps=workflow.draw_pose_overlay) as draw:
                replay=workflow.refine_point_tracks(tracks,render_options={'pose_alpha':.9,'pose_labels':False},**common)
                self.assertTrue(replay['reused'])
                self.assertEqual(draw.call_count,2)
                self.assertTrue(all(call.kwargs['render_options']['pose_alpha']==.9 for call in draw.call_args_list))
                self.assertTrue(all(not call.kwargs['render_options']['pose_labels'] for call in draw.call_args_list))
            self.assertEqual(pose.read_bytes(),saved)
            self.assertEqual(json.loads(manifest.read_text())['input_signature'],signature)
            sidecar=json.loads(overlay.with_suffix('.json').read_text())
            self.assertEqual(sidecar['settings']['renderer_version'],4)
            self.assertFalse(sidecar['settings']['render_options']['pose_labels'])


if __name__=='__main__':unittest.main()
