"""Tiny synthetic correlation and support-adaptation checks; no models or GPU."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.common.geometry import mask_geometry
from video_annotation.pose.registration import rotation
from video_annotation.pose.workflow import refine_point_tracks
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE
from video_annotation.rgb_tracking.workflow import Cancelled, file_digest


class PointPoseTests(unittest.TestCase):
    def setUp(self):
        data=tempfile.TemporaryDirectory(dir=ROOT/'data'); output=tempfile.TemporaryDirectory(dir=ROOT/'outputs')
        self.addCleanup(data.cleanup); self.addCleanup(output.cleanup)
        self.folder,self.output=Path(data.name),Path(output.name)
        self.video=self.folder/'source.mp4'; self.video.write_bytes(b'synthetic source identity')
        self.source=dict(width=128,height=128,fps=200.,number_of_frames=210,display_rotation_degrees=0)
        self.tracks=self.folder/'points.csv'

    def write_tracks(self,rows,patches,queries,ids=(7,),path=None):
        path=path or self.tracks; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['frame','point_id','x','y','visible','quality_score'])
            for offset,points in enumerate(rows):
                for identity,(xy,visible) in zip(ids,points):
                    writer.writerow([200+offset,identity,*xy,int(visible),1 if visible else -1])
        metadata=dict(kind='rgb_point_tracks',tracker='klt-template',coordinate_space=DISPLAY_COORDINATE_SPACE,
            source_video=os.path.relpath(self.video,path.parent),source_size_px=[128,128],source_metadata_fps=200.,
            source_file_stat=[self.video.stat().st_size,self.video.stat().st_mtime_ns],display_rotation_degrees=0,
            query_frame=200,frames=len(rows),point_ids=list(ids),query_points_xy=queries,query_patches=patches)
        path.with_suffix('.json').write_text(json.dumps(metadata))
        return metadata

    def run_refinement(self,images,**options):
        options.setdefault("method","similarity")
        options.setdefault("method_options",dict(center_search=False))
        with patch('video_annotation.pose.workflow.probe_video',return_value=(self.source,{})), \
             patch('video_annotation.pose.workflow.video_frames',side_effect=lambda *args,**kwargs:iter(images)) as decode:
            result=refine_point_tracks(self.tracks,**options)
        return result,decode

    def test_real_correlation_small_translation_rotation_scale_and_reuse(self):
        rng=np.random.default_rng(4)
        texture=cv2.GaussianBlur(rng.uniform(15,240,(128,128)).astype(np.float32),(0,0),1)
        base=np.rint(texture).astype(np.uint8)
        centers=[np.array([64.,64.])+[i*.6,-i*.3] for i in range(3)]
        images=[]
        for i,center in enumerate(centers):
            matrix=rotation(i*3.)*(1+i*.025)
            affine=np.c_[matrix,center-matrix@np.array([64.,64.])]
            gray=cv2.warpAffine(base,affine,(128,128),flags=cv2.INTER_LINEAR)
            images.append(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))
        self.write_tracks([[(center,True)] for center in centers],[{'radius':25}],[[64,64]])
        before_threads=cv2.getNumThreads()
        result,decode=self.run_refinement(images,overlay=False,
            config={'angles':128,'radii':32,'patch_size':64,'rotation_smoothness':0.,'scale_smoothness':0.})
        self.assertEqual(cv2.getNumThreads(),before_threads)
        decode.assert_called_once()
        pose=Path(result['poses']['7'])
        with pose.open() as stream: rows=list(csv.DictReader(stream))
        self.assertEqual([int(row['frame']) for row in rows],[200,201,202])
        self.assertTrue(all(int(row['valid']) for row in rows))
        self.assertAlmostEqual(float(rows[-1]['rotation_degrees']),6.,delta=1.2)
        self.assertAlmostEqual(float(rows[-1]['scale']),1.05,delta=.04)
        self.assertAlmostEqual(float(rows[-1]['circle_radius']),25*float(rows[-1]['scale']),places=6)
        self.assertEqual(rows[-1]['geometry_source'],'point_patch_circle')
        metadata=json.loads(pose.with_suffix('.json').read_text())
        self.assertEqual(metadata['config']['tracker_scale_weight'],0.)
        self.assertEqual(metadata['config']['registration_scale_weight'],1.)
        self.assertEqual(metadata['reference_frame'],200)
        self.assertEqual(metadata['config']['radius_blend'],0.)
        self.assertNotIn('outputs/',pose.with_suffix('.json').read_text())
        self.assertGreater(result['timing']['refinement_seconds'],0)
        reused,_=self.run_refinement([],overlay=False,
            config={'angles':128,'radii':32,'patch_size':64,'rotation_smoothness':0.,'scale_smoothness':0.})
        self.assertTrue(reused['reused'])
        self.assertEqual(reused['timing']['refinement_seconds'],result['timing']['refinement_seconds'])
        self.assertEqual(reused['current_run_timing']['refinement_seconds'],0.)

    def test_corrected_mask_offset_transport_gaps_ended_ids_and_overlay_replay(self):
        mask=np.zeros((128,128),np.uint8);mask[55:78,68:91]=255
        cv2.imwrite(str(self.folder/'support.png'),mask)
        points=[np.array([64.+i,64.]) for i in range(6)]
        visibility=[False,True,True,False,True,False]
        rows=[[(point,good),([30.,30.],False)] for point,good in zip(points,visibility)]
        metadata=self.write_tracks(rows,[{'radius':32,'mask':'support.png'},{'radius':12}],[[64,64],[30,30]],ids=(7,42))
        corrected=self.folder/'corrected/points.csv';corrected.parent.mkdir()
        corrected.write_bytes(self.tracks.read_bytes())
        corrected_metadata={**metadata,'source_video':'../source.mp4','tracker':'manual','manual_edits':{
            'base_tracks':'../points.csv','base_tracks_sha256':file_digest(self.tracks),
            'base_metadata_sha256':file_digest(self.tracks.with_suffix('.json')),'anchors':{}}}
        corrected_metadata.pop('query_patches')
        corrected.with_suffix('.json').write_text(json.dumps(corrected_metadata))
        original=self.tracks;self.tracks=corrected
        calls=[]
        class Estimator:
            def __init__(self,gray,box,config,mask=None):
                calls.append(('init',mask.copy()))
                geometry=mask_geometry(mask)
                self.center=np.array([geometry['bbox_x']+geometry['bbox_w']/2,geometry['bbox_y']+geometry['bbox_h']/2])
                self.offset=np.array(box[:2])+np.array(box[2:])/2-self.center
            def row(self,center,angle=0.,scale=1.):
                return dict(valid=1,x=center[0],y=center[1],scale=scale,similarity_scale=scale,
                    rotation_degrees=angle,rotation_valid=int(angle!=20),rotation_reason='low_observability' if angle==20 else 'ok',
                    registration_scale=scale,scale_target=scale,template_updated=0)
            def initial_result(self):return self.row(self.center)
            def estimate(self,gray,box,frame,mask=None):
                calls.append((frame,mask.copy()))
                angle,scale=(10.,1.1) if frame==1 else (20.,1.2)
                point=np.array(box[:2])+np.array(box[2:])/2
                return self.row(point-scale*rotation(angle)@self.offset,angle,scale)
            def invalid_result(self,box,frame,reason):
                calls.append(('invalid',frame));return dict(valid=0,reason=reason)
        class Writer:
            def __init__(self,path,*args,**kwargs):Path(path).write_bytes(b'preview fixture')
            def write(self,frame):pass
            def close(self):pass
        overlay=self.output/'pose.mp4'
        images=[np.zeros((128,128,3),np.uint8) for _ in range(5)]
        from video_annotation.pose import workflow
        with patch.object(workflow,'PoseEstimator',Estimator),patch.object(workflow,'H264Writer',Writer), \
             patch('video_annotation.pose.overlay.circle_geometry',wraps=workflow.circle_geometry) as geometry:
            result,decode=self.run_refinement(images,overlay=True,overlay_path=overlay)
        self.assertEqual(result['frames'],5)
        self.assertEqual(decode.call_args.args[1:],(200,5))
        self.assertEqual([call[0] for call in calls],['init',1,'invalid',3])
        with Path(result['poses']['7']).open() as stream: output=list(csv.DictReader(stream))
        self.assertEqual([int(row['frame']) for row in output],list(range(200,205)))
        self.assertEqual([int(row['valid']) for row in output],[0,1,1,0,1])
        self.assertEqual([row['reference_frame'] for row in output[1:]],['201']*4)
        np.testing.assert_allclose([float(output[-1]['x']),float(output[-1]['y'])],points[4])
        self.assertAlmostEqual(float(output[-1]['circle_radius']),38.4)
        self.assertIsNone(geometry.call_args_list[-1].args[2])
        # Last visible mask uses the preceding accepted similarity across the gap.
        initial=mask>0
        yy,xx=np.indices(initial.shape)
        initial &= (xx-64)**2+(yy-64)**2<=32**2
        matrix=1.1*rotation(10.)
        expected=cv2.warpAffine(initial.astype(np.uint8),np.c_[matrix,points[4]-matrix@np.array([64.,64.])],(128,128),flags=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(calls[-1][1],expected)
        self.assertEqual(len(Path(result['poses']['42']).read_text().splitlines()),1)
        self.assertEqual(json.loads(Path(result['poses']['7']).with_suffix('.json').read_text())['reference_frame'],201)
        overlay.unlink();overlay.with_suffix('.json').unlink()
        with patch.object(workflow,'PoseEstimator',side_effect=AssertionError('must reuse saved poses')),patch.object(workflow,'H264Writer',Writer):
            replay,_=self.run_refinement(images,overlay=True,overlay_path=overlay)
        self.assertTrue(replay['reused'])
        self.assertTrue(overlay.is_file())
        self.assertEqual(replay['current_run_timing']['refinement_seconds'],0.)
        self.assertEqual(replay['timing_source'],'recorded refinement')
        self.tracks=original

    def test_cancel_restores_threads_without_partial_pose_export(self):
        self.write_tracks([[([64.,64.],True)]],[{'radius':24}],[[64,64]])
        counter=iter([False,True])
        before=cv2.getNumThreads()
        with self.assertRaises(Cancelled):
            self.run_refinement([np.zeros((128,128,3),np.uint8)],overlay=False,cancel=lambda:next(counter))
        self.assertEqual(cv2.getNumThreads(),before)
        self.assertFalse((self.folder/'refined-correlation').exists())


if __name__=='__main__':unittest.main()
