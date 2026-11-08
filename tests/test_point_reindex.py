"""Model-free ID relabeling checks, including linked manual/pose provenance."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking.annotations import PointController
from video_annotation.rgb_tracking.manual_tracks import TrackEdits
from video_annotation.rgb_tracking.reindex import reindex_saved_points
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE
from video_annotation.rgb_tracking.workflow import file_digest


class PointReindexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT/'data')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.video = self.folder/'source.mp4'
        self.video.write_bytes(b'identity only; no video/model decode needed')

    def fixture(self):
        tracks = self.folder/'points.csv'
        tracks.write_text('frame,point_id,x,y,visible,quality_score\n0,1,20.0,21.5,1,0.9\n0,7,40,41,1,0.8\n1,1,21,22,1,0.7\n1,7,41,42,1,0.6\n')
        quality = self.folder/'quality.csv'
        quality.write_text('frame,point_id,valid,score\n0,1,1,0.9\n0,7,1,0.8\n1,1,1,0.7\n1,7,1,0.6\n')
        metadata = dict(kind='rgb_point_tracks', tracker='klt-template', source_video=self.video.name,
            coordinate_space=DISPLAY_COORDINATE_SPACE, source_size_px=[100,100], source_metadata_fps=30.,
            source_file_stat=[self.video.stat().st_size,self.video.stat().st_mtime_ns],
            query_frame=0, frames=2, point_ids=[1,7], query_points_xy=[[20,21.5],[40,41]],
            track_policy={'point_end_frames':{'1':1,'7':1}}, input_signature='old opaque signature',
            quality_file=quality.name, quality_sha256=file_digest(quality), tracks_sha256=file_digest(tracks))
        tracks.with_suffix('.json').write_text(json.dumps(metadata))
        with patch('video_annotation.rgb_tracking.manual_tracks.archive_files'):
            model=TrackEdits(tracks)
            model.set_anchor(7,1,44,45)
            corrected=model.save(self.folder/'corrected/points.csv')
        reference = self.folder/'untracked.json'
        reference.write_text(json.dumps({'kind':'rgb_point_reference','source':self.video.name}))
        reference.with_suffix('.csv').write_text('frame,point_id,x,y,radius,mask\n0,4,15.0,16,20,\n0,9,17,18,21,\n')
        pose_root=self.folder/'refined';pose_root.mkdir()
        artifacts={}
        for identity in (1,7):
            folder=pose_root/f'point_{identity}';folder.mkdir()
            csv_path=folder/'pose.csv'
            csv_path.write_text(f'frame,point_id,object_id,scale,rotation_deg\n0,{identity},{identity},1.20,30.0\n')
            (folder/'pose.raw.csv').write_bytes(csv_path.read_bytes())
            value=dict(kind='rgb_point_pose',point_id=identity,object_id=identity,point_tracks='../../points.csv',
                csv_sha256=file_digest(csv_path),raw_csv='pose.raw.csv',raw_csv_sha256=file_digest(folder/'pose.raw.csv'))
            (folder/'pose.json').write_text(json.dumps(value))
            for name in ('pose.csv','pose.json','pose.raw.csv'):
                artifacts[f'point_{identity}/{name}']=file_digest(folder/name)
        (pose_root/'refinement.json').write_text(json.dumps(dict(kind='point_pose_refinement',
            point_tracks='../points.csv',point_ids=[7],input_signature='old pose signature',artifacts=artifacts)))
        return tracks,corrected,reference,pose_root

    def test_cancel_and_delete_reuse_ids_and_preserve_patches(self):
        controller=PointController([self.video],reader=lambda *_:np.zeros((100,100,3),np.uint8))
        controller.add([20,20]);controller.add([40,40]);controller.undo();controller.add([50,50])
        self.assertEqual(controller.point_ids,[0,1])
        controller.set_patches([1],radius=23)
        controller.add([70,70]);controller.remove(0)
        self.assertEqual(controller.point_ids,[0,1])
        self.assertEqual(controller.patches[0]['radius'],23)
        self.assertEqual(controller.points[0],[50,50])
        controller.undo()
        self.assertEqual(controller.point_ids,[0,1,2])
        self.assertEqual(controller.patches[1]['radius'],23)

    def test_migrate_reference_manual_quality_and_collision_safe_pose_names(self):
        tracks,corrected,reference,pose_root=self.fixture()
        before={p:p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        preview=reindex_saved_points(self.folder,dry_run=True)
        self.assertEqual((preview['reference_sets'],preview['track_sets'],preview['pose_sets']),(1,2,2))
        self.assertTrue(all(p.read_bytes()==content for p,content in before.items()))
        result=reindex_saved_points(self.folder)
        self.assertEqual(len(result['backups']),result['changed_files'])
        self.assertTrue(all(Path(p).is_file() for p in result['backups']))
        metadata=json.loads(corrected.with_suffix('.json').read_text())
        self.assertEqual(metadata['point_ids'],[0,1])
        self.assertEqual(metadata['manual_edits']['base_tracks_sha256'],file_digest(tracks))
        self.assertEqual(metadata['manual_edits']['base_metadata_sha256'],file_digest(tracks.with_suffix('.json')))
        self.assertIn('1',metadata['manual_edits']['anchors'])
        self.assertEqual(metadata['track_policy']['point_end_frames'],{'0':1,'1':1})
        self.assertEqual(TrackEdits(corrected).point_ids,[0,1])
        for filename in ('points.csv','quality.csv','untracked.csv'):
            path=self.folder/filename
            old=list(csv.DictReader(io.StringIO(before[path].decode())))
            new=list(csv.DictReader(io.StringIO(path.read_text())))
            self.assertEqual([{k:v for k,v in r.items() if k!='point_id'} for r in old],
                             [{k:v for k,v in r.items() if k!='point_id'} for r in new])
        self.assertEqual([int(r['point_id']) for r in csv.DictReader(io.StringIO(reference.with_suffix('.csv').read_text()))],[0,1])
        manifest=json.loads((pose_root/'refinement.json').read_text())
        self.assertEqual(manifest['point_ids'],[1]) # Selected subset uses the full track domain.
        for relative,digest in manifest['artifacts'].items():self.assertEqual(file_digest(pose_root/relative),digest)
        for new,old in ((0,1),(1,7)):
            pose=pose_root/f'point_{new}'/'pose.json'
            value=json.loads(pose.read_text())
            self.assertEqual(value['point_id'],new)
            self.assertEqual(value['id_reindex']['previous_id'],old)
            self.assertEqual(value['csv_sha256'],file_digest(pose.with_suffix('.csv')))
        self.assertEqual(reindex_saved_points(self.folder,dry_run=True)['changed_files'],0)

    def test_stale_dependency_aborts_before_any_mutation(self):
        tracks,_,_,_=self.fixture()
        tracks.write_text(tracks.read_text()+'\n')
        before={p:p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError,'dependency changed'):
            reindex_saved_points(self.folder)
        self.assertTrue(all(p.read_bytes()==content for p,content in before.items()))
        self.assertFalse(list(self.folder.rglob('*.pre-reindex*')))


if __name__=='__main__':unittest.main()
