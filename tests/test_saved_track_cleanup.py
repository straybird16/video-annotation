"""Tiny CPU-only lifetime cleanup checks; no real dataset, video decode or model."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking.cleanup_tracks import cleanup_saved_tracks
from video_annotation.rgb_tracking.manual_tracks import TrackEdits
from video_annotation.rgb_tracking.render import point_groups
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE
from video_annotation.rgb_tracking.workflow import file_digest


class SavedTrackCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT/'data')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.video = self.folder/'source.mp4'; self.video.write_bytes(b'source identity only')
        self.tracks = self.folder/'points.csv'
        visible = [[1,1],[0,0],[0,1],[0,1],[0,0],[1,1],[1,0]]
        with self.tracks.open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['frame','point_id','x','y','visible','quality_score'])
            for frame, row in enumerate(visible):
                for point, good in enumerate(row):
                    writer.writerow([frame,point,20+point*20+frame,20+point*20,good,1 if good else -1])
        self.quality = self.folder/'points.quality.csv'
        with self.quality.open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['frame','point_id','valid','motion_model','registration_status','reason'])
            for frame,row in enumerate(visible):
                for point,good in enumerate(row):
                    writer.writerow([frame,point,good,'similarity','not_applicable',''])
        self.metadata = dict(kind='rgb_point_tracks',tracker='klt-template',source_video='source.mp4',
            coordinate_space=DISPLAY_COORDINATE_SPACE,display_rotation_degrees=0,source_size_px=[100,100],
            source_file_stat=[self.video.stat().st_size,self.video.stat().st_mtime_ns],source_metadata_fps=10.,
            query_frame=0,frames=7,point_ids=[0,1],query_points_xy=[[20,20],[40,40]],
            quality_file=self.quality.name,quality_sha256=file_digest(self.quality),
            tracks_sha256=file_digest(self.tracks),tracks_bytes=self.tracks.stat().st_size,
            quality_summary={'valid_point_rows':7,'total_point_rows':14})
        self.tracks.with_suffix('.json').write_text(json.dumps(self.metadata))
        history = patch('video_annotation.rgb_tracking.manual_tracks.archive_files')
        history.start(); self.addCleanup(history.stop)

    def test_legacy_cleanup_quality_backups_dry_run_and_idempotence(self):
        before = self.tracks.read_bytes(),self.tracks.with_suffix('.json').read_bytes(),self.quality.read_bytes()
        preview=cleanup_saved_tracks([self.tracks],reacquire_seconds=.1,dry_run=True)
        self.assertEqual(preview['removed_rows'],7)
        self.assertEqual(preview['suppressed_rows'],2)
        self.assertEqual(self.tracks.read_bytes(),before[0])
        self.assertFalse(list(self.folder.glob('*.pre-cleanup*')))
        result=cleanup_saved_tracks([self.tracks],reacquire_seconds=.1)
        self.assertEqual(result['changed_files'],1)
        self.assertEqual(len(result['files'][0]['backups']),3)
        metadata=json.loads(self.tracks.with_suffix('.json').read_text())
        self.assertEqual(metadata['track_policy']['point_end_frames'],{'0':0,'1':5})
        self.assertEqual(metadata['frames'],7)
        self.assertEqual(metadata['point_ids'],[0,1])
        self.assertEqual(metadata['quality_summary']['valid_point_rows'],5)
        self.assertEqual(metadata['quality_summary']['total_point_rows'],7)
        self.assertEqual(metadata['quality_sha256'],file_digest(self.quality))
        with self.tracks.open() as stream: rows=list(csv.DictReader(stream))
        with self.quality.open() as stream: quality=list(csv.DictReader(stream))
        self.assertEqual([(r['frame'],r['point_id'],r['visible']) for r in rows],
                         [(r['frame'],r['point_id'],r['valid']) for r in quality])
        self.assertTrue(all((float(r['quality_score'])>0)==(r['visible']=='1') for r in rows))
        for path,old in zip((self.tracks,self.tracks.with_suffix('.json'),self.quality),before):
            self.assertEqual(path.with_name(path.name+'.pre-cleanup').read_bytes(),old)
        repeat=cleanup_saved_tracks([self.tracks],reacquire_seconds=.1)
        self.assertEqual(repeat['changed_files'],0)
        self.assertEqual(len(list(self.folder.glob('*.pre-cleanup*'))),3)

    def test_clean_original_and_corrected_rebase_verified_hashes_and_reopen(self):
        model=TrackEdits(self.tracks,reacquire_seconds=100.)
        model.set_anchor(0,5,60,60)
        corrected=model.save(self.folder/'corrected/points.csv')
        original=corrected.read_bytes()
        result=cleanup_saved_tracks([corrected,self.tracks],reacquire_seconds=.1)
        self.assertEqual([Path(row['path']) for row in result['files']],[self.tracks,corrected])
        metadata=json.loads(corrected.with_suffix('.json').read_text())
        self.assertEqual(metadata['manual_edits']['base_tracks_sha256'],file_digest(self.tracks))
        self.assertEqual(metadata['manual_edits']['base_metadata_sha256'],file_digest(self.tracks.with_suffix('.json')))
        self.assertEqual(metadata['manual_edits']['anchors']['0'][0]['frame'],5)
        self.assertEqual(metadata['track_policy']['point_end_frames']['0'],5)
        self.assertEqual(corrected.with_name(corrected.name+'.pre-cleanup').read_bytes(),original)
        reopened=TrackEdits(corrected)
        xy,visible,_=reopened.preview(reopened.tail,reopened.join_frame)
        groups=list(point_groups(corrected,[0,1],metadata))
        stored_visible=np.array([[points[p][1] for p in [0,1]] for _,points in groups])
        stored_xy=np.array([[points[p][0] for p in [0,1]] for _,points in groups])
        np.testing.assert_array_equal(visible,stored_visible)
        np.testing.assert_allclose(xy[visible],stored_xy[visible])
        self.assertEqual(reopened.reacquire_seconds,.1)
        self.assertEqual(cleanup_saved_tracks([corrected,self.tracks],reacquire_seconds=.1)['changed_files'],0)
        metadata['manual_edits']['base_tracks_sha256']='unverified'
        corrected.with_suffix('.json').write_text(json.dumps(metadata))
        before=self.tracks.read_bytes()
        with self.assertRaisesRegex(ValueError,'unverified provenance'):
            cleanup_saved_tracks([corrected,self.tracks],reacquire_seconds=0)
        self.assertEqual(self.tracks.read_bytes(),before)

    def test_manual_preview_save_final_cleanup_zero_and_human_restart(self):
        model=TrackEdits(self.tracks)
        self.assertFalse(model.has_saved_policy)
        self.assertEqual(model.source_fps,10.)
        model.set_reacquire_seconds(0)
        self.assertTrue(model.dirty)
        self.assertFalse(model.preview()[1][5,0])
        self.assertFalse(model.original_lifecycle_visible[5,0])
        model.set_anchor(0,5,60,60)
        self.assertTrue(model.preview()[1][5,0])
        self.assertFalse(model.original_lifecycle_visible[5,0])
        output=model.save(tail='stop')
        metadata=json.loads(output.with_suffix('.json').read_text())
        self.assertEqual(metadata['track_policy']['reacquire_seconds'],0.)
        self.assertEqual(metadata['track_policy']['point_end_frames'],{'0':5,'1':0})
        with output.open() as stream: rows=list(csv.DictReader(stream))
        self.assertEqual(max(int(row['frame']) for row in rows if row['point_id']=='0'),5)
        self.assertFalse(model.dirty)
        restored=TrackEdits(output)
        self.assertTrue(restored.has_saved_policy)
        self.assertEqual(restored.reacquire_seconds,0.)
        restored.set_reacquire_seconds(.25)
        self.assertTrue(restored.dirty)
        with self.assertRaises(ValueError): restored.set_reacquire_seconds(float('nan'))
        self.assertEqual(restored.reacquire_seconds,.25)


if __name__=='__main__':
    unittest.main()
