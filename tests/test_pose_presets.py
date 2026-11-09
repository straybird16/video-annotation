"""Portable GUI presets validate before mutation and never save input paths."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from video_annotation.rgb_tracking.overlay_settings import normalize_overlay_settings
from video_annotation.rgb_tracking.pose_gui import PoseRefinementTab, pose_settings_snapshot, normalize_pose_settings
from .test_point_pose_gui import Value, values


def rendering():
    return normalize_overlay_settings(dict(kind='rgb_overlay_settings',schema_version=1,
        overlay_fps=120,max_width=1920,render_options=dict(pose_alpha=.55,pose_labels=False)))


class PosePresetTests(unittest.TestCase):
    def test_round_trip_keeps_refinement_and_overlay_and_omits_source_force(self):
        settings=dict(values(),method='klt_ransac',point_ids='1,3',center_search=True,search_radius='48',force=True,
                      tracks='/private/example.csv')
        document=pose_settings_snapshot(settings,rendering())
        result=normalize_pose_settings(json.loads(json.dumps(document)))
        self.assertEqual(result,document)
        self.assertEqual(result['refinement']['method'],'klt_ransac')
        self.assertEqual(result['refinement']['point_ids'],'1, 3')
        self.assertEqual(result['refinement']['search_radius'],48.)
        self.assertIs(result['refinement']['center_search'],True)
        self.assertNotIn('force',result['refinement']);self.assertNotIn('tracks',result['refinement'])
        self.assertEqual(result['render']['render_options']['pose_alpha'],.55)
        self.assertIs(result['render']['render_options']['pose_labels'],False)

    def test_save_load_and_invalid_document_do_not_partially_apply(self):
        tab=PoseRefinementTab.__new__(PoseRefinementTab)
        tab.require_idle=lambda:None
        initial=pose_settings_snapshot(dict(values(),method='klt_ransac'),rendering())['refinement']
        tab.variables={name:Value(value) for name,value in dict(initial,force=True).items()}
        tab.status=Value('');tab.method_label=Value('');tab.sync_method_controls=Mock()
        tab.owner=SimpleNamespace(capture_render_settings=rendering,apply_render_settings=Mock())
        tab.tracks_var=Value('preserve-current-input.csv')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'preset.json'
            tab.save_settings(path)
            tab.variables['method'].set('similarity')
            tab.load_settings(path)
            self.assertEqual(tab.variables['method'].get(),'klt_ransac')
            self.assertIs(tab.variables['force'].get(),False)
            self.assertEqual(tab.tracks_var.get(),'preserve-current-input.csv')
            tab.owner.apply_render_settings.assert_called_once()
            saved=json.loads(path.read_text())
            saved['render']['render_options']['pose_alpha']=2
            path.write_text(json.dumps(saved))
            before={name:var.get() for name,var in tab.variables.items()}
            with self.assertRaises(ValueError):tab.load_settings(path)
            self.assertEqual(before,{name:var.get() for name,var in tab.variables.items()})
            tab.owner.apply_render_settings.assert_called_once()
            saved['render']=rendering();saved['refinement']['use_recent']='false'
            with self.assertRaises(ValueError):normalize_pose_settings(saved)


if __name__=='__main__':unittest.main()
