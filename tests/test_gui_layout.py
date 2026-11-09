"""Focused overlay-preset checks and small native-Tk layout smoke tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from video_annotation.rgb_tracking.labeler import PointLabeler
from video_annotation.rgb_tracking.overlay_settings import (
    load_overlay_settings, normalize_overlay_settings, save_overlay_settings)
from video_annotation.rgb_tracking.render_options import normalize_render_options


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class OverlaySettingsTests(unittest.TestCase):
    def test_pending_fields_roundtrip_and_invalid_load_is_atomic(self):
        owner = PointLabeler.__new__(PointLabeler)
        owner.controller = SimpleNamespace(busy=False)
        owner.overlay_fps, owner.max_width = None, 1280
        owner.render_options = normalize_render_options()
        owner.render_variables = {name:Value(value) for name, value in
            dict(owner.render_options, overlay_fps='120', max_width='640').items()}
        owner.render_variables['pose_alpha'].set('.25')
        owner.render_variables['pose_halo'].set(True)
        owner.render_variables['pose_labels'].set(False)
        owner.sync_render_settings = Mock()
        value = owner.capture_render_settings()
        self.assertEqual((value['overlay_fps'], value['max_width']), (120., 640))
        self.assertEqual(value['render_options']['pose_alpha'], .25)
        self.assertIs(value['render_options']['pose_halo'], True)
        self.assertIs(value['render_options']['pose_labels'], False)
        self.assertEqual(owner.max_width, 1280)  # Capturing pending fields does not apply them.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'overlay.json'
            save_overlay_settings(path, value)
            self.assertEqual(load_overlay_settings(path), value)
        invalid = deepcopy(value)
        invalid['render_options']['pose_labels'] = 'false'
        with self.assertRaises(ValueError): owner.apply_render_settings(invalid)
        self.assertEqual(owner.max_width, 1280)
        owner.sync_render_settings.assert_not_called()
        owner.apply_render_settings(value)
        self.assertEqual(owner.max_width, 640)
        self.assertIs(owner.render_options['pose_labels'], False)
        with self.assertRaises(ValueError): normalize_overlay_settings(dict(value, schema_version=2))


class NativeLayoutTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f'Native Tk display unavailable: {error}')
        self.root.geometry('420x260')

    def tearDown(self):
        if hasattr(self, 'root'):
            self.root.destroy()

    def test_scrollable_sections_keep_native_wheels_and_remove_bindings(self):
        import tkinter as tk
        from tkinter import ttk
        from video_annotation.rgb_tracking.ui import ScrollableFrame, CollapsibleFrame
        original_all = self.root.bind_all('<MouseWheel>')
        scroll = ScrollableFrame(self.root)
        scroll.pack(fill='both', expand=True)
        section = CollapsibleFrame(scroll.body, text='Retained controls')
        section.pack(fill='x')
        value = tk.StringVar(value='kept')
        ttk.Entry(section.body, textvariable=value).pack()
        label = ttk.Label(scroll.body, text='Scroll from a regular control')
        label.pack()
        ttk.Frame(scroll.body, height=1200, width=800).pack()
        video, listing, text = tk.Canvas(scroll.body), tk.Listbox(scroll.body), tk.Text(scroll.body)
        self.root.update()
        self.assertTrue(scroll.scrollbars['y'].winfo_ismapped())
        self.assertGreater(scroll.body.winfo_height(), scroll.canvas.winfo_height())
        start = scroll.canvas.yview()
        self.assertEqual(scroll._wheel(SimpleNamespace(widget=label, state=0, delta=-120)), 'break')
        self.assertNotEqual(scroll.canvas.yview(), start)
        for widget in (video, listing, text):
            position = scroll.canvas.yview()
            self.assertIsNone(scroll._wheel(SimpleNamespace(widget=widget, state=0, delta=-120)))
            self.assertEqual(scroll.canvas.yview(), position)
        section.set_expanded(False)
        self.assertFalse(section.expanded)
        self.assertEqual(section.body.winfo_manager(), '')
        section.toggle()
        self.assertEqual(value.get(), 'kept')
        identifiers = [identifier for _, identifier in scroll._wheel_bindings]
        scroll.destroy()
        for identifier in identifiers:
            self.assertNotIn(identifier, self.root.bind('<MouseWheel>') or '')
        self.assertEqual(self.root.bind_all('<MouseWheel>'), original_all)

    def test_notebook_keeps_lazy_tab_identity_and_useful_video_height(self):
        from video_annotation.rgb_tracking.ui import ScrollableFrame, CollapsibleFrame
        controller = SimpleNamespace(busy=False, dirty=False, default_radius=32)
        with patch.object(PointLabeler, 'show'), patch.object(PointLabeler, 'redraw'), patch.object(PointLabeler, 'poll'):
            owner = PointLabeler(controller, root=self.root)
            self.root.update()
            frames = [owner.annotation_frame, owner.review_frame, owner.pose_frame,
                      owner.sam_frame, owner.mask_review_frame, owner.render_frame]
            self.assertTrue(all(isinstance(frame, ScrollableFrame) for frame in frames))
            self.assertEqual(tuple(owner.notebook.tabs()), tuple(str(frame) for frame in frames))
            self.assertEqual(owner._active_tab, str(owner.annotation_frame))
            self.assertIsNone(owner.review)
            self.assertIsNone(owner.pose_view)
            self.assertGreaterEqual(owner.canvas.winfo_height(), 400)
            self.assertIs(owner.render_variables['pose_halo'].get(), False)
            self.assertIs(owner.render_variables['pose_labels'].get(), True)
            owner.notebook.select(owner.pose_frame)
            owner.tab_changed()
            self.root.update()
            self.assertIsNotNone(owner.pose_view)
            self.assertIs(owner.pose_view.master, owner.pose_frame.body)
            for widget in owner.pose_view.winfo_children():
                if isinstance(widget, CollapsibleFrame):
                    widget.set_expanded(True)
            self.root.update()
            self.assertTrue(owner.pose_frame.scrollbars['y'].winfo_ismapped())
            owner.pose_frame.canvas.yview_moveto(1.)
            self.assertAlmostEqual(owner.pose_frame.canvas.yview()[1], 1.)


if __name__ == '__main__':
    unittest.main()
