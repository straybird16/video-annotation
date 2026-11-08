"""Model-free checks for editing one SAM instance without discarding its peers."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from video_annotation.mask_tracking.labeler import MaskLabeler


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class PreviewList:
    def __init__(self): self.selected = [0]
    def curselection(self): return self.selected


class MaskPreviewEditingTests(unittest.TestCase):
    def labeler(self):
        gui = MaskLabeler.__new__(MaskLabeler)
        gui.annotation_tools = True
        gui.reset_preview_edits()
        first, second = np.zeros((50, 80), bool), np.zeros((50, 80), bool)
        first[10:21, 10:21] = True
        second[30:41, 55:66] = True
        gui.preview_masks, gui.preview_ids = {3:first, 9:second}, [3, 9]
        gui.preview_list = PreviewList()
        gui.controller = SimpleNamespace(image=np.zeros((50, 80, 3), np.uint8), busy=False, dirty=False,
            prompt={'points_xy':[[15.,15.]], 'point_labels':[1]}, _idle=Mock(), roi_id=0)
        def edit(prompt): gui.controller.prompt = deepcopy(prompt)
        gui.controller.edit = Mock(side_effect=edit)
        gui.controller.undo = Mock()
        gui.controller.region_mask = lambda: gui.controller.prompt.get('mask')
        gui.processor = SimpleNamespace(backend='sam31')
        gui.tool_var, gui.brush_var, gui.label_var, gui.status = Value('Clicks'), Value('1'), Value(1), Value('')
        gui.transform = SimpleNamespace(to_source=lambda x,y: [float(x), float(y)])
        gui.canvas, gui.preview_info = Mock(), Mock()
        gui.brush_mask = gui.brush_last = gui.drag_start = gui.drag_end = None
        gui.vertices = []
        gui.region_cache_ready = False
        gui.redraw = gui.sync_motion_controls = gui._frame_matches = Mock()
        return gui

    def stroke(self, gui, x, y, *, erase=False):
        event = SimpleNamespace(x=x, y=y, num=3 if erase else 1)
        (gui.right_press if erase else gui.press)(event)
        gui.release(event)

    def test_selected_preview_accepts_brush_and_right_erase_without_changing_prompt_or_peers(self):
        gui = self.labeler()
        prompt, other = deepcopy(gui.controller.prompt), gui.preview_masks[9].copy()
        gui.edit_selected_mask()
        self.assertEqual(gui.tool_var.get(), 'Brush +')
        self.stroke(gui, 15, 15, erase=True)
        self.assertFalse(gui.preview_masks[3][15, 15])
        self.stroke(gui, 25, 15)
        self.assertTrue(gui.preview_masks[3][15, 25])
        np.testing.assert_array_equal(gui.preview_masks[9], other)
        self.assertEqual(gui.controller.prompt, prompt)
        gui.controller.edit.assert_not_called()
        self.assertEqual(set(gui.preview_masks), {3, 9})

    def test_undo_is_per_object_and_multiple_selection_cannot_destroy_previews(self):
        gui = self.labeler()
        original = {identity:mask.copy() for identity,mask in gui.preview_masks.items()}
        gui.edit_selected_mask()
        self.stroke(gui, 15, 15, erase=True)
        gui.preview_list.selected = [1]
        gui.preview_selection_changed()
        self.stroke(gui, 60, 35, erase=True)
        second_edited = gui.preview_masks[9].copy()
        gui.preview_list.selected = [0]
        gui.undo()
        np.testing.assert_array_equal(gui.preview_masks[3], original[3])
        np.testing.assert_array_equal(gui.preview_masks[9], second_edited)
        gui.controller.undo.assert_not_called()
        gui.preview_list.selected = [0, 1]
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            gui.edit_selected_mask()
        self.assertEqual(set(gui.preview_masks), {3, 9})

    def test_saving_selected_references_preserves_edited_pixels_and_separate_masks(self):
        gui = self.labeler()
        gui.edit_selected_mask()
        self.stroke(gui, 15, 15, erase=True)
        expected = [gui.preview_masks[identity].copy() for identity in gui.preview_ids]
        saved = []
        def new_roi(): gui.controller.roi_id += 1
        gui.controller.new_roi = new_roi
        gui.controller.save = lambda: saved.append(gui.controller.prompt['mask'].copy())
        gui.show = Mock()
        gui.preview_list.selected = [0, 1]
        gui.save_preview_references()
        self.assertEqual(len(saved), 2)
        for actual, mask in zip(saved, expected):
            np.testing.assert_array_equal(actual, mask)

    def test_new_prompt_or_regenerated_preview_resets_edit_history_intentionally(self):
        gui = self.labeler()
        gui.edit_selected_mask()
        self.stroke(gui, 15, 15, erase=True)
        self.assertTrue(gui.preview_undo)
        gui.edit({'points_xy':[[30.,20.]], 'point_labels':[1]})
        self.assertFalse(gui.preview_masks)
        self.assertFalse(gui.preview_undo)
        self.assertIsNone(gui.editing_preview_id)
        replacement = np.ones((50, 80), bool)
        gui.preview_undo = {3:[replacement.copy()]}
        gui.job_completed({'kind':'preview', 'preview':SimpleNamespace(masks={7:replacement})})
        self.assertEqual(set(gui.preview_masks), {7})
        self.assertFalse(gui.preview_undo)


if __name__ == '__main__':
    unittest.main()
