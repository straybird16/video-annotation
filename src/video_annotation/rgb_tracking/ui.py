"""Shared readable fonts, light ttk styling and a small local guide viewer."""
from pathlib import Path
import re

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = ttk = None


class ScrollableFrame(ttk.Frame if ttk is not None else object):
    """A tab-sized viewport; image canvases and native text/list wheels stay local."""

    def __init__(self, parent, **kwargs):
        if ttk is None:
            raise RuntimeError('Scrollable views require Tk')
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background='#f4f6fa')
        self.body = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), anchor='nw', window=self.body)
        self.scrollbars = {'y':ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview),
                           'x':ttk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)}
        self.canvas.configure(yscrollcommand=self.scrollbars['y'].set, xscrollcommand=self.scrollbars['x'].set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbars['y'].grid(row=0, column=1, sticky='ns')
        self.scrollbars['x'].grid(row=1, column=0, sticky='ew')
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)
        self._pending = None
        self._window_size = None
        self.canvas.bind('<Configure>', self._schedule)
        self.body.bind('<Configure>', self._schedule)
        self._top = self.winfo_toplevel()
        self._wheel_bindings = [(sequence, self._top.bind(sequence, self._wheel, add='+'))
                                for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>')]
        self.bind('<Destroy>', self._destroyed, add='+')

    def _schedule(self, event=None):
        if self._pending is None:
            self._pending = self.after_idle(self._layout)

    def _layout(self):
        self._pending = None
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        requested = (self.body.winfo_reqwidth(), self.body.winfo_reqheight())
        size = (max(width, requested[0]), max(height, requested[1]))
        if size != self._window_size:
            self.canvas.itemconfigure(self._window, width=size[0], height=size[1])
            self._window_size = size
        self.canvas.configure(scrollregion=(0, 0, *size))
        for axis, needed in (('x', requested[0] > width), ('y', requested[1] > height)):
            self.scrollbars[axis].grid() if needed else self.scrollbars[axis].grid_remove()

    def _wheel_target(self, widget):
        if widget is self.canvas:
            return True
        while widget is not None and widget is not self.body:
            if widget.winfo_class() in ('Canvas', 'Listbox', 'Text', 'Treeview', 'TCombobox', 'TSpinbox', 'TScrollbar'):
                return False
            widget = getattr(widget, 'master', None)
        return widget is self.body

    def _wheel(self, event):
        if not self._wheel_target(event.widget):
            return
        view = self.canvas.xview if event.state & 1 else self.canvas.yview
        if view() == (0., 1.):
            return
        delta = -1 if getattr(event, 'num', None) == 4 else 1 if getattr(event, 'num', None) == 5 else 0
        if not delta and getattr(event, 'delta', 0):
            delta = -int(event.delta/120) or (-1 if event.delta > 0 else 1)
        if delta:
            scroll = self.canvas.xview_scroll if event.state & 1 else self.canvas.yview_scroll
            scroll(delta, 'units')
            return 'break'

    def _destroyed(self, event):
        if event.widget is not self:
            return
        if self._pending is not None:
            self.after_cancel(self._pending)
        for sequence, identifier in self._wheel_bindings:
            try:
                self._top.unbind(sequence, identifier)
            except tk.TclError:
                pass
        self._wheel_bindings.clear()


class CollapsibleFrame(ttk.Frame if ttk is not None else object):
    """An outlined section with a retained content frame and explicit state."""

    def __init__(self, parent, *, text='', expanded=True, **kwargs):
        if ttk is None:
            raise RuntimeError('Collapsible views require Tk')
        super().__init__(parent, **kwargs)
        self.text = text
        self.header = ttk.Button(self, command=self.toggle)
        self.header.pack(fill='x')
        self.body = ttk.Frame(self, padding=(10, 8))
        self.expanded = None
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.expanded = bool(expanded)
        self.header.configure(text=f'{"▾" if self.expanded else "▸"}  {self.text}')
        if self.expanded:
            self.body.pack(fill='both', expand=True)
        else:
            self.body.pack_forget()

    def toggle(self):
        self.set_expanded(not self.expanded)


def font_family(root):
    from tkinter import font
    if not hasattr(root, '_point_font_family'):
        families = {name.casefold(): name for name in font.families(root)}
        preferred = ('Calibri', 'Arial', 'Aptos', 'Liberation Sans', 'Noto Sans', 'Helvetica', 'Segoe UI')
        root._point_font_family = next((families[name.casefold()] for name in preferred
                                        if name.casefold() in families),
                                       font.nametofont('TkDefaultFont', root=root).actual('family'))
    return root._point_font_family


def configure_style(root):
    from tkinter import font, ttk
    family = font_family(root)
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
                 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont'):
        try:
            font.nametofont(name, root=root).configure(family=family, size=13)
        except Exception:
            continue
    root.option_add('*Font', (family, 13))
    root.option_add('*Listbox.background', '#ffffff')
    root.option_add('*Listbox.foreground', '#17263b')
    root.option_add('*Listbox.selectBackground', '#d8e8ff')
    root.option_add('*Listbox.selectForeground', '#124e96')
    root.option_add('*Listbox.relief', 'flat')
    root.option_add('*Listbox.highlightThickness', 1)
    root.option_add('*Listbox.highlightBackground', '#d5dce7')
    root.option_add('*Text.background', '#ffffff')
    root.option_add('*Text.foreground', '#17263b')
    root.configure(background='#f4f6fa')
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    style.configure('.', font=(family, 13), background='#f4f6fa', foreground='#17263b')
    style.configure('TButton', padding=(10, 7), borderwidth=1, relief='solid',
                    background='#ffffff', bordercolor='#94a3b8', lightcolor='#ffffff', darkcolor='#94a3b8')
    style.map('TButton', background=[('active', '#e1ebfa'), ('pressed', '#d2e2f7')],
              bordercolor=[('focus', '#2563eb'), ('active', '#4d79b4')])
    style.configure('TEntry', fieldbackground='#ffffff', padding=6, borderwidth=1,
                    relief='solid', bordercolor='#94a3b8', lightcolor='#ffffff', darkcolor='#94a3b8')
    style.map('TEntry', bordercolor=[('focus', '#2563eb')])
    style.configure('TCombobox', fieldbackground='#ffffff', padding=5, borderwidth=1,
                    relief='solid', bordercolor='#94a3b8', lightcolor='#ffffff', darkcolor='#94a3b8')
    style.map('TCombobox', fieldbackground=[('readonly', '#ffffff')])
    style.configure('TCheckbutton', padding=(4, 5))
    style.configure('TRadiobutton', padding=(4, 5))
    style.configure('TLabelframe', borderwidth=1, relief='solid', bordercolor='#b9c5d5')
    style.configure('TLabelframe.Label', font=(family, 13, 'bold'), foreground='#26466a')
    style.configure('TNotebook', borderwidth=0, background='#f4f6fa')
    style.configure('TNotebook.Tab', padding=(20, 9))
    style.map('TNotebook.Tab', background=[('selected', '#ffffff')],
              foreground=[('selected', '#165eaa')])
    style.configure('Point.TButton', padding=(10, 7), font=(family, 13))
    style.configure('Point.Title.TLabel', font=(family, 16, 'bold'), padding=(0, 3))
    return style


def _plain(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text.replace('**', '').replace('`', '')


def show_guide(parent, path):
    """Render local Markdown headings/tables as readable paragraphs, no dependency."""
    import tkinter as tk
    from tkinter import ttk
    path = Path(path)
    window = tk.Toplevel(parent)
    window.title('Point tracking · help and parameters')
    window.geometry('980x760')
    window.transient(parent.winfo_toplevel())
    family, size = font_family(parent.winfo_toplevel()), tk.IntVar(window, value=15)
    bar = ttk.Frame(window, padding=10)
    bar.pack(fill='x')
    ttk.Label(bar, text=path.stem.replace('_', ' ').title(), style='Point.Title.TLabel').pack(side='left')
    body = ttk.Frame(window, padding=(12, 0, 12, 12))
    body.pack(fill='both', expand=True)
    text = tk.Text(body, wrap='word', padx=22, pady=18, relief='flat', borderwidth=0,
                   spacing1=3, spacing3=7, highlightthickness=1, highlightbackground='#d5dce7')
    scroll = ttk.Scrollbar(body, orient='vertical', command=text.yview)
    scroll.pack(side='right', fill='y')
    text.configure(yscrollcommand=scroll.set)
    text.pack(fill='both', expand=True)
    def change(delta):
        size.set(max(12, min(24, size.get()+delta)))
        current = size.get()
        text.configure(font=(family, current))
        for tag, extra in (('title', 7), ('heading', 3), ('parameter', 0)):
            text.tag_configure(tag, font=(family, current+extra, 'bold'), spacing1=12, spacing3=5)
        text.tag_configure('code', font=(family, current-1), background='#eef3fa', lmargin1=12, lmargin2=12)
    ttk.Button(bar, text='A−', command=lambda: change(-1), width=4).pack(side='right', padx=3)
    ttk.Button(bar, text='A+', command=lambda: change(1), width=4).pack(side='right', padx=3)
    ttk.Label(bar, text='Text size').pack(side='right', padx=8)
    code = False
    for line in path.read_text().splitlines():
        if line.startswith('```'):
            code = not code
            continue
        if code:
            text.insert('end', line+'\n', 'code')
        elif line.startswith('# '):
            text.insert('end', _plain(line[2:])+'\n', 'title')
        elif line.startswith('#'):
            text.insert('end', _plain(line.lstrip('#').strip())+'\n', 'heading')
        elif line.startswith('|'):
            columns = [_plain(part.strip()) for part in line.strip('|').split('|')]
            if not columns or columns[0] in ('Parameter', 'Artifact') or set(''.join(columns)) <= {'-', ':', ' '}:
                continue
            label = columns[0]+(f' · default {columns[1]}' if len(columns) == 3 else '')
            text.insert('end', label+'\n', 'parameter')
            text.insert('end', columns[-1]+'\n')
        else:
            text.insert('end', _plain(line)+'\n')
    change(0)
    text.configure(state='disabled')
    ttk.Button(window, text='Close', command=window.destroy).pack(pady=(0, 12))
    return window
