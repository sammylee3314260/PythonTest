import numpy as np
import napari

class LabelEditor:
    def __init__(self):
        self.viewer = None
        self.layer = None
        self.label_data = None
        self.merge_queue = []
        self.deleted_ids = []
        self.save = None
        self._installed = False
        self._on_mouse_press = None
        self._keybindings = []

    def install(self, viewer: napari.Viewer, label_layer, label_mask,save=False):
        if self._installed:
            raise RuntimeError("Already installed")
        self.viewer = viewer
        self.layer = label_layer
        self.save = save
        self.label_data = label_mask

        def on_mouse_press(event):
            btn = getattr(event,"button", None)
            pos_world = self.viewer.cursor.position
            if pos_world is None:
                return
            val = self.layer.get_value(pos_world, world = True)
            if val is None or val == 0:
                return
            if btn == 1: # left click
                if val not in self.merge_queue:
                    self.merge_queue.append(int(val))
                print('Queue',self.merge_queue)
            elif btn == 2: # right click
                self._delete_label(int(val))
                print(f'Delete label {val}')
            else: print(f'btn == {btn}'); return
        self._on_mouse_press = on_mouse_press
        self.viewer.window.qt_viewer.canvas.events.mouse_press.connect(self._on_mouse_press)
        self._bind_key('c', self.clear_queue)
        self._bind_key('q', self._quit_only)
        self._bind_key('e', self._quit_without_save)
        self._bind_key('s', self._save_and_exit)
        self._bind_key('Shift-m', self.merge_queue_now)
        self._installed = True
        print('LabelEditor installed')

    def uninstall(self):
        if not self._installed: return
        try: self.viewer.window.qt_viewer.canvas.events.mouse_press.disconnect(self._on_mouse_press)
        except Exception: pass
        self._installed = False
        print("LabelEditor uninstalled")

    def run(self):
        napari.run()
        return self.label_data, self.save, list(self.deleted_ids)
    
    def merge_queue_now(self, *args, **kwargs):
        if len(self.merge_queue) < 2: print('More than 2 labels needed to merge. Current queue:', self.merge_queue);return
        keep = self.merge_queue[0]
        for val in self.merge_queue:
            self._relabel(val,keep)
        self._refresh()
        print(f'Merge {self.merge_queue} to {keep}')
        self.merge_queue = []

    def clear_queue(self, *args, **kwargs):
        self.merge_queue.clear()
        print("Queue cleared")
    
    def _quit_only(self,viewer):
        self.viewer.close()
    def _quit_without_save(self,viewer):
        self.save = False; self.viewer.close()
    def _save_and_exit(self,viewer):
        self.save = True; self.viewer.close()
    
    def _delete_label(self,label_id:int):
        data = self.layer.data
        raw_data = self.label_data
        if np.any(data == label_id):
            data[data == label_id] = 0
            raw_data[raw_data == label_id] = 0
            self._refresh()
            self.deleted_ids.append(int(label_id))
            print(f'Deleted label {label_id}')
    
    def _relabel(self, src:int, dst:int):
        if src == dst: return
        data = self.layer.data
        raw_data = self.label_data
        mask = (data==src)
        if np.any(mask):
            data[mask] = dst
            raw_data[mask] = dst
    
    def _refresh(self):
        self.layer.data = self.layer.data.copy()
    
    def _bind_key(self, key, func, overwrite=False):
        bound = self.viewer.bind_key(key,overwrite=overwrite)
        bound(func)
        self._keybindings.append((key,func))