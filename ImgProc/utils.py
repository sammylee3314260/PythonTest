### Maintain a utility functions library
# import sys
import glob
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime

DEBUG = True

'''
# old version of get_filepath, it will crash in Windows system, bc readline and subprocess cannot be imported on windows system
# Im trying to merge windows system.
def get_filepath(try_gui:bool = True, sys_argv = None, given_path = None):
    """If try_gui, will try to run Windows gui system.\n
       sys_argv should only put in: sys.argv.
       If given_path, will return given_path.
    """
    if sys_argv[1:]:
        if isinstance(sys_argv[1], str): return sys_argv[1]
        else: print(f"sys.argv[1] {sys_argv[1]} is not string, keep going.")
    if given_path: return given_path
    import subprocess
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "& { $p = New-Object -ComObject Shell.Application; " +
        "$f = $p.BrowseForFolder(0, 'Select a folder', 0); " +
        "if ($f) { $f.Self.Path } }"
    ]
    if try_gui:
        try:
            win_path = subprocess.check_output(cmd).decode('utf-8').strip()
            if win_path:
                unix_path = subprocess.check_output(["wslpath", "-u", win_path]).decode('utf-8').strip()
                return unix_path
        except subprocess.CalledProcessError:
            print("Cannot open windows file explorer. Try cli.")
    
    import readline
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    def path_completer(text, state):
        return (glob.glob(os.path.expanduser(text) + '*') + [None])[state]
    readline.set_completer(path_completer)
    parent_folder = input("Please enter the path to the parent folder, you can use tab:\n")
    return parent_folder
'''
def get_filepath(try_gui:bool = True, sys_argv = [], given_path = None):
    """Proirity-wise:
       1. If path come with sys_argv, will return sys_argv[1], sys_argv should only put in: sys.argv.
       2. If given_path, will return given_path.
       3. If try_gui, will try to run Windows gui system.
       4. Finally Cli type-in
    """
    if sys_argv[1:]:
        if isinstance(sys_argv[1], str): return sys_argv[1]
        else: print(f"sys.argv[1] {sys_argv[1]} is not string, keep going.")
    if given_path: return given_path

    env = _detect_env()
    if DEBUG: print(f"env = {env}")
    if env == "unknown": raise ValueError(f"Unknown os for this code you can add your system to it.")
    if try_gui:
        try:
            if env == "wsl": return _gui_select_wsl()
            elif env in ("windows", "linux", "macos"): return _gui_select_tkinter()
        except Exception as e:
            print(f"GUI selection failed {e}, try Cli")
    return _cli_input(env)

def _detect_env():
    # To autodetect which environment are we in rn.
    import platform
    system = platform.system()
    if system == "Windows": return "windows"
    if system == "Linux":
        # wsl also return Linux.
        release = platform.uname().release.lower()
        if "microsoft" in release or "wsl" in release: return "wsl"
        return "linux"
    if system == "Darwin": return "macos" # I dont think anyone will use this line though
    return "unknown"

def _gui_select_wsl():
    # for wsl it is easier to use subprocess
    import subprocess
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "& { $p = New-Object -ComObject Shell.Application; " +
        "$f = $p.BrowseForFolder(0, 'Select a folder', 0); " +
        "if ($f) { $f.Self.Path } }"
    ]
    try:
        win_path = subprocess.check_output(cmd).decode('utf-8').strip()
        if win_path:
            unix_path = subprocess.check_output(["wslpath", "-u", win_path]).decode('utf-8').strip()
            return unix_path
    except subprocess.CalledProcessError:
        raise ValueError("Cannot open windows file explorer. Try cli.")

def _gui_select_tkinter():
    # The code to input filepath with tkinter, work on GUI windows Linux.
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select a folder")
    root.destroy()
    if not folder: return None
    return folder

def _cli_input(env):
    import glob, os
    if env != "windows":
        import readline
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        def path_completer(text, state):
            return (glob.glob(os.path.expanduser(text) + '*') + [None])[state]
        readline.set_completer(path_completer)
    parent_folder = input("Please enter the path to the parent folder, you can use tab if you are NOT in Windows system:\n")
    return parent_folder

def natural_sort_key(s):
    """sort filename (e.g. _2_ < _10_)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8: return frame
    # f_min, f_max = np.percentile(frame, (0.5,99.5))
    f_min, f_max = frame.min(), frame.max()
    if f_max == f_min:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame.astype(np.float32) - f_min) / (f_max - f_min) * 255
    return norm.astype(np.uint8)

def remove_date_folder(path):
    def is_date(name):
        for fmt in ("%Y-%m-%d","%Y%m%d"):
            try:
                datetime.strptime(name,fmt)
                return True
            except ValueError: pass
        return False
    return Path(*[part for part in path.parts
                  if not is_date(part)])

# if __name__ == "__main__": print(get_filepath(try_gui=False))