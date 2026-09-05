import sys
import os
import re
import subprocess

TIFF2MP4_SCRIPT = "/home/sammylee/code/PythonTest/ImgProc/tiff2mp4.py"
PYTHON = "/home/sammylee/.pyenv/versions/pythontestvenv/bin/python"

def win_to_wsl(p):
    try:
        unix_path = subprocess.check_output(["wslpath", "-u", p]).decode('utf-8').strip()
        return unix_path
    except subprocess.CalledProcessError:
        return p
    except FileNotFoundError: # What does this do???
        print(f"No wslpath found")
        return p.replace("\\","/")

def main():
    for row in sys.argv[1:]:
        path = win_to_wsl(row)
        print(row, path)
        if not os.path.exists(path):
            print(f"Path({row};{path}) not exist")
            continue
        result = subprocess.run([PYTHON, TIFF2MP4_SCRIPT, path])
        if result.returncode != 0:
            print(f"Sth occur, code: {result.returncode}")
        else: print(f"finish: {path}")

if __name__ == "__main__":
    main()