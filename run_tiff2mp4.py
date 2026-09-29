# This is a launcher function for tiff2mp4.
# The whole reason is to maintain a deeper sys.path so tiff2mp4 can find utils file
# The pyinstaller is run on this file like '''pyinstaller --onefile run_tiff2mp4.py'''

from ImgProc import tiff2mp4
if __name__ == "__main__":
    tiff2mp4.main()