"""
CZI to AVI Batch Converter
===========================
功能：
  1. 依照檔名順序批次開啟 CZI 檔案
  2. 自動從 metadata 讀取 pixel size 與時間間隔
  3. 在每幀右下角壓上 timestamp（相對時間 MM:SS）與 scale bar
  4. 輸出為 AVI（每個 CZI 對應一個 AVI）

安裝依賴：
  pip install aicspylibczi opencv-python numpy tqdm

使用方式：
  1. 修改下方 INPUT_DIR 為你的 CZI 資料夾路徑
  2. 修改下方 OUTPUT_DIR 為輸出資料夾（會自動建立）l
  3. 執行:python czi_to_avi.py
"""

import os
import re
import glob
import numpy as np
import cv2
from tqdm import tqdm
from aicspylibczi import CziFile

# ─────────────────────────────────────────────
DEBUG = False
# ─────────────────────────────────────────────
INPUT_DIR  = r"/mnt/f/Osmolarity/2026-02-18/2.recovery/2026-02-19/"   # CZI file folder
OUTPUT_DIR = r"/mnt/SammyRis/Sammy/Osmolarity_analysis/260218/mp4_output/2.recovery/"  # Output folder for mp4
# ─────────────────────────────────────────────l

# FPS for mp4 file (play rate)
OUTPUT_FPS = 10

# Scale bar target length in µm
SCALEBAR_UM = 250   # Default 50 µm

# Font setting
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)   # white
FONT_THICK = 2
# BG_COLOR = FONT_COLOR
BG_COLOR   = (0, 0, 0)         # Black Bg frame

# ─────────────────────────────────────────────

def natural_sort_key(s):
    """sort filename (e.g. _2_ < _10_)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def get_pixel_size_um(czi: CziFile):
    """Read XY pixel size from CZI metadata (µm) """
    try:
        meta = czi.meta
        # aicspylibczi 回傳 ElementTree，找 Scaling 節點
        ns = {'czi': 'http://www.zeiss.com/czi/2010/metadata'}
        # 嘗試兩種可能的路徑
        for path in [
            ".//Scaling/Items/Distance[@Id='X']/Value",
            ".//Distance[@Id='X']/Value",
        ]:
            node = meta.find(path)
            if node is not None and node.text:
                val_m = float(node.text)   # CZI 內部單位為公尺
                return val_m * 1e6         # 轉換為 µm
    except Exception:
        pass
    return None  # 無法讀取


def get_time_stamps_sec(czi: CziFile):
    """從 CZI metadata 讀取每幀時間（秒），回傳 list"""
    try:
        meta = czi.meta
        times = []
        for node in meta.findall(".//TimeStamp"):
            if node.text:
                times.append(float(node.text))
        if times:
            t0 = times[0]
            return [t - t0 for t in times]
    except Exception:
        pass
    return None


def sec_to_mmss(seconds: float) -> str:
    """將秒數轉為 HH:MM 格式"""
    total_sec = int(round(seconds))
    mm = total_sec // 60
    ss = total_sec % 60
    return f"{mm:02d}:{ss:02d}"


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """將任意位元深度影像正規化為 uint8 灰階"""
    if frame.dtype == np.uint8:
        return frame
    f_min, f_max = frame.min(), frame.max()
    if f_max == f_min:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame.astype(np.float32) - f_min) / (f_max - f_min) * 255
    return norm.astype(np.uint8)


def draw_overlay(frame_bgr: np.ndarray,
                 timestamp_str: str,
                 scalebar_px: int,
                 pixel_size_um: float,
                 scalebar_um: float) -> np.ndarray:
    """在影像右下角疊加 timestamp 與 scale bar"""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    margin = 20  # 邊距（像素）

    # ── Scale bar ──────────────────────────────
    if scalebar_px and scalebar_px > 0:
        bar_x2 = w - margin
        bar_x1 = bar_x2 - scalebar_px
        bar_y  = h - margin
        bar_thick = max(3, h // 120)

        # 黑色陰影（防白底看不見）
        cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), (0, 0, 0), bar_thick + 2)
        cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), (255, 255, 255), bar_thick)

        # Scale bar 標籤
        label = f"{int(scalebar_um)} micron"
        (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE * 0.85, FONT_THICK)
        lx = bar_x1 + (scalebar_px - lw) // 2
        ly = bar_y - bar_thick - 5
        cv2.putText(img, label, (lx, ly), FONT, FONT_SCALE * 0.85,
                    (0, 0, 0), FONT_THICK + 2, cv2.LINE_AA)
        cv2.putText(img, label, (lx, ly), FONT, FONT_SCALE * 0.85,
                    FONT_COLOR, FONT_THICK, cv2.LINE_AA)

    # ── Timestamp ──────────────────────────────
    (tw, th), _ = cv2.getTextSize(timestamp_str, FONT, FONT_SCALE, FONT_THICK)
    tx = margin + tw
    ty = margin + th
    # 半透明黑底
    pad = 5
    # cv2.rectangle(img, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad),
    #               BG_COLOR, -1)
    cv2.putText(img, timestamp_str, (tx, ty), FONT, FONT_SCALE,
                FONT_COLOR, FONT_THICK, cv2.LINE_AA)

    return img

def best_focus_z(czi, T, Z, C=0):
    scores = []
    for z in range(Z):
        frame, _ = czi.read_image(T=T, Z=z, C=C)
        frame = normalize_frame(np.squeeze(frame))
        # Laplacian
        score = cv2.Laplacian(frame, cv2.CV_64F).var()
        scores.append(score)
    return int(np.argmax(scores))

def process_czi(czi_path: str, output_path: str):
    """處理單一 CZI 檔案，輸出 mp4"""
    print(f"\n  Read file: {os.path.basename(czi_path)}")
    czi = CziFile(czi_path)

    # 讀取 pixel size
    pixel_size_um = get_pixel_size_um(czi)
    if pixel_size_um:
        scalebar_px = int(round(SCALEBAR_UM / pixel_size_um))
        print(f"     Pixel size = {pixel_size_um:.4f} micron → scale bar = {scalebar_px} px ({SCALEBAR_UM} micron)")
    else:
        scalebar_px = 0
        print("     Can't find pixel size, no scalebar presented")

    # 讀取時間戳
    time_stamps = get_time_stamps_sec(czi)
    print('time_stamps',time_stamps)

    # 取得影像維度資訊
    dims = czi.get_dims_shape()   # list of dict
    # 嘗試取第一個 scene / position
    shape_dict = dims[0] if dims else {}
    T = shape_dict.get('T', (0, 1))[1]   # time points
    Z = shape_dict.get('Z', (0, 1))[1]   # z-planes（取中間層）
    z_picked = best_focus_z(czi, T-1, Z)
    # z_picked = Z // 2

    print(f"     Time frames = {T},  Z planes = {Z}  (使用第 {z_picked} 層)")

    if T == 0:
        print("     NO T dimention found, skip.")
        return

    # 讀取第一幀取得影像尺寸
    first_frame, _ = czi.read_image(T=0, Z=z_picked, C=0)
    # first_frame shape: (1,1,...,H,W,1) or similar — squeeze
    first_frame = np.squeeze(first_frame)
    first_gray  = normalize_frame(first_frame)
    h, w = first_gray.shape

    # 建立 mp4v 編碼 for MP4 output，相容性最佳
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (w, h))

    for t in tqdm(range(T), desc="     written frame", leave=False):
        frame_data, _ = czi.read_image(T=t, Z=z_picked, C=0)
        frame_gray = normalize_frame(np.squeeze(frame_data))
        frame_bgr  = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        # Timestamp 字串
        if time_stamps and t < len(time_stamps):
            ts_str = sec_to_mmss(time_stamps[t])
        else:
            ts_str = sec_to_mmss(t*15)   # fallback：用幀號當秒數

        frame_out = draw_overlay(frame_bgr, ts_str, scalebar_px,
                                 pixel_size_um or 1, SCALEBAR_UM)
        out.write(frame_out)

    out.release()
    print(f"     output:{os.path.basename(output_path)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find and sort CZI
    pattern = os.path.join(INPUT_DIR, "*.czi")
    czi_files = sorted(glob.glob(pattern), key=lambda p: natural_sort_key(os.path.basename(p)))

    if not czi_files:
        print(f"No czi files found at {INPUT_DIR}")
        return

    print(f"Found {len(czi_files)} CZI files, processing...\n")

    for i, czi_path in enumerate(czi_files, 1):
        base_name = os.path.splitext(os.path.basename(czi_path))[0]
        out_name  = f"{base_name}.mp4"
        out_path  = os.path.join(OUTPUT_DIR, out_name)
        print(f"[{i}/{len(czi_files)}]", end="")
        process_czi(czi_path, out_path)
        if DEBUG: exit()

    print(f"\nComplete! MP4 saved at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()