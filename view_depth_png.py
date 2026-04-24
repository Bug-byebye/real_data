# -*- coding: utf-8 -*-
"""
查看 capture_femto_bolt_benchmark.py 保存的 16 位单通道深度 PNG（uint16）。

用法（在 conda activate orbbec-benchmark 下）:
  python view_depth_png.py                    # 默认使用脚本同目录下的 depth 文件夹
  python view_depth_png.py depth/00001.png   # 指定文件
  python view_depth_png.py --dir depth       # 指定目录（按数字序浏览）

按键:
  n         下一张
  p         上一张
  r         重新加载当前文件
  + / -     调整伪彩色对比度（收窄/放宽有效深度的分位区间）
  q / ESC   退出
  点击窗口关闭按钮  立即结束进程（与 q 相同）

依赖: numpy, opencv-python（与采集脚本同一 conda 环境即可）
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _list_depth_pngs(folder: str) -> List[str]:
    """按文件名中的数字排序（00000.png, 00001.png …）。"""
    if not os.path.isdir(folder):
        return []
    out: List[Tuple[int, str]] = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".png"):
            continue
        m = re.match(r"^(\d+)\.png$", fn, re.I)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, fn)))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def _load_depth_u16(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[错误] 无法读取: {path}", file=sys.stderr)
        return None
    if img.ndim == 3:
        print("[警告] 图像为多通道，将只取第一通道作为深度。", file=sys.stderr)
        img = img[:, :, 0]
    if img.dtype != np.uint16:
        print(f"[提示] 数据类型为 {img.dtype}，将转为 uint16 显示（可能有截断）。")
        img = np.clip(img, 0, 65535).astype(np.uint16)
    return img


def _depth_to_vis_bgr(
    depth_u16: np.ndarray,
    lo_pct: float,
    hi_pct: float,
) -> np.ndarray:
    """有效深度 >0 的分位数拉伸后伪彩色（与采集脚本预览思路一致）。"""
    v = depth_u16.astype(np.float32)
    mask = v > 0
    if not np.any(mask):
        h, w = depth_u16.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)
    vals = v[mask]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    u8 = np.zeros_like(v, dtype=np.uint8)
    u8[mask] = np.clip((v[mask] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _window_was_closed(win: str) -> bool:
    """
    检测用户是否已关闭 HighGUI 窗口（点击 X）。
    关闭后 getWindowProperty 一般为 -1，或抛出 cv2.error。
    """
    try:
        v = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return True
    # 窗口已销毁时一般为 -1；可见时 >= 0。仅用 <0 判断关闭，避免把最小化(0)当成退出。
    return float(v) < 0


def _print_stats(path: str, depth: np.ndarray) -> None:
    valid = depth[depth > 0]
    print(
        f"[信息] {path}\n"
        f"       形状 {depth.shape}, dtype={depth.dtype}, "
        f"有效像素 {valid.size}/{depth.size}"
    )
    if valid.size:
        print(
            f"       min={int(valid.min())}, max={int(valid.max())}, "
            f"median={float(np.median(valid)):.1f}"
        )
    else:
        print("       （无大于 0 的深度值）")


def main() -> None:
    root = os.path.abspath(os.path.dirname(__file__))
    ap = argparse.ArgumentParser(description="查看 16 位深度 PNG")
    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        help="深度 PNG 路径；省略则打开 --dir 中按数字排序的第一张",
    )
    ap.add_argument(
        "--dir",
        "-d",
        default=None,
        help="深度图目录（默认: 脚本目录下的 depth）",
    )
    args = ap.parse_args()

    depth_dir = os.path.normpath(
        args.dir if args.dir is not None else os.path.join(root, "output-test\depth")
    )

    files: List[str] = []
    idx = 0

    if args.path:
        one = os.path.abspath(args.path)
        if not os.path.isfile(one):
            print(f"[错误] 文件不存在: {one}", file=sys.stderr)
            sys.exit(1)
        # 若该文件所在目录与 depth_dir 一致，则加载整目录便于 n/p 切换
        if os.path.normpath(os.path.dirname(one)) == os.path.normpath(depth_dir):
            files = _list_depth_pngs(depth_dir)
            if one in files:
                idx = files.index(one)
            else:
                files = [one]
                idx = 0
        else:
            files = [one]
            idx = 0
    else:
        files = _list_depth_pngs(depth_dir)
        if not files:
            print(f"[错误] 目录中无 00000.png 形式的 PNG: {depth_dir}", file=sys.stderr)
            sys.exit(1)

    lo_pct, hi_pct = 5.0, 95.0
    win = "Depth viewer (16-bit PNG, pseudo color)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(files):
        path = files[idx]
        depth = _load_depth_u16(path)
        if depth is None:
            idx += 1
            continue
        _print_stats(path, depth)
        vis = _depth_to_vis_bgr(depth, lo_pct, hi_pct)
        cv2.imshow(win, vis)
        print(
            f"[信息] 伪彩色分位数: {lo_pct:.0f}%～{hi_pct:.0f}%（+/- 调整对比度，r 重载当前文件）"
        )

        while True:
            if _window_was_closed(win):
                print("[信息] 检测到窗口已关闭，退出。")
                cv2.destroyAllWindows()
                return
            _wk = cv2.waitKeyEx if hasattr(cv2, "waitKeyEx") else cv2.waitKey
            k = _wk(30)
            if _window_was_closed(win):
                print("[信息] 检测到窗口已关闭，退出。")
                cv2.destroyAllWindows()
                return
            if k == -1:
                continue
            lo = k & 0xFF
            if lo in (ord("q"), ord("Q")) or k == 27:
                cv2.destroyAllWindows()
                return
            if lo in (ord("n"), ord("N")):
                if idx < len(files) - 1:
                    idx += 1
                    break
                print("[提示] 已是最后一张。")
            if lo in (ord("p"), ord("P")):
                if idx > 0:
                    idx -= 1
                    break
                print("[提示] 已是第一张。")
            if lo in (ord("r"), ord("R")):
                break
            if lo in (ord("+"), ord("=")):
                # 收窄分位区间 → 拉伸中间细节，观感上对比度增强
                lo_pct = min(45.0, lo_pct + 1.0)
                hi_pct = max(55.0, hi_pct - 1.0)
                if hi_pct <= lo_pct + 1.0:
                    hi_pct = lo_pct + 2.0
                vis = _depth_to_vis_bgr(depth, lo_pct, hi_pct)
                cv2.imshow(win, vis)
            elif lo in (ord("-"), ord("_")):
                lo_pct = max(0.0, lo_pct - 1.0)
                hi_pct = min(100.0, hi_pct + 1.0)
                if hi_pct <= lo_pct + 1.0:
                    hi_pct = lo_pct + 2.0
                vis = _depth_to_vis_bgr(depth, lo_pct, hi_pct)
                cv2.imshow(win, vis)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
