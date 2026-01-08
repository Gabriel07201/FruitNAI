import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import cv2
import dxcam

import win32gui
import win32con
import win32api
import logging


def list_visible_window_titles(limit: int = 60) -> List[str]:
    titles: List[str] = []

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = (win32gui.GetWindowText(hwnd) or "").strip()
            if t:
                titles.append(t)

    win32gui.EnumWindows(enum_handler, None)
    return titles[:limit]

@dataclass(frozen=True)
class WindowRegion:
    left: int
    top: int
    width: int
    height: int
    
@dataclass(frozen=True)
class MonitorInfo:
    idx: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

@dataclass(frozen=True)
class CaptureInfo:
    region_abs: WindowRegion
    region_rel: WindowRegion
    monitor: MonitorInfo


class WindowLocator:
    def __init__(self, title_substring: str):
        self.title_substring = title_substring.lower()

    def _find_hwnd(self) -> Optional[int]:
        result: List[int] = []

        def enum_handler(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd) or ""
                if self.title_substring in title.lower():
                    result.append(hwnd)

        win32gui.EnumWindows(enum_handler, None)
        return result[0] if result else None

    def _get_monitor_info(self, region: WindowRegion) -> MonitorInfo:
        monitors = win32api.EnumDisplayMonitors()
        center_x = region.left + region.width // 2
        center_y = region.top + region.height // 2
        monitor_idx = 0
        monitor_rect = None
        monitor_handle = None

        for idx, (handle, _, rect) in enumerate(monitors):
            if rect[0] <= center_x < rect[2] and rect[1] <= center_y < rect[3]:
                monitor_idx = idx
                monitor_rect = rect
                monitor_handle = handle
                break

        if monitor_rect is None:
            monitor_handle = win32api.MonitorFromPoint((center_x, center_y), win32con.MONITOR_DEFAULTTONEAREST)
            for idx, (handle, _, rect) in enumerate(monitors):
                if handle == monitor_handle:
                    monitor_idx = idx
                    monitor_rect = rect
                    break

        if monitor_rect is None:
            monitor_rect = monitors[0][2]

        return MonitorInfo(
            idx=monitor_idx,
            left=monitor_rect[0],
            top=monitor_rect[1],
            right=monitor_rect[2],
            bottom=monitor_rect[3],
        )

    def get_client_region(self, bring_foreground: bool = True) -> CaptureInfo:
        hwnd = self._find_hwnd()
        if hwnd is None:
            raise RuntimeError(f"Não achei janela contendo: '{self.title_substring}'")

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.15)

        if bring_foreground:
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass

        # Área cliente (sem bordas) -> coordenadas na tela
        left_top = win32gui.ClientToScreen(hwnd, (0, 0))
        right_bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:4])

        left, top = left_top
        right, bottom = right_bottom

        width = max(0, right - left)
        height = max(0, bottom - top)

        if width == 0 or height == 0:
            raise RuntimeError("A área cliente está com tamanho 0 (janela minimizada/oculta?).")

        region_abs = WindowRegion(left=left, top=top, width=width, height=height)
        monitor = self._get_monitor_info(region_abs)
        rel_left = left - monitor.left
        rel_top = top - monitor.top
        region_rel = WindowRegion(left=rel_left, top=rel_top, width=width, height=height)

        return CaptureInfo(region_abs=region_abs, region_rel=region_rel, monitor=monitor)


class ScreenGrabber:
    def __init__(self, output_idx: int = 0, target_fps: int = 60):
        self._camera = dxcam.create(output_idx=output_idx)
        if self._camera is None:
            raise RuntimeError("Falha ao inicializar DXCAM.")
        self._camera.start(target_fps=target_fps)

    def grab_bgr(self, region: WindowRegion) -> np.ndarray:
        left = region.left
        top = region.top
        right = left + region.width
        bottom = top + region.height
        frame = self._camera.grab(region=(left, top, right, bottom))
        if frame is None:
            raise RuntimeError("Falha ao capturar frame via DXCAM.")
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


class GameCapture:
    def __init__(self, title_substring: str, bring_foreground: bool = True, target_fps: int = 60):
        self.locator = WindowLocator(title_substring)
        self.grabber: Optional[ScreenGrabber] = None
        self._monitor_idx: Optional[int] = None
        self.bring_foreground = bring_foreground
        self.target_fps = target_fps
        self._logger = logging.getLogger(__name__)

    def read(self) -> tuple[np.ndarray, WindowRegion]:
        capture = self.locator.get_client_region(bring_foreground=self.bring_foreground)
        if self.grabber is None or capture.monitor.idx != self._monitor_idx:
            self.grabber = ScreenGrabber(output_idx=capture.monitor.idx, target_fps=self.target_fps)
            self._monitor_idx = capture.monitor.idx

        region_abs = capture.region_abs
        region_rel = capture.region_rel
        monitor_rect = capture.monitor.rect
        region_abs_rect = (region_abs.left, region_abs.top, region_abs.left + region_abs.width, region_abs.top + region_abs.height)
        region_rel_rect = (region_rel.left, region_rel.top, region_rel.left + region_rel.width, region_rel.top + region_rel.height)

        self._logger.info(
            "DXCAM grab info: monitor_idx=%s monitor_rect=%s region_abs=%s region_rel=%s",
            capture.monitor.idx,
            monitor_rect,
            region_abs_rect,
            region_rel_rect,
        )

        frame = self.grabber.grab_bgr(region_rel)
        return frame, region_abs