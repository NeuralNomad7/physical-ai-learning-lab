"""Capture the viewer window of this process on Windows (best effort).

Only windows owned by the current process are considered, and the window is
asked to render itself (``PrintWindow`` with ``PW_RENDERFULLCONTENT``). The
screen is never read, so other applications cannot appear in the image.
Some OpenGL drivers do not support this path and return a black image; callers
must check the result with ``render.is_nonblank`` and report failures.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2


def capture_own_window() -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (RGB image or None, info) for the largest visible window of this process."""
    if sys.platform != "win32":
        return None, {"error": "window capture is implemented for Windows only"}
    try:
        return _capture()
    except Exception as exc:  # a failed capture must not stop the viewer
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def _capture() -> tuple[np.ndarray | None, dict[str, Any]]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL

    pid = os.getpid()
    candidates: list[tuple[int, int, int, str]] = []

    @enum_proc
    def visit(hwnd: int, _lparam: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            rect = wintypes.RECT()
            if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                candidates.append((hwnd, rect.right - rect.left, rect.bottom - rect.top, title.value))
        return True

    user32.EnumWindows(visit, 0)
    candidates = [c for c in candidates if c[1] > 0 and c[2] > 0]
    if not candidates:
        return None, {"error": "no visible window owned by this process"}
    hwnd, width, height, title = max(candidates, key=lambda c: c[1] * c[2])
    info: dict[str, Any] = {"window_title": title, "width": width, "height": height, "method": "PrintWindow"}

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    window_dc = user32.GetDC(hwnd)
    if not window_dc:
        return None, {**info, "error": "GetDC failed"}
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not user32.PrintWindow(hwnd, memory_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            return None, {**info, "error": f"PrintWindow failed (error {ctypes.get_last_error()})"}
        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height  # top-down rows
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0  # BI_RGB
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.SelectObject(memory_dc, previous)  # GetDIBits needs the bitmap deselected
        previous = None
        rows = gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
        if rows != height:
            return None, {**info, "error": f"GetDIBits copied {rows} of {height} rows"}
        bgrx = np.frombuffer(buffer.raw, dtype=np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(bgrx[:, :, 2::-1]), info
    finally:
        if previous is not None:
            gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)
