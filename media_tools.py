"""媒体与硬件工具：ffmpeg / ffprobe 探测、硬件解码/编码检测、码率读取。

本模块集中所有与 ffmpeg 命令行、显卡硬件加速相关的逻辑，
对外提供纯函数（除少量全局缓存外无状态），便于测试与复用。
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import List, Optional, Tuple

import cv2

# 支持的 H.264 硬件编码器，按自动选择的优先级排列。
HW_ENCODERS: Tuple[str, ...] = ("h264_nvenc", "h264_qsv", "h264_amf")
# 硬件解码方式优先级。
PREFERRED_DECODE: Tuple[str, ...] = ("cuda", "d3d11va", "dxva2", "vaapi")

# 避免在多线程同时探测时相互干扰的 subprocess 默认标志。
_NO_WINDOW = 0
if os.name == "nt":  # 在 Windows 下隐藏探测时弹出的黑框
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 修改 OPENCV_FFMPEG_CAPTURE_OPTIONS 是进程级操作，需串行化。
_cap_env_lock = threading.Lock()


def _run(cmd: List[str], timeout: float) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW
        )
    except Exception:
        return None


# ----------------------------------------------------------------- ffprobe / 码率
def find_ffprobe(ffmpeg_exec: Optional[str]) -> Optional[str]:
    """根据 ffmpeg 路径推断同目录下的 ffprobe。"""
    if not ffmpeg_exec:
        return None
    directory, name = os.path.split(ffmpeg_exec)
    probe_name = name.replace("ffmpeg", "ffprobe")
    candidate = os.path.join(directory, probe_name)
    if os.path.isfile(candidate):
        return candidate
    return None


def probe_video_bitrate(ffmpeg_exec: Optional[str], path: str, timeout: float = 6.0) -> Optional[int]:
    """读取源视频的码率（bit/s）。优先使用 ffprobe，失败则用 ffmpeg 兜底。

    返回 None 表示无法确定，此时调用方应回退到质量（CRF/CQ）模式。
    """
    ffprobe = find_ffprobe(ffmpeg_exec)
    if ffprobe:
        # 先取视频流码率，为空再取整个文件码率。
        for entries in ("stream=bit_rate", "format=bit_rate"):
            select = ["-select_streams", "v:0"] if entries.startswith("stream") else []
            proc = _run(
                [ffprobe, "-v", "error", *select, "-show_entries", entries,
                 "-of", "default=nokey=1:noprint_wrappers=1", path],
                timeout,
            )
            rate = _parse_bitrate(proc.stdout if proc else None)
            if rate:
                return rate
    return _bitrate_from_size(ffmpeg_exec, path, timeout)


def _parse_bitrate(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    for token in text.split():
        token = token.strip()
        if token.isdigit():
            value = int(token)
            if value > 0:
                return value
    return None


def _bitrate_from_size(ffmpeg_exec: Optional[str], path: str, timeout: float) -> Optional[int]:
    """当 ffprobe 不可用时，用文件大小 / 时长估算码率。"""
    ffprobe = find_ffprobe(ffmpeg_exec)
    duration: Optional[float] = None
    if ffprobe:
        proc = _run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            timeout,
        )
        if proc and proc.stdout:
            try:
                duration = float(proc.stdout.strip())
            except ValueError:
                duration = None
    if not duration or duration <= 0:
        return None
    try:
        size_bits = os.path.getsize(path) * 8
    except OSError:
        return None
    est = int(size_bits / duration)
    return est if est > 0 else None


# ----------------------------------------------------------------- 硬件解码
def detect_hwaccels(ffmpeg_exec: Optional[str], timeout: float = 4.0) -> List[str]:
    """列出 ffmpeg 支持的硬件解码方式。"""
    if not ffmpeg_exec:
        return []
    proc = _run([ffmpeg_exec, "-hide_banner", "-hwaccels"], timeout)
    if not proc:
        return []
    output = (proc.stdout or "") + (proc.stderr or "")
    methods: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "Hardware acceleration methods" in line:
            continue
        methods.append(line)
    return methods


def pick_decode_method(methods: List[str]) -> Optional[str]:
    for candidate in PREFERRED_DECODE:
        if candidate in methods:
            return candidate
    return None


def build_capture_options(method: Optional[str], device: Optional[str]) -> Optional[str]:
    """构造 OPENCV_FFMPEG_CAPTURE_OPTIONS 字符串。"""
    if not method:
        return None
    opts = [f"hw_acceleration;{method}"]
    if method in ("cuda", "d3d11va", "dxva2"):
        opts.append("hw_device;0")
    elif method == "vaapi":
        opts.append(f"hw_device;{device or '/dev/dri/renderD128'}")
    return "|".join(opts)


def open_capture(
    path: str, method: Optional[str] = None, device: Optional[str] = None
) -> Tuple[Optional["cv2.VideoCapture"], bool]:
    """打开视频，优先硬件解码。返回 (capture, 是否启用了硬件解码)。"""
    opts = build_capture_options(method, device)
    if opts:
        with _cap_env_lock:
            prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            finally:
                if prev is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev
        if cap is not None and cap.isOpened():
            return cap, True
    return cv2.VideoCapture(path), False


# ----------------------------------------------------------------- 硬件编码
def detect_hw_encoders(ffmpeg_exec: Optional[str], timeout: float = 3.0) -> List[str]:
    """列出 ffmpeg 声明支持的 H.264 硬件编码器（仅按名称匹配）。"""
    if not ffmpeg_exec:
        return []
    proc = _run([ffmpeg_exec, "-hide_banner", "-encoders"], timeout)
    if not proc:
        return []
    output = (proc.stdout or "") + (proc.stderr or "")
    return [code for code in HW_ENCODERS if code in output]


def encoder_usable(ffmpeg_exec: str, encoder: str, timeout: float = 4.0) -> Tuple[bool, str]:
    """通过编码一帧测试图验证编码器是否真正可用。返回 (是否可用, 失败信息)。"""
    cmd = [
        ffmpeg_exec, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=size=128x128:rate=30:duration=0.2",
        "-frames:v", "1", "-c:v", encoder, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    proc = _run(cmd, timeout)
    if proc is None:
        return False, "自检超时或无法启动"
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, err[-1] if err else "未知错误"


def detect_working_hw_encoders(ffmpeg_exec: str) -> Tuple[List[str], List[str]]:
    """返回 (可用编码器, 声明但自检失败的编码器)。"""
    usable: List[str] = []
    failed: List[str] = []
    for enc in detect_hw_encoders(ffmpeg_exec):
        ok, _ = encoder_usable(ffmpeg_exec, enc)
        (usable if ok else failed).append(enc)
    return usable, failed


def pick_auto_encoder(detected: List[str]) -> str:
    for code in HW_ENCODERS:
        if code in detected:
            return code
    return "libx264"


def encoder_params(encoder: str, bitrate: Optional[int]) -> Tuple[str, List[str], bool]:
    """返回 (编码器名, 附加参数, 是否为硬件编码)。

    当 bitrate 已知时按「保持原码率」的 VBR 模式配置（目标码率 +
    maxrate/bufsize 上限）；未知时回退到质量恒定模式，避免画质塌陷。
    """
    is_hw = encoder in HW_ENCODERS
    maxrate = int(bitrate * 1.5) if bitrate else 0
    bufsize = int(bitrate * 2) if bitrate else 0
    rate_args = (
        ["-b:v", str(bitrate), "-maxrate", str(maxrate), "-bufsize", str(bufsize)]
        if bitrate else []
    )

    if encoder == "h264_nvenc":
        extra = ["-preset", "p5", "-rc", "vbr", "-multipass", "qres"]
        extra += rate_args if bitrate else ["-cq", "20"]
        return encoder, extra, True
    if encoder == "h264_qsv":
        extra = ["-preset", "medium"]
        extra += rate_args if bitrate else ["-global_quality", "22"]
        return encoder, extra, True
    if encoder == "h264_amf":
        extra = ["-quality", "balanced"]
        extra += (["-rc", "vbr_peak", *rate_args] if bitrate
                  else ["-rc", "cqp", "-qp_i", "22", "-qp_p", "22"])
        return encoder, extra, True

    # 默认软件编码 libx264
    extra = ["-preset", "medium"]
    extra += rate_args if bitrate else ["-crf", "20"]
    return "libx264", extra, False
