"""视频导出管线：解码 -> ROI 模糊 -> 可选裁剪 -> 编码。

从 UI 中剥离出的纯处理逻辑，通过依赖注入（ffmpeg 路径、日志回调、
硬件解码参数）与界面解耦，可独立测试。
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

import blur_core as bc
import media_tools as mt

Roi = Optional[Tuple[int, int, int, int]]
LogFn = Callable[[str], None]


class VideoExporter:
    """负责单个视频从读取到写盘的完整流程，含硬件->软件编码回退。"""

    def __init__(
        self,
        ffmpeg_exec: Optional[str],
        decode_method: Optional[str] = None,
        decode_device: Optional[str] = None,
        log: Optional[LogFn] = None,
    ) -> None:
        self.ffmpeg_exec = ffmpeg_exec
        self.decode_method = decode_method
        self.decode_device = decode_device
        self._log: LogFn = log or (lambda _msg: None)

    # ------------------------------------------------------------- 对外接口
    def export(
        self,
        src_path: str,
        out_path: str,
        roi: Roi,
        method: str,
        strength: int,
        crop_enabled: bool,
        remove_audio: bool,
        encoder: str,
        preserve_bitrate: bool = True,
    ) -> str:
        """导出单个视频，返回输出路径。硬件编码失败时自动回退 libx264。"""
        bitrate = None
        if preserve_bitrate:
            bitrate = mt.probe_video_bitrate(self.ffmpeg_exec, src_path)
            if bitrate:
                self._log(f"{os.path.basename(src_path)} 源码率 ~{bitrate // 1000} kbps，导出将尽量保持")
            else:
                self._log(f"{os.path.basename(src_path)} 无法读取源码率，改用质量优先模式")

        attempts = [encoder]
        _, _, first_is_hw = mt.encoder_params(encoder, bitrate)
        if first_is_hw:
            attempts.append("libx264")

        last_error: Optional[Exception] = None
        for enc in attempts:
            try:
                return self._run_pipeline(
                    src_path, out_path, roi, method, strength,
                    crop_enabled, remove_audio, enc, bitrate,
                )
            except Exception as exc:
                last_error = exc
                self._log(f"{os.path.basename(src_path)} 使用 {enc} 失败: {exc}")
                self._safe_remove(out_path)
                if enc != attempts[-1]:
                    self._log(f"{os.path.basename(src_path)} 将回退到软件编码重试")
        raise last_error or RuntimeError("编码未完成")

    # ------------------------------------------------------------- 内部实现
    def _run_pipeline(
        self,
        src_path: str,
        out_path: str,
        roi: Roi,
        method: str,
        strength: int,
        crop_enabled: bool,
        remove_audio: bool,
        encoder: str,
        bitrate: Optional[int],
    ) -> str:
        cap, _ = mt.open_capture(src_path, self.decode_method, self.decode_device)
        if not cap or not cap.isOpened():
            raise RuntimeError("无法打开视频")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            crop_box = self._compute_crop(width, height) if crop_enabled else None
            # 无裁剪但源尺寸为奇数时，裁掉 1 像素以满足 yuv420p 的偶数要求。
            if crop_box is None and (width % 2 or height % 2):
                crop_box = (0, 0, width - (width % 2), height - (height % 2))
            out_w = crop_box[2] if crop_box else width
            out_h = crop_box[3] if crop_box else height

            # 时域擦除：先采样若干帧估计干净背景，再逐帧只替换字幕像素。
            background_roi = None
            if method == "inpaint_temporal" and roi:
                background_roi = self._build_temporal_background(cap, roi, width, height, total)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if background_roi is None:
                    self._log(f"{os.path.basename(src_path)} 时域背景采样不足，退化为空间修复")

            proc, using_hw = self._start_ffmpeg_writer(
                out_path, out_w, out_h, fps, encoder, src_path, remove_audio, bitrate
            )
            writer = None
            if proc is None:
                writer = self._open_cv_writer(out_path, fps, out_w, out_h)

            self._pump_frames(
                cap, proc, writer, roi, method, strength, crop_box, total, src_path, background_roi
            )

            if proc is not None:
                self._finish_ffmpeg(proc, using_hw)
            elif writer is not None:
                writer.release()
                if not remove_audio:
                    self._mux_audio(out_path, src_path)
        finally:
            cap.release()

        self._log(f"{os.path.basename(src_path)} {'硬件' if using_hw else '软件'}编码完成")
        return out_path

    def _build_temporal_background(self, cap, roi, width, height, total):
        """采样多帧，取中值作为 ROI 区域的干净背景（字幕多为瞬时/半透明覆盖时有效）。"""
        box = bc.clamp_roi(roi, width, height)
        if box is None or total <= 0:
            return None
        x, y, end_x, end_y = box
        sample_count = min(21, total)
        indexes = np.linspace(0, total - 1, sample_count).astype(int)
        samples = []
        for idx in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                samples.append(frame[y:end_y, x:end_x].copy())
        if len(samples) < 3:
            return None
        return np.median(np.stack(samples), axis=0).astype(np.uint8)

    def _pump_frames(
        self, cap, proc, writer, roi, method, strength, crop_box, total, src_path, background_roi=None
    ) -> None:
        frame_idx = 0
        last_log = time.time()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if background_roi is not None:
                frame = bc.composite_background(frame, background_roi, roi, strength)
            else:
                frame = bc.apply_roi_blur(frame, roi, method, strength)
            if crop_box:
                x0, y0, cw, ch = crop_box
                frame = frame[y0:y0 + ch, x0:x0 + cw]
            if proc is not None:
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(frame.tobytes())
                except Exception as exc:
                    raise RuntimeError(f"写入编码器失败: {exc}") from exc
            else:
                assert writer is not None
                writer.write(frame)
            now = time.time()
            if frame_idx == 0 or now - last_log >= 1.0:
                pct = int(frame_idx * 100 / total) if total else 0
                self._log(f"处理中 {os.path.basename(src_path)}: {frame_idx}/{total} 帧 ({pct}%)")
                last_log = now
            frame_idx += 1

    @staticmethod
    def _compute_crop(width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """将画面从中心裁剪到 9:16，比例已接近则不裁剪。

        裁剪后的宽高统一向下取偶数，避免 H.264 (yuv420p) 因奇数尺寸编码失败。
        """
        target_ratio = 9 / 16
        cur_ratio = width / height if height else target_ratio
        if abs(cur_ratio - target_ratio) <= 0.01:
            return None
        def even(v: int) -> int:
            return max(2, v - (v % 2))

        target_w = int(height * target_ratio)
        if target_w <= width:
            target_w = even(target_w)
            x0 = max(0, (width - target_w) // 2)
            return (x0, 0, target_w, even(height))
        target_h = even(int(width / target_ratio))
        y0 = max(0, (height - target_h) // 2)
        return (0, y0, even(width), target_h)

    def _start_ffmpeg_writer(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        encoder: str,
        source_path: str,
        remove_audio: bool,
        bitrate: Optional[int],
    ) -> Tuple[Optional[subprocess.Popen], bool]:
        if not self.ffmpeg_exec:
            return None, False
        enc, extra, is_hw = mt.encoder_params(encoder, bitrate)
        args = [
            self.ffmpeg_exec, "-y", "-loglevel", "error", "-threads", "0",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        ]
        if remove_audio:
            args.append("-an")
        else:
            args += ["-i", source_path, "-map", "0:v:0", "-map", "1:a?"]
        args += ["-c:v", enc, *extra, "-pix_fmt", "yuv420p"]
        if not remove_audio:
            args += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        args += ["-movflags", "+faststart", out_path]
        try:
            return subprocess.Popen(args, stdin=subprocess.PIPE), is_hw
        except Exception:
            return None, False

    @staticmethod
    def _open_cv_writer(out_path: str, fps: float, width: int, height: int) -> "cv2.VideoWriter":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer or not writer.isOpened():
            raise RuntimeError("无法创建输出文件")
        return writer

    def _finish_ffmpeg(self, proc: subprocess.Popen, using_hw: bool) -> None:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        if proc.wait() != 0:
            raise RuntimeError("硬件编码失败，请检查显卡驱动/ffmpeg" if using_hw else "ffmpeg 编码失败")

    def _mux_audio(self, video_path: str, source_path: str) -> None:
        """给纯 OpenCV 写出的无声视频补上源音轨。"""
        if not self.ffmpeg_exec:
            self._log("未找到 ffmpeg，无法为输出合成音频")
            return
        temp_out = f"{video_path}.tmp_audio.mp4"
        cmd = [
            self.ffmpeg_exec, "-y", "-loglevel", "error",
            "-i", video_path, "-i", source_path,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", temp_out,
        ]
        try:
            ret = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if ret.returncode == 0 and os.path.exists(temp_out):
                os.replace(temp_out, video_path)
                self._log("已将源音轨合成到输出文件")
            else:
                err = ((ret.stderr or "") + (ret.stdout or "")).strip().splitlines()
                detail = f": {err[-1]}" if err else ""
                self._log(f"音轨合成失败，输出为无声视频{detail}")
        finally:
            self._safe_remove(temp_out)

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
