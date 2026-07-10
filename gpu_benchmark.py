"""显卡/编码诊断与基准脚本。

在装有 1050 Ti 的目标机上运行，收集以下数据并写入报告文件，便于回传分析：
  1. ffmpeg 版本、可用硬件编码器(nvenc/qsv/amf)、硬件解码方式、CUDA 滤镜是否存在；
  2. 同一段视频用不同编码器/参数编码的耗时与速度（对比 CPU / NVENC / 单趟vs两趟 / 预设）；
  3. 一段持续 NVENC 转码期间 nvidia-smi 采样的 sm/enc/dec 真实占用；
  4. 本项目实际导出管线（融合 GPU 路径 vs 逐帧路径）的耗时，并确认是否真的用到 NVENC
     （而非静默回退到 CPU libx264）。

用法：
  python gpu_benchmark.py                     # 自动找 ffmpeg（config.json / PATH），生成合成测试视频
  python gpu_benchmark.py --ffmpeg D:/ffmpeg/bin/ffmpeg.exe
  python gpu_benchmark.py --video 我的素材.mp4  # 用真实视频更有代表性
运行结束会生成 gpu_benchmark_report.txt，请把该文件内容整段回传。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

REPORT_LINES: List[str] = []


def out(line: str = "") -> None:
    print(line)
    REPORT_LINES.append(line)


def section(title: str) -> None:
    out("")
    out("=" * 68)
    out(title)
    out("=" * 68)


# --------------------------------------------------------------------- 工具查找
def find_ffmpeg(cli_path: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if cli_path:
        candidates.append(cli_path)
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                p = json.load(f).get("ffmpeg_path")
            if p:
                candidates.append(p)
        except Exception:
            pass
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def find_ffprobe(ffmpeg: str) -> Optional[str]:
    d, n = os.path.split(ffmpeg)
    cand = os.path.join(d, n.replace("ffmpeg", "ffprobe"))
    if os.path.isfile(cand):
        return cand
    return shutil.which("ffprobe")


def find_nvidia_smi() -> Optional[str]:
    p = shutil.which("nvidia-smi")
    if p:
        return p
    for c in (r"C:\Windows\System32\nvidia-smi.exe",
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.isfile(c):
            return c
    return None


def run(cmd: List[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------- 能力探测
def probe_capabilities(ffmpeg: str) -> None:
    section("1) 环境与能力探测")
    try:
        ver = run([ffmpeg, "-hide_banner", "-version"]).stdout.splitlines()[0]
    except Exception as e:
        ver = f"(读取失败: {e})"
    out(f"ffmpeg: {ffmpeg}")
    out(f"版本  : {ver}")

    enc = run([ffmpeg, "-hide_banner", "-encoders"]).stdout
    hw_enc = [c for c in ("h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf") if c in enc]
    out(f"硬件编码器(声明): {', '.join(hw_enc) or '无'}")

    hwa = run([ffmpeg, "-hide_banner", "-hwaccels"]).stdout
    methods = [l.strip() for l in hwa.splitlines() if l.strip() and "methods" not in l.lower()]
    out(f"硬件解码方式    : {', '.join(methods) or '无'}")

    filt = run([ffmpeg, "-hide_banner", "-filters"]).stdout
    cuda_filters = [f for f in ("scale_cuda", "scale_npp", "overlay_cuda", "hwupload_cuda",
                                "hwdownload") if re.search(rf"\b{f}\b", filt)]
    out(f"CUDA 相关滤镜    : {', '.join(cuda_filters) or '无'}（决定“方案A 全GPU滤镜”是否可行）")

    smi = find_nvidia_smi()
    if smi:
        try:
            info = run([smi, "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader"]).stdout.strip()
            out(f"GPU / 驱动      : {info}")
        except Exception as e:
            out(f"nvidia-smi 查询失败: {e}")
    else:
        out("nvidia-smi      : 未找到（将跳过 GPU 占用采样）")


# --------------------------------------------------------------------- 源与探测
def make_source(ffmpeg: str, path: str, seconds: int, w: int, h: int, fps: int) -> None:
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}",
           "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", path]
    run(cmd, timeout=180)


def probe_video(ffprobe: Optional[str], path: str) -> Tuple[int, int, float, int, Optional[int]]:
    """返回 (w, h, fps, frames, bitrate)。"""
    w = h = frames = 0
    fps = 30.0
    bitrate: Optional[int] = None
    if ffprobe and os.path.isfile(ffprobe):
        r = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,nb_frames,bit_rate",
                 "-of", "json", path])
        try:
            st = json.loads(r.stdout)["streams"][0]
            w, h = int(st.get("width", 0)), int(st.get("height", 0))
            num, den = (st.get("r_frame_rate", "30/1").split("/") + ["1"])[:2]
            fps = float(num) / float(den) if float(den) else 30.0
            frames = int(st.get("nb_frames", 0) or 0)
            br = st.get("bit_rate")
            bitrate = int(br) if br and br.isdigit() else None
        except Exception:
            pass
    if not frames:
        frames = int(fps * 10)
    return w, h, fps, frames, bitrate


# --------------------------------------------------------------------- 编码基准
_SPEED_RE = re.compile(r"speed=\s*([\d.]+)x")
_FPS_RE = re.compile(r"fps=\s*([\d.]+)")


def encode_test(ffmpeg: str, label: str, in_args: List[str], venc_args: List[str],
                src: str, frames: int, src_fps: float) -> dict:
    cmd = [ffmpeg, "-y", "-hide_banner", "-nostdin", *in_args, "-i", src,
           *venc_args, "-an", "-f", "null", "-"]
    t0 = time.time()
    try:
        proc = run(cmd, timeout=600)
    except Exception as e:
        out(f"  {label:<26} 运行异常: {e}")
        return {"label": label, "ok": False}
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    stderr = proc.stderr or ""
    speed = _SPEED_RE.findall(stderr)
    enc_fps = _FPS_RE.findall(stderr)
    speed_x = float(speed[-1]) if speed else (frames / src_fps / elapsed if elapsed else 0)
    fps_out = float(enc_fps[-1]) if enc_fps else (frames / elapsed if elapsed else 0)
    if ok:
        out(f"  {label:<26} 耗时 {elapsed:6.2f}s | 速度 {speed_x:5.2f}x | ~{fps_out:6.1f} fps")
    else:
        tail = (stderr.strip().splitlines() or ["(无输出)"])[-1]
        out(f"  {label:<26} 失败! -> {tail}")
    return {"label": label, "ok": ok, "elapsed": elapsed, "speed_x": speed_x, "fps": fps_out}


def run_encode_matrix(ffmpeg: str, src: str, frames: int, src_fps: float,
                      bitrate: Optional[int], has_nvenc: bool, has_cuda_dec: bool) -> None:
    section("2) 编码基准（同一段源，-f null，纯测编码速度）")
    br = bitrate or 8_000_000
    rate = ["-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-bufsize", str(br * 2)]
    out(f"目标码率: ~{br // 1000} kbps；帧数≈{frames}，源 fps≈{src_fps:.1f}")
    out("")
    encode_test(ffmpeg, "CPU libx264 medium", [], ["-c:v", "libx264", "-preset", "medium", *rate],
                src, frames, src_fps)
    if has_nvenc:
        encode_test(ffmpeg, "NVENC p5 单趟(新默认)", [],
                    ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", *rate], src, frames, src_fps)
        encode_test(ffmpeg, "NVENC p5 两趟multipass(旧)", [],
                    ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-multipass", "qres", *rate],
                    src, frames, src_fps)
        encode_test(ffmpeg, "NVENC p1 最快", [],
                    ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", *rate], src, frames, src_fps)
        if has_cuda_dec:
            encode_test(ffmpeg, "NVDEC解码+NVENC", ["-hwaccel", "cuda"],
                        ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", *rate], src, frames, src_fps)
            encode_test(ffmpeg, "全GPU留显存(无滤镜)", ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
                        ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", *rate], src, frames, src_fps)
    else:
        out("  未检测到 h264_nvenc，跳过 NVENC 测试。")


# --------------------------------------------------------------------- GPU 采样
def sample_gpu(ffmpeg: str, smi: str, src: str, bitrate: Optional[int],
               in_args: List[str]) -> None:
    section("3) 持续 NVENC 转码期间的 GPU 占用（nvidia-smi 采样）")
    br = bitrate or 8_000_000
    # 循环输入拉长到约 25s，便于稳定采样。
    enc_cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-stream_loop", "6", *in_args, "-i", src,
               "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-b:v", str(br),
               "-an", "-f", "null", "-"]
    out(f"采样命令: {' '.join(in_args)} -c:v h264_nvenc -preset p5 (循环源~25s)")
    try:
        enc = subprocess.Popen(enc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        out(f"启动编码失败: {e}")
        return
    sm_vals, enc_vals, dec_vals = [], [], []
    try:
        smon = subprocess.Popen([smi, "dmon", "-s", "u", "-d", "1"],
                                stdout=subprocess.PIPE, text=True)
    except Exception as e:
        out(f"启动 nvidia-smi dmon 失败: {e}")
        enc.wait()
        return
    t_end = time.time() + 30
    try:
        while enc.poll() is None and time.time() < t_end:
            line = smon.stdout.readline()
            if not line:
                break
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            def num(tok: str) -> Optional[float]:
                try:
                    return float(tok)
                except ValueError:
                    return None
            sm, mem, e, d = num(parts[1]), num(parts[2]), num(parts[3]), num(parts[4])
            if sm is not None:
                sm_vals.append(sm)
            if e is not None:
                enc_vals.append(e)
            if d is not None:
                dec_vals.append(d)
    finally:
        try:
            smon.terminate()
        except Exception:
            pass
        try:
            enc.terminate()
        except Exception:
            pass
        try:
            enc.wait(timeout=10)
        except Exception:
            pass

    def stat(name: str, vals: List[float]) -> None:
        if vals:
            out(f"  {name:<10} 样本 {len(vals):2d} | 均值 {sum(vals)/len(vals):5.1f}% | 峰值 {max(vals):5.1f}%")
        else:
            out(f"  {name:<10} 无有效样本")
    out("（sm=CUDA核心, enc=NVENC编码器, dec=NVDEC解码器；关注 enc/dec 而非 sm）")
    stat("SM(3D)", sm_vals)
    stat("ENC", enc_vals)
    stat("DEC", dec_vals)


# --------------------------------------------------------------------- 项目管线
def test_project_pipeline(ffmpeg: str, src: str) -> None:
    section("4) 本项目实际导出管线（确认是否真的走 NVENC，还是回退 CPU）")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import media_tools as mt
        from exporter import VideoExporter
        from masks import Mask
    except Exception as e:
        out(f"无法导入项目模块（需在项目目录、且已装 opencv-python/numpy）: {e}")
        return

    usable, _ = mt.detect_working_hw_encoders(ffmpeg)
    encoder = mt.pick_auto_encoder(usable)
    out(f"自动选择的编码器: {encoder}（可用硬件编码器: {', '.join(usable) or '无'}）")
    methods = mt.detect_hwaccels(ffmpeg)
    decode = mt.pick_decode_method(methods)
    out(f"自动选择的硬件解码: {decode or '无（CPU 解码）'}")
    out("")

    def run_case(name: str, mask: Mask) -> None:
        logs: List[str] = []
        ex = VideoExporter(ffmpeg_exec=ffmpeg, decode_method=decode, log=lambda s: logs.append(s))
        out_path = src + f".{name}.out.mp4"
        t0 = time.time()
        try:
            ex.export(src, out_path, masks=[mask], crop_enabled=False,
                      remove_audio=True, encoder=encoder, preserve_bitrate=True)
            elapsed = time.time() - t0
            joined = " | ".join(logs)
            soft = ("软件编码" in joined) or ("libx264" in joined)
            path_used = "融合GPU管线" if any("融合" in l for l in logs) else "逐帧管线"
            out(f"  {name:<16} 耗时 {elapsed:6.2f}s | {path_used} | "
                f"{'⚠回退到CPU软件编码' if soft else '硬件/NVENC 编码'}")
            for l in logs:
                if any(k in l for k in ("管线", "编码完成", "解码", "回退", "码率")):
                    out(f"      · {l}")
        except Exception as e:
            out(f"  {name:<16} 失败: {e}")
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

    static = Mask("gaussian", 80)
    static.set_keyframe(0, (100, 100, 400, 120))       # 静态 -> 应走融合 GPU 管线
    moving = Mask("gaussian", 80)
    moving.set_keyframe(0, (50, 50, 300, 100))
    moving.set_keyframe(9999, (900, 500, 300, 100))    # 移动 -> 逐帧管线
    run_case("静态遮罩", static)
    run_case("移动遮罩", moving)


# --------------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="显卡/编码诊断与基准")
    ap.add_argument("--ffmpeg", help="ffmpeg 可执行文件路径（缺省自动从 config.json / PATH 查找）")
    ap.add_argument("--video", help="用于测试的真实视频（缺省生成合成源）")
    ap.add_argument("--seconds", type=int, default=10, help="合成源时长秒（默认10）")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out("字幕/水印遮罩工具 —— 显卡基准报告")
    out(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 平台: {sys.platform} | Python: {sys.version.split()[0]}")

    ffmpeg = find_ffmpeg(args.ffmpeg)
    if not ffmpeg:
        out("未找到 ffmpeg。请用 --ffmpeg 指定路径，或确保 config.json 里的 ffmpeg_path 有效。")
        _write_report()
        return
    ffprobe = find_ffprobe(ffmpeg)

    probe_capabilities(ffmpeg)

    # 准备测试源
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.video and os.path.isfile(args.video):
        src = args.video
        out(f"\n使用真实视频: {src}")
        cleanup = False
    else:
        src = os.path.join(base_dir, "_bench_src.mp4")
        out(f"\n生成合成测试源: {args.width}x{args.height}@{args.fps} {args.seconds}s ...")
        make_source(ffmpeg, src, args.seconds, args.width, args.height, args.fps)
        cleanup = True
    if not os.path.isfile(src):
        out("测试源不可用，终止。")
        _write_report()
        return

    w, h, fps, frames, bitrate = probe_video(ffprobe, src)
    out(f"源信息: {w}x{h} @ {fps:.2f}fps, 帧数≈{frames}, 码率≈{(bitrate//1000) if bitrate else '未知'} kbps")

    enc = run([ffmpeg, "-hide_banner", "-encoders"]).stdout
    has_nvenc = "h264_nvenc" in enc
    has_cuda_dec = "cuda" in run([ffmpeg, "-hide_banner", "-hwaccels"]).stdout

    run_encode_matrix(ffmpeg, src, frames, fps, bitrate, has_nvenc, has_cuda_dec)

    smi = find_nvidia_smi()
    if smi and has_nvenc:
        in_args = ["-hwaccel", "cuda"] if has_cuda_dec else []
        sample_gpu(ffmpeg, smi, src, bitrate, in_args)
    else:
        section("3) GPU 占用采样")
        out("跳过（无 nvidia-smi 或无 NVENC）。")

    test_project_pipeline(ffmpeg, src)

    if cleanup:
        try:
            os.remove(src)
        except OSError:
            pass

    section("完成")
    out("请把下面这个文件的内容整段发回：")
    _write_report()


def _write_report() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_benchmark_report.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT_LINES) + "\n")
        print(f"\n>>> 报告已写入: {path}")
        print(">>> 请把该文件内容整段回传。")
    except Exception as e:
        print(f"写报告失败: {e}")


if __name__ == "__main__":
    main()
