"""融合式 ffmpeg GPU 管线：解码(NVDEC) + 区域滤镜 + 编码(NVENC) 单进程完成。

相比"逐帧解码 -> Python 处理 -> rawvideo 管道给 ffmpeg 编码"，本管线：
- 不把每帧拉回 Python，去掉了每帧 Python 循环与进程间 rawvideo 拷贝；
- 解码走硬件(NVDEC/…)，编码走硬件(NVENC)，中间的矩形模糊/马赛克用 ffmpeg 原生
  滤镜(crop+gblur/pixelize+overlay)完成。

限制（超出范围者由调用方回退到逐帧 Python 管线）：
- 仅支持"静态矩形遮挡"：单关键帧、方法为模糊/马赛克族；
- 移动(多关键帧)遮罩、inpaint/时域擦除、纯色/中值/毛玻璃不在此路径；
- 不做羽化软边（导出边缘略硬于预览）。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# 可用融合管线表达的方法（其余走逐帧 Python 路径）。
FUSED_METHODS = frozenset(
    {"gaussian", "box", "double", "pixelate", "pixelate_strong", "pixelate_gaussian"}
)


def masks_expressible(masks: list) -> bool:
    """是否可用融合 ffmpeg 管线：全部为静态(单关键帧)且方法在受支持集合内。"""
    usable = False
    for m in masks:
        if getattr(m, "method", "") not in FUSED_METHODS:
            return False
        kf = getattr(m, "keyframes", [])
        if len(kf) > 1:  # 移动遮罩不在此路径
            return False
        if kf:
            usable = True
    return usable


def _even(v: int) -> int:
    v = int(v)
    return max(2, v - (v % 2))


def _clamp_rect(rect: Tuple[int, int, int, int], width: int, height: int):
    x, y, w, h = rect
    cw = _even(min(int(w), width))
    ch = _even(min(int(h), height))
    x = _even(max(0, min(int(x), width - cw)))
    y = _even(max(0, min(int(y), height - ch)))
    return x, y, cw, ch


def _region_filter(method: str, strength: int) -> str:
    strength = max(5, int(strength))
    if method in ("pixelate", "pixelate_strong", "pixelate_gaussian"):
        block = max(6, strength // 4) if method == "pixelate_strong" else max(4, strength // 8)
        return f"pixelize=w={block}:h={block}"
    sigma = max(1.5, strength / 8.0)
    if method == "double":
        sigma *= 1.4
    return f"gblur=sigma={round(sigma, 2)}"


def build_filtergraph(
    masks: list,
    width: int,
    height: int,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[str, str]]:
    """生成 -filter_complex 字符串与最终输出标签；无可用遮罩返回 None。

    每个静态遮罩：切分画面 -> 裁出区域做模糊/马赛克 -> overlay 贴回原位。
    坐标均为整数，无需转义，稳健可靠。最后按需做 9:16 裁剪。
    """
    stages: List[str] = []
    cur = "0:v"
    idx = 0
    for mask in masks:
        kf = getattr(mask, "keyframes", [])
        if not kf:
            continue
        x, y, cw, ch = _clamp_rect(kf[0][1], width, height)
        if cw < 2 or ch < 2:
            continue
        filt = _region_filter(mask.method, mask.strength)
        base, region, blurred, out = f"b{idx}", f"r{idx}", f"rb{idx}", f"v{idx}"
        stages.append(f"[{cur}]split[{base}][{region}]")
        stages.append(f"[{region}]crop={cw}:{ch}:{x}:{y},{filt}[{blurred}]")
        stages.append(f"[{base}][{blurred}]overlay=x={x}:y={y}[{out}]")
        cur = out
        idx += 1
    if idx == 0:
        return None
    if crop_box:
        cx, cy, cw, ch = crop_box
        stages.append(f"[{cur}]crop={cw}:{ch}:{cx}:{cy}[vout]")
        cur = "vout"
    return ";".join(stages), cur


def build_command(
    ffmpeg_exec: str,
    src_path: str,
    out_path: str,
    filter_complex: str,
    final_label: str,
    encoder: str,
    encoder_extra: List[str],
    remove_audio: bool,
    hwaccel: Optional[str] = None,
) -> List[str]:
    """组装单进程 ffmpeg 命令：硬件解码 + 滤镜 + 硬件编码。音轨直接拷贝，避免重转码。"""
    args = [ffmpeg_exec, "-y", "-loglevel", "error"]
    if hwaccel:
        args += ["-hwaccel", hwaccel]
    args += ["-i", src_path, "-filter_complex", filter_complex, "-map", f"[{final_label}]"]
    if remove_audio:
        args.append("-an")
    else:
        args += ["-map", "0:a?", "-c:a", "copy"]
    args += ["-c:v", encoder, *encoder_extra, "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return args
