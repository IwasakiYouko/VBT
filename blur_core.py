from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# 覆盖式遮挡算法（会遮住整块 ROI，可套用羽化软边）。
# inpaint / inpaint_temporal 为「擦除式」，只动文字像素，故不参与羽化。
_OCCLUSION_METHODS = frozenset(
    {"gaussian", "median", "box", "double", "pixelate",
     "pixelate_strong", "pixelate_gaussian", "frosted", "solid"}
)


def _safe_bgr(src):
    """确保输入为 8UC1 或 8UC3；不满足时尝试转 BGR，否则返回原始用于回退。"""
    if src is None:
        return None, 0
    if src.dtype != "uint8":
        try:
            src = src.astype("uint8", copy=False)
        except Exception:
            return None, 0
    if src.ndim == 2:
        return src, 1
    if src.ndim == 3:
        ch = src.shape[2]
        if ch == 3:
            return src, 3
        if ch == 4:
            try:
                return cv2.cvtColor(src, cv2.COLOR_BGRA2BGR), 3
            except Exception:
                return None, 0
    return None, 0


_cuda_ready: Optional[bool] = None
_cuda_filter_cache: Dict[tuple, "cv2.cuda.Filter"] = {}


def _cuda_available() -> bool:
    global _cuda_ready
    if _cuda_ready is None:
        try:
            _cuda_ready = (cv2.cuda.getCudaEnabledDeviceCount() or 0) > 0
        except Exception:
            _cuda_ready = False
    return bool(_cuda_ready)


def kernel_size(strength: int) -> int:
    k = max(3, int(strength) // 2)
    if k % 2 == 0:
        k += 1
    return min(k, 99)


def pixelate(region, block: int):
    h, w = region.shape[:2]
    block = max(2, min(block, w, h))
    small_w = max(1, w // block)
    small_h = max(1, h // block)
    temp = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def clamp_roi(roi, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """把 ROI 夹到画面范围内，返回 (x, y, end_x, end_y)；无效返回 None。"""
    if not roi:
        return None
    x, y, rw, rh = roi
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    end_x = min(int(x) + int(rw), w)
    end_y = min(int(y) + int(rh), h)
    if end_x <= x or end_y <= y:
        return None
    return x, y, end_x, end_y


_feather_cache: Dict[tuple, "np.ndarray"] = {}


def _feather_alpha(h: int, w: int, band: int) -> "np.ndarray":
    """生成中心为 1、四周线性衰减到 0 的软边 alpha（HxWx1, float32）。"""
    key = (h, w, band)
    cached = _feather_cache.get(key)
    if cached is not None:
        return cached
    mask = np.ones((h, w), np.float32)
    if band > 0 and h > 2 * band and w > 2 * band:
        ramp = np.linspace(0.0, 1.0, band, dtype=np.float32)
        mask[:band, :] *= ramp[:, None]
        mask[h - band:, :] *= ramp[::-1][:, None]
        mask[:, :band] *= ramp[None, :]
        mask[:, w - band:] *= ramp[::-1][None, :]
    alpha = mask[:, :, None]
    _feather_cache[key] = alpha
    return alpha


def detect_subtitle_mask(region, strength: int):
    """检测 ROI 内的字幕笔画：高亮白字 + 暗色描边等高对比像素。

    返回与 region 同尺寸的 uint8 掩膜（255 = 判定为字幕），无字幕返回 None。
    """
    if region is None or region.size == 0:
        return None
    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # 字幕多为近白填充 + 近黑描边；再叠加一层「明显亮于局部背景」的自适应判定。
    white = cv2.inRange(gray, 205, 255)
    dark = cv2.inRange(gray, 0, 55)
    mask = cv2.bitwise_or(white, dark)
    try:
        local = cv2.GaussianBlur(gray, (0, 0), max(3.0, region.shape[0] / 6.0))
        bright = cv2.inRange(cv2.subtract(gray, local), 40, 255)
        mask = cv2.bitwise_or(mask, bright)
    except Exception:
        pass
    # 连接笔画并向外扩张，覆盖抗锯齿边缘；扩张量随强度增大。
    grow = max(1, int(strength) // 40)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1 + grow)
    if cv2.countNonZero(mask) == 0:
        return None
    return mask


def inpaint_region(region, strength: int):
    """检测文字笔画并只对其做修复，保留其余真实背景。"""
    mask = detect_subtitle_mask(region, strength)
    if mask is None:
        return region
    # 掩膜几乎覆盖全区时 inpaint 无有效参考，退化为模糊以保证遮住字幕。
    if cv2.countNonZero(mask) / mask.size > 0.9:
        k = kernel_size(strength)
        return cv2.GaussianBlur(region, (k, k), 0)
    radius = max(3, int(strength) // 20)
    try:
        return cv2.inpaint(region, mask, radius, cv2.INPAINT_TELEA)
    except Exception:
        return region


def frosted_glass(region, strength: int):
    """毛玻璃/磨砂：强模糊叠加轻微噪声，观感比纯高斯更自然。"""
    k = kernel_size(strength)
    base = cv2.GaussianBlur(region, (k, k), 0)
    noise = np.random.randint(-8, 9, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def solid_bar(region, strength: int):
    """纯色遮块：用 ROI 平均色填充（新闻常见的马赛克条形式）。"""
    if region.ndim == 3:
        color = region.reshape(-1, region.shape[2]).mean(axis=0)
        out = np.empty_like(region)
        out[:] = color.astype(np.uint8)
        return out
    return np.full_like(region, int(region.mean()))


def composite_background(frame, background_roi, roi, strength: int):
    """时域擦除：把 ROI 内的字幕像素替换为干净背景，非字幕像素原样保留。

    background_roi 为提前算好的、与夹取后 ROI 等尺寸的干净背景图。
    """
    if frame is None or background_roi is None or not roi:
        return frame
    h, w = frame.shape[:2]
    box = clamp_roi(roi, w, h)
    if box is None:
        return frame
    x, y, end_x, end_y = box
    region = frame[y:end_y, x:end_x]
    bg = background_roi[: region.shape[0], : region.shape[1]]
    if bg.shape[:2] != region.shape[:2]:
        return frame
    mask = detect_subtitle_mask(region, strength)
    if mask is None:
        return frame
    # 掩膜羽化后做 alpha 混合，字幕处取背景、边缘平滑过渡。
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.5)
    alpha = alpha[:, :, None] if region.ndim == 3 else alpha
    region[:] = (region.astype(np.float32) * (1.0 - alpha) + bg.astype(np.float32) * alpha).astype(np.uint8)
    return frame


def blur_region(region, method: str, strength: int):
    k = kernel_size(strength)
    if method == "gaussian":
        return cv2.GaussianBlur(region, (k, k), 0)
    if method == "median":
        return cv2.medianBlur(region, k)
    if method == "box":
        return cv2.blur(region, (k, k))
    if method == "pixelate":
        return pixelate(region, block=max(4, strength // 8))
    if method == "pixelate_strong":
        return pixelate(region, block=max(6, strength // 4))
    if method == "double":
        tmp = cv2.GaussianBlur(region, (k, k), 0)
        return cv2.blur(tmp, (k, k))
    if method == "pixelate_gaussian":
        tmp = pixelate(region, block=max(4, strength // 8))
        return cv2.GaussianBlur(tmp, (k, k), 0)
    return cv2.GaussianBlur(region, (k, k), 0)


def apply_blur_to_target(target, method: str, strength: int):
    """尽量原地模糊 ROI，减少额外复制开销。"""
    if method == "surface":  # 兼容旧配置
        method = "gaussian"
    k = kernel_size(strength)
    if method == "gaussian":
        if _apply_cuda_filter(target, method, strength):
            return target
        return cv2.GaussianBlur(target, (k, k), 0, dst=target)
    if method == "box":
        if _apply_cuda_filter(target, method, strength):
            return target
        return cv2.blur(target, (k, k), dst=target)
    if method == "median":
        target[:] = cv2.medianBlur(target, k)
        return target
    if method == "double":
        if _apply_cuda_filter(target, method, strength):
            return target
        cv2.GaussianBlur(target, (k, k), 0, dst=target)
        return cv2.blur(target, (k, k), dst=target)
    if method == "inpaint" or method == "inpaint_temporal":
        # 时域擦除在无背景时退化为空间修复。
        target[:] = inpaint_region(target, strength)
        return target
    if method == "frosted":
        target[:] = frosted_glass(target, strength)
        return target
    if method == "solid":
        target[:] = solid_bar(target, strength)
        return target
    region = blur_region(target.copy(), method, strength)
    target[:] = region
    return target


def apply_roi_blur(frame, roi, method: str, strength: int, feather: bool = True):
    """在给定 ROI 内套用与预览一致的模糊/擦除算法。

    feather=True 时对覆盖式遮挡（模糊/马赛克/毛玻璃/纯色条）叠加羽化软边，
    让遮挡区与周围画面平滑过渡，避免生硬的矩形边界。
    """
    if frame is None or not roi:
        return frame
    h, w = frame.shape[:2]
    box = clamp_roi(roi, w, h)
    if box is None:
        return frame
    x, y, end_x, end_y = box
    target = frame[y:end_y, x:end_x]
    if feather and method in _OCCLUSION_METHODS:
        original = target.copy()
        processed = target.copy()
        apply_blur_to_target(processed, method, strength)
        band = max(2, int(min(processed.shape[0], processed.shape[1]) * 0.10))
        alpha = _feather_alpha(processed.shape[0], processed.shape[1], band)
        blended = processed.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
        target[:] = blended.astype(np.uint8)
    else:
        apply_blur_to_target(target, method, strength)
    return frame


def _get_cuda_filter(method: str, k: int, channels: int):
    if not _cuda_available():
        return None
    if channels not in (1, 3):
        return None
    src_type = cv2.CV_8UC3 if channels == 3 else cv2.CV_8UC1
    cache_key = (method, k, channels)
    flt = _cuda_filter_cache.get(cache_key)
    if flt is not None:
        return flt
    try:
        if method == "gaussian":
            flt = cv2.cuda.createGaussianFilter(src_type, src_type, (k, k), 0)
        elif method == "box":
            flt = cv2.cuda.createBoxFilter(src_type, src_type, (k, k))
        else:
            flt = None
    except Exception:
        flt = None
    if flt is not None:
        _cuda_filter_cache[cache_key] = flt
    return flt


def _apply_cuda_filter(target, method: str, strength: int) -> bool:
    if not _cuda_available():
        return False
    src, channels = _safe_bgr(target)
    if src is None or channels not in (1, 3):
        return False
    k = kernel_size(strength)
    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(src)
        if method == "double":
            g_flt = _get_cuda_filter("gaussian", k, channels)
            b_flt = _get_cuda_filter("box", k, channels)
            if g_flt is None or b_flt is None:
                return False
            tmp = g_flt.apply(gpu)
            gpu_out = b_flt.apply(tmp)
        else:
            flt = _get_cuda_filter(method, k, channels)
            if flt is None:
                return False
            gpu_out = flt.apply(gpu)
        gpu_out.download(target)
        return True
    except Exception:
        return False
