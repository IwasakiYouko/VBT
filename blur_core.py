from typing import Dict, Optional

import cv2


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
    region = blur_region(target.copy(), method, strength)
    target[:] = region
    return target


def apply_roi_blur(frame, roi, method: str, strength: int):
    """在给定 ROI 内套用与预览一致的模糊算法。"""
    if frame is None or not roi:
        return frame
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    end_x = min(x + rw, w)
    end_y = min(y + rh, h)
    if end_x <= x or end_y <= y:
        return frame
    target = frame[y:end_y, x:end_x]
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
