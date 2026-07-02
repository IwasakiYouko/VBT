import argparse
import time
import tracemalloc

import cv2
import numpy as np

import blur_core as bc


def run_benchmark(
    width: int,
    height: int,
    iterations: int,
    roi_ratio: float,
    strength: int,
) -> None:
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    roi_w = int(width * roi_ratio)
    roi_h = int(height * roi_ratio)
    roi = ((width - roi_w) // 2, (height - roi_h) // 2, roi_w, roi_h)

    methods = [
        "gaussian",
        "median",
        "box",
        "pixelate",
        "pixelate_strong",
        "double",
        "pixelate_gaussian",
    ]
    print(f"Frame: {width}x{height}, ROI: {roi_w}x{roi_h}, strength={strength}, iterations={iterations}")
    tracemalloc.start()
    results = []
    for method in methods:
        total = 0.0
        for _ in range(iterations):
            work = frame.copy()
            start = time.perf_counter()
            bc.apply_roi_blur(work, roi, method, strength)
            total += time.perf_counter() - start
        avg_ms = (total / iterations) * 1000.0
        fps = 1000.0 / avg_ms if avg_ms else 0.0
        results.append((method, avg_ms, fps))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nMethod                 Avg ms    Est. FPS")
    for method, avg_ms, fps in results:
        print(f"{method:<22} {avg_ms:7.2f}   {fps:7.1f}")
    print(f"\nPeak memory: {peak / 1024 / 1024:.2f} MB | Current: {current / 1024 / 1024:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="模糊算法性能基准（纯 CPU，使用合成帧）")
    parser.add_argument("--width", type=int, default=1280, help="测试帧宽度")
    parser.add_argument("--height", type=int, default=720, help="测试帧高度")
    parser.add_argument("--iterations", type=int, default=8, help="每种算法迭代次数")
    parser.add_argument("--roi-ratio", type=float, default=0.35, help="ROI 占比（0-1，按宽高中间区域）")
    parser.add_argument("--strength", type=int, default=80, help="模糊强度")
    args = parser.parse_args()
    run_benchmark(args.width, args.height, args.iterations, args.roi_ratio, args.strength)


if __name__ == "__main__":
    main()
