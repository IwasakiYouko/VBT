"""遮罩数据模型：单个视频可含多个遮罩，每个遮罩支持关键帧运动。

- 静态遮罩：只有 1 个关键帧，位置恒定。
- 移动遮罩：≥2 个关键帧，按帧号在关键帧之间线性插值，可跟随浮动水印。
每个遮罩带有独立的算法(method)与强度(strength)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Rect = Tuple[int, int, int, int]  # (x, y, w, h)
Keyframe = Tuple[int, Rect]       # (frame_index, rect)


@dataclass
class Mask:
    method: str = "gaussian"
    strength: int = 80
    # 按帧号升序排列的关键帧列表。
    keyframes: List[Keyframe] = field(default_factory=list)

    # ---------------------------------------------------------------- 关键帧
    def is_moving(self) -> bool:
        return len(self.keyframes) >= 2

    def frame_indexes(self) -> List[int]:
        return [f for f, _ in self.keyframes]

    def set_keyframe(self, frame_index: int, rect: Rect) -> None:
        """在指定帧记录/替换关键帧，并保持按帧号排序。"""
        frame_index = max(0, int(frame_index))
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        kept = [(f, r) for f, r in self.keyframes if f != frame_index]
        kept.append((frame_index, rect))
        kept.sort(key=lambda item: item[0])
        self.keyframes = kept

    def remove_keyframe(self, frame_index: int) -> bool:
        """删除指定帧的关键帧；若删除后仍有关键帧返回 True。"""
        before = len(self.keyframes)
        self.keyframes = [(f, r) for f, r in self.keyframes if f != frame_index]
        return len(self.keyframes) != before

    def make_static(self, rect: Optional[Rect] = None) -> None:
        """压缩为单关键帧（静态）。rect 为空时取第一个关键帧。"""
        if rect is None:
            rect = self.keyframes[0][1] if self.keyframes else None
        self.keyframes = [(0, tuple(int(v) for v in rect))] if rect else []  # type: ignore[arg-type]

    def rect_at(self, frame_index: int) -> Optional[Rect]:
        """返回该帧的矩形；关键帧之间线性插值，两端做保持外推。"""
        kf = self.keyframes
        if not kf:
            return None
        if len(kf) == 1:
            return kf[0][1]
        if frame_index <= kf[0][0]:
            return kf[0][1]
        if frame_index >= kf[-1][0]:
            return kf[-1][1]
        for i in range(len(kf) - 1):
            f0, r0 = kf[i]
            f1, r1 = kf[i + 1]
            if f0 <= frame_index <= f1:
                t = (frame_index - f0) / (f1 - f0) if f1 > f0 else 0.0
                return tuple(int(round(a + (b - a) * t)) for a, b in zip(r0, r1))  # type: ignore[return-value]
        return kf[-1][1]

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "strength": int(self.strength),
            "keyframes": [[int(f), [int(v) for v in r]] for f, r in self.keyframes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mask":
        mask = cls(
            method=str(data.get("method", "gaussian")),
            strength=int(data.get("strength", 80)),
        )
        for item in data.get("keyframes", []):
            try:
                f, r = item
                mask.set_keyframe(int(f), tuple(int(v) for v in r))
            except Exception:
                continue
        return mask
