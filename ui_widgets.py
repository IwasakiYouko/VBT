"""可复用 UI 小部件：文本截断、悬停提示、双向滚动容器。

从 main.py 抽出的与业务无关的界面基础件，便于复用与测试。
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional, Union

import customtkinter as ctk

import theme

TextSource = Union[str, Callable[[], str]]


def ellipsize_middle(text: str, max_units: int = 44) -> str:
    """按显示宽度做中间截断（CJK 记 2 个单位），防止长文件名把布局撑宽。"""
    def units(ch: str) -> int:
        return 2 if ord(ch) > 0x2E80 else 1

    if sum(units(c) for c in text) <= max_units:
        return text
    head_budget = int((max_units - 2) * 0.6)
    tail_budget = max_units - 2 - head_budget
    head, used = [], 0
    for ch in text:
        if used + units(ch) > head_budget:
            break
        head.append(ch)
        used += units(ch)
    tail, used = [], 0
    for ch in reversed(text):
        if used + units(ch) > tail_budget:
            break
        tail.append(ch)
        used += units(ch)
    return "".join(head) + "…" + "".join(reversed(tail))


class Tooltip:
    """悬停提示。text 可以是字符串或返回字符串的函数（动态内容）。"""

    def __init__(self, widget, text: TextSource, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip: Optional[tk.Toplevel] = None
        self._job: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        try:
            self._job = self.widget.after(self.delay_ms, self._show)
        except Exception:
            self._job = None

    def _cancel(self) -> None:
        if self._job:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _show(self) -> None:
        self._job = None
        if self._tip is not None:
            return
        text = self.text() if callable(self.text) else self.text
        if not text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            tip = tk.Toplevel(self.widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tip.attributes("-topmost", True)
            tk.Label(
                tip, text=text, justify="left", wraplength=560,
                bg=theme.SURFACE_3, fg=theme.TEXT, relief="flat", bd=0,
                padx=10, pady=6, font=("Segoe UI", 9),
            ).pack()
            self._tip = tip
        except Exception:
            self._tip = None

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


class XYScrollFrame(ctk.CTkFrame):
    """内容可上下 + 左右滚动的容器；子控件放入 .inner。

    与 CTkScrollableFrame 不同：内容比视口宽时出现横向滚动条，
    且内容再宽也不会把外层布局撑大（Canvas 视口尺寸独立于内容）。
    """

    def __init__(self, master, fg_color: str = theme.SURFACE_DIM, **kwargs) -> None:
        super().__init__(master, fg_color=fg_color, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(self, bg=fg_color, highlightthickness=0, bd=0)
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        vbar = ctk.CTkScrollbar(self, orientation="vertical", command=self._canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self._canvas.xview, height=14)
        hbar.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.inner = ctk.CTkFrame(self._canvas, fg_color=fg_color)
        self._window = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._sync, add="+")
        self._canvas.bind("<Configure>", self._sync, add="+")
        self._bind_wheel(self._canvas)
        self._bind_wheel(self.inner)

    def _sync(self, _event=None) -> None:
        """内容短时铺满视口宽度，内容长时超出并由横向滚动条接管。"""
        try:
            req = self.inner.winfo_reqwidth()
            view = self._canvas.winfo_width()
            self._canvas.itemconfigure(self._window, width=max(req, view))
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        except Exception:
            pass

    # ---------------------------------------------------------------- 滚轮
    def _on_wheel(self, event) -> Optional[str]:
        delta = 0
        if getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) in (4, 5):
            delta = -1 if event.num == 4 else 1
        if delta == 0:
            return None
        try:
            if getattr(event, "state", 0) & 0x1:  # Shift 按下 -> 横向
                self._canvas.xview_scroll(delta, "units")
            else:
                self._canvas.yview_scroll(delta, "units")
        except Exception:
            return None
        return "break"

    def _bind_wheel(self, widget) -> None:
        for seq in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(seq, self._on_wheel, add="+")
            except Exception:
                pass

    def bind_tree(self) -> None:
        """为 inner 的现有子控件绑定滚轮（每次重建列表后调用）。"""
        for child in self.inner.winfo_children():
            self._bind_wheel(child)
        self._sync()
