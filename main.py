import os
import concurrent.futures
import threading
import time
import tkinter as tk
from typing import List, Optional, Tuple

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from customtkinter import filedialog

import blur_core as bc
import media_tools as mt
from app_config import AppConfig
from exporter import VideoExporter
from masks import Mask


class SubtitleBlurApp(ctk.CTk):
    """字幕消除工具主界面。"""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("字幕消除工具")
        self.geometry("800x600")
        self.minsize(800, 600)
        self.after(50, self._maximize_window)

        # 路径与配置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, "config.json")
        self.legacy_config_path = os.path.join(self.base_dir, "subtitle_blur_config.json")
        self.video_paths: List[str] = []
        self.output_dir: Optional[str] = None
        self.ffmpeg_path: Optional[str] = None
        self.is_processing = False
        self.processing_paths: set[str] = set()
        self.list_buttons: List[ctk.CTkButton] = []
        self.list_placeholder: Optional[ctk.CTkLabel] = None
        self.list_button_styles = {
            "normal": {
                "fg_color": "#1f232a",
                "hover_color": "#2a303a",
                "text_color": "#d7dbe2",
                "border_width": 0,
                "border_color": "#1f232a",
            },
            "active": {
                "fg_color": "#1e5bbf",
                "hover_color": "#174a9e",
                "text_color": "#ffffff",
                "border_width": 0,
                "border_color": "#1e5bbf",
            },
            "processing": {
                "fg_color": "#b45309",
                "hover_color": "#92400e",
                "text_color": "#ffffff",
                "border_width": 0,
                "border_color": "#b45309",
            },
            "active_processing": {
                "fg_color": "#b45309",
                "hover_color": "#92400e",
                "text_color": "#ffffff",
                "border_width": 2,
                "border_color": "#1e5bbf",
            },
        }

        # UI 变量
        self.blur_options: List[Tuple[str, str]] = [
            ("高斯模糊", "gaussian"),
            ("中值模糊", "median"),
            ("均值模糊", "box"),
            ("像素化", "pixelate"),
            ("强像素化", "pixelate_strong"),
            ("双重模糊", "double"),
            ("像素化+高斯", "pixelate_gaussian"),
            ("毛玻璃", "frosted"),
            ("纯色条", "solid"),
            ("字幕擦除(inpaint)", "inpaint"),
            ("字幕擦除(时域)", "inpaint_temporal"),
        ]
        self.blur_method_var = ctk.StringVar(value=self.blur_options[0][0])
        self.blur_strength = ctk.IntVar(value=80)
        self.remove_audio = ctk.BooleanVar(value=False)
        self.crop_9x16 = ctk.BooleanVar(value=True)
        self.remove_after = ctk.BooleanVar(value=False)
        self.encoder_options: List[Tuple[str, str]] = [
            ("自动检测（优先 NVENC / QSV / AMF）", "auto"),
            ("NVIDIA NVENC (h264_nvenc)", "h264_nvenc"),
            ("Intel QSV (h264_qsv)", "h264_qsv"),
            ("AMD AMF (h264_amf)", "h264_amf"),
            ("软件 x264 (libx264)", "libx264"),
            ("仅拷贝码流 (copy)", "copy"),
        ]
        self.encoder_var = ctk.StringVar(value=self.encoder_options[0][0])
        self.current_encoder = "auto"
        self.detected_hw_encoders: List[str] = []

        # 预览与处理区域状态
        self.preview_size: Tuple[int, int] = (540, 960)
        self.preview_image: Optional[ImageTk.PhotoImage] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.frame_size: Optional[Tuple[int, int]] = None
        self.preview_scale = 1.0
        self.preview_offset: Tuple[int, int] = (0, 0)
        # 每个视频一组遮罩；active_mask_index 指向当前编辑的遮罩。
        self.masks_map: dict[str, List[Mask]] = {}
        self.active_mask_index: Optional[int] = None
        self.mask_var = ctk.StringVar(value="无遮罩")
        self._loading_mask = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_rect: Optional[int] = None
        self.preview_index: Optional[int] = None
        self.preview_choices: List[str] = []
        self.preview_choice_map: dict[str, int] = {}
        self.preview_var = ctk.StringVar(value="请选择视频")
        self.current_frame_idx = 0
        self.fps = 25.0
        self._save_job: Optional[str] = None
        # 并行数量：最多 CPU 核心数，上限 8
        self.worker_count = max(1, min(os.cpu_count() or 4, 8))
        self.cap_lock = threading.Lock()
        self.preview_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview")
        self._preview_generation = 0
        self._seek_job: Optional[str] = None
        self._pending_seek_frame: Optional[int] = None
        self._seek_token = 0
        self._ignore_seek_event = False
        self._seek_cooldown_job: Optional[str] = None
        self._seeking_active = False
        self._last_seek_frame: Optional[int] = None
        self.hwaccel_method: Optional[str] = None
        self.hwaccel_device: Optional[str] = None
        self._hwaccel_checked = False
        self._using_hw_preview = False
        self._resize_job: Optional[str] = None
        self._strength_job: Optional[str] = None
        self._last_frame_cache: Optional[Tuple[int, int, "cv2.Mat"]] = None
        self._preview_image_item: Optional[int] = None
        self._encoder_probe_cache: dict[str, List[str]] = {}
        self._encoder_probe_running: set[str] = set()
        self._encoder_probe_lock = threading.Lock()

        self._load_config()
        self._enable_hardware_accel()
        self._build_layout()
        self._apply_saved_paths()
        self._start_encoder_probe_if_needed()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_layout(self) -> None:
        # 预览在左（列 0，权重大），列表+设置在右（列 1，固定宽）
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=1, minsize=280)
        self.grid_rowconfigure(0, weight=1)

        # ── 右侧面板（列表 + 日志） ──
        panel_list = ctk.CTkFrame(self, corner_radius=12)
        panel_list.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        panel_list.grid_columnconfigure(0, weight=1)
        panel_list.grid_rowconfigure(1, weight=2)
        panel_list.grid_rowconfigure(4, weight=1)

        # ── 左侧面板（预览 + 设置） ──
        panel_preview = ctk.CTkFrame(self, corner_radius=12)
        panel_preview.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=12)
        panel_preview.grid_columnconfigure(0, weight=1)
        panel_preview.grid_rowconfigure(0, weight=7)
        panel_preview.grid_rowconfigure(1, weight=2, minsize=300)

        # ===== 视频列表区 =====
        ctk.CTkLabel(panel_list, text="视频列表", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 6)
        )
        self.list_frame = ctk.CTkScrollableFrame(panel_list, height=260, corner_radius=10, fg_color="#141820")
        self.list_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.list_frame.grid_columnconfigure(0, weight=1)
        self._refresh_list()

        btn_frame = ctk.CTkFrame(panel_list, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        btn_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="listbtns")
        ctk.CTkButton(btn_frame, text="添加视频", command=self.add_videos).grid(
            row=0, column=0, padx=(0, 3), pady=0, sticky="ew"
        )
        ctk.CTkButton(btn_frame, text="移除选中", command=self._remove_selected_video).grid(
            row=0, column=1, padx=3, pady=0, sticky="ew"
        )
        ctk.CTkButton(
            btn_frame, text="清空列表", command=self.clear_list,
            fg_color="#7f1d1d", hover_color="#991b1b",
        ).grid(row=0, column=2, padx=(3, 0), pady=0, sticky="ew")

        # ===== 日志区 =====
        ctk.CTkLabel(panel_list, text="日志", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=3, column=0, sticky="w", padx=14
        )
        self.log_box = ctk.CTkTextbox(panel_list, corner_radius=10, font=ctk.CTkFont(size=12))
        self.log_box.grid(row=4, column=0, sticky="nsew", padx=12, pady=(6, 4))
        self.progress_bar = ctk.CTkProgressBar(panel_list, height=8, corner_radius=4)
        self.progress_bar.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.progress_bar.set(0)
        self.status_label = ctk.CTkLabel(panel_list, text="就绪", anchor="w", text_color="#7a8fa6")
        self.status_label.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 10))

        # ===== 预览画布区 =====
        preview_frame = ctk.CTkFrame(panel_preview, corner_radius=12)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(12, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=8)
        preview_frame.grid_rowconfigure(0, weight=0)
        preview_frame.grid_rowconfigure(2, weight=0)
        preview_frame.grid_rowconfigure(3, weight=0)

        header = ctk.CTkFrame(preview_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="处理 / 预览", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.preview_menu = ctk.CTkOptionMenu(
            header,
            variable=self.preview_var,
            values=self.preview_choices or ["请选择视频"],
            command=self._on_preview_select,
            width=220,
        )
        self.preview_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        canvas_wrap = ctk.CTkFrame(preview_frame, fg_color="#0d1017", corner_radius=10)
        canvas_wrap.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 6))
        canvas_wrap.grid_columnconfigure(0, weight=1)
        canvas_wrap.grid_rowconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(
            canvas_wrap,
            width=self.preview_size[0],
            height=self.preview_size[1],
            bg="#0d1017",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.bind("<Configure>", self._on_window_resize)

        seek_row = ctk.CTkFrame(preview_frame, fg_color="transparent")
        seek_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 4))
        seek_row.grid_columnconfigure(1, weight=1)
        self.frame_label = ctk.CTkLabel(seek_row, text="帧 0/0 (00:00/00:00)", text_color="#8a9ab0")
        self.frame_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.timeline_slider = ctk.CTkSlider(seek_row, from_=0, to=1, command=self._on_seek)
        self.timeline_slider.grid(row=0, column=1, sticky="ew")

        preview_actions = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        preview_actions.grid_columnconfigure(0, weight=1)
        self.roi_label = ctk.CTkLabel(
            preview_actions, text="在画面中拖拽框选遮罩区域", text_color="#8a9ab0", anchor="w"
        )
        self.roi_label.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(preview_actions, text="删除关键帧", width=96, command=self._delete_current_keyframe).grid(
            row=0, column=1, sticky="e", padx=(8, 4)
        )
        ctk.CTkButton(
            preview_actions, text="转为静态", width=84, command=self._make_mask_static
        ).grid(row=0, column=2, sticky="e")

        # ===== 设置面板 =====
        settings = ctk.CTkScrollableFrame(panel_preview, corner_radius=12, height=300)
        settings.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=(0, 6))
        settings.grid_columnconfigure(1, weight=1)
        self.settings_frame = settings
        self._bind_mousewheel_to_settings(settings)

        def _section(label: str, row: int) -> None:
            sf = ctk.CTkFrame(settings, fg_color="transparent")
            sf.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 2))
            sf.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                sf, text=label, font=ctk.CTkFont(size=11, weight="bold"), text_color="#4a7fc1"
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkFrame(sf, height=1, fg_color="#1e3a5f").grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # ── 遮罩设置 ──
        _section("▸  遮罩设置", 0)
        ctk.CTkLabel(settings, text="遮罩").grid(row=1, column=0, sticky="w", padx=14, pady=(6, 4))
        mask_row = ctk.CTkFrame(settings, fg_color="transparent")
        mask_row.grid(row=1, column=1, sticky="ew", padx=12, pady=(6, 4))
        mask_row.grid_columnconfigure(0, weight=1)
        self.mask_menu = ctk.CTkOptionMenu(
            mask_row, variable=self.mask_var, values=["无遮罩"], command=self._on_mask_select
        )
        self.mask_menu.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(mask_row, text="＋", width=34, command=self._add_mask).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkButton(
            mask_row, text="－", width=34, command=self._delete_mask,
            fg_color="#7f1d1d", hover_color="#991b1b",
        ).grid(row=0, column=2)

        ctk.CTkLabel(settings, text="算法").grid(row=2, column=0, sticky="w", padx=14, pady=(6, 4))
        ctk.CTkOptionMenu(
            settings,
            variable=self.blur_method_var,
            values=[label for label, _ in self.blur_options],
            command=self._on_blur_method_change,
        ).grid(row=2, column=1, sticky="ew", padx=12, pady=(6, 4))

        strength_row = ctk.CTkFrame(settings, fg_color="transparent")
        strength_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        strength_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(strength_row, text="强度").grid(row=0, column=0, sticky="w", padx=(2, 8))
        ctk.CTkSlider(
            strength_row, from_=5, to=120, variable=self.blur_strength, command=self._on_blur_strength_change
        ).grid(row=0, column=1, sticky="ew")
        ctk.CTkLabel(strength_row, textvariable=self.blur_strength, width=34, anchor="e").grid(
            row=0, column=2, padx=(6, 0)
        )
        ctk.CTkLabel(
            settings,
            text="提示：先「＋」添加遮罩再框选；在不同帧重新框选即生成移动关键帧以跟随水印。",
            text_color="#5c6b82", font=ctk.CTkFont(size=11), wraplength=360, justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4))

        # ── 编码设置 ──
        _section("▸  编码设置", 5)
        ctk.CTkLabel(settings, text="编码器").grid(row=6, column=0, sticky="w", padx=14, pady=(6, 6))
        ctk.CTkOptionMenu(
            settings,
            variable=self.encoder_var,
            values=[label for label, _ in self.encoder_options],
            command=self._on_encoder_change,
        ).grid(row=6, column=1, sticky="ew", padx=12, pady=(6, 6))

        # ── 处理选项 ──
        _section("▸  处理选项", 7)
        opts = ctk.CTkFrame(settings, fg_color="transparent")
        opts.grid(row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=(4, 6))
        opts.grid_columnconfigure(0, weight=1)
        ctk.CTkCheckBox(
            opts, text="去除音轨", variable=self.remove_audio, corner_radius=8, command=self._on_option_toggle
        ).grid(row=0, column=0, sticky="w", pady=3)
        ctk.CTkCheckBox(
            opts, text="裁剪为 9:16", variable=self.crop_9x16, corner_radius=8, command=self._on_option_toggle
        ).grid(row=1, column=0, sticky="w", pady=3)
        ctk.CTkCheckBox(
            opts, text="导出后从列表移除", variable=self.remove_after, corner_radius=8, command=self._on_option_toggle
        ).grid(row=2, column=0, sticky="w", pady=3)

        # ── 路径与导出 ──
        _section("▸  路径与导出", 9)
        ctk.CTkLabel(settings, text="ffmpeg").grid(row=10, column=0, sticky="w", padx=14, pady=(6, 2))
        path_frame = ctk.CTkFrame(settings, fg_color="transparent")
        path_frame.grid(row=10, column=1, sticky="ew", padx=12, pady=(6, 2))
        path_frame.grid_columnconfigure(0, weight=1)
        self.ffmpeg_entry = ctk.CTkEntry(path_frame, placeholder_text="自动/手动选择", corner_radius=8)
        self.ffmpeg_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(path_frame, text="浏览", width=72, command=self.choose_ffmpeg).grid(row=0, column=1)

        ctk.CTkLabel(settings, text="导出目录").grid(row=11, column=0, sticky="w", padx=14, pady=(6, 6))
        out_frame = ctk.CTkFrame(settings, fg_color="transparent")
        out_frame.grid(row=11, column=1, sticky="ew", padx=12, pady=(6, 6))
        out_frame.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(out_frame, placeholder_text="默认：源文件目录", corner_radius=8)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(out_frame, text="选择", width=72, command=self.choose_output_dir).grid(row=0, column=1)

        # ── 动作按钮 ──
        action_row = ctk.CTkFrame(settings, fg_color="transparent")
        action_row.grid(row=12, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 14))
        action_row.grid_columnconfigure((0, 1), weight=1, uniform="action")
        ctk.CTkButton(
            action_row, text="开始处理", command=self.start_processing, height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            action_row, text="预览设置", command=self.preview_settings, height=42,
            fg_color="#1e4d8c", hover_color="#163d70",
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self._clear_preview_canvas()
        self._update_preview_selector()

    # ------------------------------------------------------------------ 文件操作
    def add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.mov *.mkv *.avi *.m4v *.wmv"), ("所有文件", "*.*")],
        )
        if not paths:
            return
        for p in paths:
            if p not in self.video_paths:
                self.video_paths.append(p)
        self._refresh_list()
        self._update_preview_selector()
        if self.preview_index is None and self.video_paths:
            self._select_preview_index(0)
        self._set_status(f"已添加 {len(paths)} 个视频")

    def clear_list(self) -> None:
        self.video_paths.clear()
        self.masks_map.clear()
        self.active_mask_index = None
        self._refresh_list()
        self._update_preview_selector()
        self._clear_preview_canvas()
        if self._seek_job:
            self.after_cancel(self._seek_job)
            self._seek_job = None
        if self._seek_cooldown_job:
            self.after_cancel(self._seek_cooldown_job)
            self._seek_cooldown_job = None
        self._seeking_active = False
        self._last_seek_frame = None
        self._pending_seek_frame = None
        self._seek_token += 1
        with self.cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self._preview_generation += 1
        self.total_frames = 0
        self.progress_bar.set(0)
        self._set_status("已清空列表")

    def _remove_selected_video(self) -> None:
        if not self.video_paths:
            self._set_status("列表为空")
            return
        if self.preview_index is None or not (0 <= self.preview_index < len(self.video_paths)):
            self._set_status("未选择要移除的视频")
            return
        path = self.video_paths[self.preview_index]
        self._remove_video_from_list(path)
        self._set_status(f"已移除: {os.path.basename(path)}")

    def _remove_video_from_list(self, path: str) -> None:
        if path in self.video_paths:
            idx = self.video_paths.index(path)
            self.video_paths.pop(idx)
            self.masks_map.pop(path, None)
            self.processing_paths.discard(path)
            self._refresh_list()
            self._update_preview_selector()
            if self.video_paths:
                new_idx = min(idx, len(self.video_paths) - 1)
                self._select_preview_index(new_idx)
            else:
                self._clear_preview_canvas()

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择导出目录")
        if path:
            self.output_dir = path
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)
            self._save_config()

    def choose_ffmpeg(self) -> None:
        path = filedialog.askopenfilename(title="选择 ffmpeg 可执行文件", filetypes=[("可执行文件", "*")])
        if path:
            self.ffmpeg_path = path
            self.ffmpeg_entry.delete(0, "end")
            self.ffmpeg_entry.insert(0, path)
            self._save_config()
            self._start_encoder_probe_if_needed(force=True)

    # ------------------------------------------------------------------ 预览与区域选择
    def _update_preview_selector(self) -> None:
        if not hasattr(self, "preview_menu"):
            return
        values = [f"{idx + 1}. {os.path.basename(p)}" for idx, p in enumerate(self.video_paths)]
        self.preview_choices = values
        self.preview_choice_map = {v: idx for idx, v in enumerate(values)}
        if not values:
            self.preview_menu.configure(values=["请选择视频"])
            self.preview_var.set("请选择视频")
            self.preview_index = None
            self.active_mask_index = None
            if self.cap:
                self.cap.release()
                self.cap = None
            self.total_frames = 0
            self._update_mask_selector()
            self._clear_preview_canvas()
            return
        self.preview_menu.configure(values=values)
        if self.preview_index is None or self.preview_index >= len(values):
            self.preview_index = 0
            self.preview_var.set(values[self.preview_index])
        self._select_preview_index(self.preview_index, refresh_choices=False)

    def _on_preview_select(self, value: str) -> None:
        idx = self.preview_choice_map.get(value)
        if idx is None:
            return
        self._select_preview_index(idx, refresh_choices=False)

    def _select_preview_index(self, index: int, refresh_choices: bool = True) -> None:
        if not (0 <= index < len(self.video_paths)):
            return
        if refresh_choices:
            self._update_preview_selector()
        self.preview_index = index
        if self.preview_choices and index < len(self.preview_choices):
            self.preview_var.set(self.preview_choices[index])
        self._load_video(index)
        self._apply_list_highlight()

    def _invalidate_preview_cache(self) -> None:
        self._last_frame_cache = None

    def _store_preview_cache(self, generation: int, frame_idx: int, frame) -> None:
        try:
            self._last_frame_cache = (generation, frame_idx, frame.copy())
        except Exception:
            self._last_frame_cache = None

    def _render_cached_frame(self, frame_idx: int, update_slider: bool) -> bool:
        cached = self._last_frame_cache
        if not cached:
            return False
        gen, idx, frame = cached
        if gen != self._preview_generation or idx != frame_idx:
            return False
        try:
            self._render_frame(frame_idx, frame.copy(), update_slider)
            return True
        except Exception:
            return False

    def _load_video(self, idx: int) -> None:
        if not (0 <= idx < len(self.video_paths)):
            return
        path = self.video_paths[idx]
        self._ensure_hwaccel_detected()
        if self._seek_job:
            self.after_cancel(self._seek_job)
            self._seek_job = None
        self._pending_seek_frame = None
        self._seek_token += 1
        self._invalidate_preview_cache()
        with self.cap_lock:
            if self.cap:
                self.cap.release()
            self.cap = self._open_video_capture(path, update_hw_flag=True)
            self._preview_generation += 1
            cap_obj = self.cap
        if not cap_obj or not cap_obj.isOpened():
            self._log(f"无法打开视频: {path}")
            with self.cap_lock:
                self.cap = None
            self._clear_preview_canvas()
            return
        if self._using_hw_preview and self.hwaccel_method:
            self._log(f"预览硬件解码开启 ({self.hwaccel_method})")
        else:
            self._log("预览解码使用 CPU")
        self.preview_index = idx
        self.total_frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.timeline_slider.configure(from_=0, to=max(self.total_frames - 1, 1))
        self.current_frame_idx = 0
        self._load_masks_for_path(path)
        self._seeking_active = False
        self._last_seek_frame = None
        self._show_frame(0)
        self._set_status(f"预览: {os.path.basename(path)}")

    def _show_frame(self, frame_idx: int) -> None:
        if self._render_cached_frame(frame_idx, update_slider=True):
            return
        self._request_preview_frame(frame_idx, use_async=True, update_slider=True)

    def _request_preview_frame(self, frame_idx: int, use_async: bool, update_slider: bool) -> None:
        if not self.cap:
            self._clear_preview_canvas()
            return
        if self.total_frames:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        generation = self._preview_generation
        if use_async:
            self._seek_token += 1
            token = self._seek_token
            self.preview_executor.submit(self._decode_frame_task, frame_idx, generation, token, update_slider)
            return
        result = self._read_frame(frame_idx, generation, None)
        if not result:
            self._log("读取帧失败")
            return
        idx, frame = result
        self._store_preview_cache(generation, idx, frame)
        self._render_frame(idx, frame, update_slider)

    def _read_frame(
        self, frame_idx: int, generation: int, token: Optional[int]
    ) -> Optional[Tuple[int, "cv2.Mat"]]:
        with self.cap_lock:
            if not self.cap or generation != self._preview_generation:
                return None
            if token is not None and token != self._seek_token:
                return None
            if self.total_frames:
                frame_idx = max(0, min(frame_idx, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
        if token is not None and token != self._seek_token:
            return None
        if ok and frame is not None:
            return frame_idx, frame
        return None

    def _decode_frame_task(self, frame_idx: int, generation: int, token: int, update_slider: bool) -> None:
        result = self._read_frame(frame_idx, generation, token)
        if not result:
            return
        idx, frame = result
        self.after(0, self._render_frame_if_current, idx, frame, generation, token, update_slider)

    def _render_frame_if_current(
        self, frame_idx: int, frame: "cv2.Mat", generation: int, token: int, update_slider: bool
    ) -> None:
        if generation != self._preview_generation:
            return
        if token != self._seek_token:
            return
        self._store_preview_cache(generation, frame_idx, frame)
        self._render_frame(frame_idx, frame, update_slider)

    def _render_frame(self, frame_idx: int, frame, update_slider: bool) -> None:
        orig_h, orig_w, _ = frame.shape
        self.frame_size = (orig_w, orig_h)
        disp_w, disp_h, scale, off_x, off_y = self._fit_to_preview(orig_w, orig_h)
        apply_blur = bool(self._current_masks()) and not self._seeking_active
        if apply_blur:
            frame = self._apply_preview_masks(frame, frame_idx)
        if (orig_w, orig_h) != (disp_w, disp_h):
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.preview_image = ImageTk.PhotoImage(image)
        center_x = self.preview_size[0] // 2
        center_y = self.preview_size[1] // 2
        if self._preview_image_item is None:
            self.preview_canvas.delete("all")
            self._preview_image_item = self.preview_canvas.create_image(center_x, center_y, image=self.preview_image)
        else:
            self.preview_canvas.itemconfig(self._preview_image_item, image=self.preview_image)
            self.preview_canvas.coords(self._preview_image_item, center_x, center_y)
        self.preview_canvas.delete("roi")
        self.preview_scale = scale
        self.preview_offset = (off_x, off_y)
        self._draw_mask_rects(frame_idx)
        self.current_frame_idx = frame_idx
        if update_slider:
            self._ignore_seek_event = True
            try:
                self.timeline_slider.set(frame_idx)
            finally:
                self._ignore_seek_event = False
        self._update_frame_label()

    def _update_frame_label(self) -> None:
        total = self.total_frames or 0
        cur = self.current_frame_idx
        cur_time = self._format_time(cur / self.fps if self.fps else 0)
        total_time = self._format_time(total / self.fps if self.fps else 0)
        self.frame_label.configure(text=f"帧 {cur}/{total}  ({cur_time} / {total_time})")

    def _format_time(self, seconds: float) -> str:
        seconds = max(0, seconds)
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _fit_to_preview(self, frame_w: int, frame_h: int) -> Tuple[int, int, float, int, int]:
        canvas_w, canvas_h = self.preview_size
        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        disp_w = int(frame_w * scale)
        disp_h = int(frame_h * scale)
        off_x = (canvas_w - disp_w) // 2
        off_y = (canvas_h - disp_h) // 2
        return disp_w, disp_h, scale, off_x, off_y

    def _on_seek(self, value: float) -> None:
        if self._ignore_seek_event:
            return
        if not self.cap or not self.total_frames:
            return
        self._seeking_active = True
        if self._seek_cooldown_job:
            self.after_cancel(self._seek_cooldown_job)
            self._seek_cooldown_job = self.after(200, self._end_seek_cooldown)
        frame_idx = int(float(value))
        self._pending_seek_frame = frame_idx
        self._last_seek_frame = frame_idx
        if self._seek_job is None:
            self._seek_job = self.after(60, self._flush_seek_request)

    def _flush_seek_request(self) -> None:
        self._seek_job = None
        if self._pending_seek_frame is None:
            return
        frame_idx = self._pending_seek_frame
        self._pending_seek_frame = None
        self._request_preview_frame(frame_idx, use_async=True, update_slider=False)

    def _end_seek_cooldown(self) -> None:
        self._seek_cooldown_job = None
        self._seeking_active = False
        if self.cap:
            target = self._last_seek_frame if self._last_seek_frame is not None else self.current_frame_idx
            self._last_seek_frame = None
            self._request_preview_frame(target, use_async=True, update_slider=True)

    # ------------------------------------------------------------------ 遮罩管理
    def _current_masks(self) -> List[Mask]:
        if self.preview_index is None or not (0 <= self.preview_index < len(self.video_paths)):
            return []
        path = self.video_paths[self.preview_index]
        return self.masks_map.setdefault(path, [])

    def _active_mask(self) -> Optional[Mask]:
        masks = self._current_masks()
        if self.active_mask_index is not None and 0 <= self.active_mask_index < len(masks):
            return masks[self.active_mask_index]
        return None

    def _add_mask(self) -> None:
        if self.preview_index is None:
            self._set_status("请先选择一个视频")
            return
        masks = self._current_masks()
        mask = Mask(
            method=self._normalize_blur_method(self.blur_method_var.get()),
            strength=max(int(self.blur_strength.get()), 5),
        )
        masks.append(mask)
        self.active_mask_index = len(masks) - 1
        self._update_mask_selector()
        self._log(f"已添加遮罩 {self.active_mask_index + 1}，请在画面中框选区域")
        self._show_frame(self.current_frame_idx)

    def _delete_mask(self) -> None:
        masks = self._current_masks()
        if not masks or self.active_mask_index is None:
            self._set_status("没有可删除的遮罩")
            return
        removed = self.active_mask_index + 1
        masks.pop(self.active_mask_index)
        self.active_mask_index = (len(masks) - 1) if masks else None
        self._update_mask_selector()
        self._log(f"已删除遮罩 {removed}")
        self._show_frame(self.current_frame_idx)

    def _delete_current_keyframe(self) -> None:
        mask = self._active_mask()
        if not mask:
            self._set_status("请先选择遮罩")
            return
        if mask.remove_keyframe(self.current_frame_idx):
            self._log(f"已删除遮罩 {self.active_mask_index + 1} 在第 {self.current_frame_idx} 帧的关键帧")
        else:
            self._set_status("当前帧没有关键帧")
        self._update_mask_label()
        self._show_frame(self.current_frame_idx)

    def _make_mask_static(self) -> None:
        mask = self._active_mask()
        if not mask or not mask.keyframes:
            self._set_status("请先选择带关键帧的遮罩")
            return
        rect = mask.rect_at(self.current_frame_idx)
        mask.make_static(rect)
        self._log(f"遮罩 {self.active_mask_index + 1} 已转为静态")
        self._update_mask_label()
        self._show_frame(self.current_frame_idx)

    def _on_mask_select(self, value: str) -> None:
        masks = self._current_masks()
        try:
            idx = int(value.split()[-1]) - 1
        except (ValueError, IndexError):
            return
        if 0 <= idx < len(masks):
            self.active_mask_index = idx
            self._load_active_mask_controls()
            self._update_mask_label()
            self._show_frame(self.current_frame_idx)

    def _load_masks_for_path(self, path: str) -> None:
        masks = self.masks_map.setdefault(path, [])
        self.active_mask_index = 0 if masks else None
        self._update_mask_selector()

    def _load_active_mask_controls(self) -> None:
        """把当前遮罩的算法/强度回填到控件（避免触发写回）。"""
        mask = self._active_mask()
        if not mask:
            return
        self._loading_mask = True
        try:
            self.blur_method_var.set(self._blur_label_from_value(mask.method))
            self.blur_strength.set(int(mask.strength))
        finally:
            self._loading_mask = False

    def _update_mask_selector(self) -> None:
        if not hasattr(self, "mask_menu"):
            return
        masks = self._current_masks()
        values = [f"遮罩 {i + 1}" for i in range(len(masks))] or ["无遮罩"]
        self.mask_menu.configure(values=values)
        if masks and self.active_mask_index is not None:
            self.active_mask_index = min(self.active_mask_index, len(masks) - 1)
            self.mask_var.set(values[self.active_mask_index])
            self._load_active_mask_controls()
        else:
            self.active_mask_index = None
            self.mask_var.set("无遮罩")
        self._update_mask_label()

    def _update_mask_label(self) -> None:
        mask = self._active_mask()
        if not mask:
            self.roi_label.configure(text="在画面中拖拽框选遮罩区域（先「＋」添加遮罩）")
            return
        rect = mask.rect_at(self.current_frame_idx)
        if not rect:
            self.roi_label.configure(text=f"遮罩 {self.active_mask_index + 1}: 未框选，拖拽以设置")
            return
        x, y, w, h = rect
        motion = f"移动·{len(mask.keyframes)}关键帧" if mask.is_moving() else "静态"
        self.roi_label.configure(
            text=f"遮罩 {self.active_mask_index + 1} [{motion}]  x={x} y={y} w={w} h={h}"
        )

    def _draw_mask_rects(self, frame_idx: int) -> None:
        for i, mask in enumerate(self._current_masks()):
            rect = mask.rect_at(frame_idx)
            if not rect:
                continue
            rx1, ry1, rx2, ry2 = self._video_to_display_rect(rect)
            if i == self.active_mask_index:
                self.preview_canvas.create_rectangle(
                    rx1, ry1, rx2, ry2, outline="#e74c3c", width=2, tags="roi"
                )
            else:
                self.preview_canvas.create_rectangle(
                    rx1, ry1, rx2, ry2, outline="#f59e0b", width=1, dash=(4, 3), tags="roi"
                )
            self.preview_canvas.create_text(
                rx1 + 2, ry1 + 8, anchor="w", text=str(i + 1),
                fill="#ffffff", font=("Arial", 10, "bold"), tags="roi",
            )

    def _on_canvas_press(self, event: tk.Event) -> None:
        if not self.cap:
            return
        self.drag_start = (event.x, event.y)
        if self.drag_rect:
            self.preview_canvas.delete(self.drag_rect)
        self.drag_rect = self.preview_canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#e74c3c", width=2, dash=(3, 2)
        )

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if not self.drag_start or not self.drag_rect:
            return
        self.preview_canvas.coords(self.drag_rect, self.drag_start[0], self.drag_start[1], event.x, event.y)

    def _on_canvas_release(self, event: tk.Event) -> None:
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        if self.drag_rect:
            self.preview_canvas.delete(self.drag_rect)
            self.drag_rect = None
        self.drag_start = None
        rect = self._display_to_video_rect(x0, y0, x1, y1)
        if not rect:
            self._log("区域过小或无效")
            self._show_frame(self.current_frame_idx)
            return
        mask = self._active_mask()
        if mask is None:
            # 尚无遮罩时，框选自动创建一个。
            self._add_mask()
            mask = self._active_mask()
        if mask is not None:
            mask.set_keyframe(self.current_frame_idx, rect)
            kf = "关键帧" if mask.is_moving() else "区域"
            self._log(
                f"遮罩 {self.active_mask_index + 1} 第 {self.current_frame_idx} 帧{kf}: "
                f"x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}"
            )
            self._update_mask_label()
        self._show_frame(self.current_frame_idx)

    def _display_to_video_rect(self, x0: int, y0: int, x1: int, y1: int) -> Optional[Tuple[int, int, int, int]]:
        if not self.frame_size:
            return None
        sx = self.preview_scale or 1.0
        ox, oy = self.preview_offset
        vx0 = int((x0 - ox) / sx)
        vy0 = int((y0 - oy) / sx)
        vx1 = int((x1 - ox) / sx)
        vy1 = int((y1 - oy) / sx)
        vx0, vx1 = sorted((vx0, vx1))
        vy0, vy1 = sorted((vy0, vy1))
        vx0 = max(vx0, 0)
        vy0 = max(vy0, 0)
        vx1 = min(vx1, self.frame_size[0])
        vy1 = min(vy1, self.frame_size[1])
        w = vx1 - vx0
        h = vy1 - vy0
        if w < 4 or h < 4:
            return None
        return vx0, vy0, w, h

    def _video_to_display_rect(self, roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = roi
        sx = self.preview_scale or 1.0
        ox, oy = self.preview_offset
        rx0 = int(x * sx + ox)
        ry0 = int(y * sx + oy)
        rx1 = int((x + w) * sx + ox)
        ry1 = int((y + h) * sx + oy)
        return rx0, ry0, rx1, ry1

    def _clear_preview_canvas(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            self.preview_size[0] // 2,
            self.preview_size[1] // 2,
            text="拖入或添加视频后在此预览",
            fill="#4a5568",
            font=("Arial", 14),
        )
        self.preview_image = None
        self._preview_image_item = None
        self._invalidate_preview_cache()
        self.frame_label.configure(text="帧 0/0 (00:00 / 00:00)")
        self.timeline_slider.configure(from_=0, to=1)
        self.timeline_slider.set(0)

    def _bind_mousewheel_to_settings(self, frame: ctk.CTkScrollableFrame) -> None:
        def _on_scroll(event: tk.Event) -> Optional[str]:
            canvas = getattr(frame, "_parent_canvas", None)
            if canvas is None:
                return None
            delta = 0
            if getattr(event, "delta", 0):
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) in (4, 5):
                delta = -1 if event.num == 4 else 1
            if delta == 0:
                return None
            try:
                canvas.yview_scroll(delta, "units")
            except Exception:
                return None
            return "break"

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            frame.bind(seq, _on_scroll, add="+")

    def _apply_preview_masks(self, frame, frame_idx: int) -> "cv2.Mat":
        # 预览无多帧背景，时域擦除在 apply_masks 内退化为空间 inpaint。
        return bc.apply_masks(frame, self._current_masks(), frame_idx, None)

    def _on_window_resize(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._apply_window_resize)

    def _apply_window_resize(self) -> None:
        self._resize_job = None
        if not hasattr(self, "preview_canvas"):
            return
        new_size = (
            max(200, self.preview_canvas.winfo_width()),
            max(300, self.preview_canvas.winfo_height()),
        )
        if new_size == self.preview_size:
            return
        self.preview_size = new_size
        self.preview_canvas.config(width=new_size[0], height=new_size[1])
        if self.cap:
            self._show_frame(self.current_frame_idx)
        else:
            self._clear_preview_canvas()

    def _on_blur_method_change(self, _: str) -> None:
        # 写回当前遮罩（选择遮罩时的回填不应触发写回）。
        if not self._loading_mask:
            mask = self._active_mask()
            if mask:
                mask.method = self._normalize_blur_method(self.blur_method_var.get())
                self._update_mask_label()
        if self.cap:
            self._show_frame(self.current_frame_idx)
        self._auto_save_config()

    def _on_blur_strength_change(self, value: float) -> None:
        self.blur_strength.set(int(value))
        if not self._loading_mask:
            mask = self._active_mask()
            if mask:
                mask.strength = max(int(value), 5)
        if self.cap:
            if self._strength_job:
                self.after_cancel(self._strength_job)
            self._strength_job = self.after(30, self._flush_strength_preview)
        self._auto_save_config()

    def _flush_strength_preview(self) -> None:
        self._strength_job = None
        if self.cap:
            self._show_frame(self.current_frame_idx)

    # ------------------------------------------------------------------ 编码器检测
    def _resolve_ffmpeg_path(self) -> Optional[str]:
        if self.ffmpeg_path and os.path.isfile(self.ffmpeg_path):
            return self.ffmpeg_path
        local = self._find_local_ffmpeg()
        if local:
            self.ffmpeg_path = local
            if hasattr(self, "ffmpeg_entry"):
                self.ffmpeg_entry.delete(0, "end")
                self.ffmpeg_entry.insert(0, local)
            self._save_config()
            return local
        return None

    def _probe_encoders_background(self, ffmpeg_exec: str) -> None:
        usable, _ = mt.detect_working_hw_encoders(ffmpeg_exec)
        with self._encoder_probe_lock:
            self._encoder_probe_cache[ffmpeg_exec] = usable
            self._encoder_probe_running.discard(ffmpeg_exec)
        if usable:
            labels = ", ".join(usable)
            self._log(f"后台检测到可用硬件编码器: {labels}")

    def _start_encoder_probe_if_needed(self, force: bool = False) -> None:
        ffmpeg_exec = self._resolve_ffmpeg_path()
        if not ffmpeg_exec:
            return
        with self._encoder_probe_lock:
            cached = self._encoder_probe_cache.get(ffmpeg_exec)
            if cached is not None and not force:
                return
            if ffmpeg_exec in self._encoder_probe_running:
                return
            self._encoder_probe_running.add(ffmpeg_exec)
        t = threading.Thread(target=self._probe_encoders_background, args=(ffmpeg_exec,), daemon=True)
        t.start()

    def _encoder_usable(self, ffmpeg_exec: str, encoder: str, log_failure: bool = False) -> bool:
        ok, err = mt.encoder_usable(ffmpeg_exec, encoder)
        if not ok and log_failure and err:
            self._log(f"{encoder} 自检失败: {err}")
        return ok

    def _on_encoder_change(self, _: str) -> None:
        self._auto_save_config()

    # ------------------------------------------------------------------ 处理逻辑
    def _ui_safe(self, fn, *args, **kwargs) -> None:
        if threading.current_thread() is threading.main_thread():
            fn(*args, **kwargs)
        else:
            self.after(0, lambda: fn(*args, **kwargs))

    def start_processing(self) -> None:
        if self.is_processing:
            self._log("当前有任务在运行，请稍候")
            return
        if not self.video_paths:
            self._log("请先添加至少一个视频")
            return
        typed_ffmpeg = self.ffmpeg_entry.get().strip()
        if typed_ffmpeg and typed_ffmpeg != (self.ffmpeg_path or ""):
            if os.path.isfile(typed_ffmpeg):
                self.ffmpeg_path = typed_ffmpeg
                self._save_config()
            else:
                self._log(f"ffmpeg 路径无效，已忽略: {typed_ffmpeg}")
        typed_output = self.output_entry.get().strip()
        if typed_output != (self.output_dir or ""):
            self.output_dir = typed_output or None
            self._save_config()
        self._ensure_hwaccel_detected()
        ffmpeg_exec = self._resolve_ffmpeg_path()
        encoder_pref = self._normalize_encoder(self.encoder_var.get())
        crop_enabled = bool(self.crop_9x16.get())
        remove_audio = bool(self.remove_audio.get())
        remove_after = bool(self.remove_after.get())
        self.is_processing = True
        self._set_status("处理中...")
        t = threading.Thread(
            target=self._process_worker,
            args=(ffmpeg_exec, encoder_pref, crop_enabled, remove_audio, remove_after),
            daemon=True,
        )
        t.start()

    def _resolve_effective_encoder(self, ffmpeg_exec: Optional[str], encoder_pref: str) -> str:
        """在工作线程中执行编码器解析（含 subprocess 自检，最多阻塞数秒）。"""
        effective_encoder = encoder_pref
        self.detected_hw_encoders = []
        if effective_encoder == "auto":
            cached_hw: List[str] = []
            if ffmpeg_exec:
                with self._encoder_probe_lock:
                    cached_hw = list(self._encoder_probe_cache.get(ffmpeg_exec, []))
            if cached_hw:
                self.detected_hw_encoders = cached_hw
            else:
                usable, failed = mt.detect_working_hw_encoders(ffmpeg_exec) if ffmpeg_exec else ([], [])
                self.detected_hw_encoders = usable
                if failed and not usable:
                    self._log(f"发现硬件编码器 {', '.join(failed)} 但自检失败，将改用软件编码")
                elif failed and usable:
                    self._log(f"部分硬件编码不可用: {', '.join(failed)}")
                if ffmpeg_exec:
                    with self._encoder_probe_lock:
                        self._encoder_probe_cache[ffmpeg_exec] = list(self.detected_hw_encoders)
            self._start_encoder_probe_if_needed()
            effective_encoder = mt.pick_auto_encoder(self.detected_hw_encoders)
            if self.detected_hw_encoders:
                labels = ", ".join(self.detected_hw_encoders)
                self._log(f"检测到可用硬件编码器: {labels}，已选择 {self._encoder_label_from_value(effective_encoder)}")
            else:
                self._log("未检测到硬件编码器，将使用软件编码")
        else:
            self._log(f"编码器设置为: {self._encoder_label_from_value(effective_encoder)}")
        if effective_encoder == "copy":
            self._log("当前流程会修改像素，无法直接拷贝码流，已改为软件 x264")
            effective_encoder = "libx264"
            self.after(0, self.encoder_var.set, self._encoder_label_from_value(effective_encoder))
        if effective_encoder in ("h264_nvenc", "h264_qsv", "h264_amf"):
            if not ffmpeg_exec:
                self._log("未找到 ffmpeg，无法使用硬件编码，将使用软件编码")
                effective_encoder = "libx264"
            else:
                if not self._encoder_usable(ffmpeg_exec, effective_encoder, log_failure=True):
                    self._log(f"{self._encoder_label_from_value(effective_encoder)} 不可用，已回退软件 x264")
                    effective_encoder = "libx264"
                else:
                    self._log(f"将尝试使用 {self._encoder_label_from_value(effective_encoder)}，失败会自动回退 x264")
        return effective_encoder

    def _process_worker(
        self,
        ffmpeg_exec: Optional[str],
        encoder_pref: str,
        crop_enabled: bool,
        remove_audio: bool,
        remove_after: bool,
    ) -> None:
        self.current_encoder = self._resolve_effective_encoder(ffmpeg_exec, encoder_pref)
        videos = list(self.video_paths)
        total = len(videos)
        completed = 0
        encoder_label = self._encoder_label_from_value(self.current_encoder)

        self.after(0, self.progress_bar.set, 0)

        def task(idx_path: Tuple[int, str]) -> Tuple[str, Optional[str]]:
            idx, path = idx_path
            self._set_processing_state(path, True)
            try:
                masks = list(self.masks_map.get(path, []))
                mask_text = f"{len(masks)} 个遮罩" if masks else "无遮罩"
                self._log(
                    f"[{idx}/{total}] 开始 {os.path.basename(path)} ({mask_text} | {encoder_label})"
                )
                out_path = self._process_single_video(path, masks, crop_enabled, remove_audio)
                return path, out_path
            finally:
                self._set_processing_state(path, False)

        max_workers = max(1, min(self.worker_count, total))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(task, (idx, path)): path for idx, path in enumerate(videos, start=1)
            }
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    _, out_path = future.result()
                    self._log(f"完成: {os.path.basename(out_path)}")
                    if remove_after:
                        self.after(0, self._remove_video_from_list, path)
                except Exception as exc:
                    self._log(f"失败 {os.path.basename(path)}: {exc}")
                completed += 1
                self.after(0, self.progress_bar.set, completed / total)
                self._set_status(f"处理中... {completed}/{total}")

        self.after(0, self.progress_bar.set, 1.0 if total else 0)
        self._set_status(f"处理完成（{total} 个文件）")
        self.after(0, setattr, self, "is_processing", False)

    def _build_exporter(self) -> VideoExporter:
        """按当前 ffmpeg 与硬件解码设置构造一个导出器。"""
        return VideoExporter(
            ffmpeg_exec=self._resolve_ffmpeg_path(),
            decode_method=self.hwaccel_method,
            decode_device=self.hwaccel_device,
            log=self._log,
        )

    def _process_single_video(
        self,
        path: str,
        masks: List[Mask],
        crop_enabled: bool,
        remove_audio: bool,
    ) -> str:
        base_dir = self.output_dir or os.path.dirname(path)
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        base_name, _ = os.path.splitext(os.path.basename(path))
        out_path = os.path.join(base_dir, f"{base_name}_blurred.mp4")
        return self._build_exporter().export(
            src_path=path,
            out_path=out_path,
            masks=masks,
            crop_enabled=crop_enabled,
            remove_audio=remove_audio,
            encoder=self.current_encoder,
            preserve_bitrate=True,
        )

    def preview_settings(self) -> None:
        total_masks = sum(len(m) for m in self.masks_map.values())
        moving = sum(1 for masks in self.masks_map.values() for m in masks if m.is_moving())
        info = (
            f"遮罩总数: {total_masks}（移动 {moving}） | 编码器: {self.encoder_var.get()} | "
            f"去音轨: {self.remove_audio.get()} | 裁剪9:16: {self.crop_9x16.get()} | "
            f"导出后移除: {self.remove_after.get()} | 文件数: {len(self.video_paths)} | 并行数: {self.worker_count}"
        )
        self._log(info)
        self._set_status("设置预览完成")

    # ------------------------------------------------------------------ 配置与辅助
    def _apply_saved_paths(self) -> None:
        if self.ffmpeg_path:
            self.ffmpeg_entry.delete(0, "end")
            self.ffmpeg_entry.insert(0, self.ffmpeg_path)
        else:
            local = self._find_local_ffmpeg()
            if local:
                self.ffmpeg_path = local
                self.ffmpeg_entry.delete(0, "end")
                self.ffmpeg_entry.insert(0, local)
                self._save_config()
        self._start_encoder_probe_if_needed()
        if self.output_dir:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, self.output_dir)

    def _find_local_ffmpeg(self) -> Optional[str]:
        try:
            entries = os.listdir(self.base_dir)
        except OSError:
            return None
        candidate_dirs = []
        for entry in entries:
            full = os.path.join(self.base_dir, entry)
            if os.path.isdir(full) and "ffmpeg" in entry.lower():
                candidate_dirs.append(full)
        candidate_dirs.append(self.base_dir)
        for root_dir in candidate_dirs:
            for root, dirs, files in os.walk(root_dir):
                for name in ("ffmpeg", "ffmpeg.exe"):
                    if name in files:
                        path = os.path.join(root, name)
                        if os.access(path, os.X_OK):
                            return path
                depth = root.replace(root_dir, "").count(os.sep)
                if depth >= 2:
                    dirs[:] = []
        return None

    def _enable_hardware_accel(self) -> None:
        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass
        try:
            threads = max(1, os.cpu_count() or 4)
            cv2.setNumThreads(threads)
        except Exception:
            pass
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
        except Exception:
            pass

    # ------------------------------------------------------------------ 硬件解码
    def _ensure_hwaccel_detected(self) -> None:
        if self._hwaccel_checked:
            return
        self._detect_hwaccel_support()
        self._hwaccel_checked = True

    def _detect_hwaccel_support(self) -> None:
        ffmpeg_exec = self._resolve_ffmpeg_path()
        methods = mt.detect_hwaccels(ffmpeg_exec)
        self.hwaccel_method = mt.pick_decode_method(methods)
        if self.hwaccel_method == "vaapi" and not self.hwaccel_device:
            self.hwaccel_device = "/dev/dri/renderD128"

    def _open_video_capture(self, path: str, update_hw_flag: bool = False) -> "cv2.VideoCapture":
        """打开视频解码句柄，优先尝试硬件加速。update_hw_flag=True 时更新预览状态标志。"""
        cap, used_hw = mt.open_capture(path, self.hwaccel_method, self.hwaccel_device)
        if update_hw_flag:
            self._using_hw_preview = used_hw
        return cap

    def _maximize_window(self) -> None:
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass

    def _on_option_toggle(self) -> None:
        self._auto_save_config()

    def _normalize_blur_method(self, value: str) -> str:
        for label, code in self.blur_options:
            if value == label or value == code:
                return code
        return self.blur_options[0][1]

    def _normalize_encoder(self, value: str) -> str:
        if value == "libx264":
            return "libx264"
        for label, code in self.encoder_options:
            if value == label or value == code:
                return code
        return self.encoder_options[0][1]

    def _blur_label_from_value(self, value: str) -> str:
        for label, code in self.blur_options:
            if value == label or value == code:
                return label
        return self.blur_options[0][0]

    def _encoder_label_from_value(self, value: str) -> str:
        if value == "libx264":
            return "软件 x264 (libx264)"
        for label, code in self.encoder_options:
            if value == label or value == code:
                return label
        return self.encoder_options[0][0]

    def _load_config(self) -> None:
        cfg = AppConfig.load(self.config_path, self.legacy_config_path)
        self.ffmpeg_path = cfg.ffmpeg_path
        self.output_dir = cfg.output_dir
        self.remove_audio.set(cfg.remove_audio)
        self.crop_9x16.set(cfg.crop_9x16)
        self.remove_after.set(cfg.remove_after_export)
        self.encoder_var.set(self._encoder_label_from_value(cfg.encoder))
        self.blur_method_var.set(self._blur_label_from_value(cfg.blur_method))
        self.blur_strength.set(cfg.blur_strength)
        # 若仅存在旧版配置文件，迁移到新路径。
        if not os.path.isfile(self.config_path) and os.path.isfile(self.legacy_config_path):
            self._save_config()

    def _current_config(self) -> AppConfig:
        return AppConfig(
            ffmpeg_path=self.ffmpeg_path,
            output_dir=self.output_dir,
            remove_audio=bool(self.remove_audio.get()),
            crop_9x16=bool(self.crop_9x16.get()),
            remove_after_export=bool(self.remove_after.get()),
            encoder=self._normalize_encoder(self.encoder_var.get()),
            blur_method=self._normalize_blur_method(self.blur_method_var.get()),
            blur_strength=int(self.blur_strength.get()),
        )

    def _save_config(self) -> None:
        if not self._current_config().save(self.config_path):
            self._log("保存配置失败")
        self._save_job = None

    def _auto_save_config(self, delay_ms: int = 300) -> None:
        if self._save_job:
            self.after_cancel(self._save_job)
        self._save_job = self.after(delay_ms, self._save_config)

    # ------------------------------------------------------------------ UI 辅助
    def _refresh_list(self) -> None:
        for btn in self.list_buttons:
            btn.destroy()
        self.list_buttons.clear()
        if self.list_placeholder:
            self.list_placeholder.destroy()
            self.list_placeholder = None
        if not self.video_paths:
            self.list_placeholder = ctk.CTkLabel(
                self.list_frame, text="暂无视频，点击「添加视频」开始", text_color="#4a5568"
            )
            self.list_placeholder.pack(pady=16)
            return
        for idx, p in enumerate(self.video_paths):
            btn = ctk.CTkButton(
                self.list_frame,
                text=self._list_button_label(idx, p, False),
                anchor="w",
                corner_radius=8,
                height=36,
                command=lambda i=idx: self._on_list_button_click(i),
            )
            btn.pack(fill="x", padx=6, pady=3)
            self.list_buttons.append(btn)
        self._apply_list_highlight()

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")

        def _append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"[{ts}] {text}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self._ui_safe(_append)

    def _set_status(self, text: str) -> None:
        self._ui_safe(self.status_label.configure, text=text)

    def _list_button_label(self, idx: int, path: str, processing: bool) -> str:
        prefix = "⏳ " if processing else ""
        return f"{prefix}{idx + 1}. {os.path.basename(path)}"

    def _style_list_button(self, button: ctk.CTkButton, state: str) -> None:
        style = self.list_button_styles.get(state, self.list_button_styles["normal"])
        button.configure(
            fg_color=style["fg_color"],
            hover_color=style["hover_color"],
            text_color=style["text_color"],
            border_width=style["border_width"],
            border_color=style["border_color"],
        )

    def _on_list_button_click(self, index: int) -> None:
        if 0 <= index < len(self.video_paths):
            self._select_preview_index(index)
            self._set_status(f"已切换: {os.path.basename(self.video_paths[index])}")
        else:
            self._set_status("无效的选择")

    def _set_processing_state(self, path: str, active: bool) -> None:
        def _apply() -> None:
            if active:
                self.processing_paths.add(path)
            else:
                self.processing_paths.discard(path)
            self._apply_list_highlight()

        self._ui_safe(_apply)

    def _apply_list_highlight(self) -> None:
        if not self.list_buttons:
            return
        for idx, btn in enumerate(self.list_buttons):
            if idx >= len(self.video_paths):
                continue
            path = self.video_paths[idx]
            is_active = self.preview_index == idx
            is_processing = path in self.processing_paths
            if is_active and is_processing:
                state = "active_processing"
            elif is_active:
                state = "active"
            elif is_processing:
                state = "processing"
            else:
                state = "normal"
            self._style_list_button(btn, state)
            btn.configure(text=self._list_button_label(idx, path, is_processing))

    def _on_close(self) -> None:
        try:
            self._seek_token += 1
            if self._seek_job:
                self.after_cancel(self._seek_job)
            self._pending_seek_frame = None
        except Exception:
            pass
        with self.cap_lock:
            self._preview_generation += 1
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        try:
            self.preview_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.destroy()


def main() -> None:
    app = SubtitleBlurApp()
    app.mainloop()


if __name__ == "__main__":
    main()
