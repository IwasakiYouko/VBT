"""Material Design 3 深色主题（参考 Google 设计规范）。

集中管理全部界面配色 / 圆角 / 语义色，界面代码只引用语义常量，
不再散落十六进制色值。取色对齐 Google 深色产品（grey / blue 系）：
表面用 Google Grey 900~700 做层级，主色用深色模式的 Google Blue 300。
"""

# ------------------------------------------------------------ 表面层级（Google Grey）
BG = "#202124"            # 窗口背景（grey 900）
SURFACE = "#292A2D"       # 卡片 / 面板
SURFACE_2 = "#303134"     # 输入框、列表项、次级按钮
SURFACE_3 = "#3C4043"     # 悬停态、分隔线（grey 800）
SURFACE_DIM = "#1B1C1F"   # 日志、下沉区域
CANVAS_BG = "#17181B"     # 预览画布

# ------------------------------------------------------------ 文字
TEXT = "#E8EAED"            # 主文字（grey 200）
TEXT_SECONDARY = "#9AA0A6"  # 次级文字（grey 500）
TEXT_DISABLED = "#5F6368"   # 占位 / 禁用（grey 700）
TEXT_ON_ACCENT = "#202124"  # 亮色按钮上的深色文字
OUTLINE = "#5F6368"         # 描边

# ------------------------------------------------------------ 主色（深色模式 Google Blue）
PRIMARY = "#8AB4F8"         # blue 300
PRIMARY_HOVER = "#AECBFA"   # blue 200
PRIMARY_DIM = "#1A73E8"     # blue 600，选中描边等小面积强调

# ------------------------------------------------------------ 语义容器色（M3 dark container）
ERROR_CONTAINER = "#5C2B29"
ERROR_CONTAINER_HOVER = "#713330"
ON_ERROR_CONTAINER = "#F2B8B5"
WARNING = "#FDD663"          # yellow 300，处理中状态
WARNING_HOVER = "#FFE28A"
ROI_ACTIVE = "#EA4335"       # Google Red，当前遮罩框
ROI_OTHER = "#FBBC04"        # Google Yellow，其它遮罩框

# ------------------------------------------------------------ 形状（M3 圆角习惯）
RADIUS_CARD = 16     # 卡片
RADIUS_WIDGET = 10   # 输入框 / 下拉
RADIUS_PILL = 18     # 36px 高按钮的全圆角

# ------------------------------------------------------------ 常用组合
def tonal_button() -> dict:
    """次级（tonal）按钮配色。"""
    return dict(fg_color=SURFACE_2, hover_color=SURFACE_3, text_color=TEXT)


def filled_button() -> dict:
    """主操作（filled）按钮配色。"""
    return dict(fg_color=PRIMARY, hover_color=PRIMARY_HOVER, text_color=TEXT_ON_ACCENT)


def danger_button() -> dict:
    """危险操作（error container）按钮配色。"""
    return dict(
        fg_color=ERROR_CONTAINER, hover_color=ERROR_CONTAINER_HOVER,
        text_color=ON_ERROR_CONTAINER,
    )


def option_menu() -> dict:
    return dict(
        fg_color=SURFACE_2, button_color=SURFACE_3, button_hover_color=OUTLINE,
        text_color=TEXT, dropdown_fg_color=SURFACE_2,
        dropdown_hover_color=SURFACE_3, dropdown_text_color=TEXT,
    )


def checkbox() -> dict:
    return dict(
        fg_color=PRIMARY, checkmark_color=TEXT_ON_ACCENT,
        hover_color=PRIMARY_HOVER, border_color=OUTLINE, text_color=TEXT,
    )


def slider() -> dict:
    return dict(
        button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
        progress_color=PRIMARY, fg_color=SURFACE_3,
    )


def entry() -> dict:
    return dict(
        fg_color=SURFACE_2, border_color=OUTLINE, text_color=TEXT,
        placeholder_text_color=TEXT_DISABLED,
    )
