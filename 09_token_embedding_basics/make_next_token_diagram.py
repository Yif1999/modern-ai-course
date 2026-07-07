from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


current_dir = Path(__file__).resolve().parent
output_dir = current_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

out_path = output_dir / "next_token_training_diagram.png"

W, H = 1800, 1200
BG = "#f8fafc"
INK = "#0f172a"
MUTED = "#475569"
BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"
GREEN = "#16a34a"
GREEN_LIGHT = "#dcfce7"
ORANGE = "#ea580c"
ORANGE_LIGHT = "#ffedd5"
PURPLE = "#7c3aed"
PURPLE_LIGHT = "#ede9fe"
RED = "#dc2626"
GRAY = "#e2e8f0"
WHITE = "#ffffff"


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Apple Symbols.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(50, bold=True)
FONT_H2 = font(32, bold=True)
FONT_BODY = font(25)
FONT_SMALL = font(21)
FONT_TINY = font(18)
FONT_TOKEN = font(34, bold=True)
FONT_MONO = font(24)


img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)


def text_size(text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def centered_text(box, text, fnt, fill=INK):
    x1, y1, x2, y2 = box
    tw, th = text_size(text, fnt)
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 1), text, font=fnt, fill=fill)


def box(x, y, w, h, fill, outline, radius=18, width=3):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)


def arrow(x1, y1, x2, y2, fill=INK, width=4):
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) > abs(dy):
        if dx >= 0:
            pts = [(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)]
        else:
            pts = [(x2, y2), (x2 + 16, y2 - 9), (x2 + 16, y2 + 9)]
    else:
        if dy >= 0:
            pts = [(x2, y2), (x2 - 9, y2 - 16), (x2 + 9, y2 - 16)]
        else:
            pts = [(x2, y2), (x2 - 9, y2 + 16), (x2 + 9, y2 + 16)]
    draw.polygon(pts, fill=fill)


def label(x, y, text, fnt=FONT_BODY, fill=INK):
    draw.text((x, y), text, font=fnt, fill=fill)


def token_row(x, y, tokens, fill, outline):
    tw, th, gap = 74, 66, 10
    for i, tok in enumerate(tokens):
        bx = x + i * (tw + gap)
        box(bx, y, tw, th, fill, outline, radius=14, width=3)
        centered_text((bx, y, bx + tw, y + th), tok, FONT_TOKEN)
    return tw, th, gap


# Title
label(70, 50, "Next Token Prediction：一条序列里有多个训练位置", FONT_TITLE)
label(74, 116, "核心：模型只吃 x_batch；y_batch 是答案；每个位置都输出一组 vocab logits。", FONT_BODY, MUTED)


# Left card: x/y shift
box(70, 180, 720, 390, WHITE, GRAY, radius=26, width=3)
label(110, 215, "1. 右移一位构造训练目标", FONT_H2)
label(112, 265, "原始文本片段：hello", FONT_BODY, MUTED)

label(120, 330, "输入 x：", FONT_BODY, BLUE)
token_row(235, 315, ["h", "e", "l", "l"], BLUE_LIGHT, BLUE)

label(120, 445, "目标 y：", FONT_BODY, GREEN)
token_row(235, 430, ["e", "l", "l", "o"], GREEN_LIGHT, GREEN)

for i in range(4):
    cx = 235 + i * 84 + 37
    arrow(cx, 382, cx, 430, fill=ORANGE, width=3)

label(450, 328, "位置 0", FONT_TINY, MUTED)
label(450, 445, "正确答案", FONT_TINY, MUTED)
label(110, 520, "含义：看到 h 预测 e；看到 h e 预测 l；看到 h e l 预测 l。", FONT_SMALL, MUTED)


# Middle flow: x -> model -> logits
box(850, 180, 400, 126, BLUE_LIGHT, BLUE, radius=24, width=3)
centered_text((850, 180, 1250, 230), "x_batch", FONT_H2, BLUE)
centered_text((850, 230, 1250, 286), "[batch, seq_len] = [4, 8]", FONT_BODY, INK)

arrow(1250, 242, 1345, 242, fill=INK, width=5)

box(1360, 180, 340, 126, PURPLE_LIGHT, PURPLE, radius=24, width=3)
centered_text((1360, 180, 1700, 232), "Embedding + LM Head", FONT_H2, PURPLE)
centered_text((1360, 232, 1700, 286), "只接收 x，不接收 y", FONT_BODY, INK)

arrow(1530, 306, 1530, 390, fill=INK, width=5)

box(850, 395, 850, 335, WHITE, GRAY, radius=26, width=3)
label(890, 430, "2. 输出 logits", FONT_H2)
label(890, 478, "logits shape: [batch, seq_len, vocab_size] = [4, 8, 16]", FONT_BODY, INK)
label(890, 515, "意思：4 条序列 × 每条 8 个位置 × 每个位置 16 个 token 分数", FONT_SMALL, MUTED)

# Draw a compact logits grid.
grid_x, grid_y = 900, 570
cell_w, cell_h, gap_x, gap_y = 54, 34, 8, 10
for r in range(4):
    for c in range(8):
        x0 = grid_x + c * (cell_w + gap_x)
        y0 = grid_y + r * (cell_h + gap_y)
        box(x0, y0, cell_w, cell_h, "#eef2ff", "#c7d2fe", radius=7, width=1)
        # Tiny bars represent a vocab-size vector.
        for b in range(4):
            bx = x0 + 9 + b * 9
            bh = [12, 22, 16, 27][(r + c + b) % 4]
            draw.rectangle((bx, y0 + cell_h - bh - 4, bx + 5, y0 + cell_h - 4), fill=PURPLE)

label(1410, 584, "每个小格代表：", FONT_SMALL, MUTED)
label(1410, 620, "一个位置的 16 个分数", FONT_SMALL, PURPLE)
label(1410, 666, "例如 logits[0, 0, :]", FONT_SMALL, INK)


# Right/loss card
box(70, 650, 720, 390, WHITE, GRAY, radius=26, width=3)
label(110, 685, "3. y_batch 是答案，不是输入", FONT_H2)

box(115, 755, 255, 120, GREEN_LIGHT, GREEN, radius=20, width=3)
centered_text((115, 755, 370, 810), "y_batch", FONT_H2, GREEN)
centered_text((115, 810, 370, 865), "[4, 8]", FONT_BODY, INK)

arrow(390, 815, 490, 815, fill=INK, width=5)

box(505, 755, 225, 120, ORANGE_LIGHT, ORANGE, radius=20, width=3)
centered_text((505, 755, 730, 810), "Cross Entropy", FONT_H2, ORANGE)
centered_text((505, 810, 730, 865), "选正确 target id", FONT_SMALL, INK)

label(112, 920, "每个位置是一道分类题：从 16 个 token 里选正确的下一个 token。", FONT_SMALL, MUTED)
label(112, 960, "训练时内部等价于：logits [32, 16]，targets [32]。", FONT_SMALL, MUTED)
label(112, 1000, "32 = batch 4 × seq_len 8，最后把 32 个 loss 求平均。", FONT_SMALL, MUTED)


# Bottom right takeaway
box(850, 785, 850, 255, "#fff7ed", ORANGE, radius=26, width=3)
label(890, 820, "4. 训练和生成的区别", FONT_H2, ORANGE)
label(895, 885, "训练时：每个位置都预测下一个 token，充分利用序列。", FONT_BODY, INK)
label(895, 935, "生成时：通常只取最后一个位置 logits[:, -1, :] 来采样下一个 token。", FONT_BODY, INK)
label(895, 985, "后面 Transformer 会用 causal mask，保证每个位置不能偷看未来。", FONT_BODY, MUTED)


# Footer
draw.line((70, 1095, 1730, 1095), fill=GRAY, width=2)
label(70, 1125, "一句话：不是一条序列只训练一次，而是一条序列里的每个位置都贡献一个 next-token loss。", FONT_BODY, INK)

img.save(out_path)
print(out_path)
