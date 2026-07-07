from manim import *


# Q/K/V 使用方式教学动画
# 目标：
# 1. 对比训练阶段和推理阶段 Q/K/V 的使用方式。
# 2. 说明训练阶段整段序列并行计算所有位置的 logits。
# 3. 说明推理阶段使用 KV Cache，复用历史 token 的 K/V，只计算当前 token。
# 4. 画面文字使用英文，避免本机字体缺中文导致渲染方块；代码注释保留中文。

config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720


BG = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
TEAL = "#2dd4bf"
YELLOW = "#facc15"
ORANGE = "#fb923c"
PURPLE = "#c084fc"
RED = "#fb7185"
GREEN = "#86efac"
GOLD = "#fbbf24"


class QKVUsageDemo(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.show_overview()
        self.show_training_detail()
        self.show_inference_detail()
        self.show_final_compare()

    def t(self, text, size=24, color=TEXT):
        return Text(text, font_size=size, color=color, font="Arial")

    def box(self, text, shape, color, width=3.25, height=0.62, size=18):
        rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=2,
            fill_color=color,
            fill_opacity=0.14,
        )
        label = self.t(text, size=size, color=TEXT)
        label.move_to(rect.get_center())
        shape_label = self.t(shape, size=13, color=MUTED).next_to(rect, DOWN, buff=0.05)
        return VGroup(rect, label, shape_label)

    def mat(self, title, shape, color, rows=3, cols=4, cell=0.18):
        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell)
                sq.set_fill(color, opacity=0.72)
                sq.set_stroke(BG, width=0.7)
                sq.move_to(RIGHT * c * cell + DOWN * r * cell)
                grid.add(sq)
        grid.center()
        label = self.t(title, size=17, color=color)
        shape_label = self.t(shape, size=12, color=MUTED)
        return VGroup(label, grid, shape_label).arrange(DOWN, buff=0.08)

    def down_arrow(self, top, bottom, color=MUTED):
        return Arrow(
            top.get_bottom(),
            bottom.get_top(),
            buff=0.13,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.14,
            color=color,
        )

    def side_arrow(self, left, right, color=MUTED):
        return Arrow(
            left.get_right(),
            right.get_left(),
            buff=0.13,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.14,
            color=color,
        )

    def clear_stage(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)

    def show_overview(self):
        # 第一幕：左右两侧高层对比。
        title = self.t("Q / K / V Usage in Tiny GPT", size=34)
        title.to_edge(UP, buff=0.28)

        left_head = self.t("Training", size=27, color=BLUE).move_to(LEFT * 3.6 + UP * 2.55)
        right_head = self.t("Inference + KV Cache", size=27, color=GREEN).move_to(
            RIGHT * 3.55 + UP * 2.55
        )
        divider = Line(UP * 2.85, DOWN * 2.85, color=MUTED).set_stroke(width=2)

        train_steps = VGroup(
            self.box("full token sequence", "[batch, seq_len]", BLUE, width=3.2),
            self.box("Q / K / V for all positions", "[batch, seq_len, n_embd]", YELLOW, width=3.6),
            self.box("attention over T x T", "[batch, heads, T, T]", RED, width=3.4),
            self.box("logits for every position", "[batch, seq_len, vocab_size]", GOLD, width=3.8),
        ).arrange(DOWN, buff=0.38)
        train_steps.move_to(LEFT * 3.55 + DOWN * 0.05)

        infer_steps = VGroup(
            self.box("past K/V cache", "[batch, heads, T_cache, d]", GREEN, width=3.7),
            self.box("new token -> new Q/K/V", "[batch, 1, n_embd]", TEAL, width=3.6),
            self.box("Q attends to cached K", "[batch, heads, 1, T_cache+1]", RED, width=3.9),
            self.box("last logits only", "[batch, 1, vocab_size]", GOLD, width=3.2),
        ).arrange(DOWN, buff=0.38)
        infer_steps.move_to(RIGHT * 3.55 + DOWN * 0.05)

        arrows = VGroup()
        for group in (train_steps, infer_steps):
            for i in range(len(group) - 1):
                arrows.add(self.down_arrow(group[i], group[i + 1]))

        self.play(FadeIn(title), Create(divider), FadeIn(left_head), FadeIn(right_head))
        self.play(LaggedStart(*[FadeIn(m) for m in train_steps], lag_ratio=0.12), run_time=1.1)
        self.play(Create(VGroup(*arrows[:3])), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(m) for m in infer_steps], lag_ratio=0.12), run_time=1.1)
        self.play(Create(VGroup(*arrows[3:])), run_time=0.7)
        self.wait(1.0)
        self.clear_stage()

    def show_training_detail(self):
        # 第二幕：训练阶段细节。
        title = self.t("Training phase: all positions are computed in parallel", size=31, color=BLUE)
        title.to_edge(UP, buff=0.32)

        row1 = VGroup(
            self.box("token ids", "[batch, seq_len]", BLUE, width=2.45),
            self.box("token + position embedding", "[batch, seq_len, n_embd]", TEAL, width=4.1),
        ).arrange(RIGHT, buff=0.8)
        row1.move_to(UP * 2.0)

        q = self.mat("Q", "[B, T, C]", YELLOW, rows=3, cols=4)
        k = self.mat("K", "[B, T, C]", ORANGE, rows=3, cols=4)
        v = self.mat("V", "[B, T, C]", PURPLE, rows=3, cols=4)
        row2 = VGroup(q, k, v).arrange(RIGHT, buff=1.0)
        row2.move_to(UP * 0.75)

        scores = self.mat("QK^T / sqrt(d)", "[B, heads, T, T]", RED, rows=4, cols=4)
        mask = self.mat("causal mask", "future tokens blocked", RED, rows=4, cols=4)
        weights = self.mat("softmax weights", "[B, heads, T, T]", GREEN, rows=4, cols=4)
        row3 = VGroup(scores, mask, weights).arrange(RIGHT, buff=0.8)
        row3.move_to(DOWN * 1.0)

        output = self.box("attention output = weights @ V", "[batch, seq_len, n_embd]", BLUE, width=4.1)
        logits = self.box("LM Head -> logits", "[batch, seq_len, vocab_size]", GOLD, width=3.25)
        row4 = VGroup(output, logits).arrange(RIGHT, buff=0.85)
        row4.move_to(DOWN * 2.65)

        note = self.t("Loss is computed at every sequence position.", size=20, color=BLUE)
        note.to_edge(DOWN, buff=0.24)

        arrows = VGroup(
            self.side_arrow(row1[0], row1[1], TEAL),
            Arrow(row1[1].get_bottom(), row2.get_top(), buff=0.18, color=MUTED, stroke_width=2.6),
            Arrow(row2.get_bottom(), row3.get_top(), buff=0.18, color=MUTED, stroke_width=2.6),
            self.side_arrow(scores, mask, RED),
            self.side_arrow(mask, weights, GREEN),
            Arrow(row3.get_bottom(), row4.get_top(), buff=0.18, color=MUTED, stroke_width=2.6),
            self.side_arrow(output, logits, GOLD),
        )

        self.play(FadeIn(title))
        self.play(FadeIn(row1), Create(arrows[0]), run_time=0.8)
        self.play(FadeIn(row2), Create(arrows[1]), run_time=0.8)
        self.play(FadeIn(row3), Create(VGroup(arrows[2], arrows[3], arrows[4])), run_time=1.0)
        self.play(FadeIn(row4), Create(VGroup(arrows[5], arrows[6])), FadeIn(note), run_time=1.0)
        self.wait(1.1)
        self.clear_stage()

    def show_inference_detail(self):
        # 第三幕：推理阶段细节。
        title = self.t("Inference phase: reuse cached K/V token by token", size=31, color=GREEN)
        title.to_edge(UP, buff=0.32)

        cache_old = self.box("past K/V cache", "[B, heads, T_cache, d]", GREEN, width=3.7)
        current = self.box("current token", "[batch, 1]", TEAL, width=2.6)
        row1 = VGroup(cache_old, current).arrange(RIGHT, buff=1.2)
        row1.move_to(UP * 1.8)

        q_new = self.mat("new Q", "[B, heads, 1, d]", YELLOW, rows=1, cols=4)
        kv_new = self.mat("new K/V", "append to cache", ORANGE, rows=2, cols=4)
        row2 = VGroup(q_new, kv_new).arrange(RIGHT, buff=1.0)
        row2.move_to(UP * 0.45)

        cache_grown = self.box("grown K/V cache", "[B, heads, T_cache+1, d]", GREEN, width=4.0)
        cache_grown.move_to(DOWN * 0.75)

        attn = self.mat("new Q attends to cached K", "[B, heads, 1, T_cache+1]", RED, rows=1, cols=6)
        out = self.box("weights @ cached V", "[batch, 1, n_embd]", BLUE, width=3.25)
        logits = self.box("last-position logits", "[batch, 1, vocab_size]", GOLD, width=3.15)
        row4 = VGroup(attn, out, logits).arrange(RIGHT, buff=0.65)
        row4.move_to(DOWN * 2.15)

        note = self.t("Old K/V do not change, so we avoid recomputing them.", size=20, color=GREEN)
        note.to_edge(DOWN, buff=0.28)

        arrows = VGroup(
            Arrow(current.get_bottom(), row2.get_top(), buff=0.18, color=TEAL, stroke_width=2.6),
            Arrow(kv_new.get_bottom(), cache_grown.get_top(), buff=0.16, color=GREEN, stroke_width=2.6),
            Arrow(cache_old.get_bottom(), cache_grown.get_top(), buff=0.16, color=GREEN, stroke_width=2.6),
            Arrow(cache_grown.get_bottom(), attn.get_top(), buff=0.18, color=RED, stroke_width=2.6),
            self.side_arrow(attn, out, GREEN),
            self.side_arrow(out, logits, GOLD),
        )

        self.play(FadeIn(title))
        self.play(FadeIn(row1), run_time=0.8)
        self.play(FadeIn(row2), Create(arrows[0]), run_time=0.8)
        self.play(FadeIn(cache_grown), Create(VGroup(arrows[1], arrows[2])), run_time=0.9)
        self.play(FadeIn(row4), Create(VGroup(arrows[3], arrows[4], arrows[5])), FadeIn(note), run_time=1.1)
        self.wait(1.2)
        self.clear_stage()

    def show_final_compare(self):
        # 第四幕：最终总结。
        title = self.t("Same math goal, different computation pattern", size=32)
        title.to_edge(UP, buff=0.35)

        left = VGroup(
            self.t("Training", size=28, color=BLUE),
            self.box("full sequence forward", "[B, T] -> [B, T, vocab]", BLUE, width=4.0),
            self.box("compute loss at all T positions", "parallel training", GOLD, width=4.2),
        ).arrange(DOWN, buff=0.5)
        left.move_to(LEFT * 3.25)

        right = VGroup(
            self.t("Inference", size=28, color=GREEN),
            self.box("one new token each step", "[B, 1] -> [B, 1, vocab]", GREEN, width=4.1),
            self.box("reuse K/V cache", "avoid repeated K/V work", GOLD, width=3.8),
        ).arrange(DOWN, buff=0.5)
        right.move_to(RIGHT * 3.25)

        divider = Line(UP * 2.45, DOWN * 2.45, color=MUTED).set_stroke(width=2)
        bottom = self.t(
            "KV Cache changes inference efficiency, not the model's learned parameters.",
            size=21,
            color=TEXT,
        )
        bottom.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(title), Create(divider))
        self.play(FadeIn(left), FadeIn(right), run_time=1.0)
        self.play(FadeIn(bottom, shift=UP * 0.2), run_time=0.8)
        self.wait(2.0)
