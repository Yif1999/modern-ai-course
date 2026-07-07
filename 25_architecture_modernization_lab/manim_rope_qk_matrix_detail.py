from __future__ import annotations

from manim import *


FONT = "PingFang SC"


class RoPEQKMatrixDetail(Scene):
    def txt(self, text: str, size: int = 26, color=WHITE):
        return Text(text, font=FONT, font_size=size, color=color)

    def card(self, text: str, color, width=2.0, height=0.62, size=22):
        rect = RoundedRectangle(width=width, height=height, corner_radius=0.08)
        rect.set_stroke(color, 2)
        rect.set_fill(color, opacity=0.13)
        label = self.txt(text, size=size)
        label.move_to(rect)
        return VGroup(rect, label)

    def arrow(self, start, end, color=WHITE):
        return Arrow(start, end, buff=0.08, color=color, stroke_width=2.6, max_tip_length_to_length_ratio=0.14)

    def wipe(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)

    def matrix_grid(self, name: str, rows: int, cols: int, color, values=None, cell=0.42):
        group = VGroup()
        cells = VGroup()
        labels = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell)
                sq.set_stroke(color, 1.2)
                sq.set_fill(color, opacity=0.09 + 0.04 * ((r + c) % 2))
                sq.move_to([c * cell, -r * cell, 0])
                cells.add(sq)
                if values:
                    lab = self.txt(str(values[r][c]), size=16, color=GRAY_A)
                    lab.move_to(sq)
                    labels.add(lab)
        cells.center()
        labels.move_to(cells)
        title = self.txt(name, size=22, color=color).next_to(cells, UP, buff=0.16)
        shape = self.txt(f"[{rows}, {cols}]", size=18, color=GRAY_B).next_to(cells, DOWN, buff=0.14)
        group.add(cells, labels, title, shape)
        return group

    def highlight_row(self, matrix, row: int, cols: int, color):
        cells = matrix[0]
        row_cells = VGroup(*[cells[row * cols + c] for c in range(cols)])
        return SurroundingRectangle(row_cells, color=color, buff=0.03, stroke_width=3)

    def highlight_pair_columns(self, matrix, cols: int, pair_start: int, color):
        cells = matrix[0]
        picked = VGroup()
        for r in range(4):
            picked.add(cells[r * cols + pair_start], cells[r * cols + pair_start + 1])
        return SurroundingRectangle(picked, color=color, buff=0.025, stroke_width=2)

    def construct(self):
        self.opening()
        self.projection_scene()
        self.row_rotation_scene()
        self.qk_score_scene()
        self.final_scene()

    def opening(self):
        title = self.txt("RoPE 细节：Q/K 矩阵到底怎么旋转？", size=36, color=PURPLE)
        subtitle = self.txt("例子：seq_len = 4，head_dim = 4", size=24, color=GRAY_A)
        group = VGroup(title, subtitle).arrange(DOWN, buff=0.18).move_to(ORIGIN)
        self.play(FadeIn(group, shift=DOWN * 0.1), run_time=0.8)
        self.wait(2.2)
        self.wipe()

    def projection_scene(self):
        title = self.txt("1. 先由 hidden state 投影出 Q / K / V", size=32, color=YELLOW).to_edge(UP, buff=0.35)
        x = self.matrix_grid("X hidden", 4, 4, BLUE, values=[["x"] * 4 for _ in range(4)]).scale(0.95).move_to([-4.6, 0.25, 0])
        wq = self.card("Wq", TEAL, width=1.25).move_to([-2.35, 1.4, 0])
        wk = self.card("Wk", TEAL, width=1.25).move_to([-2.35, 0.25, 0])
        wv = self.card("Wv", GRAY, width=1.25).move_to([-2.35, -0.9, 0])
        q = self.matrix_grid("Q", 4, 4, GREEN, values=[["q"] * 4 for _ in range(4)]).scale(0.82).move_to([0.1, 1.35, 0])
        k = self.matrix_grid("K", 4, 4, ORANGE, values=[["k"] * 4 for _ in range(4)]).scale(0.82).move_to([0.1, -0.65, 0])
        v = self.matrix_grid("V", 4, 4, GRAY, values=[["v"] * 4 for _ in range(4)]).scale(0.82).move_to([3.25, -0.65, 0])

        arrows = VGroup(
            self.arrow(x.get_right(), wq.get_left(), TEAL),
            self.arrow(x.get_right(), wk.get_left(), TEAL),
            self.arrow(x.get_right(), wv.get_left(), GRAY),
            self.arrow(wq.get_right(), q.get_left(), GREEN),
            self.arrow(wk.get_right(), k.get_left(), ORANGE),
            self.arrow(wv.get_right(), v.get_left(), GRAY),
        )
        note = self.txt("RoPE 只处理 Q 和 K；V 通常不旋转", size=24, color=YELLOW).to_edge(DOWN, buff=0.35)

        self.play(FadeIn(title), FadeIn(x), run_time=0.7)
        self.play(FadeIn(wq), FadeIn(wk), FadeIn(wv), Create(arrows[:3]), run_time=0.9)
        self.play(FadeIn(q), FadeIn(k), FadeIn(v), Create(arrows[3:]), run_time=1.0)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(7)
        self.wipe()

    def row_rotation_scene(self):
        title = self.txt("2. RoPE 对 Q/K 的每一行按 position 旋转", size=32, color=PURPLE).to_edge(UP, buff=0.35)
        q = self.matrix_grid(
            "Q before RoPE",
            4,
            4,
            GREEN,
            values=[
                ["q00", "q01", "q02", "q03"],
                ["q10", "q11", "q12", "q13"],
                ["q20", "q21", "q22", "q23"],
                ["q30", "q31", "q32", "q33"],
            ],
        ).scale(0.9).move_to([-4.1, 0.1, 0])
        qrot = self.matrix_grid(
            "Q_rot",
            4,
            4,
            PURPLE,
            values=[
                ["r00", "r01", "r02", "r03"],
                ["r10", "r11", "r12", "r13"],
                ["r20", "r21", "r22", "r23"],
                ["r30", "r31", "r32", "r33"],
            ],
        ).scale(0.9).move_to([4.1, 0.1, 0])

        pair_a = self.highlight_pair_columns(q, 4, 0, BLUE)
        pair_b = self.highlight_pair_columns(q, 4, 2, ORANGE)
        pair_text = self.txt("head_dim=4 被拆成两对：(0,1) 和 (2,3)", size=23, color=GRAY_A).move_to([0, 2.15, 0])

        block = VGroup(
            self.txt("pos p 的旋转矩阵 Rₚ", size=24, color=YELLOW),
            self.txt("[ cosθₚ  -sinθₚ ]", size=23, color=YELLOW),
            self.txt("[ sinθₚ   cosθₚ ]", size=23, color=YELLOW),
            self.txt("同一个 Rₚ 作用到每一对维度", size=20, color=GRAY_A),
        ).arrange(DOWN, buff=0.12).move_to([0, 0.15, 0])

        arrow = self.arrow(q.get_right(), qrot.get_left(), PURPLE)
        row_highlights = VGroup()
        for row, color in enumerate([BLUE, PURPLE, ORANGE, RED]):
            row_highlights.add(self.highlight_row(q, row, 4, color))

        pos_labels = VGroup(
            self.txt("pos0: θ=0", size=18, color=BLUE),
            self.txt("pos1: θ=1·θ", size=18, color=PURPLE),
            self.txt("pos2: θ=2·θ", size=18, color=ORANGE),
            self.txt("pos3: θ=3·θ", size=18, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([-0.15, -2.0, 0])

        self.play(FadeIn(title), FadeIn(q), FadeIn(pair_text), run_time=0.8)
        self.play(Create(pair_a), Create(pair_b), run_time=0.7)
        self.play(FadeIn(block, shift=UP * 0.1), run_time=0.8)
        self.play(Create(arrow), FadeIn(qrot), run_time=0.8)
        for row_h, label in zip(row_highlights, pos_labels):
            self.play(Create(row_h), FadeIn(label), run_time=0.42)
            self.play(row_h.animate.move_to(qrot[0][row_highlights.submobjects.index(row_h) * 4 : row_highlights.submobjects.index(row_h) * 4 + 4].get_center()), run_time=0.35)
            self.play(FadeOut(row_h), run_time=0.18)
        note = self.txt("K 矩阵做同样的逐行旋转，得到 K_rot。", size=24, color=ORANGE).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(9)
        self.wipe()

    def qk_score_scene(self):
        title = self.txt("3. 最后用旋转后的 Q/K 计算 attention scores", size=32, color=YELLOW).to_edge(UP, buff=0.35)
        qrot = self.matrix_grid("Q_rot", 4, 4, PURPLE, values=[["q′"] * 4 for _ in range(4)]).scale(0.84).move_to([-4.1, 0.25, 0])
        kt = self.matrix_grid("K_rotᵀ", 4, 4, ORANGE, values=[["k′"] * 4 for _ in range(4)]).scale(0.84).move_to([0, 0.25, 0])
        scores = self.matrix_grid("scores", 4, 4, RED, values=[["s"] * 4 for _ in range(4)]).scale(0.84).move_to([4.1, 0.25, 0])

        mult = self.txt("@", size=36, color=GRAY_A).move_to([-2.05, 0.25, 0])
        eq = self.txt("=", size=36, color=GRAY_A).move_to([2.05, 0.25, 0])

        q_row = self.highlight_row(qrot, 2, 4, BLUE)
        k_col_cells = VGroup(*[kt[0][r * 4 + 1] for r in range(4)])
        k_col = SurroundingRectangle(k_col_cells, color=ORANGE, buff=0.03, stroke_width=3)
        score_cell = SurroundingRectangle(scores[0][2 * 4 + 1], color=YELLOW, buff=0.03, stroke_width=3)

        formula = self.txt("score[2,1] = Q_rot 第 2 行 · K_rot 第 1 行", size=24, color=YELLOW).to_edge(DOWN, buff=0.75)
        rel = self.txt("(R₂ q) · (R₁ k) 里包含 θ₂ - θ₁，也就是相对位置", size=23, color=GRAY_A).to_edge(DOWN, buff=0.28)

        self.play(FadeIn(title), FadeIn(qrot), FadeIn(kt), FadeIn(mult), FadeIn(eq), FadeIn(scores), run_time=1.0)
        self.play(Create(q_row), Create(k_col), Create(score_cell), FadeIn(formula), run_time=1.0)
        self.play(FadeIn(rel), run_time=0.7)
        self.wait(12)
        self.wipe()

    def final_scene(self):
        title = self.txt("RoPE 的真实流程", size=36, color=PURPLE).to_edge(UP, buff=0.55)
        steps = VGroup(
            self.card("1. X → Q/K/V", BLUE, width=3.2),
            self.card("2. Q/K 按 position 逐行旋转", PURPLE, width=4.4),
            self.card("3. 得到 Q_rot / K_rot", ORANGE, width=4.0),
            self.card("4. scores = Q_rot @ K_rot.T", RED, width=4.5),
            self.card("5. V 不旋转，只被 weights 加权求和", GRAY, width=4.8),
        ).arrange(DOWN, buff=0.24).move_to([0, 0.05, 0])
        summary = self.txt("所以 RoPE 不是给 x 加位置，而是改变 Q/K 的匹配方式。", size=25, color=YELLOW).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(title), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.12) for s in steps], lag_ratio=0.16), run_time=1.6)
        self.play(FadeIn(summary), run_time=0.7)
        self.wait(10)
