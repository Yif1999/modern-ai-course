from __future__ import annotations

import math
import numpy as np

from manim import *


FONT = "PingFang SC"


class DetailBase(Scene):
    def txt(self, text: str, size: int = 26, color=WHITE):
        return Text(text, font=FONT, font_size=size, color=color)

    def card(self, text: str, color, width=2.2, height=0.58, size=21):
        box = RoundedRectangle(width=width, height=height, corner_radius=0.07)
        box.set_stroke(color, 2)
        box.set_fill(color, opacity=0.12)
        label = self.txt(text, size=size)
        label.move_to(box)
        return VGroup(box, label)

    def arrow(self, start, end, color=WHITE):
        return Arrow(start, end, buff=0.08, color=color, stroke_width=2.6, max_tip_length_to_length_ratio=0.14)

    def wipe(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.4)

    def title(self, main: str, sub: str, color=YELLOW):
        group = VGroup(
            self.txt(main, size=34, color=color),
            self.txt(sub, size=21, color=GRAY_B),
        ).arrange(DOWN, buff=0.12)
        group.to_edge(UP, buff=0.28)
        self.play(FadeIn(group, shift=DOWN * 0.1), run_time=0.6)
        return group

    def strip(self, text: str, color=YELLOW):
        box = RoundedRectangle(width=11.2, height=0.62, corner_radius=0.08)
        box.set_stroke(color, 1.6)
        box.set_fill(color, opacity=0.12)
        box.to_edge(DOWN, buff=0.25)
        label = self.txt(text, size=21, color=color)
        label.move_to(box)
        group = VGroup(box, label)
        self.play(FadeIn(group), run_time=0.45)
        return group

    def grid(self, name: str, values: list[list[str]], color, cell_w=0.52, cell_h=0.4, font_size=15):
        rows = len(values)
        cols = len(values[0])
        cells = VGroup()
        labels = VGroup()
        for r in range(rows):
            for c in range(cols):
                rect = Rectangle(width=cell_w, height=cell_h)
                rect.set_stroke(color, 1.2)
                rect.set_fill(color, opacity=0.08 + 0.03 * ((r + c) % 2))
                rect.move_to([c * cell_w, -r * cell_h, 0])
                cells.add(rect)
                label = self.txt(values[r][c], size=font_size, color=GRAY_A)
                label.move_to(rect)
                labels.add(label)
        cells.center()
        labels.move_to(cells)
        title = self.txt(name, size=20, color=color).next_to(cells, UP, buff=0.12)
        shape = self.txt(f"[{rows}, {cols}]", size=16, color=GRAY_B).next_to(cells, DOWN, buff=0.1)
        return VGroup(cells, labels, title, shape)

    def row_box(self, matrix: VGroup, row: int, cols: int, color):
        cells = matrix[0]
        return SurroundingRectangle(VGroup(*[cells[row * cols + c] for c in range(cols)]), color=color, buff=0.025, stroke_width=3)

    def col_box(self, matrix: VGroup, col: int, rows: int, cols: int, color):
        cells = matrix[0]
        return SurroundingRectangle(VGroup(*[cells[r * cols + col] for r in range(rows)]), color=color, buff=0.025, stroke_width=3)


class RoPEMatrixDetail(DetailBase):
    def construct(self):
        self.title("RoPE：Q/K 矩阵逐行旋转", "例子：seq_len=4，head_dim=4；每一行对应一个位置", PURPLE)

        q = self.grid(
            "Q before",
            [
                ["q00", "q01", "q02", "q03"],
                ["q10", "q11", "q12", "q13"],
                ["q20", "q21", "q22", "q23"],
                ["q30", "q31", "q32", "q33"],
            ],
            GREEN,
        ).scale(0.95).move_to([-4.25, 0.4, 0])
        k = self.grid(
            "K before",
            [
                ["k00", "k01", "k02", "k03"],
                ["k10", "k11", "k12", "k13"],
                ["k20", "k21", "k22", "k23"],
                ["k30", "k31", "k32", "k33"],
            ],
            ORANGE,
        ).scale(0.95).move_to([-4.25, -1.7, 0])

        rope = self.card("RoPE\n按 position 旋转", PURPLE, width=2.25, height=1.0, size=20).move_to([0, -0.65, 0])
        qrot = self.grid("Q_rot", [["q′"] * 4 for _ in range(4)], PURPLE).scale(0.95).move_to([4.1, 0.4, 0])
        krot = self.grid("K_rot", [["k′"] * 4 for _ in range(4)], RED).scale(0.95).move_to([4.1, -1.7, 0])

        self.play(FadeIn(q), FadeIn(k), run_time=0.8)
        self.play(FadeIn(rope), run_time=0.4)
        arrows = VGroup(
            self.arrow(q.get_right(), rope.get_left(), GREEN),
            self.arrow(k.get_right(), rope.get_left(), ORANGE),
            self.arrow(rope.get_right(), qrot.get_left(), PURPLE),
            self.arrow(rope.get_right(), krot.get_left(), RED),
        )
        self.play(Create(arrows[:2]), run_time=0.5)
        self.play(Create(arrows[2:]), FadeIn(qrot), FadeIn(krot), run_time=0.8)

        pair_note = self.txt("head_dim=4 拆成两对维度：(0,1) 和 (2,3)", size=22, color=GRAY_A).move_to([0, 2.0, 0])
        pair1 = SurroundingRectangle(VGroup(*[q[0][r * 4 + c] for r in range(4) for c in [0, 1]]), color=BLUE, buff=0.02)
        pair2 = SurroundingRectangle(VGroup(*[q[0][r * 4 + c] for r in range(4) for c in [2, 3]]), color=YELLOW, buff=0.02)
        self.play(FadeIn(pair_note), Create(pair1), Create(pair2), run_time=0.8)

        formula = VGroup(
            self.txt("position p 使用旋转矩阵 R(p)", size=21, color=YELLOW),
            self.txt("[ cos(theta_p)  -sin(theta_p) ]", size=18, color=YELLOW),
            self.txt("[ sin(theta_p)   cos(theta_p) ]", size=18, color=YELLOW),
            self.txt("同一个 R(p) 分别作用到维度对 (0,1)、(2,3)", size=17, color=GRAY_A),
        ).arrange(DOWN, buff=0.06).move_to([0, 1.18, 0])
        self.play(FadeIn(formula), run_time=0.8)

        row_colors = [BLUE, PURPLE, ORANGE, RED]
        pos_texts = [
            "当前行：pos 0\n旋转角度 theta=0",
            "当前行：pos 1\n旋转角度 theta=1*theta",
            "当前行：pos 2\n旋转角度 theta=2*theta",
            "当前行：pos 3\n旋转角度 theta=3*theta",
        ]
        active_label = None

        for i in range(4):
            q_row = self.row_box(q, i, 4, row_colors[i])
            k_row = self.row_box(k, i, 4, row_colors[i])
            q_dest = self.row_box(qrot, i, 4, row_colors[i])
            k_dest = self.row_box(krot, i, 4, row_colors[i])
            new_label = self.card(pos_texts[i], row_colors[i], width=2.55, height=0.82, size=17).move_to([0, 0.13, 0])
            if active_label is None:
                active_label = new_label
                label_anim = FadeIn(active_label)
            else:
                label_anim = Transform(active_label, new_label)
            self.play(Create(q_row), Create(k_row), label_anim, run_time=0.4)
            self.play(Transform(q_row, q_dest), Transform(k_row, k_dest), run_time=0.55)
            self.play(FadeOut(q_row), FadeOut(k_row), run_time=0.15)

        self.strip("第一层意思：Q/K 矩阵的每一行按它所在 position 使用不同旋转角度。", PURPLE)
        self.wait(2)
        self.wipe()

        self.title("RoPE：一行向量内部怎么旋转", "把 head_dim 拆成二维 pair，再分别旋转", PURPLE)
        row = self.grid(
            "Q row at pos 2",
            [["q20", "q21", "q22", "q23"]],
            GREEN,
            cell_w=0.74,
            cell_h=0.46,
            font_size=17,
        ).move_to([-3.7, 1.15, 0])
        pair_a = SurroundingRectangle(VGroup(row[0][0], row[0][1]), color=BLUE, buff=0.035, stroke_width=3)
        pair_b = SurroundingRectangle(VGroup(row[0][2], row[0][3]), color=YELLOW, buff=0.035, stroke_width=3)
        split = self.txt("拆成两个二维向量", size=22, color=GRAY_A).move_to([-3.7, 0.35, 0])

        def vector_panel(title, center, before, after, color):
            x_axis = Line(center + LEFT * 0.9, center + RIGHT * 0.9, color=GRAY_D)
            y_axis = Line(center + DOWN * 0.75, center + UP * 0.75, color=GRAY_D)
            before_arrow = Arrow(center, center + np.array([before[0], before[1], 0]), buff=0, color=BLUE, stroke_width=5)
            after_arrow = Arrow(center, center + np.array([after[0], after[1], 0]), buff=0, color=color, stroke_width=5)
            arc = Arc(radius=0.48, start_angle=math.atan2(before[1], before[0]), angle=0.65, color=color)
            arc.move_arc_center_to(center)
            label = self.txt(title, size=19, color=color).next_to(x_axis, UP, buff=0.22)
            before_label = self.txt("before", size=15, color=BLUE).next_to(before_arrow.get_end(), RIGHT, buff=0.05)
            after_label = self.txt("after", size=15, color=color).next_to(after_arrow.get_end(), RIGHT, buff=0.05)
            return VGroup(x_axis, y_axis, before_arrow, after_arrow, arc, label, before_label, after_label)

        panel_a = vector_panel("(q20, q21)", np.array([0.0, -0.7, 0.0]), (0.65, 0.20), (0.25, 0.65), BLUE)
        panel_b = vector_panel("(q22, q23)", np.array([3.0, -0.7, 0.0]), (0.55, -0.35), (0.78, 0.10), YELLOW)
        rot_note = VGroup(
            self.txt("pos=2，所以两个 pair 都使用同一个 position 角度", size=21, color=YELLOW),
            self.txt("结果写回同一行，得到 Q_rot 的第 2 行", size=21, color=GRAY_A),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.65)

        row_out = self.grid(
            "Q_rot row 2",
            [["q20′", "q21′", "q22′", "q23′"]],
            PURPLE,
            cell_w=0.74,
            cell_h=0.46,
            font_size=17,
        ).move_to([3.7, 1.15, 0])
        self.play(FadeIn(row), run_time=0.5)
        self.play(Create(pair_a), Create(pair_b), FadeIn(split), run_time=0.6)
        self.play(FadeIn(panel_a), FadeIn(panel_b), FadeIn(rot_note), run_time=1.0)
        write_arrow = self.arrow(row.get_right(), row_out.get_left(), PURPLE)
        self.play(Create(write_arrow), FadeIn(row_out), run_time=0.9)
        self.play(FadeOut(rot_note), run_time=0.3)
        self.strip("第二层意思：RoPE 不是新增一列位置向量，而是把 Q/K 的二维子向量旋转。", PURPLE)
        self.wait(5)
        self.wipe()

        self.title("RoPE 后再算 attention scores", "scores = Q_rot @ K_rot.T", YELLOW)
        q2 = self.grid("Q_rot", [["q′"] * 4 for _ in range(4)], PURPLE).scale(0.9).move_to([-4.2, 0.2, 0])
        kt = self.grid("K_rot.T", [["k′"] * 4 for _ in range(4)], ORANGE).scale(0.9).move_to([0, 0.2, 0])
        scores = self.grid("scores", [["s"] * 4 for _ in range(4)], RED).scale(0.9).move_to([4.2, 0.2, 0])
        at = self.txt("@", size=34, color=GRAY_A).move_to([-2.05, 0.2, 0])
        eq = self.txt("=", size=34, color=GRAY_A).move_to([2.05, 0.2, 0])
        self.play(FadeIn(q2), FadeIn(kt), FadeIn(scores), FadeIn(at), FadeIn(eq), run_time=0.8)

        row = self.row_box(q2, 2, 4, BLUE)
        col = self.col_box(kt, 1, 4, 4, ORANGE)
        cell = SurroundingRectangle(scores[0][2 * 4 + 1], color=YELLOW, buff=0.025, stroke_width=3)
        detail = VGroup(
            self.txt("score[2,1]", size=22, color=YELLOW),
            self.txt("= Q_rot 第 2 行 · K_rot 第 1 行", size=21, color=YELLOW),
            self.txt("角度差 θ₂ - θ₁ 进入点积", size=20, color=GRAY_A),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.45)
        self.play(Create(row), Create(col), Create(cell), FadeIn(detail), run_time=1.0)
        self.wait(7)


class RMSNormDetail(DetailBase):
    def construct(self):
        self.title("RMSNorm：一步步缩放 hidden vector", "不减均值，只控制尺度", GREEN)
        x_vals = [["3.0", "-1.0", "2.0", "4.0"]]
        sq_vals = [["9.0", "1.0", "4.0", "16.0"]]
        norm_vals = [["1.10", "-0.37", "0.73", "1.46"]]
        x = self.grid("x", x_vals, BLUE, cell_w=0.72, cell_h=0.48, font_size=18).move_to([-4.25, 0.8, 0])
        square = self.grid("x²", sq_vals, ORANGE, cell_w=0.72, cell_h=0.48, font_size=18).move_to([-1.25, 0.8, 0])
        rms = self.card("mean=7.5\nRMS≈2.74", GREEN, width=2.0, height=1.0, size=19).move_to([1.45, 0.8, 0])
        out = self.grid("x / RMS", norm_vals, GREEN, cell_w=0.72, cell_h=0.48, font_size=18).move_to([4.3, 0.8, 0])

        arrows = VGroup(
            self.arrow(x.get_right(), square.get_left(), ORANGE),
            self.arrow(square.get_right(), rms.get_left(), GREEN),
            self.arrow(rms.get_right(), out.get_left(), GREEN),
        )
        self.play(FadeIn(x), run_time=0.5)
        self.play(Create(arrows[0]), FadeIn(square), run_time=0.7)
        self.play(Create(arrows[1]), FadeIn(rms), run_time=0.7)
        self.play(Create(arrows[2]), FadeIn(out), run_time=0.7)

        bars_before = self.bars([3.0, 1.0, 2.0, 4.0], BLUE, "缩放前").move_to([-2.5, -1.5, 0])
        bars_after = self.bars([1.1, 0.37, 0.73, 1.46], GREEN, "缩放后").move_to([2.5, -1.5, 0])
        self.play(FadeIn(bars_before), run_time=0.5)
        self.play(TransformFromCopy(bars_before, bars_after), run_time=0.9)
        self.strip("RMSNorm：不改变方向关系太多，主要把 hidden state 的尺度压稳。", GREEN)
        self.wait(7)

    def bars(self, values, color, title):
        bars = VGroup()
        for value in values:
            rect = Rectangle(width=0.34, height=value * 0.35)
            rect.set_stroke(color, 1)
            rect.set_fill(color, opacity=0.65)
            rect.align_to(ORIGIN, DOWN)
            bars.add(rect)
        bars.arrange(RIGHT, buff=0.16, aligned_edge=DOWN)
        label = self.txt(title, size=20, color=color).next_to(bars, UP, buff=0.12)
        return VGroup(bars, label)


class SwiGLUDetail(DetailBase):
    def construct(self):
        self.title("SwiGLU：不是一个箭头，是两路向量相乘", "gate 控制通过多少，value 提供内容", ORANGE)

        x = self.grid("x", [["x1", "x2", "x3"]], BLUE, cell_w=0.62, cell_h=0.44).move_to([-5.0, 0.85, 0])
        wg = self.grid("W_gate", [["", "", ""], ["", "", ""], ["", "", ""]], ORANGE, cell_w=0.34, cell_h=0.34, font_size=10).move_to([-3.1, 1.45, 0])
        wv = self.grid("W_value", [["", "", ""], ["", "", ""], ["", "", ""]], TEAL, cell_w=0.34, cell_h=0.34, font_size=10).move_to([-3.1, -0.45, 0])
        gate = self.grid("gate", [["0.8", "-0.4", "1.2"]], ORANGE, cell_w=0.66, cell_h=0.44, font_size=17).move_to([-0.75, 1.45, 0])
        silu = self.grid("SiLU(gate)", [["0.55", "-0.16", "0.92"]], ORANGE, cell_w=0.78, cell_h=0.44, font_size=16).move_to([1.85, 1.45, 0])
        value = self.grid("value", [["1.4", "0.7", "-0.5"]], TEAL, cell_w=0.66, cell_h=0.44, font_size=17).move_to([-0.75, -0.45, 0])
        prod = self.grid("multiply", [["0.77", "-0.11", "-0.46"]], YELLOW, cell_w=0.72, cell_h=0.44, font_size=17).move_to([3.0, 0.35, 0])
        wout = self.grid("W_out", [[""], [""], [""]], GREEN, cell_w=0.34, cell_h=0.34, font_size=10).move_to([4.75, 0.35, 0])
        out = self.grid("out", [["o1", "o2", "o3"]], GREEN, cell_w=0.62, cell_h=0.44).move_to([4.6, -1.4, 0])

        self.play(FadeIn(x), run_time=0.5)
        self.play(FadeIn(wg), FadeIn(wv), Create(self.arrow(x.get_right(), wg.get_left(), ORANGE)), Create(self.arrow(x.get_right(), wv.get_left(), TEAL)), run_time=0.9)
        self.play(FadeIn(gate), FadeIn(value), Create(self.arrow(wg.get_right(), gate.get_left(), ORANGE)), Create(self.arrow(wv.get_right(), value.get_left(), TEAL)), run_time=0.9)
        self.play(FadeIn(silu), Create(self.arrow(gate.get_right(), silu.get_left(), ORANGE)), run_time=0.7)

        multiply_note = self.txt("逐元素相乘：同一列对同一列", size=22, color=YELLOW).move_to([1.65, -1.45, 0])
        self.play(FadeIn(multiply_note), run_time=0.4)
        self.play(FadeIn(prod), Create(self.arrow(silu.get_bottom(), prod.get_top(), ORANGE)), Create(self.arrow(value.get_right(), prod.get_left(), TEAL)), run_time=0.9)

        for i, color in enumerate([BLUE, PURPLE, RED]):
            a = SurroundingRectangle(silu[0][i], color=color, buff=0.025, stroke_width=3)
            b = SurroundingRectangle(value[0][i], color=color, buff=0.025, stroke_width=3)
            c = SurroundingRectangle(prod[0][i], color=color, buff=0.025, stroke_width=3)
            self.play(Create(a), Create(b), Create(c), run_time=0.25)
            self.play(FadeOut(a), FadeOut(b), FadeOut(c), run_time=0.18)

        self.play(FadeIn(wout), Create(self.arrow(prod.get_right(), wout.get_left(), GREEN)), run_time=0.6)
        self.play(FadeIn(out), Create(self.arrow(wout.get_bottom(), out.get_top(), GREEN)), run_time=0.6)
        self.strip("SwiGLU：Linear 分成 gate/value 两路，SiLU(gate) * value 后再投影。", ORANGE)
        self.wait(7)


class WeightTyingDetail(DetailBase):
    def construct(self):
        self.title("Weight Tying：输入和输出用同一张表", "不是两套词表参数，而是一套 embedding table 复用两次", YELLOW)

        table = self.grid(
            "embedding table",
            [
                ["e00", "e01", "e02", "e03"],
                ["e10", "e11", "e12", "e13"],
                ["e20", "e21", "e22", "e23"],
                ["e30", "e31", "e32", "e33"],
                ["e40", "e41", "e42", "e43"],
            ],
            YELLOW,
            cell_w=0.58,
            cell_h=0.36,
            font_size=14,
        ).move_to([0, 0.2, 0])
        token = self.card("token id = 2", BLUE, width=2.2).move_to([-4.7, 1.0, 0])
        vec = self.grid("token vector", [["e20", "e21", "e22", "e23"]], BLUE, cell_w=0.62, cell_h=0.44, font_size=15).move_to([-4.5, -1.25, 0])
        hidden = self.grid("hidden", [["h0", "h1", "h2", "h3"]], GREEN, cell_w=0.62, cell_h=0.44, font_size=15).move_to([4.5, -1.25, 0])
        logits = self.grid("logits", [["l0", "l1", "l2", "l3", "l4"]], RED, cell_w=0.48, cell_h=0.44, font_size=15).move_to([4.5, 1.05, 0])

        row2 = self.row_box(table, 2, 4, BLUE)
        self.play(FadeIn(table), run_time=0.5)
        self.play(FadeIn(token), Create(self.arrow(token.get_right(), table.get_left() + UP * 0.15, BLUE)), Create(row2), run_time=0.8)
        self.play(FadeIn(vec), Create(self.arrow(table.get_left() + DOWN * 0.3, vec.get_right(), BLUE)), run_time=0.7)

        transpose_label = self.txt("输出时使用同一张表的转置：hidden @ E.T", size=23, color=RED).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(hidden), FadeIn(transpose_label), run_time=0.6)
        self.play(Create(self.arrow(hidden.get_left(), table.get_right() + DOWN * 0.25, GREEN)), run_time=0.6)
        self.play(FadeIn(logits), Create(self.arrow(table.get_right() + UP * 0.25, logits.get_left(), RED)), run_time=0.8)

        for i in range(5):
            row = self.row_box(table, i, 4, RED)
            cell = SurroundingRectangle(logits[0][i], color=RED, buff=0.025, stroke_width=3)
            self.play(Create(row), Create(cell), run_time=0.22)
            self.play(FadeOut(row), FadeOut(cell), run_time=0.14)

        self.strip("Weight Tying：省掉独立 LM Head，让输入和输出共享同一套 token 表示。", YELLOW)
        self.wait(7)
