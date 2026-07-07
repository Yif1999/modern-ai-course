from manim import *


# Q/K/V 矩阵形式教学动画
# 重点解释：
# 1. 代码里的 nn.Linear(C, C) 本质上是一个可学习矩阵。
# 2. 输入 X: [seq_len, n_embd] 乘以 W_Q / W_K / W_V 后得到 Q / K / V。
# 3. Q/K/V 是激活值；W_Q/W_K/W_V 才是模型参数。
# 4. 所有 token 共用同一组投影矩阵，这就是线性层在序列上的批量矩阵乘法。

config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720


BG = "#0f172a"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
TEAL = "#2dd4bf"
YELLOW = "#facc15"
ORANGE = "#fb923c"
PURPLE = "#c084fc"
GREEN = "#86efac"
RED = "#fb7185"
GOLD = "#fbbf24"


class QKVMatrixFormDemo(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.show_linear_layer_as_matrix()
        self.show_all_tokens_matrix_multiply()
        self.show_one_token_row()
        self.show_three_projections()
        self.show_head_split()
        self.show_summary()

    def text(self, content, size=24, color=TEXT):
        return Text(content, font_size=size, color=color, font="Arial")

    def clear(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.35)

    def matrix_grid(
        self,
        rows,
        cols,
        color,
        cell=0.24,
        opacity=0.72,
        stroke="#1e293b",
    ):
        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell)
                sq.set_fill(color, opacity=opacity)
                sq.set_stroke(stroke, width=0.75)
                sq.move_to(RIGHT * c * cell + DOWN * r * cell)
                grid.add(sq)
        grid.center()
        return grid

    def labeled_matrix(self, name, shape, rows, cols, color, cell=0.24):
        title = self.text(name, size=23, color=color)
        grid = self.matrix_grid(rows, cols, color=color, cell=cell)
        shape_label = self.text(shape, size=16, color=MUTED)
        return VGroup(title, grid, shape_label).arrange(DOWN, buff=0.1)

    def box(self, content, color, width=4.2, height=0.68, size=22):
        rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=2,
            fill_color=color,
            fill_opacity=0.13,
        )
        label = self.text(content, size=size, color=TEXT)
        label.move_to(rect.get_center())
        return VGroup(rect, label)

    def arrow(self, start, end, color=MUTED):
        return Arrow(
            start.get_right(),
            end.get_left(),
            buff=0.14,
            stroke_width=3,
            color=color,
            max_tip_length_to_length_ratio=0.14,
        )

    def show_linear_layer_as_matrix(self):
        title = self.text("A Linear layer is a learned matrix multiplication", size=34)
        title.to_edge(UP, buff=0.35)

        code = VGroup(
            self.box("q_proj = nn.Linear(C, C)", BLUE, width=4.6),
            self.box("k_proj = nn.Linear(C, C)", ORANGE, width=4.6),
            self.box("v_proj = nn.Linear(C, C)", PURPLE, width=4.6),
        ).arrange(DOWN, buff=0.26)
        code.move_to(LEFT * 3.3 + UP * 0.6)

        math = VGroup(
            self.text("Math view", size=28, color=GOLD),
            self.text("Q = X @ W_Q + b_Q", size=26, color=YELLOW),
            self.text("K = X @ W_K + b_K", size=26, color=ORANGE),
            self.text("V = X @ W_V + b_V", size=26, color=PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        math.move_to(RIGHT * 3.05 + UP * 0.45)

        note = self.text(
            "W_Q / W_K / W_V are parameters. Q / K / V are activations.",
            size=22,
            color=GREEN,
        )
        note.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(x) for x in code], lag_ratio=0.15), run_time=1.0)
        self.play(FadeIn(math, shift=LEFT * 0.2), run_time=0.8)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.4)
        self.clear()

    def show_all_tokens_matrix_multiply(self):
        title = self.text("All token embeddings are multiplied by the same W_Q", size=33)
        title.to_edge(UP, buff=0.35)

        x = self.labeled_matrix("X", "[T, C] = [4, 6]", 4, 6, BLUE)
        wq = self.labeled_matrix("W_Q", "[C, C] = [6, 6]", 6, 6, YELLOW, cell=0.21)
        q = self.labeled_matrix("Q", "[T, C] = [4, 6]", 4, 6, YELLOW)

        equation = VGroup(x, self.text("@", size=36, color=TEXT), wq, self.text("=", size=36), q)
        equation.arrange(RIGHT, buff=0.42)
        equation.move_to(UP * 0.2)

        row_labels = VGroup()
        for i in range(4):
            label = self.text(f"token {i}", size=14, color=MUTED)
            label.next_to(x[1][i * 6], LEFT, buff=0.13)
            row_labels.add(label)

        note = self.text(
            "One matrix multiply computes q_0, q_1, q_2, q_3 at once.",
            size=22,
            color=GREEN,
        )
        note.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(title))
        self.play(FadeIn(equation), FadeIn(row_labels), run_time=1.0)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.5)
        self.clear()

    def show_one_token_row(self):
        title = self.text("One row of X becomes one row of Q", size=34)
        title.to_edge(UP, buff=0.35)

        x_row = self.labeled_matrix("x_2", "[1, C]", 1, 6, BLUE, cell=0.28)
        wq = self.labeled_matrix("W_Q", "[C, C]", 6, 6, YELLOW, cell=0.2)
        q_row = self.labeled_matrix("q_2", "[1, C]", 1, 6, YELLOW, cell=0.28)

        equation = VGroup(
            x_row,
            self.text("@", size=36),
            wq,
            self.text("+ b_Q =", size=28),
            q_row,
        ).arrange(RIGHT, buff=0.38)
        equation.move_to(UP * 0.35)

        explanation = VGroup(
            self.text("x_2 is the embedding vector of token 2.", size=21, color=TEXT),
            self.text("W_Q is shared by every token position.", size=21, color=GREEN),
            self.text("q_2 is the Query vector used by token 2.", size=21, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        explanation.to_edge(DOWN, buff=0.65)

        self.play(FadeIn(title))
        self.play(FadeIn(equation), run_time=1.0)
        self.play(FadeIn(explanation), run_time=0.8)
        self.wait(1.5)
        self.clear()

    def show_three_projections(self):
        title = self.text("Q, K, V come from three different learned matrices", size=33)
        title.to_edge(UP, buff=0.35)

        x = self.labeled_matrix("X", "[T, C]", 4, 6, BLUE)
        x.move_to(LEFT * 4.8 + UP * 0.15)

        wq = self.labeled_matrix("W_Q", "[C, C]", 4, 4, YELLOW, cell=0.2)
        wk = self.labeled_matrix("W_K", "[C, C]", 4, 4, ORANGE, cell=0.2)
        wv = self.labeled_matrix("W_V", "[C, C]", 4, 4, PURPLE, cell=0.2)
        weights = VGroup(wq, wk, wv).arrange(DOWN, buff=0.28)
        weights.move_to(LEFT * 1.1 + UP * 0.15)

        q = self.labeled_matrix("Q", "[T, C]", 4, 6, YELLOW, cell=0.21)
        k = self.labeled_matrix("K", "[T, C]", 4, 6, ORANGE, cell=0.21)
        v = self.labeled_matrix("V", "[T, C]", 4, 6, PURPLE, cell=0.21)
        outputs = VGroup(q, k, v).arrange(DOWN, buff=0.28)
        outputs.move_to(RIGHT * 3.35 + UP * 0.15)

        arrows = VGroup()
        for weight, out, color in [(wq, q, YELLOW), (wk, k, ORANGE), (wv, v, PURPLE)]:
            arrows.add(Arrow(x.get_right(), weight.get_left(), buff=0.15, stroke_width=2.5, color=color))
            arrows.add(Arrow(weight.get_right(), out.get_left(), buff=0.15, stroke_width=2.5, color=color))

        note = self.text(
            "Same X, three projections: attention asks different questions from the same hidden state.",
            size=20,
            color=GREEN,
        )
        note.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(title))
        self.play(FadeIn(x), FadeIn(weights), FadeIn(outputs), Create(arrows), run_time=1.2)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.5)
        self.clear()

    def show_head_split(self):
        title = self.text("Multi-head attention reshapes Q / K / V after projection", size=32)
        title.to_edge(UP, buff=0.35)

        q = self.labeled_matrix("Q", "[T, C] = [4, 6]", 4, 6, YELLOW)
        q.move_to(LEFT * 3.4 + UP * 0.55)

        head1 = self.labeled_matrix("head 0", "[T, 3]", 4, 3, YELLOW, cell=0.24)
        head2 = self.labeled_matrix("head 1", "[T, 3]", 4, 3, YELLOW, cell=0.24)
        heads = VGroup(head1, head2).arrange(RIGHT, buff=0.65)
        heads.move_to(RIGHT * 2.2 + UP * 0.55)

        arrow = Arrow(q.get_right(), heads.get_left(), buff=0.25, stroke_width=3, color=GREEN)

        formula = self.text(
            "C = num_heads * head_dim  ->  6 = 2 * 3",
            size=24,
            color=TEXT,
        )
        formula.move_to(DOWN * 1.65)

        note = self.text(
            "Projection first, split into heads second.",
            size=23,
            color=GREEN,
        )
        note.to_edge(DOWN, buff=0.62)

        self.play(FadeIn(title))
        self.play(FadeIn(q), FadeIn(heads), Create(arrow), run_time=1.1)
        self.play(FadeIn(formula), FadeIn(note), run_time=0.7)
        self.wait(1.4)
        self.clear()

    def show_summary(self):
        title = self.text("Key idea", size=38, color=GOLD)
        title.to_edge(UP, buff=0.5)

        lines = VGroup(
            self.text("nn.Linear(C, C) stores learned weights W and bias b.", size=25, color=TEXT),
            self.text("For Q/K/V: X is multiplied by three learned matrices.", size=25, color=TEXT),
            self.text("W_Q / W_K / W_V are parameters.", size=25, color=GREEN),
            self.text("Q / K / V are computed activations for the current input.", size=25, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        lines.move_to(DOWN * 0.2)

        final = self.text("Code uses Linear. Math sees matrix multiplication.", size=29, color=BLUE)
        final.to_edge(DOWN, buff=0.7)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(line, shift=RIGHT * 0.15) for line in lines], lag_ratio=0.15))
        self.play(FadeIn(final), run_time=0.6)
        self.wait(2.0)
