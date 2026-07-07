from manim import *


# Linear(6, 6) 权重连接关系动画
# 重点解释：
# 1. 输入维度 6、输出维度 6 只是节点数量 / 向量长度。
# 2. 权重不是在节点上，而是在每条输入到输出的连接线上。
# 3. 每个输出节点都连接全部 6 个输入节点，所以每个输出有 6 个权重。
# 4. 6 个输出 × 每个输出 6 条输入连接 = 36 个 weight 参数。
# 5. 如果 bias=True，还会有 6 个 bias 参数，每个输出节点一个。

config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720


BG = "#0f172a"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
GREEN = "#86efac"
YELLOW = "#facc15"
ORANGE = "#fb923c"
RED = "#fb7185"
PURPLE = "#c084fc"


class LinearWeightsConnectionsDemo(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.show_nodes_are_not_parameters()
        self.show_one_output_connections()
        self.show_all_connections()
        self.show_matrix_view()
        self.show_parameter_count()

    def t(self, text, size=24, color=TEXT):
        return Text(text, font_size=size, color=color, font="Arial")

    def clear_scene(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.35)

    def node(self, label, color):
        circle = Circle(radius=0.22, stroke_color=color, stroke_width=2)
        circle.set_fill(color, opacity=0.16)
        text = self.t(label, size=15, color=TEXT)
        text.move_to(circle.get_center())
        return VGroup(circle, text)

    def make_layer_nodes(self, labels, color, x_pos):
        nodes = VGroup()
        for i, label in enumerate(labels):
            n = self.node(label, color)
            n.move_to(RIGHT * x_pos + UP * (2.3 - i * 0.92))
            nodes.add(n)
        return nodes

    def edge(self, left_node, right_node, color=MUTED, width=1.4, opacity=0.65):
        line = Line(
            left_node.get_right(),
            right_node.get_left(),
            color=color,
            stroke_width=width,
        )
        line.set_opacity(opacity)
        return line

    def grid(self, rows, cols, color, cell=0.28):
        g = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell)
                sq.set_fill(color, opacity=0.68)
                sq.set_stroke(BG, width=0.7)
                sq.move_to(RIGHT * c * cell + DOWN * r * cell)
                g.add(sq)
        g.center()
        return g

    def show_nodes_are_not_parameters(self):
        title = self.t("Linear(6, 6): nodes are not the weights", size=34)
        title.to_edge(UP, buff=0.35)

        inputs = self.make_layer_nodes([f"x{i}" for i in range(6)], BLUE, -3.2)
        outputs = self.make_layer_nodes([f"y{i}" for i in range(6)], GREEN, 3.2)

        input_label = self.t("6 input values", size=22, color=BLUE).next_to(inputs, UP, buff=0.25)
        output_label = self.t("6 output values", size=22, color=GREEN).next_to(outputs, UP, buff=0.25)

        wrong = self.t("Not: 6 + 6 = 12 parameters", size=28, color=RED)
        wrong.to_edge(DOWN, buff=0.65)
        cross = Cross(wrong, stroke_color=RED, stroke_width=5)

        note = self.t("Input and output nodes describe shape, not parameter count.", size=21, color=MUTED)
        note.next_to(wrong, UP, buff=0.28)

        self.play(FadeIn(title))
        self.play(FadeIn(inputs), FadeIn(outputs), FadeIn(input_label), FadeIn(output_label), run_time=0.9)
        self.play(FadeIn(note), FadeIn(wrong), Create(cross), run_time=0.9)
        self.wait(1.2)
        self.clear_scene()

    def show_one_output_connections(self):
        title = self.t("One output value uses all 6 input values", size=34)
        title.to_edge(UP, buff=0.35)

        inputs = self.make_layer_nodes([f"x{i}" for i in range(6)], BLUE, -3.6)
        y0 = self.node("y0", GREEN)
        y0.move_to(RIGHT * 3.15)

        edges = VGroup()
        edge_labels = VGroup()
        for i, x_node in enumerate(inputs):
            e = self.edge(x_node, y0, color=YELLOW, width=2.5, opacity=0.85)
            edges.add(e)
            label = self.t(f"w{i}0", size=14, color=YELLOW)
            label.move_to(e.point_from_proportion(0.55) + UP * (0.08 if i % 2 == 0 else -0.08))
            edge_labels.add(label)

        formula = self.t(
            "y0 = x0*w00 + x1*w10 + ... + x5*w50 + b0",
            size=25,
            color=TEXT,
        )
        formula.to_edge(DOWN, buff=0.75)

        count = self.t("For y0: 6 connection weights + 1 bias", size=23, color=GREEN)
        count.next_to(formula, UP, buff=0.28)

        self.play(FadeIn(title))
        self.play(FadeIn(inputs), FadeIn(y0), run_time=0.7)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.08), FadeIn(edge_labels), run_time=1.2)
        self.play(FadeIn(count), FadeIn(formula), run_time=0.8)
        self.wait(1.5)
        self.clear_scene()

    def show_all_connections(self):
        title = self.t("Linear(6, 6): every output has 6 incoming weights", size=33)
        title.to_edge(UP, buff=0.35)

        inputs = self.make_layer_nodes([f"x{i}" for i in range(6)], BLUE, -3.6)
        outputs = self.make_layer_nodes([f"y{j}" for j in range(6)], GREEN, 3.6)

        all_edges = VGroup()
        for j, y_node in enumerate(outputs):
            for i, x_node in enumerate(inputs):
                color = [YELLOW, ORANGE, PURPLE, GREEN, BLUE, RED][j]
                all_edges.add(self.edge(x_node, y_node, color=color, width=1.25, opacity=0.42))

        highlight_y3_edges = VGroup()
        for x_node in inputs:
            highlight_y3_edges.add(self.edge(x_node, outputs[3], color=YELLOW, width=3.0, opacity=0.95))

        note1 = self.t("6 outputs", size=23, color=GREEN).next_to(outputs, UP, buff=0.25)
        note2 = self.t("each output receives 6 weighted connections", size=23, color=YELLOW)
        note2.to_edge(DOWN, buff=0.9)
        count = self.t("6 outputs × 6 inputs = 36 weights", size=30, color=GREEN)
        count.next_to(note2, UP, buff=0.25)

        self.play(FadeIn(title))
        self.play(FadeIn(inputs), FadeIn(outputs), FadeIn(note1), run_time=0.7)
        self.play(Create(all_edges), run_time=1.0)
        self.play(Create(highlight_y3_edges), FadeIn(note2), run_time=0.8)
        self.play(FadeIn(count), run_time=0.7)
        self.wait(1.5)
        self.clear_scene()

    def show_matrix_view(self):
        title = self.t("The 36 connection weights are stored as a 6 x 6 matrix", size=33)
        title.to_edge(UP, buff=0.35)

        left = VGroup(
            self.t("connection view", size=24, color=BLUE),
            self.t("x_i -> y_j has weight w_ij", size=20, color=MUTED),
        ).arrange(DOWN, buff=0.2)
        left.move_to(LEFT * 3.35 + UP * 2.45)

        inputs = self.make_layer_nodes([f"x{i}" for i in range(3)], BLUE, -4.4)
        outputs = self.make_layer_nodes([f"y{j}" for j in range(3)], GREEN, -2.2)
        inputs.shift(DOWN * 0.85)
        outputs.shift(DOWN * 0.85)
        mini_edges = VGroup()
        for y in outputs:
            for x in inputs:
                mini_edges.add(self.edge(x, y, color=YELLOW, width=1.6, opacity=0.65))

        matrix_title = self.t("matrix view", size=24, color=YELLOW)
        matrix = self.grid(6, 6, YELLOW, cell=0.3)
        matrix_label = self.t("W: [out_features, in_features] = [6, 6]", size=19, color=MUTED)
        matrix_group = VGroup(matrix_title, matrix, matrix_label).arrange(DOWN, buff=0.15)
        matrix_group.move_to(RIGHT * 2.7 + DOWN * 0.05)

        # 高亮一行：一个输出节点对应一组 6 个输入权重。
        highlight = SurroundingRectangle(VGroup(*matrix[18:24]), color=GREEN, buff=0.03, stroke_width=3)
        row_note = self.t("one row: weights feeding one output", size=21, color=GREEN)
        row_note.next_to(matrix_group, DOWN, buff=0.28)

        implementation_note = self.t(
            "Frameworks may store W as [out, in], then compute X @ W.T + b.",
            size=20,
            color=TEXT,
        )
        implementation_note.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(title))
        self.play(FadeIn(left), FadeIn(inputs), FadeIn(outputs), Create(mini_edges), run_time=0.9)
        self.play(FadeIn(matrix_group), run_time=0.8)
        self.play(Create(highlight), FadeIn(row_note), FadeIn(implementation_note), run_time=0.9)
        self.wait(1.6)
        self.clear_scene()

    def show_parameter_count(self):
        title = self.t("Parameter count", size=38, color=GREEN)
        title.to_edge(UP, buff=0.45)

        equations = VGroup(
            self.t("Linear(in_features, out_features)", size=28, color=TEXT),
            self.t("weight params = in_features × out_features", size=26, color=YELLOW),
            self.t("bias params = out_features", size=26, color=ORANGE),
            self.t("Linear(6, 6): 6 × 6 + 6 = 42 params", size=31, color=GREEN),
            self.t("without bias: 6 × 6 = 36 params", size=28, color=BLUE),
        ).arrange(DOWN, buff=0.35)
        equations.move_to(DOWN * 0.05)

        final = self.t("Weights live on connections. Bias lives on output nodes.", size=25, color=TEXT)
        final.to_edge(DOWN, buff=0.62)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(eq, shift=RIGHT * 0.15) for eq in equations], lag_ratio=0.14), run_time=1.5)
        self.play(FadeIn(final), run_time=0.7)
        self.wait(2.0)
