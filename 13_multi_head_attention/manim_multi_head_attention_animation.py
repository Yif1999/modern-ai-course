from manim import *


config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720


class MultiHeadAttentionExplained(Scene):
    """用分屏式动画解释 Multi-Head Attention，避免元素堆叠和箭头交叉。"""

    def make_card(
        self,
        title,
        body,
        color=BLUE,
        width=3.0,
        height=1.25,
        title_size=24,
        body_size=18,
    ):
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.12,
            color=color,
            stroke_width=3,
        )
        title_text = Text(title, font_size=title_size, color=WHITE)
        body_text = Text(body, font_size=body_size, color=GRAY_B, line_spacing=0.85)
        text_group = VGroup(title_text, body_text).arrange(DOWN, buff=0.12)
        text_group.move_to(box.get_center())
        return VGroup(box, text_group)

    def make_arrow(self, start, end, color=GRAY_A):
        return Arrow(
            start=start,
            end=end,
            buff=0.18,
            stroke_width=4,
            color=color,
            max_tip_length_to_length_ratio=0.16,
        )

    def make_matrix(self, color, side=0.22):
        grid = VGroup()
        for r in range(4):
            for c in range(4):
                cell = Square(side_length=side)
                cell.set_stroke(color, width=1.2)
                if c > r:
                    cell.set_fill(GRAY_D, opacity=0.35)
                else:
                    opacity = 0.25 + 0.12 * ((r + c) % 4)
                    cell.set_fill(color, opacity=opacity)
                cell.move_to(RIGHT * (c - 1.5) * side + DOWN * (r - 1.5) * side)
                grid.add(cell)
        return grid

    def page_title(self, text, subtitle=None):
        title = Text(text, font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.35)
        if subtitle is None:
            return VGroup(title)
        sub = Text(subtitle, font_size=22, color=GRAY_B)
        sub.next_to(title, DOWN, buff=0.18)
        return VGroup(title, sub)

    def construct(self):
        self.camera.background_color = "#080B10"

        # 第一幕：token embedding + position embedding 得到 x。
        title = self.page_title(
            "Multi-Head Attention：多头注意力",
            "先得到每个位置的隐藏向量 x，再并行拆成多个 head",
        )
        tokens = VGroup()
        token_text = list("hello ai")
        for i, token in enumerate(token_text):
            box = RoundedRectangle(
                width=0.58,
                height=0.58,
                corner_radius=0.08,
                color=GRAY_B,
                stroke_width=2,
            )
            label = Text(repr(token)[1:-1] if token != " " else "space", font_size=20)
            label.move_to(box)
            item = VGroup(box, label)
            item.move_to(LEFT * 2.2 + RIGHT * i * 0.65 + UP * 1.35)
            tokens.add(item)

        token_emb = self.make_card(
            "Token Emb",
            "[B, T, C]",
            color=TEAL,
            width=2.65,
            height=1.05,
        )
        pos_emb = self.make_card(
            "Position Emb",
            "[T, C]",
            color=BLUE,
            width=2.85,
            height=1.05,
        )
        x_card = self.make_card(
            "x = token + position",
            "[B, T, C]\nC = n_embd",
            color=YELLOW,
            width=3.35,
            height=1.25,
        )

        token_emb.move_to(LEFT * 2.0 + DOWN * 0.45)
        pos_emb.move_to(RIGHT * 2.0 + DOWN * 0.45)
        x_card.move_to(DOWN * 2.35)

        plus = Text("+", font_size=42, color=WHITE)
        plus.move_to(DOWN * 0.45)

        arrows_1 = VGroup(
            self.make_arrow(token_emb.get_bottom(), x_card.get_top() + LEFT * 0.55),
            self.make_arrow(pos_emb.get_bottom(), x_card.get_top() + RIGHT * 0.55),
        )

        self.play(FadeIn(title, shift=DOWN))
        self.play(LaggedStart(*[FadeIn(t, shift=UP) for t in tokens], lag_ratio=0.05))
        self.play(FadeIn(token_emb), FadeIn(pos_emb), FadeIn(plus))
        self.play(Create(arrows_1), FadeIn(x_card, shift=RIGHT))
        self.wait(1.8)
        self.play(FadeOut(VGroup(title, tokens, token_emb, pos_emb, plus, x_card, arrows_1)))

        # 第二幕：Q/K/V 投影，再 split heads。
        title = self.page_title(
            "一步生成 Q / K / V，然后切成 4 个 head",
            "本节例子：n_embd = 32，num_heads = 4，所以 head_size = 8",
        )
        x_card = self.make_card("x", "[B, T, 32]", color=YELLOW, width=2.1)
        qkv_card = self.make_card(
            "Linear Projections",
            "Q, K, V\n[B, T, 32]",
            color=ORANGE,
            width=3.25,
            height=1.35,
        )
        split_card = self.make_card(
            "Split Heads",
            "[B, T, 32]\n→ [B, 4, T, 8]",
            color=PURPLE,
            width=3.25,
            height=1.35,
        )
        heads_shape = self.make_card(
            "每个 head",
            "拿到 8 维子空间\n独立做 attention",
            color=GREEN,
            width=3.25,
            height=1.35,
        )

        row = VGroup(x_card, qkv_card, split_card, heads_shape).arrange(
            RIGHT,
            buff=0.55,
        )
        row.move_to(DOWN * 0.1)

        arrows_2 = VGroup(
            self.make_arrow(x_card.get_right(), qkv_card.get_left()),
            self.make_arrow(qkv_card.get_right(), split_card.get_left()),
            self.make_arrow(split_card.get_right(), heads_shape.get_left()),
        )

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP))
        self.play(Create(arrows_2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, row, arrows_2)))

        # 第三幕：四个 head 并行，每个 head 都有自己的 causal attention map。
        title = self.page_title(
            "4 个 Head 并行做 Causal Self-Attention",
            "每个 head 都有自己的 [T, T] attention map；灰色右上角表示未来 token 被 mask",
        )
        head_colors = [RED, GREEN, PURPLE, TEAL]
        head_cards = VGroup()
        positions = [
            LEFT * 3.25 + UP * 0.75,
            RIGHT * 3.25 + UP * 0.75,
            LEFT * 3.25 + DOWN * 1.85,
            RIGHT * 3.25 + DOWN * 1.85,
        ]

        for i, color in enumerate(head_colors):
            card = RoundedRectangle(
                width=5.15,
                height=2.15,
                corner_radius=0.12,
                color=color,
                stroke_width=3,
            )
            card.move_to(positions[i])

            label = Text(f"Head {i}", font_size=24, color=color)
            label.move_to(card.get_top() + DOWN * 0.33)

            steps = Text(
                "Q_h K_hᵀ / sqrt(8)\nmask → softmax\nA_h @ V_h",
                font_size=18,
                color=WHITE,
                line_spacing=0.85,
            )
            steps.move_to(card.get_center() + LEFT * 1.15 + DOWN * 0.15)

            matrix = self.make_matrix(color, side=0.19)
            matrix.move_to(card.get_center() + RIGHT * 1.35 + DOWN * 0.08)

            shape = Text("A_h: [T, T]", font_size=17, color=GRAY_B)
            shape.next_to(matrix, DOWN, buff=0.12)

            head_cards.add(VGroup(card, label, steps, matrix, shape))

        self.play(FadeIn(title, shift=DOWN))
        self.play(
            LaggedStart(
                *[FadeIn(card, scale=0.92) for card in head_cards],
                lag_ratio=0.12,
            )
        )
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, head_cards)))

        # 第四幕：concat + output projection + logits。
        title = self.page_title(
            "Concat Heads，再做 Output Projection",
            "concat 只是拼接；projection 会重新混合不同 head 的信息",
        )

        head_outputs = VGroup()
        for i, color in enumerate(head_colors):
            bar = self.make_card(
                f"out {i}",
                "[B, T, 8]",
                color=color,
                width=1.55,
                height=0.85,
                title_size=20,
                body_size=15,
            )
            head_outputs.add(bar)
        head_outputs.arrange(DOWN, buff=0.18)
        head_outputs.move_to(LEFT * 4.7 + DOWN * 0.15)

        concat_card = self.make_card(
            "Concat",
            "4 × [B, T, 8]\n→ [B, T, 32]",
            color=GOLD,
            width=2.75,
            height=1.25,
        )
        proj_card = self.make_card(
            "Output Projection",
            "Linear(32 → 32)\n混合 heads",
            color=BLUE,
            width=2.85,
            height=1.25,
            title_size=23,
            body_size=17,
        )
        logits_card = self.make_card(
            "Logits",
            "[B, T, vocab_size]\n预测下一个 token",
            color=YELLOW,
            width=2.85,
            height=1.25,
            title_size=23,
            body_size=17,
        )

        concat_card.move_to(LEFT * 2.0 + DOWN * 0.15)
        proj_card.move_to(RIGHT * 1.15 + DOWN * 0.15)
        logits_card.move_to(RIGHT * 4.35 + DOWN * 0.15)

        arrows_4 = VGroup(
            self.make_arrow(head_outputs.get_right(), concat_card.get_left()),
            self.make_arrow(concat_card.get_right(), proj_card.get_left()),
            self.make_arrow(proj_card.get_right(), logits_card.get_left()),
        )

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(head_outputs, shift=RIGHT))
        self.play(FadeIn(concat_card), Create(arrows_4[0]))
        self.play(FadeIn(proj_card), Create(arrows_4[1]))
        self.play(FadeIn(logits_card), Create(arrows_4[2]))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, head_outputs, concat_card, proj_card, logits_card, arrows_4)))

        # 总结页。
        summary = Text(
            "Multi-Head Attention = 多个并行注意力视角 + concat + 输出投影",
            font_size=32,
            color=YELLOW,
        )
        detail = Text(
            "它还不是完整 Transformer Block：还没有 residual、LayerNorm 和 FeedForward MLP",
            font_size=22,
            color=GRAY_B,
        )
        detail.next_to(summary, DOWN, buff=0.35)

        self.play(Write(summary))
        self.play(FadeIn(detail, shift=UP))
        self.wait(2.2)
