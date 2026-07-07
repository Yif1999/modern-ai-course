from manim import *


class SelfAttentionExplained(Scene):
    """用 Manim 展示 single-head causal self-attention 的核心数据流。"""

    def matrix_card(self, title, shape, color, width=3.0, height=1.1):
        """创建一个带标题、shape 和颜色边框的矩阵卡片。"""
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.12,
            color=color,
            stroke_width=4,
        )
        label = Text(title, font_size=24, weight=BOLD).move_to(box.get_center() + UP * 0.18)
        shape_text = Text(shape, font_size=19, color=GRAY_A).move_to(box.get_center() + DOWN * 0.23)
        return VGroup(box, label, shape_text)

    def token_row(self):
        """展示一个极小字符序列 hello ai。"""
        tokens = ["h", "e", "l", "l", "o", "space", "a", "i"]
        row = VGroup()
        for token in tokens:
            square = RoundedRectangle(
                width=0.78,
                height=0.62,
                corner_radius=0.08,
                color=WHITE,
                stroke_width=2,
            )
            label = Text(token, font_size=22 if token != "space" else 16)
            cell = VGroup(square, label)
            row.add(cell)
        row.arrange(RIGHT, buff=0.12)
        return row

    def causal_mask_grid(self):
        """画出下三角 causal mask：当前位置只能看自己和过去。"""
        n = 8
        grid = VGroup()
        for i in range(n):
            for j in range(n):
                allowed = j <= i
                color = GREEN_C if allowed else RED_E
                fill_opacity = 0.75 if allowed else 0.22
                cell = Square(side_length=0.28)
                cell.set_stroke(WHITE, width=0.5)
                cell.set_fill(color, opacity=fill_opacity)
                cell.move_to(RIGHT * j * 0.3 + DOWN * i * 0.3)
                grid.add(cell)
        grid.move_to(ORIGIN)
        border = SurroundingRectangle(grid, color=WHITE, buff=0.05)
        title = Text("causal mask", font_size=22, color=RED_B).next_to(border, UP, buff=0.15)
        subtitle = Text("lower triangle = visible", font_size=15, color=GRAY_A).next_to(border, DOWN, buff=0.12)
        return VGroup(grid, border, title, subtitle)

    def attention_heatmap(self):
        """画一个简化 attention weights 热力图，强调每行只分配给过去位置。"""
        n = 8
        heatmap = VGroup()
        for i in range(n):
            for j in range(n):
                visible = j <= i
                opacity = 0.18 + 0.65 * (j + 1) / (i + 1) if visible else 0.03
                cell = Square(side_length=0.28)
                cell.set_stroke(WHITE, width=0.5)
                cell.set_fill(YELLOW_C if visible else DARK_GRAY, opacity=opacity)
                cell.move_to(RIGHT * j * 0.3 + DOWN * i * 0.3)
                heatmap.add(cell)
        heatmap.move_to(ORIGIN)
        border = SurroundingRectangle(heatmap, color=YELLOW_C, buff=0.05)
        title = Text("attention weights", font_size=22, color=YELLOW_C).next_to(border, UP, buff=0.15)
        subtitle = Text("each row sums to 1", font_size=15, color=GRAY_A).next_to(border, DOWN, buff=0.12)
        return VGroup(heatmap, border, title, subtitle)

    def construct(self):
        self.camera.background_color = "#10131a"

        # 0. 标题
        title = Text("Single-Head Causal Self-Attention", font_size=38, weight=BOLD)
        subtitle = Text("one tiny sequence: hello ai", font_size=22, color=GRAY_A)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)
        self.play(VGroup(title, subtitle).animate.to_edge(UP, buff=0.35))

        # 1. token 序列
        tokens = self.token_row().move_to(UP * 1.3)
        token_caption = Text("tokens: [batch=1, seq_len=8]", font_size=22, color=GRAY_A)
        token_caption.next_to(tokens, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(cell, shift=UP * 0.2) for cell in tokens], lag_ratio=0.08))
        self.play(Write(token_caption))
        self.wait(0.5)

        # 2. token embedding + position embedding
        token_emb = self.matrix_card("token embedding", "[1, 8, 8]", TEAL_C)
        pos_emb = self.matrix_card("position embedding", "[8, 8]", BLUE_C)
        x_card = self.matrix_card("x = token + position", "[1, 8, 8]", PURPLE_C, width=3.4)
        plus = Text("+", font_size=34, color=WHITE)
        arrow_to_x = Arrow(RIGHT, RIGHT * 1.4, buff=0.1, color=WHITE)

        embed_group = VGroup(token_emb, plus, pos_emb, arrow_to_x, x_card)
        embed_group.arrange(RIGHT, buff=0.35)
        embed_group.move_to(DOWN * 0.45)
        self.play(
            ReplacementTransform(tokens.copy(), token_emb),
            FadeIn(pos_emb, shift=UP * 0.2),
            FadeIn(plus),
        )
        self.play(GrowArrow(arrow_to_x), FadeIn(x_card, shift=RIGHT * 0.2))
        self.wait(0.7)

        # 3. Q/K/V 投影
        self.play(FadeOut(token_caption), FadeOut(embed_group), tokens.animate.to_edge(UP, buff=1.35))

        x_left = self.matrix_card("x", "[1, 8, 8]", PURPLE_C).move_to(LEFT * 4.8 + UP * 0.2)
        q_card = self.matrix_card("Q = xWq", "[1, 8, 8]", ORANGE)
        k_card = self.matrix_card("K = xWk", "[1, 8, 8]", GOLD)
        v_card = self.matrix_card("V = xWv", "[1, 8, 8]", GREEN_C)
        qkv = VGroup(q_card, k_card, v_card).arrange(DOWN, buff=0.35).move_to(RIGHT * 1.15 + UP * 0.2)
        arrows = VGroup(*[
            Arrow(x_left.get_right(), card.get_left(), buff=0.12, color=GRAY_A)
            for card in qkv
        ])
        qkv_caption = Text(
            "Project each token into Query, Key, and Value",
            font_size=22,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(x_left), Write(qkv_caption))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.18))
        self.play(LaggedStart(*[FadeIn(card, shift=RIGHT * 0.25) for card in qkv], lag_ratio=0.16))
        self.wait(0.8)

        # 4. QK^T / sqrt(d) 得到 attention scores
        self.play(FadeOut(qkv_caption))
        formula = Text("scores = QK^T / sqrt(d)", font_size=21, weight=BOLD, color=WHITE)
        scores = self.matrix_card("attention scores", "[1, 8, 8]", RED_C, width=2.9, height=1.0)
        score_group = VGroup(scores, formula).arrange(DOWN, buff=0.2).move_to(RIGHT * 4.65 + DOWN * 0.05)
        q_to_score = Arrow(q_card.get_right(), scores.get_left(), buff=0.1, color=ORANGE)
        k_to_score = Arrow(k_card.get_right(), scores.get_left(), buff=0.1, color=GOLD)
        score_caption = Text(
            "Every row asks: which previous tokens should I look at?",
            font_size=18,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.85)

        self.play(FadeIn(formula, shift=UP * 0.2), GrowArrow(q_to_score), GrowArrow(k_to_score))
        self.play(FadeIn(scores, shift=RIGHT * 0.2), Write(score_caption))
        self.wait(0.8)

        # 5. causal mask 屏蔽未来
        self.play(
            FadeOut(x_left),
            FadeOut(q_card),
            FadeOut(k_card),
            FadeOut(v_card),
            FadeOut(arrows),
            FadeOut(q_to_score),
            FadeOut(k_to_score),
            FadeOut(formula),
            FadeOut(score_caption),
            scores.animate.move_to(LEFT * 3.7),
        )
        mask = self.causal_mask_grid().scale(0.95).move_to(RIGHT * 0.4)
        masked_scores = self.matrix_card("masked scores", "[1, 8, 8]", RED_B, width=3.2).move_to(RIGHT * 4.3)
        mask_arrow_1 = Arrow(scores.get_right(), mask.get_left(), buff=0.2, color=GRAY_A)
        mask_arrow_2 = Arrow(mask.get_right(), masked_scores.get_left(), buff=0.2, color=GRAY_A)
        mask_caption = Text("Future positions become -infinity before softmax", font_size=21, color=GRAY_A)
        mask_caption.to_edge(DOWN, buff=0.85)

        self.play(GrowArrow(mask_arrow_1), FadeIn(mask))
        self.play(GrowArrow(mask_arrow_2), FadeIn(masked_scores, shift=RIGHT * 0.2), Write(mask_caption))
        self.wait(0.8)

        # 6. softmax 得到 attention weights
        self.play(FadeOut(scores), FadeOut(mask), FadeOut(mask_arrow_1), FadeOut(mask_arrow_2), FadeOut(mask_caption))
        masked_scores.move_to(LEFT * 3.7)
        softmax = self.matrix_card("softmax", "axis=-1", YELLOW_C, width=2.3).move_to(ORIGIN)
        weights = self.attention_heatmap().scale(0.95).move_to(RIGHT * 3.6)
        soft_arrow_1 = Arrow(masked_scores.get_right(), softmax.get_left(), buff=0.15, color=GRAY_A)
        soft_arrow_2 = Arrow(softmax.get_right(), weights.get_left(), buff=0.15, color=GRAY_A)
        weight_caption = Text("attention weights say: look where, and how much", font_size=21, color=GRAY_A)
        weight_caption.to_edge(DOWN, buff=0.85)

        self.play(FadeIn(masked_scores), GrowArrow(soft_arrow_1), FadeIn(softmax))
        self.play(GrowArrow(soft_arrow_2), FadeIn(weights), Write(weight_caption))
        self.wait(0.9)

        # 7. attention weights @ V 加权求和
        self.play(FadeOut(masked_scores), FadeOut(softmax), FadeOut(soft_arrow_1), FadeOut(soft_arrow_2), FadeOut(weight_caption))
        weights.move_to(LEFT * 3.6)
        v_again = self.matrix_card("V", "[1, 8, 8]", GREEN_C).move_to(ORIGIN)
        output = self.matrix_card("output = weights @ V", "[1, 8, 8]", PINK, width=3.6).move_to(RIGHT * 3.9)
        weighted_arrow_1 = Arrow(weights.get_right(), v_again.get_left(), buff=0.15, color=YELLOW_C)
        weighted_arrow_2 = Arrow(v_again.get_right(), output.get_left(), buff=0.15, color=GREEN_C)
        output_caption = Text("Weighted sum mixes information from visible tokens", font_size=21, color=GRAY_A)
        output_caption.to_edge(DOWN, buff=0.85)

        self.play(GrowArrow(weighted_arrow_1), FadeIn(v_again))
        self.play(GrowArrow(weighted_arrow_2), FadeIn(output, shift=RIGHT * 0.2), Write(output_caption))
        self.wait(0.9)

        # 8. 最终 logits
        logits = self.matrix_card("logits", "[1, 8, vocab_size]", BLUE_E, width=3.8).move_to(RIGHT * 3.8)
        lm_head = self.matrix_card("lm head", "Linear", BLUE_B, width=2.2).move_to(ORIGIN)
        self.play(FadeOut(weights), FadeOut(v_again), FadeOut(weighted_arrow_1), FadeOut(weighted_arrow_2), FadeOut(output_caption))
        output.move_to(LEFT * 3.7)
        final_arrow_1 = Arrow(output.get_right(), lm_head.get_left(), buff=0.15, color=PINK)
        final_arrow_2 = Arrow(lm_head.get_right(), logits.get_left(), buff=0.15, color=BLUE_B)
        final_caption = Text(
            "Each position predicts the next token from the vocabulary",
            font_size=22,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.85)

        self.play(GrowArrow(final_arrow_1), FadeIn(lm_head))
        self.play(GrowArrow(final_arrow_2), FadeIn(logits, shift=RIGHT * 0.2), Write(final_caption))
        self.wait(1.0)

        # 9. 总结公式
        summary = Text("Attention(Q, K, V) = softmax(mask(QK^T / sqrt(d))) V", font_size=28, weight=BOLD)
        summary_bg = BackgroundRectangle(summary, color=BLACK, fill_opacity=0.75, buff=0.22)
        summary_group = VGroup(summary_bg, summary).to_edge(DOWN, buff=0.42)
        self.play(FadeOut(final_caption), FadeIn(summary_group, shift=UP * 0.15))
        self.wait(1.6)
