from manim import *


config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720


class TransformerBlockExplained(Scene):
    """用分屏式动画解释一个 Transformer Block，避免流程图堆叠过密。"""

    def make_card(
        self,
        title,
        body,
        color=BLUE,
        width=3.0,
        height=1.15,
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
            max_tip_length_to_length_ratio=0.15,
        )

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

        # 第一幕：整体结构。
        title = self.page_title(
            "Transformer Block 基础",
            "把 Multi-Head Attention 加上 Residual、LayerNorm 和 FeedForward MLP",
        )

        input_card = self.make_card("输入 x", "[B, S, n_embd]", color=GREEN, width=2.45)
        attn_part = self.make_card(
            "Attention 子层",
            "LayerNorm\nMulti-Head Attention\nResidual Add",
            color=BLUE,
            width=3.05,
            height=1.55,
            body_size=17,
        )
        mlp_part = self.make_card(
            "MLP 子层",
            "LayerNorm\nFeedForward MLP\nResidual Add",
            color=TEAL,
            width=3.05,
            height=1.55,
            body_size=17,
        )
        output_card = self.make_card("输出 out", "[B, S, n_embd]", color=GREEN, width=2.45)

        flow = VGroup(input_card, attn_part, mlp_part, output_card).arrange(
            RIGHT,
            buff=0.36,
        )
        flow.move_to(DOWN * 0.15)

        arrows = VGroup(
            self.make_arrow(input_card.get_right(), attn_part.get_left()),
            self.make_arrow(attn_part.get_right(), mlp_part.get_left()),
            self.make_arrow(mlp_part.get_right(), output_card.get_left()),
        )

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(flow, shift=UP))
        self.play(Create(arrows))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, flow, arrows)))

        # 第二幕：Attention 子层的 Pre-LN 和 residual。
        title = self.page_title(
            "第一段：Attention + Residual",
            "Pre-LN 形式：x = x + attention(layer_norm(x))",
        )

        x_card = self.make_card("x", "[B, S, C]", color=GREEN, width=2.0)
        ln_card = self.make_card("LayerNorm 1", "按最后一维 C 归一化", color=PURPLE)
        attn_card = self.make_card(
            "Multi-Head Attention",
            "读取前文上下文\n输出 [B, S, C]",
            color=BLUE,
            width=3.25,
        )
        add_card = self.make_card(
            "Residual Add",
            "x + attn_out\n保留原信息",
            color=ORANGE,
            width=3.0,
        )

        row = VGroup(x_card, ln_card, attn_card, add_card).arrange(RIGHT, buff=0.45)
        row.move_to(DOWN * 0.25)

        arrows = VGroup(
            self.make_arrow(x_card.get_right(), ln_card.get_left()),
            self.make_arrow(ln_card.get_right(), attn_card.get_left()),
            self.make_arrow(attn_card.get_right(), add_card.get_left()),
        )

        residual = CurvedArrow(
            x_card.get_top(),
            add_card.get_top(),
            angle=-TAU / 4,
            color=YELLOW,
            stroke_width=5,
        )
        res_label = Text("残差路径：原始 x 直接加回来", font_size=22, color=YELLOW)
        res_label.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP), Create(arrows))
        self.play(Create(residual), FadeIn(res_label))
        self.wait(2.2)
        self.play(FadeOut(VGroup(title, row, arrows, residual, res_label)))

        # 第三幕：FeedForward 子层。
        title = self.page_title(
            "第二段：FeedForward MLP + Residual",
            "x = x + feed_forward(layer_norm(x))",
        )

        x_card = self.make_card(
            "x",
            "[B, S, C]",
            color=GREEN,
            width=1.75,
            title_size=22,
            body_size=16,
        )
        ln_card = self.make_card(
            "LayerNorm 2",
            "稳定输入",
            color=PURPLE,
            width=2.05,
            title_size=21,
            body_size=16,
        )
        expand_card = self.make_card(
            "Linear",
            "C → 4C",
            color=TEAL,
            width=1.75,
            title_size=21,
            body_size=16,
        )
        gelu_card = self.make_card(
            "GELU",
            "非线性",
            color=TEAL,
            width=1.65,
            title_size=21,
            body_size=16,
        )
        shrink_card = self.make_card(
            "Linear",
            "4C → C",
            color=TEAL,
            width=1.75,
            title_size=21,
            body_size=16,
        )
        add_card = self.make_card(
            "Residual",
            "x + mlp",
            color=ORANGE,
            width=2.0,
            title_size=21,
            body_size=16,
        )

        row = VGroup(
            x_card,
            ln_card,
            expand_card,
            gelu_card,
            shrink_card,
            add_card,
        ).arrange(RIGHT, buff=0.28)
        row.move_to(DOWN * 0.25)

        arrows = VGroup(
            self.make_arrow(x_card.get_right(), ln_card.get_left()),
            self.make_arrow(ln_card.get_right(), expand_card.get_left()),
            self.make_arrow(expand_card.get_right(), gelu_card.get_left()),
            self.make_arrow(gelu_card.get_right(), shrink_card.get_left()),
            self.make_arrow(shrink_card.get_right(), add_card.get_left()),
        )

        residual = CurvedArrow(
            x_card.get_bottom(),
            add_card.get_bottom(),
            angle=TAU / 4,
            color=YELLOW,
            stroke_width=5,
        )
        res_label = Text("第二条残差：在原表示上叠加 MLP 变换", font_size=22, color=YELLOW)
        res_label.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP), Create(arrows))
        self.play(Create(residual), FadeIn(res_label))
        self.wait(2.2)
        self.play(FadeOut(VGroup(title, row, arrows, residual, res_label)))

        # 第四幕：为什么 shape 保持一致。
        title = self.page_title(
            "为什么输入输出 Shape 保持一致？",
            "只有 [B, S, n_embd] 不变，多个 Block 才能一层一层堆叠",
        )

        in_card = self.make_card("输入", "[B, S, 32]", color=GREEN, width=2.35)
        block_card = self.make_card(
            "Transformer Block",
            "内部有 attention / residual\nLayerNorm / MLP",
            color=BLUE,
            width=3.25,
            height=1.4,
            title_size=23,
            body_size=17,
        )
        out_card = self.make_card("输出", "[B, S, 32]", color=GREEN, width=2.35)
        logits_card = self.make_card(
            "LM Head",
            "[B, S, vocab_size]\n预测下一个 token",
            color=YELLOW,
            width=2.85,
            height=1.25,
            title_size=23,
            body_size=17,
        )

        row = VGroup(in_card, block_card, out_card, logits_card).arrange(
            RIGHT,
            buff=0.42,
        )
        row.move_to(DOWN * 0.1)

        arrows = VGroup(
            self.make_arrow(in_card.get_right(), block_card.get_left()),
            self.make_arrow(block_card.get_right(), out_card.get_left()),
            self.make_arrow(out_card.get_right(), logits_card.get_left()),
        )

        note = Text(
            "Block 改变的是表示内容，不改变张量外形",
            font_size=26,
            color=YELLOW,
        )
        note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP), Create(arrows))
        self.play(FadeIn(note, shift=UP))
        self.play(Circumscribe(in_card, color=GREEN), Circumscribe(out_card, color=GREEN))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, row, arrows, note)))

        summary = Text(
            "Transformer Block = 上下文读取 + 信息保留 + 训练稳定 + 表达增强",
            font_size=32,
            color=YELLOW,
        )
        detail = Text(
            "本节只有一个 Block；下一步才会堆叠成 Tiny GPT",
            font_size=22,
            color=GRAY_B,
        )
        detail.next_to(summary, DOWN, buff=0.35)

        self.play(Write(summary))
        self.play(FadeIn(detail, shift=UP))
        self.wait(2.3)
