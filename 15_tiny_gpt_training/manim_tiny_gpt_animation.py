from manim import *


config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720


class TinyGPTOverview(Scene):
    """用分屏式动画解释 Tiny GPT 的整体数据流，避免流程图互相重叠。"""

    def make_card(
        self,
        title,
        body,
        color=BLUE,
        width=2.45,
        height=1.12,
        title_size=22,
        body_size=16,
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
        group = VGroup(title_text, body_text).arrange(DOWN, buff=0.1)
        group.move_to(box.get_center())
        return VGroup(box, group)

    def make_arrow(self, start, end, color=GRAY_A):
        return Arrow(
            start=start,
            end=end,
            buff=0.15,
            stroke_width=4,
            color=color,
            max_tip_length_to_length_ratio=0.14,
        )

    def title_group(self, text, subtitle=None):
        title = Text(text, font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.32)
        if subtitle is None:
            return VGroup(title)
        sub = Text(subtitle, font_size=21, color=GRAY_B)
        sub.next_to(title, DOWN, buff=0.15)
        return VGroup(title, sub)

    def construct(self):
        self.camera.background_color = "#080B10"

        # 第一幕：Tiny GPT 总览。
        title = self.title_group(
            "Tiny GPT 总览",
            "把 token、embedding、多个 Transformer Blocks 和 LM Head 串起来",
        )

        text_card = self.make_card("文本", '"hello ai"', color=GREEN, width=1.85)
        ids_card = self.make_card("Token IDs", "[B, S]", color=TEAL, width=1.95)
        emb_card = self.make_card(
            "Embedding",
            "token + position\n[B, S, C]",
            color=BLUE,
            width=2.35,
            height=1.22,
        )
        blocks_card = self.make_card(
            "Blocks × N",
            "Block 1\nBlock 2",
            color=PURPLE,
            width=2.2,
            height=1.22,
        )
        head_card = self.make_card(
            "LN + LM Head",
            "[B, S, vocab]",
            color=YELLOW,
            width=2.25,
        )
        sample_card = self.make_card(
            "Sample",
            "取最后位置\n生成下个 token",
            color=ORANGE,
            width=2.25,
            height=1.22,
        )

        row = VGroup(
            text_card,
            ids_card,
            emb_card,
            blocks_card,
            head_card,
            sample_card,
        ).arrange(RIGHT, buff=0.22)
        row.move_to(DOWN * 0.1)

        arrows = VGroup(
            self.make_arrow(text_card.get_right(), ids_card.get_left()),
            self.make_arrow(ids_card.get_right(), emb_card.get_left()),
            self.make_arrow(emb_card.get_right(), blocks_card.get_left()),
            self.make_arrow(blocks_card.get_right(), head_card.get_left()),
            self.make_arrow(head_card.get_right(), sample_card.get_left()),
        )

        note = Text(
            "训练：所有位置一起预测下一个 token；生成：只用最后一个位置继续往后采样",
            font_size=22,
            color=GRAY_B,
        )
        note.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP), Create(arrows))
        self.play(FadeIn(note, shift=UP))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, row, arrows, note)))

        # 第二幕：多个 Transformer Blocks 堆叠。
        title = self.title_group(
            "多个 Transformer Blocks 堆叠",
            "每一层都有自己的 Attention、LayerNorm、Residual 和 MLP 参数",
        )

        inp = self.make_card("输入 x", "[B, S, C]", color=GREEN, width=2.1)
        b1 = self.make_card(
            "Block 1",
            "MHA + MLP\nshape 不变",
            color=PURPLE,
            width=2.55,
            height=1.25,
        )
        b2 = self.make_card(
            "Block 2",
            "新的参数\n继续加工表示",
            color=PURPLE,
            width=2.55,
            height=1.25,
        )
        out = self.make_card("输出", "[B, S, C]", color=GREEN, width=2.1)

        row = VGroup(inp, b1, b2, out).arrange(RIGHT, buff=0.55)
        row.move_to(DOWN * 0.05)

        arrows = VGroup(
            self.make_arrow(inp.get_right(), b1.get_left()),
            self.make_arrow(b1.get_right(), b2.get_left()),
            self.make_arrow(b2.get_right(), out.get_left()),
        )

        shape_note = Text(
            "Block 改变的是表示内容，不改变 [B, S, C] 这个外形",
            font_size=25,
            color=YELLOW,
        )
        shape_note.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP), Create(arrows))
        self.play(Circumscribe(b1, color=PURPLE), Circumscribe(b2, color=PURPLE))
        self.play(FadeIn(shape_note, shift=UP))
        self.wait(2.1)
        self.play(FadeOut(VGroup(title, row, arrows, shape_note)))

        # 第三幕：一个 Block 内部是什么。
        title = self.title_group(
            "一个 Transformer Block 内部",
            "两条主分支：Attention 读取上下文，MLP 增强每个 token 的表示",
        )

        x1 = self.make_card("x", "[B, S, C]", color=GREEN, width=1.7)
        ln1 = self.make_card("LayerNorm", "稳定输入", color=BLUE, width=2.0)
        mha = self.make_card("MHA", "读取前文", color=PURPLE, width=1.8)
        add1 = self.make_card("Residual", "x + attn", color=ORANGE, width=2.0)

        top = VGroup(x1, ln1, mha, add1).arrange(RIGHT, buff=0.55)
        top.move_to(UP * 0.8)

        x2 = self.make_card("x", "[B, S, C]", color=GREEN, width=1.7)
        ln2 = self.make_card("LayerNorm", "稳定输入", color=BLUE, width=2.0)
        mlp = self.make_card("FeedForward", "C → 4C → C", color=TEAL, width=2.35)
        add2 = self.make_card("Residual", "x + mlp", color=ORANGE, width=2.0)

        bottom = VGroup(x2, ln2, mlp, add2).arrange(RIGHT, buff=0.45)
        bottom.move_to(DOWN * 1.15)

        arrows = VGroup(
            self.make_arrow(x1.get_right(), ln1.get_left()),
            self.make_arrow(ln1.get_right(), mha.get_left()),
            self.make_arrow(mha.get_right(), add1.get_left()),
            self.make_arrow(x2.get_right(), ln2.get_left()),
            self.make_arrow(ln2.get_right(), mlp.get_left()),
            self.make_arrow(mlp.get_right(), add2.get_left()),
        )

        res1 = CurvedArrow(
            x1.get_top(),
            add1.get_top(),
            angle=-TAU / 4,
            color=YELLOW,
            stroke_width=4,
        )
        res2 = CurvedArrow(
            x2.get_bottom(),
            add2.get_bottom(),
            angle=TAU / 4,
            color=YELLOW,
            stroke_width=4,
        )

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(top), FadeIn(bottom), Create(arrows))
        self.play(Create(res1), Create(res2))
        self.wait(2.3)
        self.play(FadeOut(VGroup(title, top, bottom, arrows, res1, res2)))

        # 第四幕：LM Head 和生成。
        title = self.title_group(
            "从 hidden state 到下一个 token",
            "LM Head 把每个位置的隐藏向量映射成 vocab logits",
        )

        hidden = self.make_card(
            "hidden",
            "[B, S, C]",
            color=GREEN,
            width=2.2,
            height=1.1,
        )
        head = self.make_card(
            "LM Head",
            "C → vocab",
            color=YELLOW,
            width=2.15,
            height=1.1,
        )
        logits = self.make_card(
            "logits",
            "[B, S, vocab]",
            color=ORANGE,
            width=2.55,
            height=1.1,
        )
        last = self.make_card(
            "最后位置",
            "[vocab]",
            color=RED,
            width=2.05,
            height=1.1,
        )
        next_token = self.make_card(
            "sample",
            "next token",
            color=TEAL,
            width=2.1,
            height=1.1,
        )

        row = VGroup(hidden, head, logits, last, next_token).arrange(RIGHT, buff=0.38)
        row.move_to(UP * 0.35)

        arrows = VGroup(
            self.make_arrow(hidden.get_right(), head.get_left()),
            self.make_arrow(head.get_right(), logits.get_left()),
            self.make_arrow(logits.get_right(), last.get_left(), color=RED),
            self.make_arrow(last.get_right(), next_token.get_left()),
        )

        train_text = Text(
            "训练时：[B×S] 个位置都参与 cross entropy",
            font_size=23,
            color=GRAY_B,
        )
        gen_text = Text(
            "生成时：只取当前上下文的最后一个位置，采样出下一个 token",
            font_size=23,
            color=YELLOW,
        )
        notes = VGroup(train_text, gen_text).arrange(DOWN, buff=0.22)
        notes.to_edge(DOWN, buff=0.65)

        self.play(FadeIn(title, shift=DOWN))
        self.play(FadeIn(row, shift=UP), Create(arrows))
        self.play(FadeIn(notes, shift=UP))
        self.play(Circumscribe(last, color=RED))
        self.wait(2.4)
        self.play(FadeOut(VGroup(title, row, arrows, notes)))

        summary = Text(
            "Tiny GPT = 字符 token + 位置 + 多层 Transformer Blocks + 下一个 token 预测",
            font_size=29,
            color=YELLOW,
        )
        detail = Text(
            "这已经是 GPT 的最小完整雏形，只是数据和模型都很小",
            font_size=22,
            color=GRAY_B,
        )
        detail.next_to(summary, DOWN, buff=0.32)

        self.play(Write(summary))
        self.play(FadeIn(detail, shift=UP))
        self.wait(2.3)
