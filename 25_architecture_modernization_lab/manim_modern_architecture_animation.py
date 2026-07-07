from __future__ import annotations

from manim import *


class ModernArchitectureOverview(Scene):
    def make_box(self, text: str, color, width: float = 2.55, height: float = 0.55):
        box = RoundedRectangle(width=width, height=height, corner_radius=0.08)
        box.set_stroke(color, width=2)
        box.set_fill(color, opacity=0.12)
        label = Text(text, font_size=20)
        label.move_to(box.get_center())
        return VGroup(box, label)

    def vertical_stack(self, items, x: float):
        group = VGroup(*items).arrange(DOWN, buff=0.32)
        group.move_to([x, -0.15, 0])
        return group

    def connect_stack(self, group):
        arrows = VGroup()
        for i in range(len(group) - 1):
            arrows.add(
                Arrow(
                    group[i].get_bottom(),
                    group[i + 1].get_top(),
                    buff=0.05,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.12,
                )
            )
        return arrows

    def construct(self):
        title = Text("Tiny GPT Architecture Modernization", font_size=34)
        subtitle = Text("same shape: [batch, seq_len, n_embd]", font_size=22, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.12)
        header = VGroup(title, subtitle).to_edge(UP, buff=0.35)

        left_title = Text("Baseline", font_size=28, color=BLUE).move_to([-3.55, 2.55, 0])
        right_title = Text("Modern", font_size=28, color=ORANGE).move_to([3.55, 2.55, 0])

        baseline = self.vertical_stack(
            [
                self.make_box("Token IDs", GRAY, 2.6),
                self.make_box("Token Emb + Learned Pos", BLUE, 3.15),
                self.make_box("LayerNorm", TEAL, 2.6),
                self.make_box("MHA", PURPLE, 2.6),
                self.make_box("GELU FFN", GREEN, 2.6),
                self.make_box("Separate LM Head", RED, 3.0),
                self.make_box("Logits", YELLOW, 2.6),
            ],
            x=-3.55,
        )
        modern = self.vertical_stack(
            [
                self.make_box("Token IDs", GRAY, 2.6),
                self.make_box("Token Emb", BLUE, 2.6),
                self.make_box("RMSNorm", TEAL, 2.6),
                self.make_box("RoPE on Q / K", ORANGE, 2.9),
                self.make_box("SwiGLU FFN", GREEN, 2.75),
                self.make_box("Tied LM Head", RED, 2.8),
                self.make_box("Logits", YELLOW, 2.6),
            ],
            x=3.55,
        )

        baseline_arrows = self.connect_stack(baseline)
        modern_arrows = self.connect_stack(modern)

        divider = Line([0, 2.35, 0], [0, -3.2, 0], color=GRAY_D, stroke_width=2)

        note1 = Text("RoPE injects position into Q/K, not token embedding", font_size=20, color=ORANGE)
        note2 = Text("Weight tying: logits = hidden @ token_embedding.T", font_size=20, color=RED)
        notes = VGroup(note1, note2).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.28)

        self.play(FadeIn(header))
        self.play(FadeIn(left_title), FadeIn(right_title), Create(divider))
        self.play(LaggedStart(*[FadeIn(item, shift=DOWN * 0.1) for item in baseline], lag_ratio=0.08))
        self.play(Create(baseline_arrows))
        self.play(LaggedStart(*[FadeIn(item, shift=DOWN * 0.1) for item in modern], lag_ratio=0.08))
        self.play(Create(modern_arrows))
        self.play(FadeIn(notes))

        highlight_rope = SurroundingRectangle(modern[3], color=ORANGE, buff=0.08)
        highlight_tying = SurroundingRectangle(modern[5], color=RED, buff=0.08)
        self.play(Create(highlight_rope))
        self.wait(0.6)
        self.play(ReplacementTransform(highlight_rope, highlight_tying))
        self.wait(0.8)

        conclusion = Text(
            "Modern components keep the tensor shape compatible,\n"
            "but change position handling, normalization, FFN, and parameter sharing.",
            font_size=22,
            line_spacing=0.95,
        )
        conclusion.move_to(ORIGIN)
        backdrop = Rectangle(width=11.5, height=2.0, color=BLACK, fill_opacity=0.88, stroke_opacity=0)
        backdrop.move_to(conclusion)
        self.play(FadeIn(backdrop), FadeIn(conclusion))
        self.wait(1.5)
