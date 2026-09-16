#!/usr/bin/env python
"""Generate a short, plain-English PDF that explains the V2 features and why they are worth adding.

Written for a non-technical audience (pitch / stakeholder) rather than engineers.

Usage:
    python scripts/make_pitch_pdf.py [output_path]

Default output: ~/Downloads/Cognitive Load Predictor V2 - Feature Pitch.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#1F4E79")
ACCENT_DARK = colors.HexColor("#13314D")
TINT = colors.HexColor("#EAF1F8")
TINT_WARM = colors.HexColor("#FDF3E3")
MUTED = colors.HexColor("#5A5A5A")
LINE = colors.HexColor("#C9D6E2")

BODY = ParagraphStyle(
    "Body", fontName="Helvetica", fontSize=9.6, leading=13.4, textColor=colors.HexColor("#1A1A1A")
)
BODY_TIGHT = ParagraphStyle("BodyTight", parent=BODY, spaceBefore=1.5)
TITLE = ParagraphStyle(
    "Title", fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=ACCENT_DARK, spaceAfter=2
)
SUBTITLE = ParagraphStyle(
    "Subtitle", fontName="Helvetica", fontSize=11.5, leading=15, textColor=MUTED, spaceAfter=6
)
H1 = ParagraphStyle(
    "H1", fontName="Helvetica-Bold", fontSize=13.5, leading=16, textColor=ACCENT,
    spaceBefore=11, spaceAfter=4,
)
H2 = ParagraphStyle(
    "H2", fontName="Helvetica-Bold", fontSize=10.6, leading=13.4, textColor=ACCENT_DARK,
    spaceBefore=0, spaceAfter=1.5,
)
CALL = ParagraphStyle("Call", parent=BODY, fontSize=10.2, leading=14.2, textColor=ACCENT_DARK)
NOTE = ParagraphStyle("Note", parent=BODY, fontName="Helvetica-Oblique", fontSize=9, leading=12.4, textColor=MUTED)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=10, bulletIndent=1, spaceBefore=1.5)


def bullet(text: str) -> Paragraph:
    return Paragraph(text, BULLET, bulletText="\u2022")


def box(paragraphs, bg=TINT, border=ACCENT):
    """A tinted callout box."""
    table = Table([[paragraphs]], colWidths=[168 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.6, border),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def feature(number: int, name: str, what: str, why: str):
    return KeepTogether(
        [
            Paragraph(f"{number}. {name}", H2),
            Paragraph(f"<b>What it is:</b> {what}", BODY_TIGHT),
            Paragraph(f"<b>Why it matters:</b> {why}", BODY_TIGHT),
            Spacer(1, 5),
        ]
    )


def build_story():
    s = []

    # ---- Cover header ------------------------------------------------
    s.append(Paragraph("Cognitive Load Predictor", TITLE))
    s.append(Paragraph("Version 2 &mdash; new features, explained simply", SUBTITLE))
    s.append(HRFlowable(width="100%", thickness=0.8, color=LINE, spaceAfter=8))
    s.append(
        box(
            [
                Paragraph(
                    "The one-line version: today the tool tells you <b>how hard a session felt</b>. "
                    "Version 2 also tells you <b>how sure it is, why it says so, what to do about it, "
                    "whether the data can be trusted, and what is likely to happen next</b>.",
                    CALL,
                )
            ]
        )
    )
    s.append(Spacer(1, 9))

    s.append(Paragraph("What the tool is", H1))
    s.append(
        Paragraph(
            "It is a short computer-based assessment that estimates a person's mental workload. "
            "The person completes four quick exercises: reacting to a signal (reaction time), switching "
            "between two rules (task switching), remembering a short sequence of items (working memory), "
            "and naming the ink colour of a word instead of reading it (Stroop test).",
            BODY,
        )
    )
    s.append(
        Paragraph(
            "Those results are turned into a single <b>cognitive load score from 0 to 100</b> and a plain-language "
            "band: <b>Low, Manageable, Elevated, or High</b>. Every session is saved, so results can be compared "
            "over time. Today it is a working research prototype used locally &mdash; no cloud, no account needed.",
            BODY,
        )
    )

    s.append(Paragraph("The three gaps we are closing", H1))
    s.append(
        bullet(
            "<b>It sounds equally sure of everything.</b> A result it is confident about and a result that is "
            "close to a coin-flip look exactly the same on screen."
        )
    )
    s.append(
        bullet(
            "<b>It gives a number, not a reason.</b> A person sees &quot;Elevated&quot; and has no idea which "
            "exercise drove it, so they cannot act on it."
        )
    )
    s.append(
        bullet(
            "<b>It only knows about right now.</b> Each session is judged in isolation &mdash; there is no sense "
            "of what is normal for this person, whether things are improving, and no warning when a session's "
            "data looks unreliable."
        )
    )
    s.append(
        Paragraph(
            "Version 2 keeps everything that already works and adds nine focused pieces that close those gaps. "
            "Nothing is thrown away or rewritten from scratch.",
            NOTE,
        )
    )

    # ---- Features ----------------------------------------------------
    s.append(Paragraph("The nine new features", H1))

    s.append(
        feature(
            1,
            "It remembers your history",
            "Each new session is compared with that same person's previous sessions &mdash; what changed, "
            "whether it is drifting up or down, and what their recent average looks like. Only earlier sessions "
            "are ever used, so the tool never accidentally 'knows the future'.",
            "A score of 60 means something different for someone who usually sits at 40 than for someone who "
            "usually sits at 75. Comparing people to their own normal is far more useful than comparing them to "
            "a generic number.",
        )
    )
    s.append(
        feature(
            2,
            "A stronger prediction engine",
            "The single simple classifier used today becomes one option among several standard, well-understood "
            "models. They are all trained and tested on the same data, and the best performer is kept &mdash; with "
            "the old model retained as a benchmark to prove the new one is actually better.",
            "Better accuracy, and evidence that it is better. Keeping the old model as a baseline means any "
            "improvement claim is measured, not assumed.",
        )
    )
    s.append(
        feature(
            3,
            "Confidence, shown honestly",
            "Every prediction now carries a confidence level: how strongly the model favours one band over the "
            "others, how much history it had to work with, and how consistent the recent sessions are.",
            "A 55-to-45 result is a very different message from a 95-to-5 result. Saying so openly is what makes "
            "the confident results believable.",
        )
    )
    s.append(
        feature(
            4,
            "Plain-English explanations",
            "The tool shows which measured behaviours pushed the result up or down, described in ordinary "
            "language (for example, responses slowed down but accuracy held up). It only reports contributors it "
            "can actually verify, and admits when it cannot pin the result on anything specific.",
            "This turns a black-box number into something a person can understand and act on &mdash; and it "
            "protects credibility, because a wrong 'reason' is worse than no reason.",
        )
    )
    s.append(
        feature(
            5,
            "Bad-data warnings",
            "Sessions are checked for signs they cannot be trusted: impossibly fast reactions, answers given too "
            "quickly to be genuine, and repetitive or contradictory patterns. Anything suspicious is flagged and "
            "each session is given a quality rating.",
            "Bad data produces confident wrong answers. Flagging instead of deleting keeps the record transparent "
            "and protects the model from being pulled off course by a handful of meaningless sessions.",
        )
    )
    s.append(
        feature(
            6,
            "A look ahead",
            "Based on the trend, the tool estimates how the next session is likely to go &mdash; better, worse, or "
            "about the same &mdash; and gives it as a range rather than a single number. If there is not enough "
            "history, it clearly says so instead of guessing.",
            "It turns a measurement into an early-warning system. Knowing a difficult period is likely is useful "
            "before it arrives, not after.",
        )
    )
    s.append(
        feature(
            7,
            "It tells you what to do next",
            "The current state, the reasons behind it, and the trend are mapped to a short list of concrete "
            "suggested actions, each labelled with the observation that triggered it.",
            "People rarely want a number on its own &mdash; they want to know what to do about it. Showing the "
            "reason behind each suggestion also makes the advice easy to judge and easy to ignore when it does "
            "not apply.",
        )
    )
    s.append(
        feature(
            8,
            "Fair comparison between tasks",
            "The tool records how demanding each version of an exercise is and adjusts the score accordingly, so "
            "a harder run is not mistaken for worse performance.",
            "Without this, a harder task and a worse day look identical. With it, comparisons between sessions "
            "are genuinely like-for-like.",
        )
    )
    s.append(
        feature(
            9,
            "Usable by other software",
            "A small, well-documented service exposes the three core questions &mdash; assess a session, explain "
            "a result, forecast the next one &mdash; with predictable, clearly-described responses and errors.",
            "It turns a desktop research tool into a building block. A dashboard, a wearable app, or a partner's "
            "platform could use the same engine. This is the first step from prototype towards product.",
        )
    )

    # ---- Scope --------------------------------------------------------
    s.append(Paragraph("What we deliberately left out &mdash; and why", H1))
    s.append(
        Paragraph(
            "Being selective is part of the value. The following are all reasonable ideas that would cost time "
            "and money without adding anything at the current scale:",
            BODY,
        )
    )
    s.append(
        bullet(
            "<b>Database servers, cloud hosting, logins, dashboards.</b> With roughly twenty real sessions on "
            "record, simple files work fine. Infrastructure now would be cost without benefit."
        )
    )
    s.append(
        bullet(
            "<b>Deep learning (LSTM / Transformer models).</b> These need far more data than we have. The "
            "engineered features capture most of the benefit at a fraction of the cost. Worth revisiting after "
            "more data is collected."
        )
    )
    s.append(
        bullet(
            "<b>Eye tracking and physiological sensors.</b> Promising, but that is a separate hardware project, "
            "not a software feature."
        )
    )

    # ---- Honesty ------------------------------------------------------
    s.append(Paragraph("What the tool does not claim", H1))
    s.append(
        box(
            [
                bullet("It is a <b>research prototype and screening instrument</b>, not a medical or diagnostic device."),
                bullet("Scores are <b>behavioural indicators</b> &mdash; they estimate workload from performance, not from brain activity directly."),
                bullet("<b>Suggestions are informed advice, not proven causality.</b> The tool says what is worth trying, not what will certainly work."),
                bullet("<b>Forecasts and predictions always carry a range and a confidence level</b>, never false certainty."),
            ],
            bg=TINT_WARM,
            border=colors.HexColor("#C98A2E"),
        )
    )

    # ---- Close --------------------------------------------------------
    s.append(Paragraph("Why this is worth building", H1))
    s.append(
        Paragraph(
            "Version 1 can produce a number. Version 2 makes that number <b>trustworthy and usable</b>: it "
            "compares each person to their own normal, states how confident it is, explains itself, spots "
            "unreliable data, anticipates what is coming, and suggests what to do about it.",
            BODY,
        )
    )
    s.append(
        Paragraph(
            "Each piece is small, independently testable, and additive. Value arrives incrementally, and none of "
            "it risks the working prototype that already exists. Together they close the trust gap that usually "
            "stops people from acting on a score.",
            BODY,
        )
    )

    s.append(Paragraph("How we will know it worked", H1))
    s.append(bullet("Every prediction shows a confidence level and at least one grounded reason."))
    s.append(bullet("Unreliable sessions are visibly flagged rather than quietly trusted."))
    s.append(bullet("Forecasts openly refuse to guess when there is too little history."))
    s.append(bullet("Everything that works today &mdash; the command-line tool, the report, the existing tests &mdash; keeps working unchanged."))
    s.append(bullet("The service answers assess, explain, and forecast with clean, documented responses."))

    s.append(Paragraph("Natural next steps after this release", H1))
    s.append(bullet("Gather more real sessions so the models are proven on real data, not only on simulated data."))
    s.append(bullet("Put a simple web view on top of the new service so results are easy to browse."))
    s.append(bullet("Make the exercises adapt to the person's ability &mdash; the full version of feature 8."))
    s.append(bullet("Run a small longitudinal study to check whether the scores track real-world workload."))

    s.append(Spacer(1, 8))
    s.append(
        Paragraph(
            "Note: this document is a plain-language summary for a non-technical audience. "
            "The engineering breakdown of the same nine features is available separately.",
            NOTE,
        )
    )
    return s


def main() -> int:
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).expanduser()
    else:
        out = Path.home() / "Downloads" / "Cognitive Load Predictor V2 - Feature Pitch.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        title="Cognitive Load Predictor V2 - Feature Pitch",
        author="Cognitive Load Predictor project",
        subject="Plain-English summary of the Version 2 features and the reasons for adding them",
        leftMargin=21 * mm,
        rightMargin=21 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
    )

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(21 * mm, 12.5 * mm, A4[0] - 21 * mm, 12.5 * mm)
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(21 * mm, 9 * mm, "Cognitive Load Predictor V2 - feature pitch")
        canvas.drawRightString(A4[0] - 21 * mm, 9 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
