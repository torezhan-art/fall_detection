"""Build the capstone weekly-progress deck (dataset + baseline results) as a .pptx.

Same structure as the RA weekly-progress deck (plain title slide, title-at-top content
slides, one-line caption, gridded tables, bold takeaway), in black / white / grey only.

Run from the project folder:  .venv\\Scripts\\python.exe presentation\\build_deck.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

PROJECT = Path(__file__).resolve().parents[1]
FIGS = PROJECT / "figures"
OUT = PROJECT / "presentation" / "Capstone_Week_Progress_ML_Baseline.pptx"

FONT = "Arial"
BLACK = RGBColor(0x00, 0x00, 0x00)
TEXT = RGBColor(0x21, 0x21, 0x21)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
LINE = RGBColor(0xA6, 0xA6, 0xA6)

TABLE_STYLE_GRID = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid": plain black grid

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(10), Inches(5.625)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def text(slide, x, y, w, h, runs, size=14, bold=False, color=TEXT, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, margin=0.0):
    """Add a text box. `runs` is a string, or a list of paragraphs; a paragraph is a string
    or a list of (text, {bold, color, size}) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    paragraphs = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for chunk, opts in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = chunk
            f = r.font
            f.name, f.size = FONT, Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.color.rgb = opts.get("color", color)
    return box


def content_slide(title, caption=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.5, 0.3, 9.0, 0.55, title, size=26, bold=True, color=BLACK)
    if caption:
        text(s, 0.5, 0.88, 9.0, 0.35, caption, size=12, color=GREY)
    n = len(prs.slides)
    text(s, 9.0, 5.2, 0.5, 0.3, str(n), size=10, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(FIGS / name), Inches(x), Inches(y), **kw)


def table(slide, x, y, w, rows, col_w, size=11, header=True, bold_rows=(), row_h=0.3,
          align_right_from=1):
    """Plain gridded table. rows: list of lists of str. col_w: column widths in inches."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(w), Inches(row_h * len(rows)))
    tbl = shape.table
    tbl_pr = shape._element.graphic.graphicData.tbl.tblPr
    tbl_pr.find(qn("a:tableStyleId")).text = TABLE_STYLE_GRID
    tbl.first_row = tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if header and i == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT
            else:
                cell.fill.background()
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j >= align_right_from else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(size), TEXT
            r.font.bold = (header and i == 0) or i in bold_rows
    return shape


def no_style(shape):
    """Drop the default shape style (it adds a shadow to the shape and its text)."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


def step_box(slide, x, y, w, h, title, detail):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    no_style(box)
    box.adjustments[0] = 0.12
    box.fill.background()
    box.line.color.rgb = LINE
    box.line.width = Pt(1)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (chunk, size, bold, color) in enumerate([(title, 13, True, BLACK), (detail, 10, False, GREY)]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = chunk
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), bold, color


def arrow(slide, x, y, w=0.28, h=0.22):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    no_style(a)
    a.fill.solid(); a.fill.fore_color.rgb = LINE
    a.line.fill.background()


def stat(slide, x, y, w, number, label):
    text(slide, x, y, w, 0.6, number, size=30, bold=True, color=BLACK)
    text(slide, x, y + 0.62, w, 0.4, label, size=12, color=GREY)


# ---------------------------------------------------------------- 1. title
s = prs.slides.add_slide(BLANK)
text(s, 0.5, 1.55, 9.0, 0.7, "Weekly Progress Presentation", size=30, color=TEXT, align=PP_ALIGN.CENTER)
text(s, 0.5, 2.55, 9.0, 0.5, "Wearable Edge-AI Fall Detection: Dataset and Baseline Model",
     size=18, color=TEXT, align=PP_ALIGN.CENTER)
text(s, 0.5, 4.3, 9.0, 0.6, ["Torezhan Abdykhamitov  ·  dataset and TinyML model",
                             "ENG 400 Capstone  ·  Supervisor: Prof. Mohammad Hashmi"],
     size=12, color=GREY, align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = (
    "Progress on my part of the capstone: the dataset and the fall-detection model. "
    "This week: a clean data pipeline on a wrist dataset and a first, fairly evaluated baseline.")

# ---------------------------------------------------------------- 2. agenda
s = content_slide("Agenda", notes="Six parts, following the ML pipeline from raw data to results.")
items = [("Dataset", "WEDA-FALL: wrist-worn, 50 Hz, 25 participants"),
         ("Data pipeline", "cleaning, unit conversion, resampling"),
         ("Windowing and labels", "2 s windows, 1 s hop, as on the device"),
         ("Evaluation protocol", "5-fold cross-validation split by person; metrics"),
         ("Baseline results", "peak-acceleration threshold"),
         ("Next steps", "1D-CNN, int8 model on the XIAO nRF52840")]
for i, (head, sub) in enumerate(items):
    y = 1.15 + i * 0.66
    text(s, 0.7, y, 0.5, 0.5, f"{i + 1}", size=22, bold=True, color=BLACK)
    text(s, 1.3, y, 7.5, 0.5, [[(head, {"bold": True, "size": 16, "color": BLACK}),
                               (f"   {sub}", {"size": 13, "color": GREY})]], anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 3. dataset
s = content_slide("Dataset: WEDA-FALL",
                  "Wrist Elderly Daily Activity and Fall dataset, github.com/joaojtmarques/WEDA-FALL",
                  notes="We switched from SisFall to WEDA-FALL. SisFall was recorded at the waist; our device "
                        "is on the wrist, and wrist motion is very different. WEDA-FALL matches our sensor "
                        "position and our 50 Hz sampling rate, and gives a hand-marked fall interval.")
table(s, 0.5, 1.35, 5.3, [
    ["Property", "Value"],
    ["Device / position", "Fitbit Sense smartwatch, wrist"],
    ["Sampling rate", "50 Hz"],
    ["Sensors used", "3-axis accelerometer + 3-axis gyroscope"],
    ["Participants", "25: 14 young (20-46 y), 11 elderly (77-95 y)"],
    ["Trials", "969: 350 falls, 619 daily activities (ADLs)"],
    ["Activity types", "8 fall types, 11 ADL types"],
    ["Labels", "annotated fall start / end time per fall"],
], col_w=[1.7, 3.6], size=11, align_right_from=9)
text(s, 6.2, 1.35, 3.3, 0.4, "Why this dataset", size=15, bold=True, color=BLACK)
text(s, 6.2, 1.8, 3.3, 3.0, [
    "Recorded on the wrist, like our device (SisFall is waist-worn)",
    "Same 50 Hz rate as our design (DR-1)",
    "Includes elderly people's daily activities",
    "Fall interval is annotated, so windows can be labelled precisely",
], size=13)
for p in s.shapes[-1].text_frame.paragraphs:
    p.space_after = Pt(8)

# ---------------------------------------------------------------- 4. dataset overview
s = content_slide("Dataset Composition",
                  "Trials per activity; elderly participants performed ADLs only, for safety",
                  notes="Falls come only from the 14 young participants. The elderly add daily activities, "
                        "which is useful for testing false alarms but means falls of elderly people are not "
                        "represented. F08 has 4 repetitions: falls towards and away from the watch side.")
picture(s, "03_dataset_overview.png", 0.5, 1.3, w=9.0)
text(s, 0.5, 4.55, 9.0, 0.6, [[("Falls are performed by the 14 young participants only. ", {"bold": True}),
                              ("Our own wrist data (after IREC approval) will be needed to cover more people.", {})]],
     size=13)

# ---------------------------------------------------------------- 5. pipeline
s = content_slide("Data Pipeline", "From raw smartwatch files to model-ready windows",
                  notes="The raw data needed cleaning. The watch sends samples in bursts, so the real rate "
                        "varies between about 40 and 50 Hz, and accelerometer and gyroscope have different "
                        "numbers of rows. We interpolate both onto one exact 50 Hz grid. 57 trials have "
                        "timestamps that jump backwards and are excluded rather than guessed.")
steps = [("Raw files", "accel + gyro CSV, 969 trials"),
         ("Clean", "drop 57 trials with broken time"),
         ("Units", "m/s² → g, rad/s → °/s (as on XIAO)"),
         ("Resample", "uniform 50 Hz grid"),
         ("Windows", "2 s, 1 s hop, labelled")]
bw, gap = 1.55, 0.31
for i, (t, d) in enumerate(steps):
    x = 0.5 + i * (bw + gap)
    step_box(s, x, 1.4, bw, 1.1, t, d)
    if i < len(steps) - 1:
        arrow(s, x + bw + 0.015, 1.84)
text(s, 0.5, 2.85, 9.0, 0.4, "Issues found in the raw data and how they were handled", size=14, bold=True, color=BLACK)
table(s, 0.5, 3.25, 9.0, [
    ["Issue", "Measured", "Handling"],
    ["Samples arrive in bursts", "raw rate 39.8-50.2 Hz (5th-95th pct.)", "interpolate onto exact 50 Hz grid"],
    ["Accel and gyro rows differ", "e.g. 776 vs 760 rows in one trial", "common time span, one grid"],
    ["Time jumps backwards", "57 of 969 trials (20 falls, 37 ADLs)", "excluded, not reordered"],
], col_w=[2.6, 3.4, 3.0], size=11, align_right_from=9)

# ---------------------------------------------------------------- 6. one fall
s = content_slide("Example Signal: Forward Fall",
                  "WEDA-FALL F01/U01_R01 after resampling; shaded = annotated fall interval",
                  notes="Top: acceleration on three axes; middle: rotation rate; bottom: acceleration "
                        "magnitude. A fall has three parts: irregular motion, a sharp impact, then stillness "
                        "at about 1 g. The model must see this whole pattern, not only the spike.")
picture(s, "01_one_fall_6axis.png", 0.5, 1.3, h=3.95)
text(s, 6.0, 1.45, 3.5, 0.4, "What a fall looks like", size=15, bold=True, color=BLACK)
text(s, 6.0, 1.9, 3.5, 3.2, [
    [("Before: ", {"bold": True}), ("walking, |a| varies around 1 g", {})],
    [("Impact: ", {"bold": True}), ("sharp peak of 5.7 g, rotation up to 490 °/s", {})],
    [("After: ", {"bold": True}), ("lying still, |a| ≈ 1 g and almost no rotation", {})],
    [("Window: ", {"bold": True}), ("2 s = 100 samples × 6 axes per decision", {})],
], size=13)
for p in s.shapes[-1].text_frame.paragraphs:
    p.space_after = Pt(10)

# ---------------------------------------------------------------- 7. falls vs ADLs
s = content_slide("Falls vs Fall-like Daily Activities",
                  "Acceleration magnitude, same participant; red = fall, blue = daily activity",
                  notes="This is why the wrist is hard: hand movements create big spikes. Hitting a table "
                        "reaches 4.9 g and a stumble 4.7 g, more than a fainting fall at 2.4 g.")
picture(s, "02_falls_vs_hard_adls.png", 1.5, 1.25, h=3.5)
text(s, 0.5, 4.8, 8.3, 0.5, [[("Hit table (4.9 g) and stumble (4.7 g) spike harder than a fainting fall (2.4 g): ", {"bold": True}),
                             ("the peak value alone cannot separate them.", {})]], size=13)

# ---------------------------------------------------------------- 8. windows
s = content_slide("Windowing and Labels",
                  "2 s windows (100 samples × 6 axes) every 1 s, the same as the on-device design (DR-1)",
                  notes="Every trial is cut into overlapping 2-second windows. A window is a fall if it "
                        "contains the impact. Windows that touch the fall but miss the impact are ambiguous "
                        "and are dropped. Each fall trial gives exactly 2 fall windows. There are about "
                        "13 normal windows for every fall window, so accuracy alone is misleading.")
table(s, 0.5, 1.4, 5.6, [
    ["Label", "Rule"],
    ["Fall (1)", "window contains the impact (peak |a| inside the annotated fall)"],
    ["Dropped", "touches the annotated fall but misses the impact (ambiguous)"],
    ["Not fall (0)", "all other windows: every ADL, and walking before / lying after a fall"],
], col_w=[1.3, 4.3], size=11, row_h=0.55, align_right_from=9)
stat(s, 6.6, 1.35, 2.9, "9,175", "windows from 912 trials")
stat(s, 6.6, 2.5, 1.4, "660", "fall")
stat(s, 8.1, 2.5, 1.4, "8,515", "not fall")
stat(s, 6.6, 3.65, 2.9, "1 : 12.9", "fall : not-fall imbalance")
text(s, 0.5, 3.85, 5.6, 0.8, "Every one of the 330 usable fall trials gives exactly 2 fall windows.",
     size=13)

# ---------------------------------------------------------------- 9. evaluation protocol
s = content_slide("Evaluation Protocol",
                  "5-fold cross-validation split by person (StratifiedGroupKFold): no one is in both train and test",
                  notes="If the same person were in training and test, the model could memorise that person's "
                        "movement style and look better than it is. Splitting by person tests unseen people, "
                        "which is what happens on campus. Fold 2 has fewer falls because only 2 of its users "
                        "are young. Metrics: recall is the same as sensitivity; precision is the share of "
                        "alarms that are real falls; specificity is the share of normal windows ignored.")
table(s, 0.5, 1.4, 4.9, [
    ["Fold", "Test users", "Windows", "Fall %"],
    ["0", "3, 7, 8, 29", "1,807", "7.9"],
    ["1", "1, 6, 12, 25, 27", "1,860", "7.4"],
    ["2", "2, 11, 22, 23, 24, 28, 30", "1,842", "5.4"],
    ["3", "4, 10, 14, 26, 31", "1,865", "7.9"],
    ["4", "5, 9, 13, 21", "1,801", "7.3"],
], col_w=[0.6, 2.4, 1.0, 0.9], size=11, align_right_from=2)
table(s, 5.7, 1.4, 3.8, [
    ["Metric", "Meaning"],
    ["Recall = sensitivity", "falls caught"],
    ["Precision", "alarms that are real falls"],
    ["F1", "balance of precision and recall"],
    ["Specificity", "normal windows ignored"],
    ["Trial specificity", "ADL trials with no alarm (T1)"],
], col_w=[1.6, 2.2], size=11, align_right_from=9)
text(s, 0.5, 3.45, 4.9, 0.8, "Users 21-31 are elderly (ADLs only). Every fold has 2-3 young users with falls.",
     size=12, color=GREY)

# ---------------------------------------------------------------- 10. threshold baseline
s = content_slide("Baseline: Peak-Acceleration Threshold",
                  "Rule: fall if max |a| in the 2 s window ≥ T; T chosen on 4 training folds, tested on the 5th",
                  notes="The simplest detector, used by several prior systems. As T goes up, false alarms "
                        "drop but more falls are missed. The two curves cross at about 90 % / 90 %: no "
                        "threshold reaches 95 % on both, so a threshold cannot meet REQ-1.")
picture(s, "04_threshold_baseline.png", 0.5, 1.3, h=3.6)
text(s, 7.1, 1.45, 2.4, 0.4, "Reading the plot", size=15, bold=True, color=BLACK)
text(s, 7.1, 1.9, 2.4, 3.0, [
    "Lower T: more falls caught, more false alarms",
    "Training folds choose T ≈ 2.95 g",
    "Best possible balance: 90 % / 90 % at 3.6 g",
    "No T gives ≥ 95 % on both",
], size=12)
for p in s.shapes[-1].text_frame.paragraphs:
    p.space_after = Pt(8)

# ---------------------------------------------------------------- 11. results table
s = content_slide("Baseline Results",
                  "Window-level metrics on held-out users (%); trial columns follow test T1",
                  notes="It catches about 96 % of falls, but only about 1 in 3 alarms is a real fall, and "
                        "4 in 10 daily-activity trials raise at least one false alarm. Accuracy of 86.6 % "
                        "looks fine but hides this, because normal windows are 13 times more common.")
table(s, 0.5, 1.35, 9.0, [
    ["Fold", "T (g)", "Precision", "Recall", "F1", "Specificity", "Accuracy", "Trial spec."],
    ["0", "2.95", "37.6", "97.2", "54.2", "86.2", "87.1", "58.3"],
    ["1", "2.95", "35.1", "98.6", "51.8", "85.4", "86.4", "53.1"],
    ["2", "2.95", "28.7", "96.0", "44.1", "86.3", "86.8", "63.4"],
    ["3", "2.95", "32.3", "90.5", "47.6", "83.6", "84.2", "56.0"],
    ["4", "3.15", "38.2", "97.0", "54.8", "87.6", "88.3", "56.9"],
    ["Mean ± std", "", "34.4 ± 4.0", "95.8 ± 3.1", "50.5 ± 4.6", "85.8 ± 1.5", "86.6 ± 1.5", "57.5 ± 3.8"],
], col_w=[1.25, 0.75, 1.15, 1.15, 1.15, 1.2, 1.15, 1.2], size=11, bold_rows=(6,), row_h=0.32)
text(s, 0.5, 3.85, 9.0, 1.2, [
    [("Catches 96 % of falls, but only 1 in 3 alarms is a real fall.", {"bold": True})],
    [("4 in 10 daily-activity trials raise a false alarm; accuracy (86.6 %) is below the 95 % target of REQ-1.", {})],
], size=13)

# ---------------------------------------------------------------- 12. false alarms by activity
s = content_slide("Where the False Alarms Come From",
                  "Share of each daily activity's windows flagged as a fall at T = 2.95 g",
                  notes="Jogging, clapping, hitting a table and jumping cause most false alarms: on the "
                        "wrist, the hand moves much harder than the body. Walking and sitting are almost "
                        "never flagged. A learned model must use the pattern after the impact (stillness), "
                        "not only the peak.")
table(s, 0.5, 1.35, 4.6, [
    ["Activity", "Flagged (%)"],
    ["D02 Jogging", "86.5"],
    ["D10 Clapping hands", "46.7"],
    ["D09 Hit table with hand", "32.9"],
    ["D08 Gentle jump", "31.7"],
    ["D07 Stumble while walking", "15.8"],
    ["D06 Crouch, tie shoes", "6.4"],
    ["D05 Collapse into chair", "4.5"],
    ["D03 Stairs", "3.5"],
    ["D11 Open / close door", "1.8"],
    ["D01 Walking", "0.2"],
    ["D04 Sit down, get up", "0.0"],
], col_w=[3.2, 1.4], size=11, row_h=0.29)
text(s, 5.5, 1.45, 4.0, 0.4, "Takeaway", size=15, bold=True, color=BLACK)
text(s, 5.5, 1.9, 4.0, 3.0, [
    "Wrist-specific movements (jogging, clapping, table hits) produce fall-sized peaks",
    "Walking and sitting are almost never flagged",
    "A learned model can use what happens after the impact: a fall ends in stillness, clapping does not",
], size=13)
for p in s.shapes[-1].text_frame.paragraphs:
    p.space_after = Pt(10)

# ---------------------------------------------------------------- 13. next steps
s = content_slide("Next Steps",
                  "Same windows, same 5 folds, same metrics, so every model is compared fairly",
                  notes="Next: a Random Forest on hand-made features and a small 1D-CNN. The question is "
                        "whether the CNN keeps recall at 95 % or more while raising precision well above "
                        "34 %. Then int8 quantisation and the memory check for M2.")
table(s, 0.5, 1.35, 9.0, [
    ["Step", "Goal", "Milestone"],
    ["Random Forest on window features", "classical-ML comparison point", "late Oct"],
    ["1D-CNN (Keras), class-weighted", "recall ≥ 95 % with precision well above 34 %", "M1, late Oct"],
    ["Trial-level evaluation", "falls detected / ADL trials without false alarm (T1)", "M1"],
    ["int8 quantisation", "≤ 1 point accuracy drop vs float", "M2, mid Nov"],
    ["Run on XIAO nRF52840 Sense", "TFLite Micro arena < 15 KB (T4)", "M2, mid Nov"],
], col_w=[3.2, 4.3, 1.5], size=11, row_h=0.36, align_right_from=2)
text(s, 0.5, 3.75, 9.0, 0.4, "Questions for discussion", size=14, bold=True, color=BLACK)
text(s, 0.5, 4.1, 9.0, 1.0, [
    "What false-alarm rate per day is acceptable for Campus Security?",
    "Timeline for IREC approval and our own wrist data collection",
], size=13)

prs.save(OUT)
print(f"Saved {OUT} ({len(prs.slides)} slides)")
