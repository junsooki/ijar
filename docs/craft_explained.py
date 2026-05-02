"""Generate an ELI4 illustrated PDF explaining how CRAFT works end-to-end."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ── Helpers ──────────────────────────────────────────────────────────────
def new_page(pdf, title, figsize=(11, 8.5)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.97)
    return fig


def draw_grid(ax, H, W, walls=None, agents=None, goals=None, title="",
              heat=None, arrows=None):
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for r in range(H + 1):
        ax.axhline(r - 0.5, color="gray", lw=0.5)
    for c in range(W + 1):
        ax.axvline(c - 0.5, color="gray", lw=0.5)
    if heat is not None:
        vmax = max(heat.max(), 1e-9)
        for r in range(H):
            for c in range(W):
                v = heat[r, c] / vmax
                if v > 0:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        fc=(1.0, 1.0 - v, 1.0 - v, 0.6), ec="none"))
    if walls:
        for (r, c) in walls:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc="black"))
    if goals:
        for i, (r, c) in enumerate(goals):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1, fc="#d4edda", ec="green", lw=1.5))
            ax.text(c, r, f"G{i+1}", ha="center", va="center",
                    fontsize=7, color="green", fontweight="bold")
    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6"]
    if agents:
        for i, (r, c) in enumerate(agents):
            col = colors[i % len(colors)]
            ax.add_patch(plt.Circle((c, r), 0.35, fc=col, ec="black", lw=1.5, zorder=5))
            ax.text(c, r, f"A{i+1}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=6)
    if arrows:
        for (r, c, dr, dc, col) in arrows:
            ax.annotate("", xy=(c + dc * 0.7, r + dr * 0.7),
                        xytext=(c, r),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.set_xticks([])
    ax.set_yticks([])


def txt(fig, y, body, fc="#f8f9fa", ec="#6c757d", size=11):
    fig.text(0.05, y, body, fontsize=size, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec))


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1 — THE BIG PICTURE
# ═══════════════════════════════════════════════════════════════════════
def page1_big_picture(pdf):
    fig = new_page(pdf, "Page 1: What Is CRAFT Trying to Do?")

    txt(fig, 0.88,
        "Imagine a maze with little robots.  Each robot starts somewhere and\n"
        "needs to walk to its own special star.\n\n"
        "Easy if there's only ONE robot — just walk toward the star!\n"
        "Hard when there are MANY robots — they bump into each other and get stuck.\n\n"
        "A smart brain (called the UNet) already knows how to guide ONE robot.\n"
        "But it was never taught to share the road with others.\n\n"
        "CRAFT's job: teach it to cooperate — without rebuilding the whole brain.",
        fc="#fff3cd", ec="#ffc107", size=12)

    gs = gridspec.GridSpec(1, 2, left=0.08, right=0.95,
                           bottom=0.15, top=0.62, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_grid(ax1, 6, 6,
              walls=[(1, 1), (1, 2), (3, 4)],
              agents=[(0, 0)],
              goals=[(5, 5)],
              title="One robot — easy!",
              arrows=[(0, 0, 0, 1, "#e74c3c")])

    ax2 = fig.add_subplot(gs[0, 1])
    draw_grid(ax2, 6, 6,
              walls=[(1, 1), (1, 2), (3, 4)],
              agents=[(2, 2), (2, 3), (3, 2), (3, 3)],
              goals=[(0, 5), (5, 0), (0, 0), (5, 5)],
              title="Four robots — crash!")
    ax2.plot([2.7, 3.3], [1.7, 2.3], "r-", lw=3, zorder=10)
    ax2.plot([2.7, 3.3], [2.3, 1.7], "r-", lw=3, zorder=10)
    ax2.text(3, 1.1, "STUCK!", ha="center", fontsize=11,
             color="red", fontweight="bold")

    fig.text(0.5, 0.07,
             "CRAFT = add a tiny helper on top of the frozen brain that says:\n"
             "'Hey A2, careful — A1 is right there!'",
             ha="center", fontsize=12, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#d4edda", ec="#28a745"))

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 2 — WHAT THE BRAIN SEES (6-CHANNEL INPUT)
# ═══════════════════════════════════════════════════════════════════════
def page2_input_channels(pdf):
    fig = new_page(pdf, "Page 2: What the Brain Sees — 6 Layers of Information")

    txt(fig, 0.90,
        "Before a robot can decide where to move, it has to LOOK at the world.\n"
        "The world is given as 6 pictures stacked on top of each other — like 6\n"
        "transparencies on an overhead projector, each showing something different.",
        fc="#cce5ff", ec="#007bff", size=11)

    channels = [
        ("#2c3e50", "white",  "Ch 0\nWalls",
         "Black = wall\nWhite = free\n\n'Where can\nI go?'"),
        ("#e74c3c", "white",  "Ch 1\nRobot positions",
         "Each robot gets\na different number\n\n'Who is where?'"),
        ("#2ecc71", "white",  "Ch 2\nGoal positions",
         "Each robot's star\ngets its number\n\n'Where does\neach go?'"),
        ("#3498db", "white",  "Ch 3\nDistance to star",
         "Close = bright\nFar  = dark\n\n'How far am\nI from home?'"),
        ("#f39c12", "black",  "Ch 4\nDistance slope ↔",
         "Bright = goal is\nto the right\nDark = to the left\n\n'Which column?'"),
        ("#9b59b6", "white",  "Ch 5\nDistance slope ↕",
         "Bright = goal is\nbelow\nDark = above\n\n'Which row?'"),
    ]

    gs = gridspec.GridSpec(2, 3, left=0.05, right=0.98,
                           bottom=0.04, top=0.72, wspace=0.35, hspace=0.45)

    for idx, (bg, fg, title, note) in enumerate(channels):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(bg)
        ax.set_xlim(0, 4); ax.set_ylim(0, 4)
        ax.set_title(title, fontsize=10, fontweight="bold", color=bg,
                     bbox=dict(fc="white", ec=bg, pad=2, boxstyle="round"))
        ax.text(2, 2.4, note, ha="center", va="center",
                fontsize=8.5, color=fg, family="monospace",
                multialignment="center")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(bg); spine.set_linewidth(2)

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3 — THE FROZEN BRAIN (UNET)
# ═══════════════════════════════════════════════════════════════════════
def page3_unet(pdf):
    fig = new_page(pdf, "Page 3: The Frozen Brain — UNet")

    txt(fig, 0.88,
        "The UNet is a big neural network with 7.8 MILLION numbers inside it.\n"
        "Think of it as a very experienced navigator who has seen millions of mazes.\n\n"
        "It reads the 6-layer picture and outputs ONE thing:\n"
        "For EVERY single cell in the grid, 5 scores:\n\n"
        "   [Stay,  Go Right,  Go Left,  Go Up,  Go Down]\n\n"
        "Higher score = 'this direction is better from this cell.'\n\n"
        "We do NOT change the UNet.  It's frozen, like a textbook.\n"
        "CRAFT only adds a thin sticky note on top.",
        fc="#cce5ff", ec="#007bff", size=11)

    gs = gridspec.GridSpec(1, 3, left=0.06, right=0.96,
                           bottom=0.12, top=0.60, wspace=0.4)

    # Input picture
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#ecf0f1")
    ax0.text(0.5, 0.75, "6-layer\npicture", ha="center", va="center",
             fontsize=13, fontweight="bold", transform=ax0.transAxes)
    ax0.text(0.5, 0.35, "[6 × H × W]", ha="center", va="center",
             fontsize=10, family="monospace", transform=ax0.transAxes)
    ax0.set_title("INPUT", fontsize=11, fontweight="bold")
    ax0.set_xticks([]); ax0.set_yticks([])

    # Arrow
    fig.text(0.38, 0.35, "→", fontsize=30, ha="center", va="center")

    # UNet box
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#007bff")
    ax1.text(0.5, 0.70, "UNet", ha="center", va="center",
             fontsize=18, fontweight="bold", color="white", transform=ax1.transAxes)
    ax1.text(0.5, 0.45, "7,800,000\nparameters", ha="center", va="center",
             fontsize=10, color="#cce5ff", family="monospace", transform=ax1.transAxes)
    ax1.text(0.5, 0.20, "FROZEN ❄️", ha="center", va="center",
             fontsize=10, color="#aed6f1", fontweight="bold", transform=ax1.transAxes)
    ax1.set_title("BRAIN", fontsize=11, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # Arrow
    fig.text(0.70, 0.35, "→", fontsize=30, ha="center", va="center")

    # Output
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#ecf0f1")
    ax2.text(0.5, 0.75, "5 scores\nper cell", ha="center", va="center",
             fontsize=12, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.5, 0.40,
             "[5 × H × W]\n\nStay / Right\nLeft / Up\nDown",
             ha="center", va="center",
             fontsize=9, family="monospace", transform=ax2.transAxes)
    ax2.set_title("OUTPUT (raw logits)", fontsize=11, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    txt(fig, 0.06,
        "Problem: UNet was trained alone — it ignores other robots.\n"
        "The scores it gives don't say 'but watch out, A2 is going left!'",
        fc="#f8d7da", ec="#dc3545", size=11)

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 4 — THE HELPER (HEURISTIC PHI)
# ═══════════════════════════════════════════════════════════════════════
def page4_heuristic_phi(pdf):
    fig = new_page(pdf, "Page 4: The Helper — Heuristic Phi (Traffic Sensor)")

    txt(fig, 0.89,
        "CRAFT adds a HELPER that computes a 'danger score' (called phi, φ) for every robot.\n"
        "High φ = 'this robot is in a crowded or dangerous spot — others should avoid it.'\n"
        "Low  φ = 'this robot is fine, nothing special going on.'\n\n"
        "The helper uses THREE hand-crafted signals (no learning needed — pure math):",
        fc="#fff3cd", ec="#ffc107", size=11)

    boxes = [
        ("#e74c3c", "SIGNAL 1: Local Density",
         "Count how many other robots\n"
         "are within 3 steps of this robot.\n\n"
         "Many neighbors → high cost.\n"
         "Alone → low cost.\n\n"
         "Weight: 2.0\n\n"
         "Analogy: a crowded hallway\n"
         "vs. an empty one."),
        ("#f39c12", "SIGNAL 2: Bottleneck Score",
         "Some cells on the map are\n"
         "CHOKE POINTS — almost every\n"
         "path passes through them.\n\n"
         "Pre-computed once per map\n"
         "using graph betweenness.\n\n"
         "Weight: 3.0\n\n"
         "Analogy: a narrow bridge\n"
         "everyone must cross."),
        ("#3498db", "SIGNAL 3: Direction Conflict",
         "Is another robot walking\n"
         "straight toward THIS robot?\n\n"
         "If yes → head-on collision\n"
         "risk → raise the cost!\n\n"
         "Weight: 5.0\n\n"
         "Analogy: two cars on a\n"
         "one-lane road facing each\n"
         "other."),
    ]

    gs = gridspec.GridSpec(1, 3, left=0.04, right=0.98,
                           bottom=0.08, top=0.68, wspace=0.3)

    for idx, (col, title, body) in enumerate(boxes):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(col)
        ax.text(0.5, 0.92, title, ha="center", va="top",
                fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes, wrap=True)
        ax.text(0.5, 0.68, body, ha="center", va="top",
                fontsize=9, color="white", family="monospace",
                transform=ax.transAxes, multialignment="center")
        ax.set_xticks([]); ax.set_yticks([])

    fig.text(0.5, 0.03,
             "φᵢ  =  2.0 × density  +  3.0 × bottleneck  +  5.0 × conflict",
             ha="center", fontsize=13, fontweight="bold", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#e2e3e5", ec="#333"))

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 5 — COST SHAPING: BENDING THE BRAIN'S SCORES
# ═══════════════════════════════════════════════════════════════════════
def page5_cost_shaping(pdf):
    fig = new_page(pdf, "Page 5: Cost Shaping — How the Helper Bends the Brain's Scores")

    txt(fig, 0.89,
        "The UNet gives raw scores.  The helper gives a danger number φ per robot.\n"
        "Cost shaping COMBINES them:\n\n"
        "   shaped_score[robot, action]  =  UNet_score  −  penalty\n\n"
        "   penalty = sum over nearby robots j of:  weight(dist) × φⱼ\n"
        "   weight(dist) = max(0,  (radius+1 − dist) / (radius+1))\n\n"
        "   dist=0 → weight=1.0    dist=1 → weight=0.67    dist=2 → weight=0.33\n\n"
        "If robot A1 is thinking about going RIGHT and A2 (φ=4.0) is 1 step to the right:\n"
        "   penalty ≈ 0.67 × 4.0 = 2.68 subtracted from the 'go right' score.\n"
        "   The 'go up' score is untouched.  So A1 will probably go up instead. ✓",
        fc="#d4edda", ec="#28a745", size=11)

    # Visual grid showing penalty reaching out
    gs = gridspec.GridSpec(1, 2, left=0.05, right=0.95,
                           bottom=0.09, top=0.55, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    H, W = 7, 7
    heat = np.zeros((H, W))
    dest = (3, 4)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r2, c2 = dest[0] + dr, dest[1] + dc
            if 0 <= r2 < H and 0 <= c2 < W:
                d = max(abs(dr), abs(dc))
                heat[r2, c2] = max(0, (3 - d) / 3)
    draw_grid(ax1, H, W,
              agents=[(3, 3)],
              title="A1 thinking about going RIGHT\n(destination = orange cell)",
              heat=heat)
    ax1.add_patch(plt.Rectangle((3.5, 2.5), 1, 1,
                                fc="none", ec="orange", lw=3, ls="--", zorder=7))
    ax1.text(4, 3, "dest", ha="center", va="center",
             fontsize=7, color="darkorange", fontweight="bold", zorder=8)
    ax1.add_patch(plt.Circle((5, 3), 0.3, fc="#3498db", ec="black", lw=1.5, zorder=5))
    ax1.text(5, 3, "A2", ha="center", va="center",
             fontsize=7, color="white", fontweight="bold", zorder=6)
    ax1.text(5, 3.55, "φ=4.0", ha="center", fontsize=8,
             color="#3498db", fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    actions = ["Stay", "Right", "Left", "Up", "Down"]
    raw     = [1.2,    2.5,   0.8,   1.8,  0.6]
    shaped  = [1.2,    2.5 - 0.67 * 4.0,  0.8,  1.8, 0.6]
    x = np.arange(len(actions))
    w = 0.35
    ax2.bar(x - w/2, raw,    w, label="Raw UNet score",    color="#007bff", alpha=0.85)
    ax2.bar(x + w/2, shaped, w, label="After shaping",     color="#28a745", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(actions, fontsize=9)
    ax2.set_ylabel("Score", fontsize=9)
    ax2.set_title("Scores before & after shaping\n(A2 with φ=4.0 is 1 step right of dest)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.annotate("'Right' score\ndropped a lot!", xy=(1 + w/2, shaped[1]),
                 xytext=(2.3, 2.0),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=9, color="red", fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 6 — PPO: LEARNING FROM EXPERIENCE
# ═══════════════════════════════════════════════════════════════════════
def page6_ppo(pdf):
    fig = new_page(pdf, "Page 6: PPO — Learning From Experience")

    txt(fig, 0.89,
        "The heuristic phi uses fixed math.  But we can also LEARN better weights.\n"
        "PPO (Proximal Policy Optimization) is a technique for training by trial and error:\n\n"
        "  1. Let the robots run around the maze for a while  (ROLLOUT)\n"
        "  2. Give rewards: +1 when a robot reaches its star, -small every step\n"
        "  3. Ask: which decisions led to good outcomes?  (GAE — see below)\n"
        "  4. Nudge the policy a tiny bit toward those decisions  (UPDATE)\n"
        "  5. Repeat thousands of times\n\n"
        "We train TWO things with PPO:\n"
        "  • Policy (action scores) — the shaped UNet + fine-tuned top layers\n"
        "  • Value head — a tiny MLP that predicts 'how good is this situation?'",
        fc="#f8d7da", ec="#dc3545", size=11)

    txt(fig, 0.40,
        "GAE — Generalized Advantage Estimation  (γ=0.99, λ=0.95)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "'Did this action turn out better or worse than I expected?'\n\n"
        "  advantage = (actual reward I got) − (what the value head predicted)\n\n"
        "  γ=0.99: care about future rewards almost as much as immediate ones\n"
        "  λ=0.95: balance between short-horizon accuracy and long-horizon reach\n\n"
        "  Positive advantage → do this action MORE often\n"
        "  Negative advantage → do this action LESS often",
        fc="#e2e3e5", ec="#6c757d", size=11)

    txt(fig, 0.12,
        "PPO clip (ε=0.2):  Never change the policy too much in one step.\n"
        "Like saying 'improve yourself, but don't completely reinvent yourself overnight.'",
        fc="#fff3cd", ec="#ffc107", size=11)

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 7 — CURRICULUM LEARNING
# ═══════════════════════════════════════════════════════════════════════
def page7_curriculum(pdf):
    fig = new_page(pdf, "Page 7: Curriculum Learning — Start Easy, Get Harder")

    txt(fig, 0.89,
        "We don't throw the robots into the hardest level immediately.\n"
        "Like a video game, we start EASY and get HARDER when they're ready.\n\n"
        "We measure readiness with ISR — Individual Success Rate:\n"
        "  ISR = (number of robots that reached their star) / (total robots)\n\n"
        "When ISR >= 0.50 (50% success), we move to the next level.",
        fc="#cce5ff", ec="#007bff", size=12)

    gs = gridspec.GridSpec(1, 3, left=0.04, right=0.98,
                           bottom=0.22, top=0.70, wspace=0.3)

    stages = [
        ("#2ecc71",  "Stage 1",  "4 robots",  "16×16 map",  "128 steps",  "ISR ≥ 0.50 → next"),
        ("#f39c12",  "Stage 2",  "8 robots",  "16×16 map",  "192 steps",  "ISR ≥ 0.50 → next"),
        ("#e74c3c",  "Stage 3", "16 robots",  "32×32 map",  "256 steps",  "Final stage"),
    ]

    for idx, (col, stage, agents, msize, steps, cond) in enumerate(stages):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(col)
        ax.text(0.5, 0.88, stage, ha="center", va="top",
                fontsize=16, fontweight="bold", color="white",
                transform=ax.transAxes)
        for i, line in enumerate([agents, msize, steps, "", cond]):
            ax.text(0.5, 0.68 - i * 0.14, line, ha="center", va="top",
                    fontsize=10, color="white", family="monospace",
                    transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    # Arrow between stages
    fig.text(0.355, 0.45, "→", fontsize=28, ha="center", va="center", color="#555")
    fig.text(0.645, 0.45, "→", fontsize=28, ha="center", va="center", color="#555")

    txt(fig, 0.12,
        "Why curriculum?\n"
        "If you start with 16 robots on a big map, the reward signal is so noisy\n"
        "that the model barely learns anything.  Starting small makes learning stable.",
        fc="#d4edda", ec="#28a745", size=11)

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 8 — DIAGNOSTIC PHASE
# ═══════════════════════════════════════════════════════════════════════
def page8_diagnostic(pdf):
    fig = new_page(pdf, "Page 8: Before Training — Diagnosing the Problem")

    txt(fig, 0.89,
        "Before fixing anything, we first PROVE that the problem actually exists.\n"
        "We run the frozen UNet on many maps and log every decision it makes.\n\n"
        "Then we ask five questions:",
        fc="#fff3cd", ec="#ffc107", size=12)

    questions = [
        ("Q1", "Does the UNet disagree with the expert (LaCAM) MORE\n"
               "when there are more robots nearby?\n"
               "→ Measures: does density predict mistakes?"),
        ("Q2", "Where on the map do mistakes cluster?\n"
               "→ Spatial heatmap of errors.\n"
               "→ Do errors bunch near chokepoints?"),
        ("Q3", "Do cells with high 'betweenness' (bottlenecks) have\n"
               "more mistakes than open cells?\n"
               "→ Confirms bottleneck hypothesis."),
        ("Q4", "Can density ALONE predict whether the UNet will\n"
               "make a mistake?  (AUC of logistic regression)\n"
               "→ AUC > 0.6 confirms density matters."),
        ("Q5", "Is there systematic directional conflict?\n"
               "Do two robots often 'want' the same cell at the same time?\n"
               "→ Measures head-on collision rate."),
    ]

    for i, (label, body) in enumerate(questions):
        y = 0.74 - i * 0.135
        fc = "#d4edda" if i % 2 == 0 else "#cce5ff"
        ec = "#28a745" if i % 2 == 0 else "#007bff"
        fig.text(0.05, y, f"{label}: {body}", fontsize=10, va="top",
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec))

    txt(fig, 0.04,
        "All five analyses are implemented in diagnostic.ipynb (Colab) and\n"
        "diagnostic_local.ipynb (local).  Results go into results/diagnostic_report.tex.",
        fc="#e2e3e5", ec="#6c757d", size=10)

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 9 — FULL PIPELINE AT A GLANCE
# ═══════════════════════════════════════════════════════════════════════
def page9_pipeline(pdf):
    fig = new_page(pdf, "Page 9: The Full Pipeline at a Glance")

    steps = [
        ("#2c3e50", "white",
         "ENVIRONMENT  (envs/pogema_railgun_env.py)",
         "Grid world.  Produces the 6-layer picture every step.\n"
         "Runs up to 256 steps per episode."),
        ("#007bff", "white",
         "FROZEN UNet  (pretrained RAILGUN checkpoint)",
         "Reads the 6-layer picture.\n"
         "Outputs 5 raw scores per cell.  Never updated."),
        ("#f39c12", "black",
         "HEURISTIC PHI  (tools/heuristic_cost.py)",
         "Computes φ per robot from density + bottleneck + conflict.\n"
         "Subtracts proximity-weighted penalty from the raw UNet scores."),
        ("#28a745", "white",
         "ACTION SAMPLING  (train_rl.py)",
         "Each robot picks its action from the shaped scores.\n"
         "Stochastic during training, greedy during evaluation."),
        ("#e74c3c", "white",
         "PPO UPDATE  (train_rl.py — ppo_update())",
         "Collect T-step rollout → compute GAE → update policy + value head.\n"
         "Gradient clipping at 0.5.  Entropy bonus 0.01 keeps exploration alive."),
        ("#9b59b6", "white",
         "CURRICULUM CHECK  (train_rl.py — evaluate_isr())",
         "Every eval_interval: measure ISR.  Save checkpoint if improved.\n"
         "Advance to next curriculum stage when ISR ≥ threshold."),
    ]

    for i, (bg, fg, title, body) in enumerate(steps):
        y = 0.86 - i * 0.132
        fig.text(0.05, y,
                 f"{'─'*3} {title}\n    {body}",
                 fontsize=9.5, va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc=bg, ec="black"))
        fig.texts[-1].set_color(fg)
        if i < len(steps) - 1:
            fig.text(0.075, y - 0.025, "↓", fontsize=14, color="#555")

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 10 — KEY NUMBERS AND VOCAB CHEATSHEET
# ═══════════════════════════════════════════════════════════════════════
def page10_cheatsheet(pdf):
    fig = new_page(pdf, "Page 10: Key Numbers & Vocab Cheatsheet")

    txt(fig, 0.89,
        "VOCAB\n"
        "─────\n"
        "UNet         Big frozen brain (7.8M params).  Reads map, gives direction scores.\n"
        "Phi (φ)      Danger number per robot.  High = congested / bottleneck / conflict.\n"
        "Cost shaping Subtract a φ-based penalty from UNet scores before picking action.\n"
        "PPO          Trial-and-error training algorithm.\n"
        "GAE          Method to figure out which decisions were actually good ones.\n"
        "ISR          Individual Success Rate = how many robots reached their star.\n"
        "Curriculum   Train easy first, get harder when ISR passes 50%.",
        fc="#e2e3e5", ec="#6c757d", size=11)

    txt(fig, 0.53,
        "KEY NUMBERS\n"
        "───────────\n"
        "7,800,000   Parameters in the frozen UNet backbone\n"
        "6           Input channels per cell (walls, agents, goals, dist, grad_x, grad_y)\n"
        "5           Actions per robot  (stay, right, left, up, down)\n"
        "3           Phi signals  (density ×2.0, bottleneck ×3.0, conflict ×5.0)\n"
        "2           Proximity radius for cost shaping (L-inf distance)\n"
        "3           Curriculum stages  (4 → 8 → 16 robots)\n"
        "0.50        ISR threshold to advance curriculum stage\n"
        "0.99        γ  — how much we care about future rewards\n"
        "0.95        λ  — GAE balance between short/long horizon\n"
        "0.20        PPO clip ε  — maximum policy change per step\n"
        "3e-5        Learning rate",
        fc="#fff3cd", ec="#ffc107", size=11)

    txt(fig, 0.11,
        "To run training:   python train_rl.py --config configs/rl_ppo.yaml\n"
        "To generate PDFs:  python docs/craft_explained.py",
        fc="#d4edda", ec="#28a745", size=11)

    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    import os
    out_path = os.path.join(os.path.dirname(__file__), "craft_explained.pdf")
    with PdfPages(out_path) as pdf:
        page1_big_picture(pdf)
        page2_input_channels(pdf)
        page3_unet(pdf)
        page4_heuristic_phi(pdf)
        page5_cost_shaping(pdf)
        page6_ppo(pdf)
        page7_curriculum(pdf)
        page8_diagnostic(pdf)
        page9_pipeline(pdf)
        page10_cheatsheet(pdf)
    print(f"PDF saved to {out_path}")


if __name__ == "__main__":
    main()
