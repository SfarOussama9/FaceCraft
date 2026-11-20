"""Generate the FaceCraft architecture diagram (Deliverable 2) as an editable .drawio file AND a PNG.

    python docs/diagrams/make_architecture.py

One description below (containers, nodes, edges) feeds both outputs so they always match:
  docs/diagrams/architecture.drawio   -> open/edit for free at https://app.diagrams.net (File > Open)
  docs/diagrams/architecture.png      -> for the report and slides
"""
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
W, H = 1880, 800

# colour themes: (fill, stroke)
BLUE, GREEN, YELLOW = ("#dae8fc", "#6c8ebf"), ("#d5e8d4", "#82b366"), ("#fff2cc", "#d6b656")
ORANGE, PURPLE, RED, GREY = ("#ffe6cc", "#d79b00"), ("#e1d5e7", "#9673a6"), ("#f8cecc", "#b85450"), ("#f5f5f5", "#999999")

# containers: id, title, x, y, w, h
CONTAINERS = [
    ("c1", "1. DATA SOURCES", 20, 60, 260, 640),
    ("c2", "2. PRE-PROCESSING  (data/)", 320, 60, 330, 640),
    ("c3", "3. MODELS — all trained from scratch  (models/, training/)", 920, 60, 560, 640),
    ("cg", "GAN baseline", 940, 240, 520, 445),
    ("c4", "4. EVALUATION & RESULTS  (evaluation/)", 1520, 60, 340, 640),
]

# nodes: id, label (\n = new line; first line bold), x, y, w, h, colours, dashed
NODES = [
    ("celeba", "CelebA\n202,599 face photos\n40 yes/no attributes\nofficial train/val/test split", 40, 100, 220, 110, BLUE, False),
    ("dialog", "CelebA-Dialog\n0–5 levels: bangs, glasses,\nbeard, smile, age", 40, 290, 220, 95, BLUE, False),
    ("f2t", "Face2Text v2.1\n18,690 descriptions\nwritten by people", 40, 460, 220, 95, BLUE, False),

    ("img", "Image pipeline\ncrop 140×140 → 64×64\nscale to [-1, 1], random flip", 340, 95, 290, 85, GREEN, False),
    ("cache", "Image cache\n2.5 GB uint8 array, memory-mapped", 340, 205, 290, 60, GREEN, False),
    ("cap", "Caption engine\ntemplate · Dialog degrees · human\nnew random caption every time\nrules: never contradictory", 340, 300, 290, 105, GREEN, False),
    ("tok", "Tokenizer\nCLIP BPE vocabulary → 77 ids", 340, 435, 290, 60, GREEN, False),
    ("held", "Held-out pairs (compositional test)\n3 attribute pairs removed from training", 340, 530, 290, 75, GREEN, False),

    ("ds", "FaceCraftDataset\n(image, token ids,\n40 attributes, source)\n\n155,309 training items", 680, 250, 200, 165, YELLOW, False),

    ("clf", "Attribute classifier — 'the judge'\nResNet-18, 40 outputs\nused ONLY to check generated faces", 940, 100, 300, 100, ORANGE, False),
    ("txt", "Text encoder\nTransformer (4 layers)\n→ sentence vector", 960, 290, 200, 90, ORANGE, False),
    ("z", "Random noise z\n(128 numbers)", 960, 420, 200, 60, ORANGE, False),
    ("gen", "Generator (the forger)\nconditional BatchNorm\n4 × upsample → 64×64 face", 1230, 290, 210, 100, ORANGE, False),
    ("disc", "Discriminator (the detective)\nspectral norm + projection\nreal & matching caption?", 1230, 470, 210, 100, ORANGE, False),
    ("loss", "Training rule\nhinge loss: real ✓ · fake ✗\nwrong caption ✗\n5 D steps per G step · EMA", 960, 560, 200, 110, ORANGE, False),

    ("acc", "Attribute accuracy\ndoes the face have what\nthe prompt asked for?", 1540, 100, 300, 90, PURPLE, False),
    ("fid", "FID / KID (realism)\nInceptionV3 features,\n5,000 generated vs real test faces", 1540, 230, 300, 90, PURPLE, False),
    ("res", "Results — Deliverable 2\nClassifier: 91.5 % (baseline 79.9 %)\nGAN: FID 48.7 · KID 0.037\nfollows gender, hair, smile, beard;\nmisses glasses, bald, gray hair", 1540, 370, 300, 140, RED, False),
    ("next", "Next — Deliverable 3\nDiffusion U-Net + cross-attention\n+ classifier-free guidance\n→ compared against the GAN", 1540, 570, 300, 110, GREY, True),
]

# edges: source, target, label, waypoints [(x, y), ...], dashed
EDGES = [
    ("celeba", "img", "pixels", [], False),
    ("img", "cache", "", [], False),
    ("celeba", "cap", "labels", [], False),
    ("dialog", "cap", "", [], False),
    ("f2t", "cap", "", [], False),
    ("cap", "tok", "", [], False),
    ("cache", "ds", "", [], False),
    ("tok", "ds", "", [(660, 465), (660, 385)], False),
    ("held", "ds", "", [(670, 567), (670, 405)], False),
    ("ds", "clf", "images + labels", [], False),
    ("ds", "txt", "token ids", [], False),
    ("ds", "disc", "real faces", [(780, 740), (1335, 740)], False),
    ("txt", "gen", "sentence", [], False),
    ("z", "gen", "", [], False),
    ("gen", "disc", "fake face", [], False),
    ("txt", "disc", "", [], False),
    ("disc", "loss", "", [], False),
    ("clf", "acc", "judges faces", [], False),
    ("gen", "fid", "generated faces", [], False),
    ("fid", "res", "", [], False),
    ("acc", "res", "", [(1855, 145), (1855, 440)], False),
    ("next", "res", "will beat?", [], True),
]

NODE = {n[0]: n for n in NODES}


def _center(n):
    _, _, x, y, w, h, *_ = n
    return x + w / 2, y + h / 2


def _border_point(n, toward):
    """Point where the line from the node centre toward `toward` leaves the rectangle."""
    _, _, x, y, w, h, *_ = n
    cx, cy = _center(n)
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


# ---------------------------------------------------------------- draw.io XML
def write_drawio(path: Path):
    cells, nid = [], 2

    def cid():
        nonlocal nid
        nid += 1
        return f"n{nid}"

    ids = {}
    for key, title, x, y, w, h in CONTAINERS:
        ids[key] = cid()
        style = ("rounded=1;arcSize=3;whiteSpace=wrap;html=1;verticalAlign=top;align=left;spacingLeft=10;"
                 "spacingTop=4;fontStyle=1;fontSize=13;fillColor=#fbfbfb;strokeColor=#b0b0b0;dashed=0;")
        if key == "cg":
            style = style.replace("#fbfbfb", "#fff8ef").replace("#b0b0b0", "#d79b00")
        cells.append(f'<mxCell id="{ids[key]}" value="{escape(title)}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for key, label, x, y, w, h, (fill, stroke), dashed in NODES:
        ids[key] = cid()
        first, *rest = label.split("\n")
        html = f"<b>{escape(first)}</b>" + "".join(f"<br>{escape(r)}" for r in rest)
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=12;"
                 f"{'dashed=1;' if dashed else ''}")
        cells.append(f'<mxCell id="{ids[key]}" value="{escape(html, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for src, dst, label, points, dashed in EDGES:
        style = ("endArrow=block;endFill=1;html=1;strokeColor=#555555;fontSize=11;labelBackgroundColor=#ffffff;"
                 f"edgeStyle=none;{'dashed=1;' if dashed else ''}")
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        array = f'<Array as="points">{pts}</Array>' if pts else ""
        geo = f'<mxGeometry relative="1" as="geometry">{array}</mxGeometry>'
        cells.append(f'<mxCell id="{cid()}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
                     f'source="{ids[src]}" target="{ids[dst]}">{geo}</mxCell>')
    title = ('<mxCell id="n2" value="&lt;b&gt;FaceCraft — Deliverable 2 architecture&lt;/b&gt;: text-to-face data pipeline, '
             'judge and GAN baseline" style="text;html=1;fontSize=18;align=left;" vertex="1" parent="1">'
             f'<mxGeometry x="20" y="10" width="{W - 40}" height="36" as="geometry"/></mxCell>')
    xml = (f'<mxfile host="FaceCraft"><diagram name="Architecture"><mxGraphModel dx="{W}" dy="{H}" grid="1" '
           f'gridSize="10" page="1" pageWidth="{W}" pageHeight="{H}"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           f'{title}{"".join(cells)}</root></mxGraphModel></diagram></mxfile>')
    path.write_text(xml, encoding="utf-8")


# ---------------------------------------------------------------- PNG (matplotlib)
def write_png(path: Path):
    fig = plt.figure(figsize=(W / 100, (H + 30) / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H + 30, 0)
    ax.axis("off")
    ax.text(20, 30, "FaceCraft — Deliverable 2 architecture", fontsize=15, fontweight="bold", va="center")
    ax.text(470, 30, ": text-to-face data pipeline, judge and GAN baseline", fontsize=13, va="center")
    for key, title, x, y, w, h in CONTAINERS:
        fill, stroke = ("#fff8ef", "#d79b00") if key == "cg" else ("#fbfbfb", "#b0b0b0")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=fill, ec=stroke, lw=1.2))
        ax.text(x + 10, y + 16, title, fontsize=9.5, fontweight="bold", va="center", color="#333333")
    for key, label, x, y, w, h, (fill, stroke), dashed in NODES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc=fill, ec=stroke,
                                    lw=1.4, ls="--" if dashed else "-"))
        first, *rest = label.split("\n")
        n_lines = 1 + len(rest)
        line_h = 15
        top = y + h / 2 - (n_lines - 1) * line_h / 2
        ax.text(x + w / 2, top, first, ha="center", va="center", fontsize=8.6, fontweight="bold")
        for k, r in enumerate(rest, start=1):
            ax.text(x + w / 2, top + k * line_h, r, ha="center", va="center", fontsize=7.8)
    for src, dst, label, points, dashed in EDGES:
        s, d = NODE[src], NODE[dst]
        first_target = points[0] if points else _center(d)
        last_origin = points[-1] if points else _center(s)
        path_pts = [_border_point(s, first_target)] + list(points) + [_border_point(d, last_origin)]
        style = dict(color="#555555", lw=1.2, ls=(0, (4, 3)) if dashed else "-")
        for a, b in zip(path_pts[:-2], path_pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], **style)
        ax.add_patch(FancyArrowPatch(path_pts[-2], path_pts[-1], arrowstyle="-|>", mutation_scale=12,
                                     color="#555555", lw=1.2, linestyle="--" if dashed else "-", shrinkA=0, shrinkB=1))
        if label:
            # put the label on the longest segment
            seg = max(zip(path_pts[:-1], path_pts[1:]), key=lambda p: (p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2)
            mx, my = (seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2
            ax.text(mx, my, label, fontsize=7.2, ha="center", va="center", style="italic",
                    bbox=dict(fc="white", ec="none", pad=1.2))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    write_drawio(OUT / "architecture.drawio")
    write_png(OUT / "architecture.png")
    print("wrote", OUT / "architecture.drawio", "and", OUT / "architecture.png")
