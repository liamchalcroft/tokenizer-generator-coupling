from pathlib import Path

import matplotlib as mpl

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TABLES = RESULTS / "tables"
INPUTS = REPO / "figures" / "inputs"
OUT = REPO / "figures" / "out"

PALETTES = {
    "oxide": {"VQ": "#B4472E", "LFQ": "#2B6CB0", "FSQ": "#B8860B", "Continuous": "#8A3D6B"},
    "lightbox": {"VQ": "#D06A4C", "LFQ": "#2D5E9A", "FSQ": "#8C8A2A", "Continuous": "#9A4F7A"},
}
MARKERS = {"VQ": "^", "LFQ": "o", "FSQ": "s", "Continuous": "D"}

INK = "#1F1F1F"
MUTED = "#6B6B6B"
RULE = "#D9D6D0"

# NeurIPS text width is 5.5 in; figures are drawn at final size so 8 pt text stays 8 pt.
TEXT_WIDTH = 5.5


def apply(palette="oxide"):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "text.color": INK,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
    return PALETTES[palette]
