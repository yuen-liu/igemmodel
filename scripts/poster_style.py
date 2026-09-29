"""Shared palette and type sizes for the poster figures.

Swap the hexes here and every poster figure follows.
"""

NAVY = "#1D4F91"
COLUMBIA = "#75AADB"
NOISE_GRAY = "#8C8C8C"

# Accents, used where a figure needs more series than the three-color palette
# covers (e.g. the UMAP's micro-clusters).
ACCENTS = ["#E9A03B", "#2A9D8F", "#8E4585"]

# Tuned for reading at poster distance.
TICK_SIZE = 18
LABEL_SIZE = 22
LEGEND_SIZE = 17


def strip_spines(ax, keep=("left", "bottom")):
    """Drop the frame box, keep only the axes we want, and thicken those."""
    for side in ("top", "right", "bottom", "left"):
        if side in keep:
            ax.spines[side].set_linewidth(1.2)
        else:
            ax.spines[side].set_visible(False)
