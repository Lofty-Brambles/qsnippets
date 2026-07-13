import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---- parameters ----
A = 0.25  # scale factor for wall half-height, adjust to taste
shock_x = (
    0.75  # location of the normal shock wave (must be > 0.5, in diverging section)
)
n_pts = 300
x = np.linspace(0, 1, n_pts)

half_height = A * (1 + 2.2 * (x - 0.5) ** 2)  # A*(1+2.2(x-0.5)^2)
y_top = half_height
y_bot = -half_height

wall_thickness = 0.35  # how far the hatched "solid wall" extends beyond the duct wall

# ---- colors ----
wall_fill = "white"
wall_edge = "black"
flow_fill = "white"
shock_color = "black"
axis_color = "black"
flow_color = "black"

plt.rcParams["hatch.linewidth"] = 0.45

fig, ax = plt.subplots(figsize=(8.4, 4.6))

# --- flow region background tint ---
ax.fill_between(x, y_bot, y_top, facecolor=flow_fill, zorder=1)

# --- hatched wall regions (solid material) ---
ax.fill_between(
    x,
    y_top,
    y_top + wall_thickness,
    facecolor=wall_fill,
    edgecolor="black",
    hatch="////",
    linewidth=0.0,
    zorder=2,
)
ax.fill_between(
    x,
    y_bot - wall_thickness,
    y_bot,
    facecolor=wall_fill,
    edgecolor="black",
    hatch="////",
    linewidth=0.0,
    zorder=2,
)

# --- wall boundary lines ---
ax.plot(x, y_top, color=wall_edge, linewidth=1.0, zorder=3)
ax.plot(x, y_bot, color=wall_edge, linewidth=1.0, zorder=3)

# --- throat marker ---
throat_top = A * 1.0 + wall_thickness
throat_bot = -A * 1.0 - wall_thickness
axis_y = -(A * 1.55 + wall_thickness + 0.22)
ax.plot(
    [0.5, 0.5],
    [A, axis_y - 0.025],
    color=axis_color,
    linestyle="--",
    linewidth=1.0,
    zorder=1,
)

# --- normal shock wave (two thin lines) ---
shock_half = A * (1 + 2.2 * (shock_x - 0.5) ** 2)
shock_dx = 0.0025
ax.plot(
    [shock_x - shock_dx, shock_x - shock_dx],
    [-shock_half, shock_half],
    color=shock_color,
    linewidth=0.9,
    zorder=4,
)
ax.plot(
    [shock_x + shock_dx, shock_x + shock_dx],
    [-shock_half, shock_half],
    color=shock_color,
    linewidth=0.9,
    zorder=4,
)
ax.text(
    shock_x - 0.03,
    throat_top + 0.10,
    "Normal shock wave",
    ha="center",
    va="bottom",
    fontsize=10,
    color=shock_color,
)
ax.annotate(
    "",
    xy=(shock_x, shock_half + 0.015),
    xytext=(shock_x - 0.03, throat_top + 0.082),
    arrowprops=dict(arrowstyle="-", color=shock_color, linewidth=0.9),
)

# --- flow arrow ---
arrow_y = A * 0.5
ax.annotate(
    "",
    xy=(0.88, arrow_y),
    xytext=(0.77, arrow_y),
    arrowprops=dict(arrowstyle="-|>", color=flow_color, linewidth=1.2),
)
ax.text(
    0.825,
    arrow_y + 0.04,
    "Flow",
    ha="center",
    va="bottom",
    fontsize=10,
    color=flow_color,
)

# --- dashed guides and distance d (figure-1 style) ---
ax.plot(
    [shock_x, shock_x],
    [shock_half, axis_y - 0.025],
    color=axis_color,
    linestyle="--",
    linewidth=1.0,
    alpha=1.0,
)

y_dim = axis_y + 0.018
dim_arrow = FancyArrowPatch(
    (0.5, y_dim),
    (shock_x, y_dim),
    arrowstyle="<->",
    mutation_scale=11,
    color=axis_color,
    lw=1.0,
)
ax.add_patch(dim_arrow)
ax.plot([0.5, 0.5], [axis_y - 0.02, y_dim + 0.015], color=axis_color, linewidth=1.0)
ax.plot(
    [shock_x, shock_x], [axis_y - 0.02, y_dim + 0.015], color=axis_color, linewidth=1.0
)
ax.text(
    (0.5 + shock_x) / 2,
    y_dim + 0.014,
    "d",
    ha="center",
    va="bottom",
    fontsize=10,
    color=axis_color,
)

# --- x-axis arrow (figure-1 style) ---
ax.annotate(
    "",
    xy=(1.04, axis_y),
    xytext=(-0.03, axis_y),
    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=axis_color),
)
ax.text(1.05, axis_y, "x", ha="left", va="center", fontsize=10, color=axis_color)

# --- formatting ---
ax.set_xlim(-0.05, 1.15)
ax.set_ylim(axis_y - 0.11, throat_top + 0.15)
ax.axis("off")

plt.tight_layout()
plt.savefig("nozzle_profile.png", dpi=300, bbox_inches="tight")
plt.show()
