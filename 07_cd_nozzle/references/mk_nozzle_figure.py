"""
Draws the converging-diverging (CD) nozzle geometry with a normal shock,
in the style of the textbook schematic (hatched duct walls, a vertical
normal-shock line, a "Flow" arrow, and a "d" dimension between the throat
and the shock).

This is a schematic / geometry figure only (panel "a" style) -- it does not
plot flow quantities; use nozzle_shock.py for the M/M0, p/p0, rho/rho0, T/T0
analytical curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ----------------------------------------------------------------------
# Nozzle geometry (same area law used in nozzle_shock.py)
#   A(x)/A_throat = 1 + 2.2*(x - 0.5)^2 ,  x in [0, 1], throat at x = 0.5
# ----------------------------------------------------------------------
x_min, x_throat, x_max = 0.0, 0.5, 1.0
x_shock = 0.75  # normal shock location


def area_ratio(x):
    return 0.05 * (1.0 + 2.2 * (x - 0.5) ** 2)


def half_height(x):
    """Duct half-height, taken proportional to sqrt(A/A*) (axisymmetric duct)."""
    return np.sqrt(area_ratio(x))


x = np.linspace(x_min, x_max, 400)
h = half_height(x)

# Outer (solid wall) boundary, drawn a bit further out than the duct wall
h_outer = h + 0.35

fig, ax = plt.subplots(figsize=(8, 5))

# ---- Hatched solid walls (top and bottom) ----
ax.fill_between(x, h, h_outer, facecolor="white", edgecolor="black",
                 hatch="////", linewidth=1.0)
ax.fill_between(x, -h_outer, -h, facecolor="white", edgecolor="black",
                 hatch="////", linewidth=1.0)

# ---- Duct wall outlines ----
ax.plot(x, h, color="black", lw=1.5)
ax.plot(x, -h, color="black", lw=1.5)

# ---- Normal shock: vertical hatched line spanning the duct at x_shock ----
h_shock = half_height(np.array([x_shock]))[0]
ax.fill_betweenx([-h_shock, h_shock], x_shock - 0.008, x_shock + 0.008,
                  facecolor="black", edgecolor="black", hatch="||||")

# ---- Flow direction arrow ----
ax.annotate("", xy=(0.93, 0.55 * h.max()), xytext=(0.78, 0.55 * h.max()),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"))
ax.text(0.855, 0.55 * h.max() + 0.08, "Flow", ha="center", fontsize=11)

# ---- Label for the normal shock wave ----
ax.annotate("Normal shock wave", xy=(x_shock, h_shock + 0.35),
            xytext=(x_shock - 0.05, h_outer.max() + 0.35),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", lw=1))

# ---- Dashed vertical guide lines from throat and shock down to axis ----
y_bottom = -h_outer.max() - 0.55
ax.plot([x_throat, x_throat], [-h[np.argmin(np.abs(x - x_throat))], y_bottom],
        color="gray", ls="--", lw=1)
ax.plot([x_shock, x_shock], [-h_shock, y_bottom],
        color="gray", ls="--", lw=1)

# ---- "d" dimension arrow between throat and shock ----
y_dim = y_bottom + 0.15
arrow = FancyArrowPatch((x_throat, y_dim), (x_shock, y_dim),
                         arrowstyle="<|-|>", mutation_scale=12,
                         color="black", lw=1.2)
ax.add_patch(arrow)
ax.text((x_throat + x_shock) / 2, y_dim + 0.08, "d", ha="center", fontsize=11)

# ---- x-axis ----
ax.annotate("", xy=(x_max + 0.05, y_bottom), xytext=(x_min - 0.03, y_bottom),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
ax.text(x_max + 0.06, y_bottom, "x", fontsize=11, va="center")

ax.set_xlim(x_min - 0.05, x_max + 0.12)
ax.set_ylim(y_bottom - 0.2, h_outer.max() + 0.7)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout()
fig.savefig("nozzle_geometry.svg")
print("Saved nozzle_geometry.svg")