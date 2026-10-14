import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""qa_bridge2.py -- QA figures for the REDESIGNED davinci_bridge2 generator
(n_bays=24, span=20.0, width=6.0, lam=0.25; see
reciprocal_geom_v2.davinci_bridge2's docstring and
results/bridge2_verification.json). Thick lines / clear support
markers per the redesign QA spec (this bridge is small enough -- 118
physical members -- that the earlier project's thin-line convention made
supports/members hard to read at print size).

Produces:
  figures_v2/qa_bridge2_3d.png
  figures_v2/qa_bridge2_elev.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from reciprocal_geom_v2 import davinci_bridge2, physical_members_v2
from reciprocal_geom import true_node_positions

OUT_FIG = _os_bootstrap.path.join(ROOT, "figures_v2")


def make_model():
    return davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)


def plot_bridge2(model, mode, fname, elev=20, azim=-58):
    true_pos = true_node_positions(model)
    members = physical_members_v2(model)
    supports = np.array([model.nodes[i] for i in model.support_idx])

    fig = plt.figure(figsize=(9, 8) if mode == "3d" else (11, 4.6))
    if mode == "3d":
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    n_long = sum(1 for m in members if m['member_id'][0] == 'long')
    n_rung = sum(1 for m in members if m['member_id'][0] == 'rung')

    for m in members:
        p0 = true_pos[m['chain'][0]]
        p1 = true_pos[m['chain'][-1]]
        is_long = (m['member_id'][0] == 'long')
        color = 'tab:blue' if is_long else 'tab:orange'
        lw = 2.4 if is_long else 2.0
        if mode == "3d":
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, lw=lw)
        elif mode == "elev":
            ax.plot([p0[0], p1[0]], [p0[2], p1[2]], color=color, lw=lw)

    if mode == "3d":
        ax.plot([], [], color='tab:blue', lw=2.4, label=f'longitudinal rail ({n_long})')
        ax.plot([], [], color='tab:orange', lw=2.0, label=f'rung half-segment ({n_rung})')
        ax.scatter(supports[:, 0], supports[:, 1], supports[:, 2], color='red',
                   marker='^', s=160, edgecolor='black', linewidth=1.2,
                   label=f'pinned abutment ({len(supports)})', depthshade=False, zorder=10)
        rng = true_pos.max(axis=0) - true_pos.min(axis=0)
        rng = np.maximum(rng, 1e-6)
        aspect = rng / rng.max()
        aspect[2] = max(aspect[2], 0.28)
        ax.set_box_aspect(list(aspect))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x [m]  (span)')
        ax.set_ylabel('y [m]  (width)')
        ax.set_zlabel('z [m]  (rise)')
        ax.set_title(f"davinci_bridge2 -- 3D view (n_bays=24, {len(members)} physical members, "
                     f"{model.n_groups} groups, {len(model.nodes)} nodes)", fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.plot([], [], color='tab:blue', lw=2.4, label=f'longitudinal rail ({n_long})')
        ax.scatter(supports[:, 0], supports[:, 2], color='red', marker='^', s=200,
                   edgecolor='black', linewidth=1.2,
                   label=f'pinned abutment ({len(supports)})', zorder=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x [m]  (span)')
        ax.set_ylabel('z [m]  (rise)')
        ax.set_title(f"davinci_bridge2 -- elevation (n_bays=24, {len(members)} physical members, "
                     f"3 rails superposed)", fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.text(0.02, 0.04, f"{n_rung} rung half-segments span across width (y) at each "
                             "station -- perpendicular to this x-z view, so they project "
                             "to points and are not separately visible here (see the 3D "
                             "view for the rung weave).",
                transform=ax.transAxes, fontsize=7.5, style='italic',
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/{fname}", dpi=170)
    plt.close(fig)
    print("saved", f"{OUT_FIG}/{fname}")


def main():
    model = make_model()
    plot_bridge2(model, "3d", "qa_bridge2_3d.png", elev=20, azim=-58)
    plot_bridge2(model, "elev", "qa_bridge2_elev.png")


if __name__ == "__main__":
    main()
