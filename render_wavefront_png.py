"""
render_wavefront_png.py
Renders the wavefront_data.json as a high-quality static PNG
that closely matches the WebGL WavefrontViewer appearance.

Usage: python render_wavefront_png.py
Output: resultviewer.png  (in project root)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────

with open('brain-viewer/public/wavefront_data.json', 'r') as f:
    data = json.load(f)

verts     = np.array(data['vertices'])    # (N, 3)
faces     = np.array(data['faces'])       # (F, 3)
t_peak    = np.array(data['t_peak'])      # (N,)
t_norm    = np.array(data['t_peak_norm']) # (N,) in [0,1]
regions   = data['regions']
meta      = data['meta']

t_min = meta['t_peak_min']
t_max = meta['t_peak_max']

# ── Custom cool-warm colormap matching GLSL shader ───────────────────────

CMAP_COLORS = [
    (0.00, (0.00, 0.20, 1.00)),  # deep blue  (V1 / fast)
    (0.20, (0.00, 0.75, 1.00)),  # cyan
    (0.45, (0.00, 1.00, 0.55)),  # mint green
    (0.65, (0.90, 1.00, 0.00)),  # yellow
    (0.85, (1.00, 0.45, 0.00)),  # orange
    (1.00, (1.00, 0.08, 0.05)),  # red        (V4 / slow)
]
cmap = LinearSegmentedColormap.from_list(
    'wavefront',
    [(pos, col) for pos, col in CMAP_COLORS],
    N=512,
)

# ── Compute per-face colours (average of vertex t_norms) ─────────────────

face_t = t_norm[faces].mean(axis=1)         # (F,)
face_colors_rgba = cmap(face_t)             # (F, 4)

# ── Wavefront "freeze" time: show wavefront at 55% sweep ─────────────────
# Mimics the glowing band mid-animation
WAVEFRONT_T = t_min + 0.50 * (t_max - t_min)
SIGMA        = 0.22   # seconds

# Scale down alpha for faces ahead of wavefront (unrevealed)
reveal = np.clip((WAVEFRONT_T - face_t * (t_max - t_min) - t_min) / (SIGMA * 3), 0, 1)

# Per-face wavefront glow
face_t_actual = t_min + face_t * (t_max - t_min)
d_glow        = face_t_actual - WAVEFRONT_T
glow          = np.exp(-(d_glow ** 2) / (2 * SIGMA ** 2))

# Blend base colour toward white at wavefront
r = face_colors_rgba[:, 0] * (1 - glow * 0.65) + glow * 0.65
g = face_colors_rgba[:, 1] * (1 - glow * 0.65) + glow * 0.65
b = face_colors_rgba[:, 2] * (1 - glow * 0.65) + glow * 0.65
alpha_reveal = 0.12 + 0.78 * np.clip(
    (WAVEFRONT_T - face_t_actual) / (SIGMA * 4) + 0.5, 0, 1
) + glow * 0.10
alpha_reveal = np.clip(alpha_reveal, 0.05, 0.95)

face_colors_final = np.stack([r, g, b, alpha_reveal], axis=1)

# ── Figure setup ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10), facecolor='#02010a')
ax  = fig.add_subplot(111, projection='3d', facecolor='#02010a')

# ── Build Poly3DCollection (solid mesh) ───────────────────────────────────

tris = verts[faces]   # (F, 3, 3) — triangle vertex coordinates

# Sort faces back-to-front by mean Z for painter's algorithm
mean_z = tris[:, :, 2].mean(axis=1)
order  = np.argsort(mean_z)

poly = Poly3DCollection(
    tris[order],
    facecolors=face_colors_final[order],
    edgecolors='none',
    linewidth=0,
    antialiased=False,
    zsort='average',
)
ax.add_collection3d(poly)

# ── Iso-contour edge overlay ──────────────────────────────────────────────
# Find edges where t_peak crosses iso levels

ISO_LEVELS = np.linspace(t_min, t_max, 14)[1:-1]

for lvl in ISO_LEVELS:
    revealed = lvl <= WAVEFRONT_T + SIGMA * 1.0
    if not revealed:
        continue
    alpha_iso = 0.45 * np.clip((WAVEFRONT_T - lvl) / (SIGMA * 5) + 1.0, 0, 1)
    if alpha_iso < 0.05:
        continue

    # Find edges crossing this iso level
    t_v  = t_peak[faces]                   # (F, 3)
    a_lt = t_v < lvl
    b_lt = ~a_lt

    # Each face can contribute 0 or 2 crossing points
    segs = []
    for fi in range(len(faces)):
        vti = t_v[fi]
        vi  = faces[fi]
        pts = []
        for e0, e1 in [(0,1),(1,2),(2,0)]:
            ta, tb = vti[e0], vti[e1]
            if (ta < lvl) != (tb < lvl):
                frac = (lvl - ta) / (tb - ta + 1e-12)
                p = verts[vi[e0]] + frac * (verts[vi[e1]] - verts[vi[e0]])
                pts.append(p)
        if len(pts) == 2:
            segs.append(pts)

    if segs:
        segs = np.array(segs)   # (S, 2, 3)
        xs = segs[:, :, 0].T
        ys = segs[:, :, 1].T
        zs = segs[:, :, 2].T
        for i in range(segs.shape[0]):
            ax.plot(xs[:, i], ys[:, i], zs[:, i],
                    color='white', alpha=alpha_iso * 0.55, linewidth=0.5)

# ── Wavefront ring (glow) ─────────────────────────────────────────────────
# Draw the active wavefront iso-contour extra-bright

wf_segs = []
for fi in range(len(faces)):
    vti = t_v[fi]
    vi  = faces[fi]
    pts = []
    for e0, e1 in [(0,1),(1,2),(2,0)]:
        ta, tb = vti[e0], vti[e1]
        if (ta < WAVEFRONT_T) != (tb < WAVEFRONT_T):
            frac = (WAVEFRONT_T - ta) / (tb - ta + 1e-12)
            p = verts[vi[e0]] + frac * (verts[vi[e1]] - verts[vi[e0]])
            pts.append(p)
    if len(pts) == 2:
        wf_segs.append(pts)

if wf_segs:
    wf_segs = np.array(wf_segs)
    for i in range(wf_segs.shape[0]):
        ax.plot(wf_segs[i, :, 0], wf_segs[i, :, 1], wf_segs[i, :, 2],
                color='#fffaaa', alpha=0.75, linewidth=1.8, zorder=5)
    # Glow pass (slightly wider, lower alpha)
    for i in range(wf_segs.shape[0]):
        ax.plot(wf_segs[i, :, 0], wf_segs[i, :, 1], wf_segs[i, :, 2],
                color='#ffeeaa', alpha=0.20, linewidth=4.5, zorder=4)

# ── Region labels ─────────────────────────────────────────────────────────

REGION_COL = {'V1': '#ff2dca', 'V2': '#ff8c00', 'V3': '#00e5ff', 'V4': '#39ff14'}
for r in regions:
    cx, cy, cz = r['centroid']
    name       = r['name']
    col        = REGION_COL.get(name, '#ffffff')
    ax.scatter([cx], [cy], [cz], color=col, s=32, zorder=10,
               edgecolors='white', linewidths=0.4)
    ax.text(cx + 0.08, cy + 0.08, cz + 0.10,
            f"{name}  {r['t_peak']:.1f}s",
            color=col, fontsize=8.5, fontfamily='monospace',
            fontweight='bold', zorder=11)

# ── Axis / view settings ──────────────────────────────────────────────────

ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.8, 1.8)
ax.set_axis_off()
ax.view_init(elev=22, azim=-60)
ax.set_box_aspect([1.4, 1.0, 1.2])
ax.dist = 7.0

# ── Colour-bar ────────────────────────────────────────────────────────────

cbar_ax = fig.add_axes([0.36, 0.06, 0.28, 0.018])
sm      = plt.cm.ScalarMappable(cmap=cmap,
                                 norm=plt.Normalize(vmin=t_min, vmax=t_max))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.ax.set_facecolor('#02010a')
cb.outline.set_edgecolor('#444')
cb.set_label('Hemodynamic Delay  t_peak  (s)', color='#aaa',
             fontsize=8, fontfamily='monospace', labelpad=4)
cb.ax.tick_params(colors='#999', labelsize=7)
plt.setp(cb.ax.get_xticklabels(), fontfamily='monospace')

# ── Info panel (text) ─────────────────────────────────────────────────────

info_lines = [
    'SPATIOTEMPORAL WAVEFRONT',
    '─────────────────────────────',
    '■  Icosphere mesh  2562v · 5120f',
    '■  Geodesic Dijkstra  (V1→V4)',
    '■  Laplacian-smoothed t_peak field',
    '■  Wavefront + iso-contour GLSL',
    '■  16 iso levels  · 5 s cycle',
    '',
    f'  t_peak  [{t_min:.2f} s → {t_max:.2f} s]',
]
fig.text(0.015, 0.97, '\n'.join(info_lines),
         va='top', ha='left', color='#cccccc',
         fontsize=7.5, fontfamily='monospace',
         linespacing=1.65,
         bbox=dict(boxstyle='round,pad=0.6',
                   facecolor='#0a0a14', edgecolor='#333', alpha=0.85))

# Region legend
legend_patches = [
    mpatches.Patch(color=REGION_COL['V1'], label='V1 — Primary Visual   4.0 s'),
    mpatches.Patch(color=REGION_COL['V2'], label='V2 — Secondary Visual 5.5 s'),
    mpatches.Patch(color=REGION_COL['V3'], label='V3 — Tertiary Visual  6.8 s'),
    mpatches.Patch(color=REGION_COL['V4'], label='V4 — Ventral Stream   8.0 s'),
]
leg = fig.legend(handles=legend_patches,
                 loc='lower right',
                 framealpha=0.82,
                 facecolor='#0a0a14',
                 edgecolor='#333',
                 fontsize=8,
                 prop={'family': 'monospace', 'size': 8},
                 labelcolor='white')

# ── Title ─────────────────────────────────────────────────────────────────

fig.text(0.50, 0.96,
         'Spatiotemporal Hemodynamic Wavefront  ·  V1 → V4  propagation',
         ha='center', va='top', color='white',
         fontsize=12, fontfamily='monospace', fontweight='bold')
fig.text(0.50, 0.93,
         'Geodesic Dijkstra  ·  Laplacian Surface Smoothing  ·  GLSL Iso-Contour Rendering',
         ha='center', va='top', color='#888',
         fontsize=8, fontfamily='monospace')

# ── Save ──────────────────────────────────────────────────────────────────

out = 'resultviewer.png'
plt.savefig(out, dpi=130, bbox_inches='tight',
            facecolor='#02010a', edgecolor='none')
plt.close()
print(f"Saved: {out}")
