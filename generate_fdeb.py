"""
SIGGRAPH 2026 — fMRI Brain Network FDEB Pre-computation
========================================================
Generates a mock 200-voxel fMRI dataset spread across four sequential
visual cortex regions (V1–V4), creates directed edges between adjacent
regions, runs a simplified Force-Directed Edge Bundling (FDEB) simulation,
and exports a lightweight JSON file for the React Three Fiber renderer.

Reference: Holten & van Wijk, "Force-Directed Edge Bundling for Graph
           Visualization", EuroVis 2009.

Output
------
  network_data.json
    nodes         : [{id, x, y, z, region, t_peak}, …]
    bundled_edges : [{src_id, tgt_id, t_peak_src, t_peak_tgt,
                      control_points: [[x,y,z], …]}, …]
"""

import json
import numpy as np

np.random.seed(42)

# ── 1. VOXEL GENERATION ───────────────────────────────────────────────────────

REGIONS      = ['V1', 'V2', 'V3', 'V4']
N_PER_REGION = 50       # voxels per region → 200 total
SPREAD       = 0.55     # Gaussian std-dev of each cluster (world units)

# Cluster centres define the spatial V1 → V4 flow (left-to-right on x-axis)
CENTERS = {
    'V1': np.array([-3.0,  0.0,  0.0]),
    'V2': np.array([-1.0,  0.45, 0.3]),
    'V3': np.array([ 1.0,  0.45, 0.3]),
    'V4': np.array([ 3.0,  0.0,  0.0]),
}

# HRF t_peak (s): V1 activates first, V4 last — models visual information flow.
# Values chosen so the sweep spans ~4 s (typical BOLD response).
T_PEAK_MEAN  = {'V1': 4.0, 'V2': 5.33, 'V3': 6.67, 'V4': 8.0}
T_PEAK_SIGMA = 0.25     # per-voxel biological variability (seconds)

nodes = []
for region in REGIONS:
    center = CENTERS[region]
    for _ in range(N_PER_REGION):
        pos    = center + np.random.randn(3) * SPREAD
        t_peak = T_PEAK_MEAN[region] + np.random.randn() * T_PEAK_SIGMA
        t_peak = float(np.clip(t_peak, 2.0, 10.0))
        nodes.append({
            'id':     len(nodes),
            'x':      round(float(pos[0]), 4),
            'y':      round(float(pos[1]), 4),
            'z':      round(float(pos[2]), 4),
            'region': region,
            't_peak': round(t_peak, 4),
        })

print(f"Nodes generated : {len(nodes)}")

# ── 2. DIRECTED EDGES (V1→V2, V2→V3, V3→V4) ─────────────────────────────────
N_NEIGHBORS = 3   # each source voxel connects to its K nearest targets

by_region = {r: [n for n in nodes if n['region'] == r] for r in REGIONS}
raw_edges  = []   # list of (src_dict, tgt_dict)

for i in range(len(REGIONS) - 1):
    src_reg, tgt_reg = REGIONS[i], REGIONS[i + 1]
    for src in by_region[src_reg]:
        sp = np.array([src['x'], src['y'], src['z']])
        ranked = sorted(
            by_region[tgt_reg],
            key=lambda t: np.linalg.norm(sp - np.array([t['x'], t['y'], t['z']]))
        )
        for tgt in ranked[:N_NEIGHBORS]:
            raw_edges.append((src, tgt))

print(f"Edges created   : {len(raw_edges)}")

# ── 3. FDEB SIMULATION ────────────────────────────────────────────────────────
#
# Each edge e_i is discretised into P+1 control points  p_0 … p_P
# (endpoints p_0 and p_P are fixed).  Two forces act on each interior point:
#
#   (a) Spring tension (discrete Laplacian):
#         F_s[pi] = k_s · (p[pi-1] + p[pi+1] - 2·p[pi])
#       Pulls each interior point toward the midpoint of its neighbours,
#       damping over-bundling and preserving the edge's original direction.
#
#   (b) Electrostatic bundling attraction:
#         F_e[pi] = k_e · Σ_j  C(e_i, e_j) · (q_j[pi] - p_i[pi])
#       Attracts compatible edge pairs toward each other.
#       C(e_i, e_j) ∈ [0,1] weights the pull by geometric similarity.
#       Edges below C_THRESH are ignored to prevent unrelated bundles merging.
#
#   Combined compatibility  C = C_a · C_s · C_p  where:
#     C_a  (angle)    = |cos θ|  — parallel edges score highest
#     C_s  (scale)    = harmonic mean of length ratio — similar-length edges
#     C_p  (position) = avg_len / (avg_len + dist_between_midpoints) — nearby edges

P_LEVELS   = [2, 4, 8]   # segment counts for each hierarchy level (coarse → fine)
ITERS      = 40           # force integration steps per level
K_SPRING   = 0.12         # spring tension coefficient
K_ELECTRO  = 0.06         # electrostatic bundling coefficient
C_THRESH   = 0.35         # minimum compatibility to attract
MAX_STEP   = 0.5          # max control-point displacement per iteration (clamping)


def make_ctrl_pts(src: dict, tgt: dict, p: int) -> np.ndarray:
    """Linearly interpolate p+1 control points from src to tgt."""
    s = np.array([src['x'], src['y'], src['z']], dtype=float)
    t = np.array([tgt['x'], tgt['y'], tgt['z']], dtype=float)
    return np.array([s + alpha * (t - s) for alpha in np.linspace(0, 1, p + 1)])


def subdivide(pts: np.ndarray) -> np.ndarray:
    """
    Double the number of segments by inserting midpoints between every
    consecutive pair of control points (endpoints stay fixed).
    P segments (P+1 points) -> 2P segments (2P+1 points).
    """
    result = []
    for k in range(len(pts) - 1):
        result.append(pts[k])
        result.append((pts[k] + pts[k + 1]) / 2.0)
    result.append(pts[-1])
    return np.array(result)


def compatibility(pts_i: np.ndarray, pts_j: np.ndarray) -> float:
    """
    Combined compatibility score C in [0, 1].

    C_a (angle):    |cos theta| between edge direction vectors.
    C_s (scale):    2*l1*l2 / (l1^2 + l2^2)  -- Holten eq. (5).
                    Equals 1 when lengths match; penalises length mismatches.
    C_p (position): avg_len / (avg_len + dist(mid_i, mid_j)).
    """
    v1 = pts_i[-1] - pts_i[0]
    v2 = pts_j[-1] - pts_j[0]
    l1 = np.linalg.norm(v1) + 1e-9
    l2 = np.linalg.norm(v2) + 1e-9

    # Angle compatibility
    C_a = abs(float(np.dot(v1, v2)) / (l1 * l2))

    # Scale compatibility -- Holten's original formula
    C_s = 2.0 * l1 * l2 / (l1 * l1 + l2 * l2)

    # Position compatibility
    mid1  = (pts_i[0] + pts_i[-1]) / 2.0
    mid2  = (pts_j[0] + pts_j[-1]) / 2.0
    avg_l = (l1 + l2) / 2.0
    C_p   = avg_l / (avg_l + np.linalg.norm(mid1 - mid2))

    return C_a * C_s * C_p


def run_fdeb_level(
    ctrl: list, E: int,
    compat_adj: dict,
    P: int, iters: int,
    k_spring: float, k_electro: float,
    max_step: float,
) -> list:
    """Run `iters` force-integration steps at segment count P. Returns updated ctrl."""
    for it in range(iters):
        new_ctrl = [pts.copy() for pts in ctrl]

        for i in range(E):
            pts = ctrl[i]
            v_i = pts[-1] - pts[0]   # edge direction for anti-parallel detection

            for pi in range(1, P):   # skip fixed endpoints (0 and P)
                p = pts[pi]

                # (a) Spring tension -- discrete Laplacian
                f_spring = k_spring * (pts[pi - 1] + pts[pi + 1] - 2.0 * p)

                # (b) Electrostatic bundling -- compatibility-weighted *average*
                #     Dividing by the sum of weights prevents force accumulation
                #     from causing numerical divergence (coordinates -> infinity).
                f_num   = np.zeros(3)
                w_total = 0.0
                for j, c in compat_adj[i]:
                    # Anti-parallel: reverse the matching index to avoid tangling
                    v_j = ctrl[j][-1] - ctrl[j][0]
                    pj  = (P - pi) if np.dot(v_i, v_j) < 0 else pi
                    f_num   += c * (ctrl[j][pj] - p)
                    w_total += c

                if w_total > 1e-9:
                    f_electro = (f_num / w_total) * k_electro
                else:
                    f_electro = np.zeros(3)

                # Clamp displacement to prevent overshooting and divergence
                delta = f_spring + f_electro
                dist  = np.linalg.norm(delta)
                if dist > max_step:
                    delta = delta * (max_step / dist)

                new_ctrl[i][pi] = p + delta

        ctrl = new_ctrl
        if (it + 1) % 20 == 0:
            print(f"    ... iteration {it + 1}/{iters}")

    return ctrl


# Initialise control-point arrays at the coarsest hierarchy level
P_init = P_LEVELS[0]
ctrl   = [make_ctrl_pts(s, t, P_init) for s, t in raw_edges]
E      = len(raw_edges)

# Pre-compute compatibility adjacency list -- O(E^2) once, reused every level
print("Computing edge compatibility...")
compat_adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(E)}
for i in range(E):
    for j in range(i + 1, E):
        c = compatibility(ctrl[i], ctrl[j])
        if c >= C_THRESH:
            compat_adj[i].append((j, c))
            compat_adj[j].append((i, c))

n_pairs = sum(len(v) for v in compat_adj.values()) // 2
print(f"Compatible pairs: {n_pairs}")

# Hierarchical simulation: coarse -> fine (avoids local minima)
print(f"Running hierarchical FDEB  levels={P_LEVELS}, {ITERS} iters/level...")
for level_idx, P in enumerate(P_LEVELS):
    print(f"  Level P={P} ({P - 1} interior control points)...")
    ctrl = run_fdeb_level(
        ctrl, E, compat_adj, P, ITERS,
        K_SPRING, K_ELECTRO, MAX_STEP,
    )
    if level_idx < len(P_LEVELS) - 1:
        ctrl = [subdivide(pts) for pts in ctrl]
        print(f"  Subdivided -> P={P_LEVELS[level_idx + 1]}")

print("FDEB complete.")

# ── 4. JSON EXPORT ────────────────────────────────────────────────────────────
bundled_edges = [
    {
        'src_id':         src['id'],
        'tgt_id':         tgt['id'],
        't_peak_src':     src['t_peak'],
        't_peak_tgt':     tgt['t_peak'],
        # Control points as [[x, y, z], …] — CatmullRomCurve3-ready
        'control_points': [[round(float(v), 4) for v in pt] for pt in ctrl[idx]],
    }
    for idx, (src, tgt) in enumerate(raw_edges)
]

out = {'nodes': nodes, 'bundled_edges': bundled_edges}
with open('network_data.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, separators=(',', ':'))

print(f"\n✓  network_data.json written")
print(f"   nodes         : {len(nodes)}")
print(f"   bundled_edges : {len(bundled_edges)}")
print(f"\nNext step: copy network_data.json → <react-app>/public/")
