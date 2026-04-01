"""
generate_wavefront.py — SIGGRAPH 2026
Spatiotemporal Hemodynamic Wavefront Generator

Pipeline:
  1. Synthesise a brain-shaped ellipsoid mesh (icosphere → deformed)
  2. Place V1–V4 seed regions on the mesh surface
  3. Compute geodesic t_peak propagation via Dijkstra on the mesh graph
     (Laplacian-smoothed distances → iso-contour levels)
  4. Export: mesh vertices/faces + per-vertex t_peak → wavefront_data.json
"""

import json
import math
import heapq
import random
import numpy as np
from typing import List, Tuple

# ── 1. Icosphere mesh generation (2 subdivisions ≈ 320 faces) ────────────

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def icosphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of an icosphere on the unit sphere."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw_verts = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ]
    verts = [normalize(np.array(v, dtype=float)) for v in raw_verts]

    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]

    midpoint_cache = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = normalize((verts[i] + verts[j]) / 2.0)
        verts.append(mid)
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        for tri in faces:
            a, b, c = tri
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    return np.array(verts), np.array(faces, dtype=int)


# ── 2. Deform sphere → brain-like ellipsoid with surface bumps ──────────

def deform_brain(verts: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Deform unit sphere into a smooth brain-like ellipsoid.
    Strategy:
      1. Anisotropic ellipsoid scaling (wider than tall, elongated front-back)
      2. Flatten the ventral (bottom) face slightly
      3. Very gentle low-frequency gyri ripples (amplitude < 3% of radius)
      4. Laplacian smooth of vertex positions to remove any angular artefacts
    """
    rng = np.random.default_rng(seed)
    v = verts.copy()

    # ── 1. Smooth brain ellipsoid ──────────────────────────────────────
    # x: left-right (widest),  y: dorsal-ventral,  z: anterior-posterior
    v[:, 0] *= 1.40   # slightly wide
    v[:, 1] *= 1.00   # normal height
    v[:, 2] *= 1.20   # slightly elongated front-back

    # ── 2. Flatten bottom (ventral surface) ───────────────────────────
    # Vertices below the equator get their Y squished
    below = v[:, 1] < 0
    v[below, 1] *= 0.70

    # ── 3. Subtle posterior occipital bulge ───────────────────────────
    post = v[:, 2] < -0.5
    v[post, 2] *= 1.10

    # ── 4. Very gentle gyri: only 5 modes, tiny amplitude (<0.025) ────
    directions = rng.standard_normal((5, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    amplitudes  = rng.uniform(0.010, 0.022, 5)   # << much smaller
    frequencies = rng.uniform(1.5, 3.0, 5)        # << lower frequency

    for d, a, f in zip(directions, amplitudes, frequencies):
        proj  = verts @ d
        noise = a * np.sin(f * proj * math.pi)
        v    += noise[:, None] * verts            # displace along unit normal

    return v


# ── 3. Laplacian smooth of the displacement field ───────────────────────

def build_adjacency(faces: np.ndarray, n_verts: int):
    adj = [set() for _ in range(n_verts)]
    for f in faces:
        a, b, c = f
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    return adj

def laplacian_smooth_scalar(values: np.ndarray, adj, iterations: int = 3,
                            lam: float = 0.5) -> np.ndarray:
    v = values.copy()
    for _ in range(iterations):
        new_v = v.copy()
        for i, neighbours in enumerate(adj):
            if neighbours:
                new_v[i] = (1 - lam) * v[i] + lam * np.mean([v[j] for j in neighbours])
        v = new_v
    return v


# ── 4. Geodesic Dijkstra on mesh graph ───────────────────────────────────

def geodesic_dijkstra(seeds: List[int], seed_times: List[float],
                      adj, verts: np.ndarray) -> np.ndarray:
    """
    Multi-source Dijkstra: each seed ROI starts at its t_peak value.
    Distance ~= propagation time away from seed (speed = 1 unit/s here).
    """
    INF = float('inf')
    t_peak = np.full(len(verts), INF)

    heap = []
    for idx, t in zip(seeds, seed_times):
        t_peak[idx] = t
        heapq.heappush(heap, (t, idx))

    while heap:
        cur_t, u = heapq.heappop(heap)
        if cur_t > t_peak[u]:
            continue
        for v in adj[u]:
            dist = np.linalg.norm(verts[u] - verts[v])
            new_t = t_peak[u] + dist * 1.0   # propagation speed = 1 unit/s
            if new_t < t_peak[v]:
                t_peak[v] = new_t
                heapq.heappush(heap, (new_t, v))

    return t_peak


# ── 5. Assign V1–V4 seed regions on the posterior brain surface ──────────

def find_region_seeds(verts: np.ndarray, n_seeds_per_region: int = 8):
    """
    Visual areas V1–V4 cluster along the posterior occipital lobe.
    V1 is most posterior; V2, V3, V4 fan forward.
    Returns: dict { region: (seed_indices, base_t_peak) }
    """
    # Most posterior vertices (smallest z in our coord system)
    z = verts[:, 2]
    y = verts[:, 1]

    regions = {
        'V1': {'z_range': (-np.inf, -0.8), 'y_range': (-0.3, 0.3), 't_peak': 4.0},
        'V2': {'z_range': (-1.2, -0.4),    'y_range': (-0.5, 0.6), 't_peak': 5.5},
        'V3': {'z_range': (-0.9, 0.0),     'y_range': (-0.7, 0.8), 't_peak': 6.8},
        'V4': {'z_range': (-0.5, 0.5),     'y_range': (-0.9, 0.9), 't_peak': 8.0},
    }

    result = {}
    for name, cfg in regions.items():
        zlo, zhi = cfg['z_range']
        ylo, yhi = cfg['y_range']
        mask = (
            (z >= zlo) & (z <= zhi) &
            (y >= ylo) & (y <= yhi)
        )
        idxs = np.where(mask)[0]

        if len(idxs) == 0:
            # Fallback: closest vertices to expected centroid
            z_mid = (max(zlo, -2.5) + min(zhi, 2.5)) / 2
            y_mid = (ylo + yhi) / 2
            target = np.array([0.0, y_mid, z_mid])
            dists = np.linalg.norm(verts - target, axis=1)
            idxs = np.argsort(dists)[:n_seeds_per_region]
        else:
            # Pick n_seeds_per_region spread evenly by random sampling
            rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
            idxs = rng.choice(idxs, size=min(n_seeds_per_region, len(idxs)), replace=False)

        result[name] = {'indices': idxs.tolist(), 't_peak': cfg['t_peak']}

    return result


# ── 6. Compute iso-contour levels ────────────────────────────────────────

def compute_iso_levels(t_peak: np.ndarray, n_levels: int = 12):
    """
    Returns list of dicts: each level has a t value and vertex indices
    that lie within ± half the level spacing of that t value.
    """
    t_min = float(np.min(t_peak))
    t_max = float(np.max(t_peak))
    levels = np.linspace(t_min, t_max, n_levels)
    half = (levels[1] - levels[0]) * 0.4

    iso_contours = []
    for lvl in levels:
        mask = np.abs(t_peak - lvl) < half
        iso_contours.append({
            't': float(lvl),
            'vertex_indices': np.where(mask)[0].tolist(),
        })
    return iso_contours


# ── 7. Colour map: t_peak → RGB (cool-warm: blue → orange → red) ─────────

def tpeak_to_color(t_peak_norm: np.ndarray) -> np.ndarray:
    """Map normalised t_peak [0,1] to a vibrant cool→warm RGB colour."""
    t = t_peak_norm.copy()
    # Viridis-inspired: blue(0) → cyan → green → yellow → red(1)
    r = np.clip(1.5 * t - 0.2, 0, 1)
    g = np.clip(np.sin(t * math.pi) * 1.2, 0, 1)
    b = np.clip(1.0 - 2.0 * t, 0, 1)
    return np.stack([r, g, b], axis=1)


# ── 8. Main ───────────────────────────────────────────────────────────────

def main():
    print("Building icosphere mesh ...")
    verts_unit, faces = icosphere(subdivisions=4)

    print("Deforming into brain shape ...")
    verts = deform_brain(verts_unit)

    print(f"Mesh: {len(verts)} vertices, {len(faces)} faces")

    adj = build_adjacency(faces, len(verts))

    print("Locating V1–V4 seed regions ...")
    regions = find_region_seeds(verts)

    # Flatten seeds for multi-source Dijkstra
    all_seed_idxs = []
    all_seed_times = []
    for name, info in regions.items():
        for idx in info['indices']:
            all_seed_idxs.append(idx)
            all_seed_times.append(info['t_peak'])

    print("Computing geodesic t_peak propagation (Dijkstra) ...")
    t_peak_raw = geodesic_dijkstra(all_seed_idxs, all_seed_times, adj, verts)

    # Handle any unreached vertices (should be none on closed mesh)
    max_finite = np.max(t_peak_raw[np.isfinite(t_peak_raw)])
    t_peak_raw = np.where(np.isfinite(t_peak_raw), t_peak_raw, max_finite)

    print("Laplacian smoothing t_peak field ...")
    t_peak = laplacian_smooth_scalar(t_peak_raw, adj, iterations=5, lam=0.6)

    # Normalise for colour mapping
    t_min, t_max = float(t_peak.min()), float(t_peak.max())
    t_norm = (t_peak - t_min) / (t_max - t_min + 1e-9)

    print("Computing vertex colours ...")
    colors = tpeak_to_color(t_norm)

    print("Computing iso-contour wavefront levels ...")
    iso_contours = compute_iso_levels(t_peak, n_levels=16)

    # Region metadata for rendering
    region_info = []
    for name, info in regions.items():
        centroid = verts[info['indices']].mean(axis=0).tolist()
        region_info.append({
            'name': name,
            't_peak': info['t_peak'],
            'indices': info['indices'],
            'centroid': centroid,
        })

    # Vertex normals (per-face averaged)
    print("Computing vertex normals ...")
    normals = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i, f in enumerate(faces):
        normals[f[0]] += fn[i]
        normals[f[1]] += fn[i]
        normals[f[2]] += fn[i]
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms_len, 1e-9)

    # Assembly
    output = {
        'meta': {
            'n_verts': int(len(verts)),
            'n_faces': int(len(faces)),
            't_peak_min': t_min,
            't_peak_max': t_max,
            'description': 'Spatiotemporal hemodynamic wavefront — V1→V4',
        },
        'vertices':   verts.tolist(),
        'normals':    normals.tolist(),
        'faces':      faces.tolist(),
        't_peak':     t_peak.tolist(),
        't_peak_norm': t_norm.tolist(),
        'colors':     colors.tolist(),
        'iso_contours': iso_contours,
        'regions':    region_info,
    }

    out_path = 'brain-viewer/public/wavefront_data.json'
    print(f"Writing {out_path} ...")
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    kb = len(json.dumps(output)) // 1024
    print(f"Done. Output size: ~{kb} KB")
    print(f"  Vertices : {len(verts)}")
    print(f"  Faces    : {len(faces)}")
    print(f"  t_peak   : [{t_min:.2f}, {t_max:.2f}] s")
    print(f"  Iso levels: {len(iso_contours)}")


if __name__ == '__main__':
    main()
