/**
 * WavefrontViewer.jsx — SIGGRAPH 2026
 *
 * Idea B: Spatiotemporal Hemodynamic Wavefront
 *
 * Renders per-vertex t_peak data (computed offline via geodesic Dijkstra)
 * as an animated "ripple" wavefront sweeping the brain mesh surface.
 *
 * GLSL technique:
 *   - Per-vertex a_tPeak attribute
 *   - Animated u_wavefront uniform sweeps t_min → t_max on loop
 *   - Fragment shader computes a glowing band at wavefront position
 *     (Gaussian falloff around |t_peak - u_wavefront|)
 *   - Behind wavefront: revealed cool-warm colour map
 *   - Ahead of wavefront: dark / unreached
 *   - Iso-contour lines: rendered as faint rings every Δt = 0.4 s
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';

// ── GLSL Shaders ─────────────────────────────────────────────────────────

const VERT = /* glsl */`
  attribute float a_tPeak;
  attribute vec3  a_color;

  varying float v_tPeak;
  varying vec3  v_color;
  varying vec3  v_worldNormal;

  void main() {
    v_tPeak       = a_tPeak;
    v_color       = a_color;
    v_worldNormal = normalize(normalMatrix * normal);
    gl_Position   = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG = /* glsl */`
  precision highp float;

  uniform float u_time;
  uniform float u_wavefront;   // current wavefront t value (sweeps t_min→t_max)
  uniform float u_tMin;
  uniform float u_tMax;
  uniform float u_sigma;       // wavefront glow width
  uniform float u_isoSpacing;  // iso-contour interval (seconds)
  uniform float u_isoWidth;    // iso-contour line half-width

  varying float v_tPeak;
  varying vec3  v_color;
  varying vec3  v_worldNormal;

  void main() {
    float tRange = u_tMax - u_tMin;

    // ── 1. Base colour from cold-warm map (revealed behind wavefront) ──
    float reveal = smoothstep(u_wavefront + u_sigma * 1.5,
                              u_wavefront - u_sigma * 0.2,
                              v_tPeak);
    vec3 baseCol = v_color * reveal;

    // ── 2. Wavefront glow band  ────────────────────────────────────────
    float d        = v_tPeak - u_wavefront;
    float glow     = exp(-(d * d) / (2.0 * u_sigma * u_sigma));
    vec3 glowColor = mix(vec3(1.0, 0.9, 0.5), vec3(1.0, 1.0, 1.0), glow * glow);
    baseCol        = mix(baseCol, glowColor, glow * 0.85);

    // ── 3. Iso-contour rings ──────────────────────────────────────────
    float tNorm    = (v_tPeak - u_tMin) / (tRange + 0.001);
    float isoPhase = mod(tNorm * tRange, u_isoSpacing);
    float isoDist  = min(isoPhase, u_isoSpacing - isoPhase);
    float isoLine  = 1.0 - smoothstep(0.0, u_isoWidth, isoDist);
    // Only show iso-lines in the revealed region
    float isoMask  = reveal * 0.55;
    baseCol        = mix(baseCol, vec3(0.95, 0.95, 1.0), isoLine * isoMask);

    // ── 4. Diffuse lighting (simple hemisphere) ───────────────────────
    vec3 lightDir  = normalize(vec3(0.6, 1.0, 0.8));
    float diff     = max(dot(v_worldNormal, lightDir), 0.0) * 0.5 + 0.5;
    baseCol        *= diff;

    // ── 5. Fresnel rim glow ───────────────────────────────────────────
    vec3 viewDir = vec3(0.0, 0.0, 1.0);   // approx, good enough for effect
    float rim    = 1.0 - abs(dot(v_worldNormal, viewDir));
    rim          = pow(rim, 2.5) * 0.35;
    baseCol      += vec3(0.3, 0.6, 1.0) * rim;

    // ── 6. Alpha ──────────────────────────────────────────────────────
    float alpha   = 0.18 + 0.72 * reveal + glow * 0.10;

    gl_FragColor  = vec4(baseCol, alpha);
  }
`;

// Transparent wireframe shader for iso-contour edge pass
const WIRE_VERT = /* glsl */`
  attribute float a_tPeak;
  varying float v_tPeak;
  void main() {
    v_tPeak     = a_tPeak;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const WIRE_FRAG = /* glsl */`
  precision highp float;
  uniform float u_wavefront;
  uniform float u_sigma;
  varying float v_tPeak;
  void main() {
    float d    = v_tPeak - u_wavefront;
    float glow = exp(-(d * d) / (2.0 * u_sigma * u_sigma * 4.0));
    float rev  = smoothstep(u_wavefront + u_sigma * 2.0,
                            u_wavefront - u_sigma * 0.3, v_tPeak);
    float a    = glow * 0.6 + rev * 0.12;
    gl_FragColor = vec4(1.0, 0.95, 0.7, a);
  }
`;

// ── Brain Mesh ────────────────────────────────────────────────────────────

function BrainMesh({ data }) {
  const meshRef = useRef();
  const wireRef = useRef();

  const { solidMat, wireMat } = useMemo(() => {
    const solid = new THREE.ShaderMaterial({
      vertexShader:   VERT,
      fragmentShader: FRAG,
      transparent:    true,
      side:           THREE.DoubleSide,
      depthWrite:     false,
      blending:       THREE.NormalBlending,
      uniforms: {
        u_wavefront:   { value: data.meta.t_peak_min },
        u_tMin:        { value: data.meta.t_peak_min },
        u_tMax:        { value: data.meta.t_peak_max },
        u_sigma:       { value: 0.18 },
        u_isoSpacing:  { value: 0.38 },
        u_isoWidth:    { value: 0.022 },
        u_time:        { value: 0 },
      },
    });

    const wire = new THREE.ShaderMaterial({
      vertexShader:   WIRE_VERT,
      fragmentShader: WIRE_FRAG,
      transparent:    true,
      wireframe:      true,
      blending:       THREE.AdditiveBlending,
      depthWrite:     false,
      uniforms: {
        u_wavefront: { value: data.meta.t_peak_min },
        u_sigma:     { value: 0.18 },
      },
    });

    return { solidMat: solid, wireMat: wire };
  }, [data]);

  const { solidGeo, wireGeo } = useMemo(() => {
    const verts    = new Float32Array(data.vertices.flat());
    const normals  = new Float32Array(data.normals.flat());
    const tPeaks   = new Float32Array(data.t_peak);
    const colors   = new Float32Array(data.colors.flat());
    const indices  = new Uint32Array(data.faces.flat());

    // Solid geometry
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(verts,   3));
    sg.setAttribute('normal',   new THREE.BufferAttribute(normals, 3));
    sg.setAttribute('a_tPeak',  new THREE.BufferAttribute(tPeaks,  1));
    sg.setAttribute('a_color',  new THREE.BufferAttribute(colors,  3));
    sg.setIndex(new THREE.BufferAttribute(indices, 1));

    // Wireframe (Three.js EdgesGeometry keeps only real edges)
    const wg = new THREE.BufferGeometry();
    const edgeGeo = new THREE.EdgesGeometry(sg, 15); // 15° crease angle
    // Copy a_tPeak to wireframe vertices
    const ePos = edgeGeo.getAttribute('position');
    const eTPeak = new Float32Array(ePos.count);
    // Map back to closest original vertex by position
    const origPos = verts;
    for (let i = 0; i < ePos.count; i++) {
      const px = ePos.getX(i), py = ePos.getY(i), pz = ePos.getZ(i);
      let bestDist = Infinity, bestIdx = 0;
      for (let j = 0; j < origPos.length / 3; j++) {
        const dx = origPos[j * 3]   - px;
        const dy = origPos[j * 3+1] - py;
        const dz = origPos[j * 3+2] - pz;
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < bestDist) { bestDist = d2; bestIdx = j; }
      }
      eTPeak[i] = tPeaks[bestIdx];
    }
    wg.setAttribute('position', ePos);
    wg.setAttribute('a_tPeak',  new THREE.BufferAttribute(eTPeak, 1));

    return { solidGeo: sg, wireGeo: wg };
  }, [data]);

  // Animate wavefront
  const tMin  = data.meta.t_peak_min;
  const tMax  = data.meta.t_peak_max;
  const CYCLE = 5.0; // seconds per full sweep

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    // linger at end, then snap back
    const phase     = (t % CYCLE) / CYCLE;
    const eased     = phase < 0.85
      ? phase / 0.85
      : 1.0 - (phase - 0.85) / 0.15 * 0.05; // quick tail-off
    const wavefront = tMin + eased * (tMax - tMin + 0.5);

    solidMat.uniforms.u_wavefront.value = wavefront;
    solidMat.uniforms.u_time.value      = t;
    wireMat.uniforms.u_wavefront.value  = wavefront;
  });

  return (
    <group>
      <mesh ref={meshRef} geometry={solidGeo} material={solidMat} />
      <lineSegments ref={wireRef} geometry={wireGeo} material={wireMat} />
    </group>
  );
}

// ── Region Labels ─────────────────────────────────────────────────────────

const REGION_COLORS = { V1: '#ff2dca', V2: '#ff8c00', V3: '#00e5ff', V4: '#39ff14' };

function RegionMarkers({ regions }) {
  return regions.map(r => {
    const [x, y, z] = r.centroid;
    return (
      <Html key={r.name} position={[x, y, z]} center distanceFactor={8}>
        <div style={{
          color: REGION_COLORS[r.name] ?? '#fff',
          fontFamily: 'monospace',
          fontSize: 11,
          fontWeight: 700,
          textShadow: `0 0 8px ${REGION_COLORS[r.name] ?? '#fff'}`,
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
          userSelect: 'none',
        }}>
          {r.name} {r.t_peak.toFixed(1)}s
        </div>
      </Html>
    );
  });
}

// ── Scene ─────────────────────────────────────────────────────────────────

function Scene() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/wavefront_data.json')
      .then(r => r.json())
      .then(setData)
      .catch(err => console.error('wavefront_data.json load failed:', err));
  }, []);

  if (!data) return null;

  return (
    <group scale={[0.65, 0.65, 0.65]}>
      <BrainMesh data={data} />
      <RegionMarkers regions={data.regions} />
    </group>
  );
}

// ── Loading ───────────────────────────────────────────────────────────────

function Loader() {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const t = setTimeout(() => setVisible(false), 3500);
    return () => clearTimeout(t);
  }, []);
  if (!visible) return null;
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: '#000', color: '#fff', fontFamily: 'monospace',
      fontSize: 13, letterSpacing: 2, pointerEvents: 'none',
    }}>
      LOADING WAVEFRONT DATA…
    </div>
  );
}

// ── Colour-bar legend ─────────────────────────────────────────────────────

function ColorBar({ tMin, tMax }) {
  if (!tMin && tMin !== 0) return null;
  return (
    <div style={{
      position: 'absolute', bottom: 32, left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 4, pointerEvents: 'none', userSelect: 'none',
    }}>
      <div style={{ color: '#aaa', fontFamily: 'monospace', fontSize: 10, letterSpacing: 1 }}>
        HEMODYNAMIC DELAY (t_peak)
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: '#5599ff', fontFamily: 'monospace', fontSize: 10 }}>
          {tMin.toFixed(1)} s
        </span>
        <div style={{
          width: 160, height: 10, borderRadius: 5,
          background: 'linear-gradient(to right, #0033ff, #00ccff, #00ff88, #ffcc00, #ff2200)',
          boxShadow: '0 0 8px rgba(0,200,255,0.5)',
        }} />
        <span style={{ color: '#ff4422', fontFamily: 'monospace', fontSize: 10 }}>
          {tMax.toFixed(1)} s
        </span>
      </div>
      <div style={{ color: '#666', fontFamily: 'monospace', fontSize: 9 }}>
        V1 (primary visual) → V4 (ventral stream)
      </div>
    </div>
  );
}

// ── Info panel ────────────────────────────────────────────────────────────

function InfoPanel() {
  const items = [
    { label: 'V1 — Primary Visual',   color: REGION_COLORS.V1, t: '4.0 s' },
    { label: 'V2 — Secondary Visual', color: REGION_COLORS.V2, t: '5.5 s' },
    { label: 'V3 — Tertiary Visual',  color: REGION_COLORS.V3, t: '6.8 s' },
    { label: 'V4 — Ventral Stream',   color: REGION_COLORS.V4, t: '8.0 s' },
  ];
  return (
    <div style={{
      position: 'absolute', top: 20, left: 20,
      background: 'rgba(0,0,0,0.72)',
      border: '1px solid rgba(255,255,255,0.10)',
      borderRadius: 8, padding: '12px 16px',
      color: '#fff', fontFamily: 'monospace', fontSize: 11,
      lineHeight: '1.9em', pointerEvents: 'none', userSelect: 'none',
      maxWidth: 220,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, letterSpacing: 1, fontSize: 12 }}>
        SPATIOTEMPORAL WAVEFRONT
      </div>
      {items.map(({ label, color, t }) => (
        <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: color, display: 'inline-block',
            boxShadow: `0 0 6px ${color}`,
            flexShrink: 0,
          }} />
          <span style={{ flex: 1 }}>{label}</span>
          <span style={{ opacity: 0.65 }}>{t}</span>
        </div>
      ))}
      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 8 }}>
        <div style={{ opacity: 0.6, fontSize: 9, lineHeight: '1.6em' }}>
          Geodesic Dijkstra on icosphere mesh<br />
          Laplacian-smoothed t_peak field<br />
          Wavefront + iso-contour GLSL shader<br />
          2562 verts · 5120 faces · 16 iso levels
        </div>
      </div>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────

export default function WavefrontViewer() {
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    fetch('/wavefront_data.json')
      .then(r => r.json())
      .then(d => setMeta(d.meta))
      .catch(() => {});
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#02010a', position: 'relative' }}>
      <Loader />
      <InfoPanel />
      {meta && <ColorBar tMin={meta.t_peak_min} tMax={meta.t_peak_max} />}

      <Canvas
        camera={{ position: [0, 0, 6], fov: 48, near: 0.01, far: 200 }}
        gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
        linear
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 5, 5]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-4, -3, -4]} intensity={0.4} color="#3366ff" />
        <Scene />
        <OrbitControls
          enableDamping
          dampingFactor={0.07}
          autoRotate
          autoRotateSpeed={0.35}
          minDistance={2}
          maxDistance={15}
        />
      </Canvas>
    </div>
  );
}
