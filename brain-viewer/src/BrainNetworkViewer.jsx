/**
 * BrainNetworkViewer.jsx — SIGGRAPH 2026
 *
 * KEY ARCHITECTURE CHANGE: All 450 edges are merged into a SINGLE
 * BufferGeometry drawn with LineSegments → one draw call → 60 fps.
 *
 * Each line-segment vertex carries:
 *   a_tPeak  : float — expected BOLD peak time at this point
 *   a_color  : vec3  — neon region colour (V1 magenta … V4 green)
 *
 * The GLSL fragment shader fires a Gaussian pulse when u_time ≈ a_tPeak,
 * creating the V1→V4 travelling-wave effect with zero CPU work per frame.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// ── Region colours (neon palette) ─────────────────────────────────────────
const REGION_COLOR_HEX = {
  V1: '#ff2dca', // magenta
  V2: '#ff8c00', // amber
  V3: '#00e5ff', // cyan
  V4: '#39ff14', // neon green
};
const REGION_COLORS_THREE = Object.fromEntries(
  Object.entries(REGION_COLOR_HEX).map(([k, v]) => [k, new THREE.Color(v)])
);

// ── GLSL ──────────────────────────────────────────────────────────────────
const VERT = /* glsl */`
  attribute float a_tPeak;
  attribute vec3  a_color;

  varying float v_tPeak;
  varying vec3  v_color;

  void main() {
    v_tPeak = a_tPeak;
    v_color = a_color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG = /* glsl */`
  precision highp float;

  uniform float u_time;
  uniform float u_period;
  uniform float u_sigma;

  varying float v_tPeak;
  varying vec3  v_color;

  void main() {
    float t     = mod(u_time, u_period);
    float d     = t - v_tPeak;
    float pulse = exp(-(d * d) / (2.0 * u_sigma * u_sigma));

    float alpha = 0.22 + 0.78 * pulse;
    vec3 col = mix(v_color, vec3(1.0), pulse * 0.6);

    gl_FragColor = vec4(col, alpha);
  }
`;

// ── Single-mesh edge network ───────────────────────────────────────────────
function EdgeNetwork({ data }) {
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader:   VERT,
    fragmentShader: FRAG,
    transparent:    true,
    depthWrite:     false,
    blending:       THREE.AdditiveBlending,
    uniforms: {
      u_time:   { value: 0 },
      u_period: { value: 10.0 },
      u_sigma:  { value: 0.9 },
    },
  }), []);

  const geometry = useMemo(() => {
    if (!data) return null;

    const nodeMap = {};
    data.nodes.forEach(n => { nodeMap[n.id] = n; });

    const positions = [];
    const tPeaks   = [];
    const colors   = [];

    data.bundled_edges.forEach(edge => {
      const srcNode   = nodeMap[edge.src_id];
      const col       = REGION_COLORS_THREE[srcNode?.region] ?? REGION_COLORS_THREE.V3;
      const [r, g, b] = [col.r, col.g, col.b];

      const pts   = edge.control_points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
      const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.5);

      const SAMPLES = 20;
      const sampled = curve.getPoints(SAMPLES);

      for (let i = 0; i < SAMPLES; i++) {
        const t0    = i       / SAMPLES;
        const t1    = (i + 1) / SAMPLES;
        const peak0 = edge.t_peak_src * (1 - t0) + edge.t_peak_tgt * t0;
        const peak1 = edge.t_peak_src * (1 - t1) + edge.t_peak_tgt * t1;
        const p0    = sampled[i];
        const p1    = sampled[i + 1];

        positions.push(p0.x, p0.y, p0.z,  p1.x, p1.y, p1.z);
        tPeaks.push(peak0, peak1);
        colors.push(r, g, b,  r, g, b);
      }
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('a_tPeak',  new THREE.Float32BufferAttribute(tPeaks,    1));
    geo.setAttribute('a_color',  new THREE.Float32BufferAttribute(colors,    3));
    return geo;
  }, [data]);

  useFrame(({ clock }) => {
    material.uniforms.u_time.value = clock.getElapsedTime();
  });

  if (!geometry) return null;
  return <lineSegments geometry={geometry} material={material} />;
}

// ── Nodes ──────────────────────────────────────────────────────────────────
function NodeCloud({ nodes }) {
  return nodes.map(node => (
    <mesh key={node.id} position={[node.x, node.y, node.z]}>
      <sphereGeometry args={[0.045, 8, 8]} />
      <meshBasicMaterial
        color={REGION_COLOR_HEX[node.region] ?? '#ffffff'}
        transparent
        opacity={0.7}
      />
    </mesh>
  ));
}

// ── Scene ──────────────────────────────────────────────────────────────────
function Scene() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/network_data.json')
      .then(r => r.json())
      .then(setData)
      .catch(err => console.error('network_data.json load failed:', err));
  }, []);

  if (!data) return null;

  return (
    <group>
      <NodeCloud nodes={data.nodes} />
      <EdgeNetwork data={data} />
    </group>
  );
}

// ── Legend ─────────────────────────────────────────────────────────────────
function Legend() {
  const items = [
    { label: 'V1 — Primary Visual',   color: REGION_COLOR_HEX.V1 },
    { label: 'V2 — Secondary Visual', color: REGION_COLOR_HEX.V2 },
    { label: 'V3 — Tertiary Visual',  color: REGION_COLOR_HEX.V3 },
    { label: 'V4 — Ventral Stream',   color: REGION_COLOR_HEX.V4 },
  ];
  return (
    <div style={{
      position: 'absolute', top: 20, left: 20,
      background: 'rgba(0,0,0,0.65)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: 8, padding: '12px 16px',
      color: '#fff', fontFamily: 'monospace', fontSize: 12,
      lineHeight: '1.9em', pointerEvents: 'none', userSelect: 'none',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, letterSpacing: 1 }}>
        fMRI BOLD NETWORK
      </div>
      {items.map(({ label, color }) => (
        <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 10, height: 10, borderRadius: '50%',
            background: color, display: 'inline-block',
            boxShadow: `0 0 6px ${color}`,
          }} />
          {label}
        </div>
      ))}
      <div style={{ marginTop: 8, opacity: 0.5, fontSize: 10 }}>
        HRF t_peak: V1 ≈ 4 s → V4 ≈ 8 s
      </div>
    </div>
  );
}

// ── Root ───────────────────────────────────────────────────────────────────
export default function BrainNetworkViewer() {
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', position: 'relative' }}>
      <Legend />
      <Canvas
        camera={{ position: [0, 0, 9], fov: 50, near: 0.01, far: 100 }}
        gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
        linear
      >
        <Scene />
        <OrbitControls
          enableDamping dampingFactor={0.07}
          autoRotate autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
