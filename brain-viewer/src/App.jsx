import { useState } from 'react'
import BrainNetworkViewer from './BrainNetworkViewer'
import WavefrontViewer from './WavefrontViewer'

const VIEWS = [
  { key: 'fdeb',       label: 'A · FDEB Network',       desc: 'Force-Directed Edge Bundling' },
  { key: 'wavefront',  label: 'B · Wavefront',           desc: 'Spatiotemporal Geodesic Mapping' },
]

function NavBar({ active, setActive }) {
  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0,
      display: 'flex', justifyContent: 'center', gap: 4,
      padding: '8px 0', zIndex: 100,
      background: 'rgba(0,0,0,0.55)',
      backdropFilter: 'blur(6px)',
      borderBottom: '1px solid rgba(255,255,255,0.08)',
    }}>
      {VIEWS.map(v => (
        <button
          key={v.key}
          onClick={() => setActive(v.key)}
          style={{
            background:    active === v.key ? 'rgba(255,255,255,0.12)' : 'transparent',
            border:        active === v.key ? '1px solid rgba(255,255,255,0.3)'
                                            : '1px solid rgba(255,255,255,0.1)',
            borderRadius:  6,
            color:         active === v.key ? '#fff' : 'rgba(255,255,255,0.45)',
            fontFamily:    'monospace',
            fontSize:      11,
            padding:       '5px 14px',
            cursor:        'pointer',
            letterSpacing: 0.5,
            transition:    'all 0.2s',
          }}
        >
          {v.label}
        </button>
      ))}
    </div>
  )
}

export default function App() {
  const [active, setActive] = useState('wavefront')

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000' }}>
      <NavBar active={active} setActive={setActive} />
      <div style={{ paddingTop: 36, height: '100%', boxSizing: 'border-box' }}>
        {active === 'fdeb'      && <BrainNetworkViewer />}
        {active === 'wavefront' && <WavefrontViewer />}
      </div>
    </div>
  )
}

