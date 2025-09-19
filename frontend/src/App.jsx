import React, { useEffect, useMemo, useRef, useState } from 'react'
import { getState, getMetrics, startTraining, startDemo, stopDemo, makeWS, getTasks, selectTasks, pause, resume, reset, exportAll, getKnowledgeGraph, updatePSOSettings } from './api'

const box = {
  fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial',
  margin: 20
}

const row = { display: 'flex', gap: 12, margin: '10px 0 20px' }
const btn = { padding: '10px 14px', border: 0, borderRadius: 6, cursor: 'pointer' }
const ok = { ...btn, background: '#27AE60', color: '#fff' }
const info = { ...btn, background: '#3498DB', color: '#fff' }
const warn = { ...btn, background: '#F39C12', color: '#fff' }
const pre = { background: '#f6f8fa', padding: 12, borderRadius: 6, maxHeight: '40vh', overflow: 'auto' }
const grid = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }
const gridWide = { display: 'grid', gridTemplateColumns: '65% 35%', gridTemplateRows: 'auto auto', gap: 16, alignItems: 'start' }

export default function App() {
  const [state, setState] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [synth, setSynth] = useState(null)
  const [tasksCatalog, setTasksCatalog] = useState({ categories: {}, recommended: [], options: [] })
  const [selectedCategory, setSelectedCategory] = useState('')
  const [activeTasks, setActiveTasks] = useState([])
  const [kg, setKg] = useState(null)
  const streamRef = useRef(null)
  // UI-tunable settings
  const [iterPause, setIterPause] = useState(0.8)
  const [revealMs, setRevealMs] = useState(220)
  const [neighborSpeed, setNeighborSpeed] = useState(0.004)
  const [spread, setSpread] = useState(1.35)
  const [dotSize, setDotSize] = useState(6)

  async function refresh() {
    try {
      const [s, m] = await Promise.all([getState(), getMetrics()])
      setState(s)
      setMetrics(m)
    } catch (e) {
      // ignore for now
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 1000)
    return () => clearInterval(id)
  }, [])

  // Debounced push of iteration pause to backend
  useEffect(() => {
    const t = setTimeout(() => {
      updatePSOSettings({ pause: iterPause }).catch(() => {})
    }, 300)
    return () => clearTimeout(t)
  }, [iterPause])

  useEffect(() => {
    getTasks().then((t) => {
      setTasksCatalog(t || { categories: {}, recommended: [], options: [] })
      const firstCat = Object.keys(t?.categories || {})[0] || ''
      setSelectedCategory(firstCat)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const ws = makeWS()
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (['agent_state', 'progress', 'thought', 'metrics', 'demo_progress', 'demo_response'].includes(msg.type)) {
          refresh()
        }
        if (msg.type === 'progress' && msg.data?.type === 'synthesis_iteration') {
          setSynth(msg.data)
        }
        const line = typeof msg === 'string' ? msg : JSON.stringify(msg)
        if (streamRef.current) {
          streamRef.current.textContent = (streamRef.current.textContent + '\n' + line)
            .split('\n').slice(-200).join('\n')
        }
      } catch {
        if (streamRef.current) {
          streamRef.current.textContent = (streamRef.current.textContent + '\n' + ev.data)
            .split('\n').slice(-200).join('\n')
        }
      }
    }
    ws.onclose = () => {
      // reconnect after a moment
      setTimeout(() => window.location.reload(), 2000)
    }
    return () => ws.close()
  }, [])

  const isTraining = state?.agent_state?.is_training
  const progress = Math.round(state?.agent_state?.training_progress || 0)

  return (
    <div style={box}>
      <h1>SwarmAgentic Web Demo</h1>
      <div style={row}>
        <button style={ok} onClick={() => startTraining().then(refresh)}>Start Training</button>
        <button style={info} onClick={() => startDemo().then(refresh)} disabled={!state?.learned_concepts?.length}>Start Demo</button>
        <button style={warn} onClick={() => stopDemo().then(refresh)}>Stop Demo</button>
        <button style={btn} onClick={() => pause().then(refresh)}>Pause</button>
        <button style={btn} onClick={() => resume().then(refresh)}>Resume</button>
        <button style={btn} onClick={() => reset().then(() => { setSynth(null); refresh() })}>Reset</button>
        <button style={btn} onClick={() => exportAll().then((r) => alert('Exported: ' + JSON.stringify(r.files || r)))}>Export Results</button>
      </div>
      <div style={{ ...row, alignItems: 'center' }}>
        <div style={{ fontWeight: 600 }}>Task Preset:</div>
        <button style={btn} onClick={async () => { await selectTasks({ preset: 'default' }); setActiveTasks([]); refresh() }}>Default (Built-in)</button>
        <button style={btn} onClick={async () => { await selectTasks({ preset: 'from_training' }); refresh() }}>From Training</button>
        <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)}>
          {Object.keys(tasksCatalog.categories || {}).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
        <button style={info} onClick={async () => {
          const tasks = (tasksCatalog.categories || {})[selectedCategory] || []
          setActiveTasks(tasks)
          await selectTasks({ tasks })
          refresh()
        }}>Apply Tasks</button>
        {activeTasks?.length ? <span style={{ color: '#555' }}>Applied {activeTasks.length} tasks</span> : null}
      </div>
      {/* Visualization controls */}
      <div style={{ ...row, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ fontWeight: 600, marginRight: 8 }}>Viz Controls:</div>
        <label>Iter Pause (s)
          <input type="range" min="0.1" max="2.0" step="0.05" value={iterPause}
            onChange={(e)=>setIterPause(parseFloat(e.target.value))} style={{ marginLeft: 6 }} />
          <span style={{ marginLeft: 6 }}>{iterPause.toFixed(2)}</span>
        </label>
        <label style={{ marginLeft: 16 }}>Reveal ms
          <input type="range" min="40" max="600" step="10" value={revealMs}
            onChange={(e)=>setRevealMs(parseInt(e.target.value,10))} style={{ marginLeft: 6 }} />
          <span style={{ marginLeft: 6 }}>{revealMs}</span>
        </label>
        <label style={{ marginLeft: 16 }}>Neighbor Speed
          <input type="range" min="0.001" max="0.02" step="0.001" value={neighborSpeed}
            onChange={(e)=>setNeighborSpeed(parseFloat(e.target.value))} style={{ marginLeft: 6 }} />
          <span style={{ marginLeft: 6 }}>{neighborSpeed.toFixed(3)}</span>
        </label>
        <label style={{ marginLeft: 16 }}>Spread
          <input type="range" min="1.0" max="1.8" step="0.05" value={spread}
            onChange={(e)=>setSpread(parseFloat(e.target.value))} style={{ marginLeft: 6 }} />
          <span style={{ marginLeft: 6 }}>{spread.toFixed(2)}</span>
        </label>
        <label style={{ marginLeft: 16 }}>Dot px
          <input type="range" min="2" max="10" step="1" value={dotSize}
            onChange={(e)=>setDotSize(parseInt(e.target.value,10))} style={{ marginLeft: 6 }} />
          <span style={{ marginLeft: 6 }}>{dotSize}px</span>
        </label>
      </div>
      <div style={row}>
        <div style={{ width: 240, background: '#eee', height: 12, borderRadius: 6, overflow: 'hidden' }}>
          <div style={{ width: `${progress}%`, height: '100%', background: isTraining ? '#27AE60' : '#bbb' }} />
        </div>
        <div>{isTraining ? `Training… ${progress}%` : 'Idle'}</div>
      </div>
      <div style={gridWide}>
        {/* Top-left: PSO Visualization (65%) */}
        <div>
          <h3>PSO Visualization</h3>
          <PSOVis2 data={synth || { teams: [], iteration: 0, total_iters: 0 }}
                   demoRunning={!!state?.demo_running}
                   revealMs={revealMs}
                   neighborSpeed={neighborSpeed}
                   spread={spread}
                   dotSize={dotSize}
          />
          {Array.isArray(synth?.top_k) && synth.top_k.length > 0 ? (
            <div>
              <h4>Top Teams</h4>
              <ul>
                {synth.top_k.map((t, i) => (
                  <li key={i}>
                    fitness {t.fitness.toFixed(3)} | cov {t.coverage.toFixed(2)} | roles {t.role_count} | wf {t.workflow_len}
                  </li>
                ))}
              </ul>
              <details>
                <summary>Global Best Spec</summary>
                <pre style={pre}>{synth.gbest_spec}</pre>
              </details>
            </div>
          ) : null}
        </div>
        {/* Top-right: State (35%) */}
        <div>
          <h3>State</h3>
          <pre style={pre}>{JSON.stringify(state, null, 2)}</pre>
        </div>
        {/* Bottom-left: Live Stream (65%) */}
        <div>
          <h3>Live Stream</h3>
          <pre style={pre} ref={streamRef} />
        </div>
        {/* Bottom-right: Metrics (35%) */}
        <div>
          <h3>Metrics</h3>
          <pre style={pre}>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      </div>
      <div style={row}>
        <button style={btn} onClick={() => getKnowledgeGraph().then(setKg).catch(() => setKg({ status: 'not_ready' }))}>Load Knowledge Graph</button>
        <div style={{ color: '#777' }}>Preset: {state?.task_preset || 'Default'} | Active tasks: {(state?.pso_tasks || []).length}</div>
      </div>
      {kg ? (
        <div>
          <h3>Knowledge Graph</h3>
          <pre style={pre}>{JSON.stringify(kg, null, 2)}</pre>
        </div>
      ) : null}
    </div>
  )
}

function PSOVis({ data }) {
  const w = 560, h = 320, pad = 36
  const bg = { width: w, height: h, border: '1px solid #ddd', background: '#fff' }
  if (!data || !Array.isArray(data.teams)) {
    return <div style={{ color: '#777' }}>Waiting for synthesis updates… Start Demo to see motion.</div>
  }
  // Dynamic scaling so clusters are not cramped; pad margins
  const tms = Array.isArray(data?.teams) ? data.teams : []
  const xs = tms.map((s) => s.center_x ?? 0)
  const ys = tms.map((s) => s.center_y ?? 0)
  const minX = Math.min(0, ...xs)
  const maxX = Math.max(1, ...xs)
  const minY = Math.min(0, ...ys)
  const maxY = Math.max(1, ...ys)
  const spanX = Math.max(0.2, maxX - minX)
  const spanY = Math.max(0.2, maxY - minY)
  const margin = 0.08
  const toXY = (x, y) => {
    const xn = ((x ?? 0) - minX) / (spanX || 1)
    const yn = ((y ?? 0) - minY) / (spanY || 1)
    const sx = Math.min(1, Math.max(0, xn * (1 - 2 * margin) + margin))
    const sy = Math.min(1, Math.max(0, yn * (1 - 2 * margin) + margin))
    return [pad + (w - 2 * pad) * sx, pad + (h - 2 * pad) * (1 - sy)]
  }
  const gbest = data.gbest_team || {}
  const [gx, gy] = toXY(gbest.center_x ?? data.gbest?.coverage ?? 0, gbest.center_y ?? data.gbest?.fitness ?? 0)
  return (
    <svg width={w} height={h} style={bg}>
      {/* axes */}
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#ddd" />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#ddd" />
      {/* grid ticks */}
      {([0,0.25,0.5,0.75,1]).map((tick,i)=>{
        const [tx,_y0]=toXY(tick,0); const [_x0,ty]=toXY(0,tick)
        return (
          <g key={`g-${i}`}> 
            <line x1={tx} y1={pad} x2={tx} y2={h-pad} stroke="#f1f1f1" />
            <text x={tx} y={h - pad + 14} fontSize={11} textAnchor="middle" fill="#999">{tick}</text>
            <line x1={pad} y1={ty} x2={w-pad} y2={ty} stroke="#f1f1f1" />
            <text x={pad - 12} y={ty+4} fontSize={11} textAnchor="end" fill="#999">{tick}</text>
          </g>
        )
      })}
      {/* labels */}
      <text x={w/2} y={h - 8} textAnchor="middle" fontSize={12} fill="#666">coverage</text>
      <text x={12} y={h/2} transform={`rotate(-90 12 ${h/2})`} textAnchor="middle" fontSize={12} fill="#666">fitness</text>
      {/* teams as role particles and legend */}
      {(() => {
        const palette = { Coordinator: '#9b59b6', Planner: '#2980b9', Executor: '#2ecc71', Verifier: '#e67e22', Researcher: '#16a085', Critic: '#c0392b' }
        const unitToPx = (dx) => { const [x1] = toXY(0,0), [x2] = toXY(dx,0); return Math.abs(x2-x1) }
        const ring = 0.12
        // 5x larger dots: min ~6px, scale with width
        const baseR = Math.max(6, unitToPx(0.006))
        const elems = []
        const tms = Array.isArray(data?.teams) ? data.teams : []
        for (let i=0;i<tms.length;i++){
          const t = tms[i]
          const roles = Array.isArray(t.roles) ? t.roles : []
          const n = Math.max(roles.length, 1)
          for (let k=0;k<n;k++){
            const theta = (2*Math.PI*k)/n
            const rx = (t.center_x ?? 0) + ring * Math.cos(theta)
            const ry = (t.center_y ?? 0) + ring * Math.sin(theta)
            const [px, py] = toXY(rx, ry)
            const role = roles[k] || 'Executor'
            const fill = palette[role] || '#3498DB'
            elems.push(<circle key={`p-${i}-${k}`} cx={px} cy={py} r={baseR} fill={fill} fillOpacity="0.85" />)
          }
        }
        // legend
        elems.push(
          <g key="legend" transform={`translate(${w - pad - 140}, ${pad})`}>
            <rect x={-8} y={-8} width={140} height={110} fill="#fff" stroke="#eee" />
            {Object.entries(palette).map(([name, color], idx) => (
              <g key={name} transform={`translate(0, ${idx * 18})`}>
                <circle cx={0} cy={0} r={5} fill={color} />
                <text x={12} y={4} fontSize={11} fill="#555">{name}</text>
              </g>
            ))}
          </g>
        )
        return elems
      })()}

      {/* faint seeker particles drifting from random positions toward gbest */}
      {(() => {
        const unitToPx = (dx) => { const [x1] = toXY(0,0), [x2] = toXY(dx,0); return Math.abs(x2-x1) }
        const seekers = Math.min(40, Math.max(10, Math.floor((visibleN || 0) * 1.5)))
        const nodes = []
        for (let i=0;i<seekers;i++){
          const rseed = (i * 9301 + (wanderTick||0) * 17) % 10000
          const rx = (rseed % 1000) / 1000
          const ry = (((rseed / 1000)|0) % 1000) / 1000
          const u = Math.min(1, Math.max(0, (data?.iteration || 0) / Math.max(1, data?.total_iters || 1)))
          const ux = rx * (1 - u) + (gxUnit ?? 0.5) * u
          const uy = ry * (1 - u) + (gyUnit ?? 0.5) * u
          const [px, py] = toXY(ux, uy)
          nodes.push(<circle key={`sk-${i}`} cx={px} cy={py} r={Math.max(0.4, unitToPx(0.0003))} fill="#7f8c8d" fillOpacity={0.25} />)
        }
        return nodes
      })()}
      {/* gbest with golden ring */}
      <circle cx={gx} cy={gy} r={8} fill="#E74C3C" />
      {(() => { const unitToPx = (dx) => { const [x1] = toXY(0,0), [x2] = toXY(dx,0); return Math.abs(x2-x1) }; const rr = unitToPx(0.12); return <circle cx={gx} cy={gy} r={rr} fill="none" stroke="#FFD700" strokeDasharray="4 4" strokeWidth={2} /> })()}
      <text x={gx + 10} y={gy} fontSize={12} fill="#FFD700">best cluster</text>
    </svg>
  )
}

// Enhanced PSO visualization with tweening and dynamic scaling
function PSOVis2({ data, demoRunning, revealMs = 220, neighborSpeed = 0.004, spread = 1.35, dotSize = 6 }) {
  const w = 900, h = 520, pad = 60
  const bg = { width: w, height: h, border: '1px solid #ddd', background: '#fff' }
  const palette = { Coordinator: '#9b59b6', Planner: '#2980b9', Executor: '#2ecc71', Verifier: '#e67e22', Researcher: '#16a085', Critic: '#c0392b' }

  const prevRef = React.useRef({ teams: [], best: {} })
  const fromRef = React.useRef({ teams: [], best: {} })
  const [t, setT] = React.useState(1)
  const [wanderTick, setWanderTick] = React.useState(0)

  // On new snapshot, move current -> prev and start tween
  React.useEffect(() => {
    fromRef.current = prevRef.current
    prevRef.current = {
      teams: Array.isArray(data?.teams) ? data.teams : [],
      best: data?.gbest_team || {}
    }
    setT(0)
    let raf
    const start = performance.now()
    const dur = 450
    const step = (now) => {
      const k = Math.min(1, (now - start) / dur)
      setT(k)
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [data && data.teams])
  React.useEffect(() => {
    // Slow background tick so neighbor communication is more readable
    const id = setInterval(() => setWanderTick((v) => (v + 1) % 1000000), 120)
    return () => clearInterval(id)
  }, [])

  const lerp = (a, b, k) => (a ?? b ?? 0) * (1 - k) + (b ?? a ?? 0) * k
  const a = fromRef.current.teams || []
  const b = prevRef.current.teams || []
  const n = Math.max(a.length, b.length)
  const teams = Array.from({ length: n }).map((_, i) => {
    const p = a[i] || a[a.length - 1] || {}
    const c = b[i] || b[b.length - 1] || {}
    return {
      roles: c.roles || p.roles || [],
      center_x: lerp(p.center_x, c.center_x, t),
      center_y: lerp(p.center_y, c.center_y, t),
      workflow_len: c.workflow_len ?? p.workflow_len ?? 0,
    }
  })
  const gb0 = fromRef.current.best || {}
  const gb1 = prevRef.current.best || {}
  const gxUnit = lerp(gb0.center_x, gb1.center_x, t)
  const gyUnit = lerp(gb0.center_y, gb1.center_y, t)

  // Dynamic scaling (coverage/fitness domain -> canvas) with margins
  const xs = (prevRef.current.teams || []).map((s) => s.center_x ?? 0)
  const ys = (prevRef.current.teams || []).map((s) => s.center_y ?? 0)
  const minX = Math.min(0, ...xs), maxX = Math.max(1, ...xs)
  const minY = Math.min(0, ...ys), maxY = Math.max(1, ...ys)
  const spanX = Math.max(0.2, maxX - minX), spanY = Math.max(0.2, maxY - minY)
  const margin = 0.08
  // Linear normalization to [0,1] with margins
  const to01 = (x, y) => {
    const xn = ((x ?? 0) - minX) / (spanX || 1)
    const yn = ((y ?? 0) - minY) / (spanY || 1)
    const sx = Math.min(1, Math.max(0, xn * (1 - 2 * margin) + margin))
    const sy = Math.min(1, Math.max(0, yn * (1 - 2 * margin) + margin))
    return [sx, sy]
  }
  // Plot mapping with spread factor to visually separate clusters
  const toXY = (x, y) => {
    const [sx, sy] = to01(x, y)
    const sxs = Math.min(1, Math.max(0, 0.5 + (sx - 0.5) * spread))
    const sys = Math.min(1, Math.max(0, 0.5 + (sy - 0.5) * spread))
    return [pad + (w - 2 * pad) * sxs, pad + (h - 2 * pad) * (1 - sys)]
  }
  // Tick mapping (no spread) to keep even spacing
  const tickXY = (x, y) => {
    const [sx, sy] = to01(x, y)
    return [pad + (w - 2 * pad) * sx, pad + (h - 2 * pad) * (1 - sy)]
  }
  const [gx, gy] = toXY(gxUnit ?? 0, gyUnit ?? 0)
  const unitToPx = (dx) => { const [x1] = tickXY(0,0), [x2] = tickXY(dx,0); return Math.abs(x2-x1) }
  const ring = 0.16
  // 5x larger dots: min ~6px, scale with width
  const baseR = Math.max(dotSize, unitToPx(dotSize/1000))

  // Progressive reveal of teams (one-by-one) per snapshot
  const revealRef = React.useRef(null)
  const [visibleN, setVisibleN] = React.useState(0)
  React.useEffect(() => {
    // Reset and start revealing teams one at a time
    if (revealRef.current) clearInterval(revealRef.current)
    setVisibleN(0)
    const target = (prevRef.current.teams || []).length
    const stepMs = revealMs // UI-controlled
    if (target > 0) {
      revealRef.current = setInterval(() => {
        setVisibleN((n) => {
          const next = Math.min(n + 1, target)
          if (next >= target && revealRef.current) {
            clearInterval(revealRef.current)
            revealRef.current = null
          }
          return next
        })
      }, stepMs)
    }
    return () => { if (revealRef.current) { clearInterval(revealRef.current); revealRef.current = null } }
  }, [prevRef.current && prevRef.current.teams])

  return (
    <svg width={w} height={h} style={bg}>
      {/* axes */}
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#bbb" strokeWidth={1.2} />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#bbb" strokeWidth={1.2} />
      <text x={w/2} y={h - 12} textAnchor="middle" fontSize={14} fill="#444">coverage</text>
      <text x={20} y={h/2} transform={`rotate(-90 20 ${h/2})`} textAnchor="middle" fontSize={14} fill="#444">fitness</text>
      {([0,0.25,0.5,0.75,1]).map((tck,i)=>{
        const vx = (minX + tck*spanX), vy = (minY + tck*spanY)
        const [tx] = tickXY(vx, 0), [,ty] = tickXY(0, vy)
        return (
          <g key={`tick-${i}`}>
            <line x1={tx} y1={pad} x2={tx} y2={h - pad} stroke="#eee" />
            <text x={tx} y={h - pad + 18} fontSize={12} textAnchor="middle" fill="#666">{vx.toFixed(2)}</text>
            <line x1={pad} y1={ty} x2={w - pad} y2={ty} stroke="#eee" />
            <text x={pad - 16} y={ty + 4} fontSize={12} textAnchor="end" fill="#666">{vy.toFixed(2)}</text>
          </g>
        )
      })}

      {/* clusters */}
      {/* Start random across space, then converge across first few iterations */}
      {(() => {
        const spawnIters = 8
        const iter = (data?.iteration || 1)
        const s = Math.min(1, Math.max(0, iter / spawnIters))
        // Show all teams each iteration (no progressive slice)
        const shown = teams
        const keyOf = (tt) => (Array.isArray(tt.roles)?tt.roles.join('|'):'') + ':' + (tt.workflow_len||0)
        const hash = (str) => { let h=0; for (let i=0;i<str.length;i++){ h=((h<<5)-h)+str.charCodeAt(i)|0 } return Math.abs(h) }
        const settle = Math.min(1, (data?.iteration || 0) / Math.max(1, data?.total_iters || 1))
        return shown.map((t, i) => {
          const roles = Array.isArray(t.roles) ? t.roles : []
          const nroles = Math.max(roles.length, 1)
          // Random seeded start -> converge to target with slight attraction to gbest
          const kk = keyOf(t)
          const h = hash(kk)
          const rx0 = ((h % 1000) / 1000)
          const ry0 = (((Math.floor(h / 1000)) % 1000) / 1000)
          const targetX = (t.center_x ?? 0)
          const targetY = (t.center_y ?? 0)
          const attract = Math.min(0.35, 0.10 + 0.25 * settle)
          const sxTarget = targetX * (1 - attract) + (gxUnit ?? targetX) * attract
          const syTarget = targetY * (1 - attract) + (gyUnit ?? targetY) * attract
          let sx = rx0 * (1 - s) + sxTarget * s
          let sy = ry0 * (1 - s) + syTarget * s
          const k = hash(kk) % 10000
          const phase = (k * 0.0003 + wanderTick * 0.12)
          const amp = (1 - settle) * 0.04 + 0.008
          sx += Math.cos(phase) * amp
          sy += Math.sin(phase * 1.3) * amp
          const [cx, cy] = toXY(sx, sy)
          const dots = []
          for (let k = 0; k < nroles; k++) {
            const theta = (2 * Math.PI * k) / nroles
            // scatter roles (avoid perfect ring) with per-role jitter and iteration-dependent compactness
            const rrnd = (hash((roles[k]||'') + ':' + i + ':' + k) % 100) / 100
            const rr = ring * (0.35 + 1.2 * rrnd) * (1 - 0.4 * settle)
            const ang = theta + ((hash('a'+i+'-'+k) % 314) / 100)
            const rx = sx + rr * Math.cos(ang)
            const ry = sy + rr * Math.sin(ang)
            const [px, py] = toXY(rx, ry)
            const role = roles[k] || 'Executor'
            const fill = palette[role] || '#3498DB'
            dots.push(<circle key={`p2-${i}-${k}`} cx={px} cy={py} r={baseR} fill={fill} stroke="#fff" strokeWidth={1.2} fillOpacity={0.95} />)
          }
          if (!roles.length) dots.push(<circle key={`t2-${i}`} cx={cx} cy={cy} r={baseR} fill="#3498DB" stroke="#fff" strokeWidth={1.2} fillOpacity={0.9} />)
          return <g key={`team2-${i}`}>{dots}</g>
        })
      })()}

      {/* communication lines removed per request */}

      {/* Show best cluster highlight ONLY at the very end; remove red dot */}
      {(() => {
        const total = data?.total_iters || 0
        const iter = data?.iteration || 0
        if (!total || iter < total) return null
        // Draw a gold diamond hull around the best team instead of a perfect circle
        const rx = unitToPx(ring * 0.9)
        const ry = unitToPx(ring * 0.7)
        const pts = [
          `${gx - rx},${gy}`,
          `${gx},${gy - ry}`,
          `${gx + rx},${gy}`,
          `${gx},${gy + ry}`,
        ].join(' ')
        return (
          <g>
            <polygon points={pts} fill="none" stroke="#FFD700" strokeDasharray="6 4" strokeWidth={2.2} />
            <text x={gx + rx + 8} y={gy} fontSize={12} fill="#FFD700">best cluster</text>
          </g>
        )
      })()}

      {/* legend */}
      <g transform={`translate(${w - pad - 140}, ${pad})`}>
        <rect x={-8} y={-8} width={140} height={110} fill="#fff" stroke="#eee" />
        {Object.entries(palette).map(([name, color], idx) => (
          <g key={name} transform={`translate(0, ${idx * 18})`}>
            <circle cx={0} cy={0} r={5} fill={color} />
            <text x={12} y={4} fontSize={11} fill="#555">{name}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}
