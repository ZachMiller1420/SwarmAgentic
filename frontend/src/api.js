// Auto-detect dev server and default API base to FastAPI on 8000 when using Vite (e.g., :5173)
const isLocal = ['localhost', '127.0.0.1'].includes(location.hostname)
const isDevHost = isLocal && location.port && location.port !== '8000'
const DEFAULT_DEV_API = isDevHost ? 'http://localhost:8000' : ''

const API_BASE = import.meta.env.VITE_API_BASE_URL || DEFAULT_DEV_API // proxy via nginx if empty

export async function getState() {
  const r = await fetch(`${API_BASE}/api/state`)
  if (!r.ok) throw new Error('state fetch failed')
  return r.json()
}

export async function getMetrics() {
  const r = await fetch(`${API_BASE}/api/metrics`)
  if (!r.ok) throw new Error('metrics fetch failed')
  return r.json()
}

export async function startTraining() {
  const r = await fetch(`${API_BASE}/api/train`, { method: 'POST' })
  if (!r.ok) throw new Error('start training failed')
  return r.json()
}

export async function startDemo() {
  const r = await fetch(`${API_BASE}/api/demo`, { method: 'POST' })
  if (!r.ok) throw new Error('start demo failed')
  return r.json()
}

export async function stopDemo() {
  const r = await fetch(`${API_BASE}/api/stop_demo`, { method: 'POST' })
  if (!r.ok) throw new Error('stop demo failed')
  return r.json()
}

export async function pause() {
  const r = await fetch(`${API_BASE}/api/pause`, { method: 'POST' })
  if (!r.ok) throw new Error('pause failed')
  return r.json()
}

export async function resume() {
  const r = await fetch(`${API_BASE}/api/resume`, { method: 'POST' })
  if (!r.ok) throw new Error('resume failed')
  return r.json()
}

export async function reset() {
  const r = await fetch(`${API_BASE}/api/reset`, { method: 'POST' })
  if (!r.ok) throw new Error('reset failed')
  return r.json()
}

export function makeWS() {
  const url = import.meta.env.VITE_WS_URL
  if (url) return new WebSocket(url)
  const proto = (isDevHost ? 'ws' : (location.protocol === 'https:' ? 'wss' : 'ws'))
  const host = isDevHost ? 'localhost:8000' : location.host
  return new WebSocket(`${proto}://${host}/ws/stream`)
}

export async function getTasks() {
  const r = await fetch(`${API_BASE}/api/tasks`)
  if (!r.ok) throw new Error('tasks fetch failed')
  return r.json()
}

export async function selectTasks(payload) {
  const r = await fetch(`${API_BASE}/api/tasks/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  })
  if (!r.ok) throw new Error('select tasks failed')
  return r.json()
}

export async function updatePSOSettings(payload) {
  const r = await fetch(`${API_BASE}/api/pso_settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  })
  if (!r.ok) throw new Error('update settings failed')
  return r.json()
}

export async function exportAll() {
  const r = await fetch(`${API_BASE}/api/export`, { method: 'POST' })
  if (!r.ok) throw new Error('export failed')
  return r.json()
}

export async function getKnowledgeGraph() {
  const r = await fetch(`${API_BASE}/api/knowledge_graph`)
  if (!r.ok) throw new Error('kg fetch failed')
  return r.json()
}
