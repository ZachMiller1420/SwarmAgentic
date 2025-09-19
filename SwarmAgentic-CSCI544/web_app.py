"""
FastAPI Web Server for SwarmAgentic Demo (Container Path)
Mirrors the root web_app.py to ensure the Docker image uses the same implementation.
"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from src.core.ai_agent import PhDLevelAIAgent
from src.training.learning_system import AdaptiveLearningSystem
from src.monitoring.quality_metrics import RealTimeMetricsCollector
from src.demonstration.interactive_demo import InteractiveDemonstrationSystem


class AppState:
    def __init__(self):
        # Use a small model by default for web environment; override via env BERT_MODEL_ID
        self.config = {
            "bert_model_path": "prajjwal1/bert-tiny",
            "academic_paper_path": "academic_summary.md",
            "training_text_path": "data/agent_ops_handbook.txt",
            "quality_metrics_window": 1000,
            "enable_real_time_monitoring": True,
        }

        # Components
        self.metrics: Optional[RealTimeMetricsCollector] = None
        self.agent: Optional[PhDLevelAIAgent] = None
        self.learning: Optional[AdaptiveLearningSystem] = None
        self.demo: Optional[InteractiveDemonstrationSystem] = None

    def init_metrics(self):
        if self.metrics is None:
            self.metrics = RealTimeMetricsCollector(
                window_size=self.config["quality_metrics_window"],
            )
        if self.config["enable_real_time_monitoring"] and not self.metrics.is_monitoring:
            self.metrics.start_monitoring(update_interval=1.0)

    def ensure_agent(self):
        # Light env setup to match desktop behavior
        try:
            import os
            # Load OpenAI API key from local file if present
            if not os.environ.get("OPENAI_API_KEY"):
                key_path = Path("OpenAI-APIkey.txt")
                if key_path.exists():
                    key_text = key_path.read_text(encoding="utf-8")
                    api_key = next((ln.strip() for ln in key_text.splitlines() if ln.strip()), "")
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key
            # Prefer enabling LLM for PSO if key is available
            if os.environ.get("OPENAI_API_KEY"):
                os.environ.setdefault("USE_LLM_PSO", "1")
            # Prefer lighter model if not explicitly set
            os.environ.setdefault("BERT_MODEL_ID", self.config.get("bert_model_path", "prajjwal1/bert-tiny"))
        except Exception:
            pass

        if self.agent is None:
            self.agent = PhDLevelAIAgent(
                bert_model_path=self.config["bert_model_path"],
                academic_paper_path=self.config["academic_paper_path"],
            )
            # If a custom training text file exists, use it
            try:
                tpath = Path(self.config.get("training_text_path", "")).resolve()
                if tpath and tpath.exists():
                    text = tpath.read_text(encoding="utf-8")
                    self.agent.set_training_text(text)
            except Exception:
                pass
        if self.learning is None:
            self.learning = AdaptiveLearningSystem(self.agent.bert_engine)
        if self.demo is None:
            self.demo = InteractiveDemonstrationSystem(self.agent, self.metrics)


state = AppState()

app = FastAPI(title="SwarmAgentic Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Simple broadcast hub for WebSocket streaming ---
class StreamHub:
    def __init__(self):
        self.clients = set()

    async def register(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)

    def unregister(self, websocket: WebSocket):
        try:
            self.clients.remove(websocket)
        except KeyError:
            pass

    async def send_all(self, message: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)


hub = StreamHub()


def _wire_stream_callbacks():
    """Register callbacks exactly once to stream events to clients."""
    a = state.agent
    d = state.demo
    m = state.metrics
    if not a:
        return

    if getattr(a, "_stream_callbacks_registered", False):
        return
    a._stream_callbacks_registered = True

    def on_state_change(data):
        try:
            asyncio.create_task(hub.send_all({"type": "agent_state", "data": jsonable_encoder(data)}))
        except Exception:
            pass

    def on_progress(data):
        try:
            asyncio.create_task(hub.send_all({"type": "progress", "data": jsonable_encoder(data)}))
        except Exception:
            pass

    def on_thought(data):
        try:
            asyncio.create_task(hub.send_all({"type": "thought", "data": jsonable_encoder(data)}))
        except Exception:
            pass

    a.register_callback('state_change', on_state_change)
    a.register_callback('progress', on_progress)
    a.register_callback('thought', on_thought)

    if d:
        try:
            d.register_callback('progress', lambda ev: asyncio.create_task(hub.send_all({"type": "demo_progress", "data": jsonable_encoder(ev)})))
            d.register_callback('response', lambda ev: asyncio.create_task(hub.send_all({"type": "demo_response", "data": jsonable_encoder(ev)})))
            d.register_callback('completion', lambda ev: asyncio.create_task(hub.send_all({"type": "demo_complete", "data": jsonable_encoder(ev)})))
        except Exception:
            pass

    if m and hasattr(m, 'update_callbacks'):
        try:
            def on_metrics_update():
                try:
                    payload = m.get_comprehensive_metrics()
                    asyncio.create_task(hub.send_all({"type": "metrics", "data": jsonable_encoder(payload)}))
                except Exception:
                    pass
            if not any(getattr(cb, '__name__', '') == 'on_metrics_update' for cb in m.update_callbacks):
                m.update_callbacks.append(on_metrics_update)
        except Exception:
            pass


@app.on_event("startup")
async def on_startup():
    state.init_metrics()
    try:
        _wire_stream_callbacks()
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def root_page():
    candidate_paths = [
        Path("web/index.html"),
        Path(__file__).parent / "web/index.html",
        Path(__file__).parent.parent / "web/index.html",
    ]
    for p in candidate_paths:
        if p.exists():
            return FileResponse(str(p))
    return HTMLResponse(
        """
        <!doctype html>
        <html>
          <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>SwarmAgentic Web Demo</title>
            <style>
              body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; }
              h1 { margin-bottom: 8px; }
              .row { display: flex; gap: 12px; margin: 10px 0 20px; }
              button { padding: 10px 14px; border: 0; border-radius: 6px; cursor: pointer; }
              .ok { background: #27AE60; color: #fff; }
              .info { background: #3498DB; color: #fff; }
              .warn { background: #F39C12; color: #fff; }
              pre { background: #f6f8fa; padding: 12px; border-radius: 6px; max-height: 40vh; overflow: auto; }
              .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
              @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
            </style>
          </head>
          <body>
            <h1>SwarmAgentic Web Demo</h1>
            <div class=\"row\">
              <button class=\"ok\" onclick=\"post('/api/train')\">Start Training</button>
              <button class=\"info\" onclick=\"post('/api/demo')\">Start Demo</button>
              <button class=\"warn\" onclick=\"post('/api/stop_demo')\">Stop Demo</button>
            </div>
            <div class=\"grid\">
              <div>
                <h3>State</h3>
                <pre id=\"state\"></pre>
              </div>
              <div>
                <h3>Metrics</h3>
                <pre id=\"metrics\"></pre>
              </div>
            </div>
            <div>
              <h3>Live Stream</h3>
              <pre id=\"stream\"></pre>
            </div>
            <script>
              async function post(url) {
                await fetch(url, { method: 'POST' });
                await load();
              }
              async function load() {
                const [s, m] = await Promise.all([
                  fetch('/api/state').then(r => r.json()),
                  fetch('/api/metrics').then(r => r.json()),
                ]);
                document.getElementById('state').textContent = JSON.stringify(s, null, 2);
                document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);
              }
              setInterval(load, 1000);
              load();
              function connectWS() {
                try {
                  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
                  const out = document.getElementById('stream');
                  ws.onmessage = (ev) => {
                    try { const msg = JSON.parse(ev.data);
                      if (msg.type === 'agent_state' || msg.type === 'progress' || msg.type === 'thought' || msg.type === 'metrics') { load(); }
                      const line = typeof msg === 'string' ? msg : JSON.stringify(msg);
                      out.textContent = (out.textContent + '\\n' + line).split('\\n').slice(-200).join('\\n');
                    } catch { out.textContent = (out.textContent + '\\n' + ev.data).split('\\n').slice(-200).join('\\n'); }
                  };
                  ws.onclose = () => setTimeout(connectWS, 2000);
                } catch (e) { console.error('WS error', e); setTimeout(connectWS, 5000); }
              }
              connectWS();
            </script>
          </body>
        </html>
        """
    )


@app.get("/api/state")
async def get_state():
    a = state.agent
    d = state.demo
    payload = {
        "agent_state": (a.state.__dict__ if a else None),
        "learning_progress": (a.learning_progress.__dict__ if a else None),
        "learned_concepts": (a.learned_concepts if a else []),
        "demo_running": (d.is_running if d else False),
        "demo_stats": (d.get_demo_statistics() if d else None),
    }
    return jsonable_encoder(payload)


@app.get("/api/metrics")
async def get_metrics():
    m = state.metrics
    return jsonable_encoder(m.get_comprehensive_metrics() if m else {})


@app.post("/api/train")
async def start_training():
    state.init_metrics()
    state.ensure_agent()
    agent = state.agent
    if not agent:
        return {"status": "error", "message": "agent unavailable"}

    if agent.state.is_training:
        return {"status": "already_running"}

    try:
        _wire_stream_callbacks()
        agent.state.is_training = True
        agent.state.current_task = "training"
        await hub.send_all({"type": "agent_state", "data": jsonable_encoder(agent.state.__dict__)})
        task = asyncio.create_task(agent.start_training())
        def _on_done(t: asyncio.Task):
            try:
                ok = t.result()
                msg = {"type": "train_complete", "data": {"success": bool(ok)}}
            except Exception as e:
                msg = {"type": "train_complete", "data": {"success": False, "error": str(e)}}
            asyncio.create_task(hub.send_all(jsonable_encoder(msg)))
        task.add_done_callback(_on_done)
        return {"status": "started"}
    except Exception as e:
        return {"status": "error", "message": f"failed to start: {e}"}


@app.post("/api/demo")
async def start_demo():
    state.init_metrics()
    state.ensure_agent()
    demo = state.demo
    agent = state.agent
    if not demo or not agent:
        return {"status": "error", "message": "components unavailable"}

    if demo.is_running:
        return {"status": "already_running"}
    if not agent.learned_concepts:
        return {"status": "not_ready", "message": "Train the agent first"}

    _wire_stream_callbacks()
    await hub.send_all({"type": "demo_state", "data": jsonable_encoder({"running": True})})
    task = asyncio.create_task(demo.start_demonstration())
    def _on_demo_done(t: asyncio.Task):
        try:
            ok = t.result()
            msg = {"type": "demo_complete", "data": {"success": bool(ok)}}
        except Exception as e:
            msg = {"type": "demo_complete", "data": {"success": False, "error": str(e)}}
        asyncio.create_task(hub.send_all(jsonable_encoder(msg)))
    task.add_done_callback(_on_demo_done)
    return {"status": "started"}


@app.post("/api/stop_demo")
async def stop_demo():
    if state.demo:
        state.demo.stop_demonstration()
    return {"status": "stopped"}


@app.get("/api/tasks")
async def get_tasks():
    tasks_payload = {"source": "file", "categories": {}, "recommended": []}
    try:
        import json
        tpath = Path("data/tasks.json")
        if tpath.exists():
            with open(tpath, 'r', encoding='utf-8') as f:
                tasks_payload["categories"] = json.load(f)
    except Exception:
        tasks_payload["categories"] = {}

    try:
        if state.agent and getattr(state.agent, 'recommended_concepts', None):
            tasks_payload["recommended"] = list(state.agent.recommended_concepts)
    except Exception:
        tasks_payload["recommended"] = []
    return jsonable_encoder(tasks_payload)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await hub.register(websocket)
    try:
        try:
            a = state.agent
            d = state.demo
            m = state.metrics
            await hub.send_all({"type": "agent_state", "data": jsonable_encoder(a.state.__dict__ if a else None)})
            await hub.send_all({"type": "metrics", "data": jsonable_encoder(m.get_comprehensive_metrics() if m else {})})
            await hub.send_all({"type": "demo_state", "data": jsonable_encoder({"running": d.is_running if d else False})})
        except Exception:
            pass

        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(websocket)
