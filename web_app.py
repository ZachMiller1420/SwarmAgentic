"""
FastAPI Web Server for SwarmAgentic Demo
Wraps the existing agent, learning, demo, and metrics into a web API
and serves a simple web UI.
"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
import os
from fastapi.staticfiles import StaticFiles

from src.core.ai_agent import PhDLevelAIAgent
from src.training.learning_system import AdaptiveLearningSystem
from src.monitoring.quality_metrics import RealTimeMetricsCollector
from src.demonstration.interactive_demo import InteractiveDemonstrationSystem


class AppState:
    def __init__(self):
        # Config (aligned with main.py defaults)
        self.config = {
            # Use a small model for fast web startup; can override via env BERT_MODEL_ID
            # Align with desktop GUI default; will gracefully fall back to a Hub model
            "bert_model_path": "bert-base-uncased-mrpc/bert-base-uncased-mrpc/huggingface_Intel_bert-base-uncased-mrpc_v1",
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
        # Main asyncio loop (captured on startup for thread-safe dispatch)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Track last applied task preset for UI visibility
        self.task_preset: Optional[str] = None

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
            from pathlib import Path
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
            # Model selection parity with desktop, with robust fallback to a Hub model
            desired = os.environ.get("BERT_MODEL_ID") or self.config.get("bert_model_path", "")
            # If looks like a local path but doesn't exist, fall back to a public fine-tuned MRPC model on the Hub
            def _is_hub_repo(s: str) -> bool:
                # crude check: org/name pattern without path separators beyond one slash
                import re
                return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", s or ""))
            try:
                fallback = os.environ.get("BERT_MODEL_FALLBACK_HUB", "textattack/bert-base-uncased-MRPC")
                if desired:
                    p = Path(desired)
                    if p.exists() or _is_hub_repo(desired):
                        # honor user/env-provided value when valid
                        os.environ["BERT_MODEL_ID"] = desired
                    else:
                        # Force fallback if provided value is invalid (fixes bad local paths)
                        os.environ["BERT_MODEL_ID"] = fallback
                else:
                    # Final fallback to a tiny model if nothing else is specified
                    os.environ["BERT_MODEL_ID"] = "prajjwal1/bert-tiny"
            except Exception:
                os.environ["BERT_MODEL_ID"] = "prajjwal1/bert-tiny"
        except Exception:
            pass

        if self.agent is None:
            self.agent = PhDLevelAIAgent(
                bert_model_path=self.config["bert_model_path"],
                academic_paper_path=self.config["academic_paper_path"],
            )
            # If a custom training text file exists, use it (parity with desktop app)
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

# Serve built React UI if present (frontend/dist)
try:
    dist_dir = Path("frontend/dist")
    if dist_dir.exists():
        # Serve assets used by Vite build
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
except Exception:
    pass


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

    # Avoid duplicate registration by checking a flag
    if getattr(a, "_stream_callbacks_registered", False):
        return
    a._stream_callbacks_registered = True

    # Agent callbacks
    def _dispatch(message: dict):
        try:
            # Prefer current loop if in async context
            loop = asyncio.get_running_loop()
            loop.create_task(hub.send_all(message))
        except RuntimeError:
            # Likely called from worker thread; use app loop
            loop = getattr(state, 'loop', None)
            if loop:
                asyncio.run_coroutine_threadsafe(hub.send_all(message), loop)

    def on_state_change(data):
        try:
            _dispatch({"type": "agent_state", "data": jsonable_encoder(data)})
        except Exception:
            pass

    def on_progress(data):
        try:
            _dispatch({"type": "progress", "data": jsonable_encoder(data)})
        except Exception:
            pass

    def on_thought(data):
        try:
            _dispatch({"type": "thought", "data": jsonable_encoder(data)})
        except Exception:
            pass

    a.register_callback('state_change', on_state_change)
    a.register_callback('progress', on_progress)
    a.register_callback('thought', on_thought)

    # Demo callbacks
    if d:
        try:
            d.register_callback('progress', lambda ev: asyncio.create_task(hub.send_all({"type": "demo_progress", "data": jsonable_encoder(ev)})))
            d.register_callback('response', lambda ev: asyncio.create_task(hub.send_all({"type": "demo_response", "data": jsonable_encoder(ev)})))
            d.register_callback('completion', lambda ev: asyncio.create_task(hub.send_all({"type": "demo_complete", "data": jsonable_encoder(ev)})))
        except Exception:
            pass

    # Metrics periodic push (use collector's callback mechanism if available)
    if m:
        try:
            def on_metrics_update(event=None, *args, **kwargs):
                try:
                    payload = m.get_comprehensive_metrics()
                    coro = hub.send_all({"type": "metrics", "data": jsonable_encoder(payload)})
                    try:
                        # If in async context, schedule directly
                        loop = asyncio.get_running_loop()
                        loop.create_task(coro)
                    except RuntimeError:
                        # Likely called from metrics background thread; use captured app loop
                        loop = getattr(state, 'loop', None)
                        if loop:
                            asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception:
                    pass

            # Register once; support either explicit registrar or direct list
            if getattr(m, "_web_stream_cb_registered", False) is not True:
                if hasattr(m, 'register_update_callback'):
                    m.register_update_callback(on_metrics_update)
                elif hasattr(m, 'update_callbacks'):
                    # Avoid duplicate registration by name check where possible
                    if not any(getattr(cb, '__name__', '') == 'on_metrics_update' for cb in getattr(m, 'update_callbacks', [])):
                        m.update_callbacks.append(on_metrics_update)
                setattr(m, "_web_stream_cb_registered", True)
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def root_page():
    # Try multiple locations for index.html to support both root and subfolder layouts
    candidate_paths = [
        Path("frontend/dist/index.html"),
        Path("web/index.html"),
        Path(__file__).parent / "web/index.html",
        Path(__file__).parent.parent / "web/index.html",
    ]
    for p in candidate_paths:
        if p.exists():
            return FileResponse(str(p))
    # Fallback rich page with State + Metrics + Live Stream
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
            <div class=\"row\">
              <button onclick=\"selectPreset('default')\">Use Default Tasks</button>
              <button onclick=\"selectPreset('from_training')\">Use From Training</button>
              <button onclick=\"exportAll()\">Export Results</button>
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
              async function selectPreset(name){
                await fetch('/api/tasks/select', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({preset:name})});
                await load();
              }
              async function exportAll(){
                const r = await fetch('/api/export', {method:'POST'}).then(r=>r.json());
                alert('Exported files: '+ JSON.stringify(r));
              }
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
@app.on_event("startup")
async def on_startup():
    # Capture the server event loop for thread-safe callbacks
    try:
        state.loop = asyncio.get_running_loop()
    except Exception:
        state.loop = None
    state.init_metrics()
    # If the agent is pre-created, ensure callbacks are wired
    try:
        _wire_stream_callbacks()
    except Exception:
        pass


@app.get("/api/state")
async def get_state():
    a = state.agent
    d = state.demo
    payload = {
        "agent_state": (a.state.__dict__ if a else None),
        "learning_progress": (a.learning_progress.__dict__ if a else None),
        # Prefer task-focused displayed concepts when available
        "learned_concepts": (getattr(a, 'displayed_concepts', None) if a else None) or (a.learned_concepts if a else []),
        "pso_tasks": (a.pso_tasks if a and a.pso_tasks else []),
        "task_preset": state.task_preset,
        "recommended_concepts": (getattr(a, 'recommended_concepts', []) if a else []),
        "demo_running": (d.is_running if d else False),
        "demo_stats": (d.get_demo_statistics() if d else None),
    }
    # Let FastAPI handle JSON encoding (datetime safe)
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

    # Avoid duplicate runs
    if agent.state.is_training:
        return {"status": "already_running"}

    try:
        _wire_stream_callbacks()
        # Kick off training; the agent will update state and stream progress
        task = asyncio.create_task(agent.start_training())
        # Log task completion or error to stream
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
    agent = state.agent
    if not agent:
        return {"status": "error", "message": "components unavailable"}

    if not agent.learned_concepts:
        return {"status": "not_ready", "message": "Train the agent first"}

    _wire_stream_callbacks()
    # Immediate broadcast of demo state
    await hub.send_all({"type": "demo_state", "data": jsonable_encoder({"running": True})})
    task = asyncio.create_task(agent.start_demonstration())
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
    if state.agent:
        try:
            state.agent.stop()
        except Exception:
            pass
    return {"status": "stopped"}


@app.post("/api/pso_settings")
async def api_pso_settings(payload: dict = Body(default={})):
    """Update PSO runtime settings (applies to subsequent iterations).

    Payload keys (all optional):
      - pause: float seconds between iterations (TEXT_PSO_PAUSE)
      - pop: int population size (TEXT_PSO_POP)
      - iters: int total iterations (TEXT_PSO_ITERS)
    """
    try:
        changed = {}
        if isinstance(payload, dict):
            if 'pause' in payload and payload['pause'] is not None:
                try:
                    val = float(payload['pause'])
                    os.environ['TEXT_PSO_PAUSE'] = str(max(0.0, val))
                    changed['pause'] = os.environ['TEXT_PSO_PAUSE']
                except Exception:
                    pass
            if 'pop' in payload and payload['pop'] is not None:
                try:
                    val = int(payload['pop'])
                    if val > 0:
                        os.environ['TEXT_PSO_POP'] = str(val)
                        changed['pop'] = os.environ['TEXT_PSO_POP']
                except Exception:
                    pass
            if 'iters' in payload and payload['iters'] is not None:
                try:
                    val = int(payload['iters'])
                    if val > 0:
                        os.environ['TEXT_PSO_ITERS'] = str(val)
                        changed['iters'] = os.environ['TEXT_PSO_ITERS']
                except Exception:
                    pass
        return {"status": "ok", "changed": changed, "effective": {
            "pause": os.environ.get('TEXT_PSO_PAUSE'),
            "pop": os.environ.get('TEXT_PSO_POP'),
            "iters": os.environ.get('TEXT_PSO_ITERS'),
        }}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Additional controls to mimic desktop GUI ---
@app.post("/api/pause")
async def api_pause():
    if state.agent:
        state.agent.pause()
        await hub.send_all({"type": "agent_state", "data": jsonable_encoder(state.agent.state.__dict__)})
        return {"status": "paused"}
    return {"status": "error", "message": "no agent"}


@app.post("/api/resume")
async def api_resume():
    if state.agent:
        state.agent.resume()
        await hub.send_all({"type": "agent_state", "data": jsonable_encoder(state.agent.state.__dict__)})
        return {"status": "resumed"}
    return {"status": "error", "message": "no agent"}


@app.post("/api/reset")
async def api_reset():
    if state.agent:
        state.agent.reset()
        await hub.send_all({"type": "agent_state", "data": jsonable_encoder(state.agent.state.__dict__)})
        return {"status": "reset"}
    return {"status": "error", "message": "no agent"}


@app.get("/api/thoughts")
async def api_thoughts():
    if not state.agent:
        return []
    try:
        return jsonable_encoder(state.agent.get_thought_process())
    except Exception:
        return []


@app.get("/api/memory")
async def api_memory():
    if not state.agent:
        return {}
    try:
        return jsonable_encoder(state.agent.get_working_memory())
    except Exception:
        return {}


@app.get("/api/tasks")
async def get_tasks():
    """Return task suggestions for synthesis/demo from data/tasks.json and agent recs."""
    tasks_payload = {"source": "file", "categories": {}, "recommended": [], "options": []}
    try:
        import json
        from pathlib import Path
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
    # Provide selection options: presets + category names
    try:
        cat_names = list(tasks_payload["categories"].keys())
    except Exception:
        cat_names = []
    tasks_payload["options"] = ["Default (Built-in)", "From Training"] + cat_names
    return jsonable_encoder(tasks_payload)


@app.post("/api/tasks/select")
async def select_tasks(payload: dict = Body(default={})):  # accept categories or explicit tasks
    """Apply selected tasks to guide synthesis demo (PSO).

    Payload examples:
    - {"categories": ["Research", "Software Delivery"]}
    - {"tasks": ["plan tasks, execute plan, verify results", ...]}
    """
    try:
        state.ensure_agent()
        agent = state.agent
        if not agent:
            return {"status": "error", "message": "agent unavailable"}

        selected_tasks = []
        preset = (payload or {}).get("preset")
        # Handle presets first
        if isinstance(preset, str):
            key = preset.strip().lower()
            if key in ("default", "default (built-in)"):
                agent.set_pso_tasks(None)
                state.task_preset = "Default"
                await hub.send_all({"type": "agent_state", "data": jsonable_encoder(agent.state.__dict__)})
                return {"status": "ok", "applied": 0, "preset": state.task_preset}
            if key in ("from_training", "from training"):
                recs = list(getattr(agent, 'recommended_concepts', []) or [])
                # Derive tasks from recommendations (mirror training seeding logic)
                derived = []
                if recs:
                    derived.append(f"plan tasks, execute plan, verify results for {recs[0]}")
                if len(recs) > 1:
                    derived.append(f"optimize workflow to address {recs[1]}")
                if len(recs) > 2:
                    derived.append(f"coverage of {recs[2]} with verification")
                agent.set_pso_tasks(derived if derived else None)
                state.task_preset = "From Training"
                await hub.send_all({"type": "agent_state", "data": jsonable_encoder(agent.state.__dict__)})
                return {"status": "ok", "applied": len(derived), "preset": state.task_preset}
        # Collect tasks from categories file
        try:
            import json
            cats = (payload or {}).get("categories") or []
            if cats:
                tpath = Path("data/tasks.json")
                if tpath.exists():
                    with open(tpath, 'r', encoding='utf-8') as f:
                        allcats = json.load(f)
                    for c in cats:
                        selected_tasks.extend(allcats.get(c, []))
        except Exception:
            pass

        # Include any explicit tasks provided
        try:
            explicit = (payload or {}).get("tasks") or []
            if explicit:
                selected_tasks.extend(explicit)
        except Exception:
            pass

        # De-duplicate while preserving order
        seen = set()
        selected_tasks = [t for t in selected_tasks if not (t in seen or seen.add(t))]

        if selected_tasks:
            agent.set_pso_tasks(selected_tasks)
            try:
                agent.apply_task_focus(selected_tasks)
            except Exception:
                pass
        else:
            agent.set_pso_tasks(None)
            try:
                agent.apply_task_focus(None)
            except Exception:
                pass

        # Let UI know tasks changed via state broadcast
        try:
            await hub.send_all({"type": "agent_state", "data": jsonable_encoder(agent.state.__dict__)})
        except Exception:
            pass

        state.task_preset = None
        return {"status": "ok", "applied": len(selected_tasks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/knowledge_graph")
async def api_knowledge_graph():
    """Return the exported knowledge graph if available."""
    try:
        state.ensure_agent()
        a = state.agent
        if not a:
            return {"status": "error", "message": "agent unavailable"}
        export = getattr(a, 'knowledge_graph_export', None)
        if not export:
            return {"status": "not_ready"}
        return jsonable_encoder({"status": "ok", "graph": export})
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/export")
async def api_export_all():
    """Export metrics, demo results, knowledge base, and knowledge graph to results/."""
    try:
        from datetime import datetime
        import json as _json
        results = {}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("results"); outdir.mkdir(exist_ok=True)
        # Metrics
        if state.metrics:
            mfile = outdir / f"quality_metrics_{ts}.json"
            state.metrics.export_metrics(str(mfile))
            results["metrics"] = str(mfile)
        # Demo results
        if state.demo:
            dfile = outdir / f"demo_results_{ts}.json"
            state.demo.export_demo_results(str(dfile))
            results["demo_results"] = str(dfile)
        # Knowledge base
        if state.agent and hasattr(state.agent, 'knowledge_base'):
            kbfile = outdir / f"knowledge_base_{ts}.json"
            with open(kbfile, 'w', encoding='utf-8') as f:
                _json.dump(state.agent.knowledge_base, f, indent=2, default=str)
            results["knowledge_base"] = str(kbfile)
        # Knowledge graph export (if available)
        if state.agent and getattr(state.agent, 'knowledge_graph_export', None):
            kgfile = outdir / f"knowledge_graph_{ts}.json"
            with open(kgfile, 'w', encoding='utf-8') as f:
                _json.dump(state.agent.knowledge_graph_export, f, indent=2, default=str)
            results["knowledge_graph"] = str(kgfile)
        return {"status": "ok", "files": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await hub.register(websocket)
    try:
        # Send initial snapshot
        try:
            a = state.agent
            d = state.demo
            m = state.metrics
            await hub.send_all({"type": "agent_state", "data": jsonable_encoder(a.state.__dict__ if a else None)})
            await hub.send_all({"type": "metrics", "data": jsonable_encoder(m.get_comprehensive_metrics() if m else {})})
            await hub.send_all({"type": "demo_state", "data": jsonable_encoder({"running": d.is_running if d else False})})
        except Exception:
            pass

        # Keep alive without requiring client messages
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(websocket)
