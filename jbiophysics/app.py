"""
jbiophysics interactive web app.

Serves a network builder UI + runs simulations via HTTP.
Usage: PYTHONPATH=. python jbiophysics/app.py
"""

import json
import sys
import os
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure jbiophysics is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_simulation(config):
    """Run a simulation from a JSON config dict."""
    from jbiophysics.compose import NetBuilder
    from jbiophysics.export import ResultsReport
    
    seed = config.get("seed", 42)
    dt = config.get("dt", 0.1)
    t_max = config.get("t_max", 1000.0)
    
    builder = NetBuilder(seed=seed)
    
    for pop in config.get("populations", []):
        builder.add_population(
            name=pop["name"],
            n=pop["n"],
            cell_type=pop["cell_type"],
            noise_amp=pop.get("noise_amp", 0.05),
            noise_tau=pop.get("noise_tau", 20.0),
        )
    
    for conn in config.get("connections", []):
        builder.connect(
            pre=conn["pre"],
            post=conn["post"],
            synapse=conn["synapse"],
            p=conn.get("p", 0.1),
            g=conn.get("g", None),
        )
    
    for param in config.get("trainable", []):
        builder.make_trainable(param)
    
    net = builder.build()
    
    # Record and simulate
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")
    
    import jaxley as jx
    traces = jx.integrate(net, delta_t=dt, t_max=t_max)
    traces_np = np.array(traces)
    
    report = ResultsReport(
        traces=traces_np,
        dt=dt,
        t_max=t_max,
        metadata=config,
    )
    return report


def build_dashboard_json(report):
    """Build Plotly JSON for the dashboard."""
    from jbiophysics.viz.dashboard import generate_dashboard
    fig = generate_dashboard(report, title="Jbiophysics — Live Simulation")
    return fig.to_json()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jbiophysics — Interactive Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --accent: #58a6ff;
    --accent-glow: rgba(88,166,255,0.15);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border: #30363d;
    --success: #3fb950;
    --warning: #d29922;
    --danger: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
  }
  .app-header {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, #1a1f2e 100%);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .app-header h1 {
    font-size: 20px;
    font-weight: 600;
    background: linear-gradient(135deg, #58a6ff, #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .app-header .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid rgba(88,166,255,0.3);
  }
  .app-layout {
    display: grid;
    grid-template-columns: 380px 1fr;
    min-height: calc(100vh - 56px);
  }
  .sidebar {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    padding: 20px;
    overflow-y: auto;
    max-height: calc(100vh - 56px);
  }
  .main-panel {
    padding: 20px;
    overflow-y: auto;
    max-height: calc(100vh - 56px);
  }
  .section-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title::before {
    content: '';
    width: 3px;
    height: 14px;
    background: var(--accent);
    border-radius: 2px;
  }
  .card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .card-header h3 {
    font-size: 14px;
    font-weight: 500;
  }
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
  }
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .form-group label {
    font-size: 11px;
    color: var(--text-secondary);
    font-weight: 500;
  }
  .form-group input, .form-group select {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    border: 1px solid var(--border);
    transition: all 0.2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border-color: #238636;
  }
  .btn-primary:hover { background: linear-gradient(135deg, #2ea043, #3fb950); transform: translateY(-1px); }
  .btn-primary:active { transform: translateY(0); }
  .btn-sm {
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 5px;
  }
  .btn-accent { background: var(--accent-glow); color: var(--accent); border-color: rgba(88,166,255,0.3); }
  .btn-accent:hover { background: rgba(88,166,255,0.25); }
  .btn-danger { background: rgba(248,81,73,0.1); color: var(--danger); border-color: rgba(248,81,73,0.3); }
  .btn-danger:hover { background: rgba(248,81,73,0.2); }
  .run-btn {
    width: 100%;
    padding: 12px;
    font-size: 15px;
    font-weight: 600;
    justify-content: center;
    margin-top: 8px;
  }
  .run-btn.running {
    background: linear-gradient(135deg, var(--warning), #e3b341);
    border-color: var(--warning);
    pointer-events: none;
  }
  .pop-item, .conn-item {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 8px;
    position: relative;
    animation: slideIn 0.2s ease;
  }
  @keyframes slideIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
  .pop-item .remove-btn, .conn-item .remove-btn {
    position: absolute;
    top: 6px;
    right: 6px;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 16px;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .pop-item .remove-btn:hover, .conn-item .remove-btn:hover { color: var(--danger); background: rgba(248,81,73,0.1); }
  .status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-secondary);
  }
  .status-dot.ready { background: var(--success); box-shadow: 0 0 6px var(--success); }
  .status-dot.running { background: var(--warning); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  #dashboard-container {
    min-height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-size: 14px;
  }
  .empty-state {
    text-align: center;
    padding: 60px 20px;
  }
  .empty-state .icon { font-size: 48px; margin-bottom: 12px; }
  .empty-state h3 { font-size: 16px; margin-bottom: 8px; color: var(--text-primary); }
  .sim-params .form-row { grid-template-columns: 1fr 1fr 1fr; }
  .presets { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
</style>
</head>
<body>

<div class="app-header">
  <h1>⚡ Jbiophysics</h1>
  <span class="badge">v0.2.0</span>
  <span style="flex:1"></span>
  <span style="font-size:12px;color:var(--text-secondary)">Interactive Biophysical Simulator</span>
</div>

<div class="app-layout">
  <div class="sidebar">
    <!-- Presets -->
    <div class="section-title">Quick Presets</div>
    <div class="presets">
      <button class="btn btn-accent btn-sm" onclick="loadPreset('minimal')">Minimal E-I</button>
      <button class="btn btn-accent btn-sm" onclick="loadPreset('column')">Cortical Column</button>
      <button class="btn btn-accent btn-sm" onclick="loadPreset('gamma')">Gamma Circuit</button>
    </div>

    <!-- Simulation Parameters -->
    <div class="section-title">Simulation</div>
    <div class="card sim-params">
      <div class="form-row">
        <div class="form-group">
          <label>Duration (ms)</label>
          <input type="number" id="t_max" value="1000" min="100" step="100">
        </div>
        <div class="form-group">
          <label>dt (ms)</label>
          <input type="number" id="dt" value="0.1" min="0.01" step="0.01">
        </div>
        <div class="form-group">
          <label>Seed</label>
          <input type="number" id="seed" value="42">
        </div>
      </div>
    </div>

    <!-- Populations -->
    <div class="section-title">Populations</div>
    <div class="card">
      <div id="pop-list"></div>
      <button class="btn btn-accent btn-sm" onclick="addPopulation()">+ Add Population</button>
    </div>

    <!-- Connections -->
    <div class="section-title">Connections</div>
    <div class="card">
      <div id="conn-list"></div>
      <button class="btn btn-accent btn-sm" onclick="addConnection()">+ Add Connection</button>
    </div>

    <!-- Run Button -->
    <button class="btn btn-primary run-btn" id="run-btn" onclick="runSimulation()">
      ▶ Run Simulation
    </button>
  </div>

  <div class="main-panel">
    <div class="status-bar">
      <div class="status-dot" id="status-dot"></div>
      <span id="status-text">Ready — configure network and press Run</span>
    </div>
    <div id="dashboard-container">
      <div class="empty-state">
        <div class="icon">🧠</div>
        <h3>No simulation yet</h3>
        <p>Define your network on the left and click Run</p>
      </div>
    </div>
  </div>
</div>

<script>
let populations = [];
let connections = [];
let popId = 0;
let connId = 0;

function addPopulation(name='', n=10, cellType='pyramidal', noiseAmp=0.05, noiseTau=20.0) {
  const id = popId++;
  populations.push({id, name: name || `pop_${id}`, n, cellType, noiseAmp, noiseTau});
  renderPops();
}

function removePopulation(id) {
  populations = populations.filter(p => p.id !== id);
  renderPops();
}

function renderPops() {
  const list = document.getElementById('pop-list');
  list.innerHTML = populations.map(p => `
    <div class="pop-item" data-id="${p.id}">
      <button class="remove-btn" onclick="removePopulation(${p.id})">×</button>
      <div class="form-row">
        <div class="form-group">
          <label>Name</label>
          <input type="text" value="${p.name}" onchange="updatePop(${p.id},'name',this.value)">
        </div>
        <div class="form-group">
          <label>Count</label>
          <input type="number" value="${p.n}" min="1" onchange="updatePop(${p.id},'n',+this.value)">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Cell Type</label>
          <select onchange="updatePop(${p.id},'cellType',this.value)">
            ${['pyramidal','pv','sst','vip'].map(t => `<option value="${t}" ${t===p.cellType?'selected':''}>${t}</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label>Noise Amp</label>
          <input type="number" value="${p.noiseAmp}" step="0.01" onchange="updatePop(${p.id},'noiseAmp',+this.value)">
        </div>
      </div>
    </div>
  `).join('');
}

function updatePop(id, field, val) {
  const p = populations.find(x => x.id === id);
  if (p) p[field] = val;
}

function addConnection(pre='', post='all', synapse='AMPA', prob=0.1, g=null) {
  const id = connId++;
  connections.push({id, pre: pre || populations[0]?.name || '', post, synapse, p: prob, g});
  renderConns();
}

function removeConnection(id) {
  connections = connections.filter(c => c.id !== id);
  renderConns();
}

function renderConns() {
  const list = document.getElementById('conn-list');
  const popNames = populations.map(p => p.name);
  const targets = ['all', ...popNames];
  list.innerHTML = connections.map(c => `
    <div class="conn-item" data-id="${c.id}">
      <button class="remove-btn" onclick="removeConnection(${c.id})">×</button>
      <div class="form-row">
        <div class="form-group">
          <label>Pre</label>
          <select onchange="updateConn(${c.id},'pre',this.value)">
            ${popNames.map(n => `<option value="${n}" ${n===c.pre?'selected':''}>${n}</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label>Post</label>
          <select onchange="updateConn(${c.id},'post',this.value)">
            ${targets.map(n => `<option value="${n}" ${n===c.post?'selected':''}>${n}</option>`).join('')}
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Synapse</label>
          <select onchange="updateConn(${c.id},'synapse',this.value)">
            ${['AMPA','GABAa','GABAb','NMDA'].map(s => `<option value="${s}" ${s===c.synapse?'selected':''}>${s}</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label>Probability</label>
          <input type="number" value="${c.p}" min="0" max="1" step="0.05" onchange="updateConn(${c.id},'p',+this.value)">
        </div>
      </div>
    </div>
  `).join('');
}

function updateConn(id, field, val) {
  const c = connections.find(x => x.id === id);
  if (c) c[field] = val;
}

function loadPreset(name) {
  populations = []; connections = []; popId = 0; connId = 0;
  if (name === 'minimal') {
    addPopulation('E', 8, 'pyramidal', 0.05, 20);
    addPopulation('I', 4, 'pv', 0.05, 10);
    addConnection('E', 'all', 'AMPA', 0.2);
    addConnection('I', 'E', 'GABAa', 0.4);
  } else if (name === 'column') {
    addPopulation('Pyr', 20, 'pyramidal', 0.05, 20);
    addPopulation('PV', 5, 'pv', 0.05, 10);
    addPopulation('SST', 3, 'sst', 0.05, 15);
    addPopulation('VIP', 2, 'vip', 0.05, 15);
    addConnection('Pyr', 'all', 'AMPA', 0.1);
    addConnection('PV', 'Pyr', 'GABAa', 0.4);
    addConnection('SST', 'Pyr', 'GABAb', 0.3);
    addConnection('VIP', 'SST', 'GABAa', 0.5);
  } else if (name === 'gamma') {
    addPopulation('E', 15, 'pyramidal', 0.08, 20);
    addPopulation('I', 5, 'pv', 0.05, 10);
    addConnection('E', 'all', 'AMPA', 0.15);
    addConnection('I', 'E', 'GABAa', 0.5);
    addConnection('E', 'E', 'NMDA', 0.1);
    document.getElementById('t_max').value = 1500;
  }
}

function setStatus(state, text) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  dot.className = 'status-dot ' + state;
  txt.textContent = text;
}

async function runSimulation() {
  const btn = document.getElementById('run-btn');
  btn.classList.add('running');
  btn.innerHTML = '⏳ Simulating...';
  setStatus('running', 'Building network and running simulation...');

  const config = {
    seed: +document.getElementById('seed').value,
    dt: +document.getElementById('dt').value,
    t_max: +document.getElementById('t_max').value,
    populations: populations.map(p => ({
      name: p.name, n: p.n, cell_type: p.cellType,
      noise_amp: p.noiseAmp, noise_tau: p.noiseTau,
    })),
    connections: connections.map(c => ({
      pre: c.pre, post: c.post, synapse: c.synapse, p: c.p, g: c.g,
    })),
    trainable: [],
  };

  try {
    const resp = await fetch('/api/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(config),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err);
    }
    const plotlyJson = await resp.json();
    const container = document.getElementById('dashboard-container');
    container.innerHTML = '';
    Plotly.newPlot(container, plotlyJson.data, plotlyJson.layout, {responsive: true});

    setStatus('ready', `Simulation complete — ${config.populations.reduce((s,p)=>s+p.n,0)} cells, ${config.t_max} ms`);
  } catch (e) {
    setStatus('', `Error: ${e.message}`);
    console.error(e);
  }

  btn.classList.remove('running');
  btn.innerHTML = '▶ Run Simulation';
}

// Load default preset
loadPreset('minimal');
</script>
</body>
</html>"""


class SimHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/simulate':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            config = json.loads(body)

            try:
                print(f"🧪 Simulating: {json.dumps(config, indent=2)[:200]}...")
                report = run_simulation(config)
                dashboard_json = build_dashboard_json(report)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(dashboard_json.encode())
                print(f"✅ Simulation complete: {report.num_cells} cells, {report.t_max} ms")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"  [{self.client_address[0]}] {format % args}")


if __name__ == '__main__':
    port = 8888
    print(f"🧠 Jbiophysics Interactive Simulator")
    print(f"   → http://localhost:{port}")
    server = HTTPServer(('', port), SimHandler)
    server.serve_forever()
