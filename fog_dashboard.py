import json
import time
import threading
import os
from datetime import datetime, timezone
from collections import deque, Counter

from flask import Flask, jsonify, render_template_string, request
import paho.mqtt.client as mqtt
from supabase import create_client, Client

# ------------ CONFIG ------------
MQTT_HOST = "localhost"              # broker running on this laptop
MQTT_PORT = 1883
MQTT_TOPIC = "Home/package_events"   # keep your topic exactly as-is

MAX_EVENTS = 200
DEDUP_WINDOW_SEC = 2.0               # fog-side dedup

# Supabase env vars
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connected.")
else:
    print("Supabase not configured yet; cloud upload disabled.")

app = Flask(__name__)

events = deque(maxlen=MAX_EVENTS)    # newest last
counts = Counter()
last_seen = {}                       # event_type -> last_time
lock = threading.Lock()

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Team 20 Fog Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #0b1220; }
    .card { border: 0; border-radius: 16px; }
    .card-bg { background: #111a2e; color: #e8eefc; }
    .muted { color: #aab6d6; }
    .badge-soft { background: rgba(255,255,255,0.08); }
    .table thead th { color:#aab6d6; border-bottom: 1px solid rgba(255,255,255,0.12); }
    .table tbody td { color:#e8eefc; border-top: 1px solid rgba(255,255,255,0.08); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body class="p-3 p-md-4">
  <div class="container-fluid" style="max-width: 1200px;">
    <div class="d-flex align-items-center justify-content-between mb-3">
      <div>
        <h2 class="text-white mb-1">Team 20 – Fog Dashboard</h2>
        <div class="muted">Listening to <span class="mono">{{topic}}</span> @ <span class="mono">{{host}}:{{port}}</span></div>
      </div>
      <div class="text-end">
        <div class="muted">Status</div>
        <div id="status" class="badge badge-soft rounded-pill px-3 py-2">Connecting…</div>
      </div>
    </div>

    <div class="row g-3 mb-3">
      <div class="col-12 col-md-4">
        <div class="card card-bg p-3">
          <div class="muted">Total Events</div>
          <div class="display-6 fw-semibold" id="totalEvents">0</div>
          <div class="muted">Last update: <span id="lastUpdate">—</span></div>
        </div>
      </div>
      <div class="col-12 col-md-4">
        <div class="card card-bg p-3">
          <div class="muted">Delivered</div>
          <div class="display-6 fw-semibold" id="deliveredCount">0</div>
          <div class="muted">Removed</div>
          <div class="h4 fw-semibold mb-0" id="removedCount">0</div>
        </div>
      </div>
      <div class="col-12 col-md-4">
        <div class="card card-bg p-3">
          <div class="muted">Filters</div>
          <div class="d-flex gap-2 mt-2 flex-wrap">
            <select id="filterEvent" class="form-select form-select-sm" style="max-width: 220px;">
              <option value="ALL">All events</option>
              <option value="DELIVERED">DELIVERED only</option>
              <option value="REMOVED">REMOVED only</option>
            </select>
            <select id="filterPerson" class="form-select form-select-sm" style="max-width: 220px;">
              <option value="ALL">All (person yes/no)</option>
              <option value="YES">person = YES</option>
              <option value="NO">person = NO</option>
            </select>
            <button class="btn btn-sm btn-outline-light" onclick="clearEvents()">Clear</button>
          </div>
          <div class="muted mt-2">Auto-refresh every 1s</div>
        </div>
      </div>
    </div>

    <div class="card card-bg p-3">
      <div class="d-flex align-items-center justify-content-between mb-2">
        <div>
          <div class="h5 mb-0">Live Event Stream</div>
          <div class="muted">Newest at top</div>
        </div>
      </div>
      <div class="table-responsive">
        <table class="table table-dark table-borderless align-middle mb-0">
          <thead>
            <tr>
              <th style="width:160px;">Time</th>
              <th style="width:140px;">Event</th>
              <th style="width:120px;">Packages</th>
              <th style="width:120px;">Person</th>
              <th>Device</th>
              <th class="mono">Raw JSON</th>
            </tr>
          </thead>
          <tbody id="rows">
            <tr><td colspan="6" class="muted">Waiting for events…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

<script>
let lastServerTs = 0;

function fmtTime(ts) {
  try {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
  } catch { return "—"; }
}

function badgeForEvent(e) {
  if (e === "DELIVERED") return "bg-success";
  if (e === "REMOVED") return "bg-danger";
  return "bg-secondary";
}

function badgeForPerson(p) {
  return p ? "bg-info" : "bg-secondary";
}

async function fetchEvents() {
  const fe = document.getElementById("filterEvent").value;
  const fp = document.getElementById("filterPerson").value;

  const res = await fetch(`/api/events?event=${encodeURIComponent(fe)}&person=${encodeURIComponent(fp)}`);
  const data = await res.json();

  document.getElementById("status").textContent = "Live";
  document.getElementById("status").className = "badge bg-success rounded-pill px-3 py-2";

  document.getElementById("totalEvents").textContent = data.stats.total;
  document.getElementById("deliveredCount").textContent = data.stats.delivered;
  document.getElementById("removedCount").textContent = data.stats.removed;
  document.getElementById("lastUpdate").textContent = data.stats.last_update ? fmtTime(data.stats.last_update) : "—";

  const rows = document.getElementById("rows");
  rows.innerHTML = "";

  if (!data.events.length) {
    rows.innerHTML = `<tr><td colspan="6" class="muted">No events (filtered or empty).</td></tr>`;
    return;
  }

  // Newest first
  for (const ev of data.events) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${fmtTime(ev.ts)}</td>
      <td><span class="badge ${badgeForEvent(ev.event)}">${ev.event}</span></td>
      <td class="mono">${ev.pkg_count ?? "-"}</td>
      <td><span class="badge ${badgeForPerson(ev.person)}">${ev.person ? "YES" : "NO"}</span></td>
      <td class="mono">${ev.device ?? "-"}</td>
      <td class="mono muted" style="max-width: 520px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${ev.raw}</td>
    `;
    rows.appendChild(tr);
  }
}

async function clearEvents() {
  await fetch("/api/clear", { method: "POST" });
  await fetchEvents();
}

setInterval(fetchEvents, 1000);
fetchEvents().catch(() => {
  const s = document.getElementById("status");
  s.textContent = "Disconnected";
  s.className = "badge bg-warning text-dark rounded-pill px-3 py-2";
});
</script>
</body>
</html>
"""

def should_accept(evt: dict) -> bool:
    evt_type = evt.get("event", "UNKNOWN")
    ts = float(evt.get("ts", time.time()))
    prev = last_seen.get(evt_type, 0)
    if (ts - prev) < DEDUP_WINDOW_SEC:
        return False
    last_seen[evt_type] = ts
    return True

def upload_event_to_supabase(evt: dict):
    if supabase is None:
        return

    try:
        ts = float(evt.get("ts", time.time()))
        row = {
            "event_time": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "event_type": evt.get("event", "UNKNOWN"),
            "pkg_count": evt.get("pkg_count"),
            "person": bool(evt.get("person", False)),
            "device": evt.get("device", "unknown"),
            "e2e_ms": evt.get("e2e_ms"),
            "raw_json": {
                "event": evt.get("event"),
                "pkg_count": evt.get("pkg_count"),
                "person": evt.get("person"),
                "device": evt.get("device"),
                "ts": evt.get("ts"),
                "raw": evt.get("raw"),
            },
        }
        supabase.table("package_events").insert(row).execute()
    except Exception as e:
        print(f"Supabase upload failed: {e}")

def on_message(client, userdata, msg):
    raw = msg.payload.decode("utf-8", errors="replace")
    try:
        evt = json.loads(raw)
        if "ts" not in evt:
            evt["ts"] = time.time()
    except Exception:
        evt = {"ts": time.time(), "event": "RAW", "raw": raw}

    e2e_ms = (time.time() - float(evt.get("ts", time.time()))) * 1000.0
    evt["e2e_ms"] = round(e2e_ms, 1)
    print(f"E2E latency: {evt['e2e_ms']} ms | event={evt.get('event')} | device={evt.get('device')}")

    evt.setdefault("pkg_count", None)
    evt.setdefault("person", False)
    evt.setdefault("device", "unknown")
    evt["raw"] = raw[:800]

    accepted = False
    with lock:
        if should_accept(evt):
            events.append(evt)
            counts[evt.get("event", "UNKNOWN")] += 1
            accepted = True

    if accepted:
        upload_event_to_supabase(evt)

def mqtt_thread():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC, qos=1)
    client.loop_forever()

@app.get("/")
def index():
    return render_template_string(HTML, topic=MQTT_TOPIC, host=MQTT_HOST, port=MQTT_PORT)

@app.get("/api/events")
def api_events():
    event_filter = request.args.get("event", "ALL")
    person_filter = request.args.get("person", "ALL")

    with lock:
        evs = list(events)

    evs = list(reversed(evs))

    if event_filter != "ALL":
        evs = [e for e in evs if e.get("event") == event_filter]

    if person_filter == "YES":
        evs = [e for e in evs if bool(e.get("person"))]
    elif person_filter == "NO":
        evs = [e for e in evs if not bool(e.get("person"))]

    with lock:
        total = len(events)
        delivered = counts.get("DELIVERED", 0)
        removed = counts.get("REMOVED", 0)
        last_update = events[-1]["ts"] if events else None

    return jsonify({
        "events": evs[:200],
        "stats": {
            "total": total,
            "delivered": delivered,
            "removed": removed,
            "last_update": last_update,
        }
    })

@app.post("/api/clear")
def api_clear():
    with lock:
        events.clear()
        counts.clear()
        last_seen.clear()
    return jsonify({"ok": True})

if __name__ == "__main__":
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8000, debug=False)

# def should_accept(evt: dict) -> bool:
#     evt_type = evt.get("event", "UNKNOWN")
#     ts = float(evt.get("ts", time.time()))
#     prev = last_seen.get(evt_type, 0)
#     if (ts - prev) < DEDUP_WINDOW_SEC:
#         return False
#     last_seen[evt_type] = ts
#     return True

# def on_message(client, userdata, msg):
#     raw = msg.payload.decode("utf-8", errors="replace")
#     try:
#         evt = json.loads(raw)
#         if "ts" not in evt:
#             evt["ts"] = time.time()
#     except Exception:
#         evt = {"ts": time.time(), "event": "RAW", "raw": raw}

#     # --- E2E LATENCY MEASURE (ms) ---
#     e2e_ms = (time.time() - float(evt.get("ts", time.time()))) * 1000.0
#     evt["e2e_ms"] = round(e2e_ms, 1)
#     print(f"E2E latency: {evt['e2e_ms']} ms | event={evt.get('event')} | device={evt.get('device')}")

#     evt.setdefault("pkg_count", None)
#     evt.setdefault("person", False)
#     evt.setdefault("device", "unknown")
#     evt["raw"] = raw[:800]

#     with lock:
#         if should_accept(evt):
#             events.append(evt)
#             counts[evt.get("event", "UNKNOWN")] += 1

# def mqtt_thread():
#     client = mqtt.Client()
#     client.on_message = on_message
#     client.connect(MQTT_HOST, MQTT_PORT, 60)
#     client.subscribe(MQTT_TOPIC, qos=1)
#     client.loop_forever()

# @app.get("/")
# def index():
#     return render_template_string(HTML, topic=MQTT_TOPIC, host=MQTT_HOST, port=MQTT_PORT)

# @app.get("/api/events")
# def api_events():
#     event_filter = request.args.get("event", "ALL")
#     person_filter = request.args.get("person", "ALL")

#     with lock:
#         evs = list(events)

#     # newest first
#     evs = list(reversed(evs))

#     if event_filter != "ALL":
#         evs = [e for e in evs if e.get("event") == event_filter]

#     if person_filter == "YES":
#         evs = [e for e in evs if bool(e.get("person"))]
#     elif person_filter == "NO":
#         evs = [e for e in evs if not bool(e.get("person"))]

#     with lock:
#         total = len(events)
#         delivered = counts.get("DELIVERED", 0)
#         removed = counts.get("REMOVED", 0)
#         last_update = events[-1]["ts"] if events else None

#     return jsonify({
#         "events": evs[:200],
#         "stats": {
#             "total": total,
#             "delivered": delivered,
#             "removed": removed,
#             "last_update": last_update,
#         }
#     })

# @app.post("/api/clear")
# def api_clear():
#     with lock:
#         events.clear()
#         counts.clear()
#         last_seen.clear()
#     return jsonify({"ok": True})

# if __name__ == "__main__":
#     t = threading.Thread(target=mqtt_thread, daemon=True)
#     t.start()
#     # Flask UI
#     app.run(host="0.0.0.0", port=8000, debug=False)