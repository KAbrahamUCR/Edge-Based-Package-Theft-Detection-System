from flask import Flask, request, jsonify
import time
import csv
from pathlib import Path

app = Flask(__name__)

# Simple fog policy knobs
COOLDOWN_SEC = 3.0          # dedup window
QUIET_HOURS = None          # example: (22, 7) means 10pm-7am, or None to disable

last_event_time = {}        # event_type -> last_time
log_path = Path("fog_events.csv")

def in_quiet_hours():
    if not QUIET_HOURS:
        return False
    start, end = QUIET_HOURS
    h = time.localtime().tm_hour
    # Handles overnight windows like 22->7
    return (h >= start) or (h < end) if start > end else (start <= h < end)

def should_accept(evt_type: str, ts: float) -> bool:
    prev = last_event_time.get(evt_type, 0)
    if (ts - prev) < COOLDOWN_SEC:
        return False
    last_event_time[evt_type] = ts
    return True

@app.post("/event")
def event():
    data = request.get_json(force=True)
    evt_type = data.get("event", "UNKNOWN")
    ts = float(data.get("ts", time.time()))

    if in_quiet_hours():
        return jsonify({"ok": True, "ignored": "quiet_hours"})

    if not should_accept(evt_type, ts):
        return jsonify({"ok": True, "ignored": "dedup"})

    # Print for demo
    print(f"[FOG] {time.strftime('%H:%M:%S')}  {evt_type}  pkg={data.get('pkg_count')} person={data.get('person')} device={data.get('device')}")

    # Append to CSV log
    new_file = not log_path.exists()
    with log_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","event","pkg_count","person","device"])
        if new_file:
            w.writeheader()
        w.writerow({
            "ts": ts,
            "event": evt_type,
            "pkg_count": data.get("pkg_count"),
            "person": data.get("person"),
            "device": data.get("device"),
        })

    return jsonify({"ok": True})

if __name__ == "__main__":
    # listen on all interfaces so Jetson can reach it
    app.run(host="0.0.0.0", port=5000)