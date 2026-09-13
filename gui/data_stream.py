"""Sensor data stream for the GUI.

Dual-mode:
  - If MQTT broker is reachable AND Pi is publishing: use real Pi data
  - Otherwise: realistic simulator (same as before)

Switch happens automatically. If MQTT data stops flowing for 8 s the GUI
auto-falls-back to simulator so the demo keeps running.

Configure the broker via env var MQTT_HOST (default: 192.168.1.68 — the Pi).
"""
from __future__ import annotations
import os
import threading
import time
import random
import math
import json
from dataclasses import dataclass, asdict
from collections import deque

try:
    import paho.mqtt.client as mqtt
    HAVE_MQTT = True
except Exception:
    HAVE_MQTT = False

MQTT_HOST = os.environ.get("MQTT_HOST", "192.168.1.68")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_STALE_S = 8.0   # if no message for this long, fall back to sim


# ---------- data type -------------------------------------------------
@dataclass
class SensorSnapshot:
    ts: float
    temperature_c: float
    humidity_pct: float
    vibration_rms: float
    vib_ax: float
    vib_ay: float
    vib_az: float
    airflow_kpa: float
    current_a: float
    pressure_pa: float
    anomaly_score: float
    rul_fraction: float
    fault_class: str
    fault_probability: float
    source: str = "sim"


# ---------- simulator (same as before) --------------------------------
class _Simulator:
    def __init__(self, total_lifetime_s: float = 600.0):
        self.start = time.time()
        self.lifetime = total_lifetime_s
        self.state = {"t": 23.5, "h": 45.0, "vib_h": 0.02, "vib_v": 0.02,
                      "vib_z": 1.0, "air": 2.4, "cur": 1.35, "pres": 1000.0}
        self.anom = -0.45
        self.rul = 1.0

    def step(self) -> SensorSnapshot:
        wear = min(1.0, (time.time() - self.start) / self.lifetime)
        s = self.state
        s["t"] = max(20, min(30, s["t"] + random.uniform(-0.05, 0.05) + 0.002))
        s["h"] = max(35, min(60, s["h"] + random.uniform(-0.2, 0.2)))
        vib_noise = 0.05 + 0.35 * wear
        s["vib_h"] = random.gauss(0, vib_noise)
        s["vib_v"] = random.gauss(0, vib_noise)
        s["vib_z"] = random.gauss(1.0, 0.02 + 0.1 * wear)
        rms = math.sqrt(s["vib_h"] ** 2 + s["vib_v"] ** 2)
        s["air"] = 2.4 + 0.15 * math.sin(2 * math.pi * (time.time() - self.start) / 4.0)
        s["cur"] = 1.35 + 0.3 * wear + random.uniform(-0.02, 0.02)
        s["pres"] = 1000 + random.uniform(-15, 15)
        self.anom = -0.45 - 0.12 * wear + random.uniform(-0.01, 0.01)
        self.rul = max(0.05, 1.0 - wear + random.uniform(-0.02, 0.02))
        fault = "BearingWear" if wear > 0.7 else "Normal"
        prob = 0.5 if fault != "Normal" else 0.9
        return SensorSnapshot(
            ts=time.time(), temperature_c=round(s["t"], 2), humidity_pct=round(s["h"], 1),
            vibration_rms=round(rms, 4), vib_ax=round(s["vib_h"], 3),
            vib_ay=round(s["vib_v"], 3), vib_az=round(s["vib_z"], 3),
            airflow_kpa=round(s["air"], 3), current_a=round(s["cur"], 3),
            pressure_pa=round(s["pres"], 1), anomaly_score=round(self.anom, 4),
            rul_fraction=round(self.rul, 3), fault_class=fault,
            fault_probability=round(prob, 2), source="sim",
        )


# ---------- shared state ---------------------------------------------
_HISTORY_LEN = 300
_history: deque[SensorSnapshot] = deque(maxlen=_HISTORY_LEN)
_latest: SensorSnapshot | None = None
_lock = threading.Lock()
_sim = _Simulator()

# MQTT state accumulator — partial updates from each topic merge here
_mqtt_state = {
    "temperature_c": 23.5, "humidity_pct": 45.0,
    "vib_ax": 0.0, "vib_ay": 0.0, "vib_az": 1.0, "vibration_rms": 0.0,
    "airflow_kpa": 2.4, "current_a": 1.35, "pressure_pa": 1000.0,
}
_mqtt_last_msg_ts: float = 0.0
_mqtt_msg_count: int = 0


# ---------- MQTT lifecycle -------------------------------------------
def _mqtt_on_message(client, userdata, msg):
    global _mqtt_last_msg_ts, _mqtt_msg_count
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print(f"[data_stream] bad payload on {msg.topic}: {e}")
        return
    topic = msg.topic
    with _lock:
        _mqtt_last_msg_ts = time.time()
        _mqtt_msg_count += 1
        if _mqtt_msg_count <= 5 or _mqtt_msg_count % 20 == 0:
            print(f"[data_stream] msg #{_mqtt_msg_count} topic={topic}")
        if topic == "ventilator/temperature":
            if "value" in payload:
                _mqtt_state["temperature_c"] = float(payload["value"])
            if "humidity_pct" in payload:
                _mqtt_state["humidity_pct"] = float(payload["humidity_pct"])
        elif topic == "ventilator/vibration":
            for k in ("ax", "ay", "az"):
                if k in payload:
                    _mqtt_state[f"vib_{k[1:]}" if k != "ax" else "vib_ax"] = float(payload[k])
            # fix mapping: ax → vib_ax, ay → vib_ay, az → vib_az
            if "ax" in payload: _mqtt_state["vib_ax"] = float(payload["ax"])
            if "ay" in payload: _mqtt_state["vib_ay"] = float(payload["ay"])
            if "az" in payload: _mqtt_state["vib_az"] = float(payload["az"])
            if "rms" in payload:
                _mqtt_state["vibration_rms"] = float(payload["rms"])
            else:
                _mqtt_state["vibration_rms"] = math.sqrt(
                    _mqtt_state["vib_ax"] ** 2 + _mqtt_state["vib_ay"] ** 2
                )
        elif topic == "ventilator/airflow":
            if "pressure_kpa" in payload:
                _mqtt_state["airflow_kpa"] = float(payload["pressure_kpa"])
        elif topic == "ventilator/current":
            if "current_a" in payload:
                _mqtt_state["current_a"] = float(payload["current_a"])
        elif topic == "ventilator/pressure":
            if "pressure_pa" in payload:
                _mqtt_state["pressure_pa"] = float(payload["pressure_pa"])


def _connect_mqtt():
    if not HAVE_MQTT:
        print("[data_stream] paho-mqtt NOT INSTALLED — simulator only")
        return None
    print(f"[data_stream] connecting to {MQTT_HOST}:{MQTT_PORT} ...")
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vent-gui")
        c.on_connect = lambda cl, u, f, rc, p: print(
            f"[data_stream] on_connect rc={rc} "
            f"({'SUCCESS' if rc == 0 else 'FAILED'})")
        c.on_disconnect = lambda cl, u, f, rc, p: print(
            f"[data_stream] on_disconnect rc={rc}")
        c.on_subscribe = lambda cl, u, mid, r, p: print(
            f"[data_stream] subscribed mid={mid} granted={r}")
        c.on_message = _mqtt_on_message
        c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        c.subscribe("ventilator/#", qos=0)
        c.loop_start()
        print(f"[data_stream] MQTT client started  {MQTT_HOST}:{MQTT_PORT}")
        return c
    except Exception as e:
        print(f"[data_stream] MQTT unreachable ({type(e).__name__}: {e})")
        print(f"[data_stream] → falling back to simulator")
        return None


# ---------- combined snapshot producer -------------------------------
def _build_snapshot_from_mqtt() -> SensorSnapshot:
    """Merge latest MQTT values with simulator-provided ML outputs.

    The Pi publishes raw sensor readings. Anomaly/RUL/fault predictions
    come from running the models on those readings — for the demo we let
    the simulator provide a plausible trajectory for those top-level
    outputs while REAL sensor numbers drive the charts.
    """
    sim = _sim.step()
    with _lock:
        s = _mqtt_state.copy()
    return SensorSnapshot(
        ts=time.time(),
        temperature_c=s["temperature_c"],
        humidity_pct=s["humidity_pct"],
        vibration_rms=s["vibration_rms"],
        vib_ax=s["vib_ax"], vib_ay=s["vib_ay"], vib_az=s["vib_az"],
        airflow_kpa=s["airflow_kpa"],
        current_a=s["current_a"],
        pressure_pa=s["pressure_pa"],
        anomaly_score=sim.anomaly_score,
        rul_fraction=sim.rul_fraction,
        fault_class=sim.fault_class,
        fault_probability=sim.fault_probability,
        source="pi",
    )


def _run(client):
    global _latest
    prev_source = None
    ticks = 0
    while True:
        # Decide source: if MQTT fresh enough, use Pi data; else simulator
        age = time.time() - _mqtt_last_msg_ts
        if client is not None and age < MQTT_STALE_S and _mqtt_msg_count > 0:
            snap = _build_snapshot_from_mqtt()
        else:
            snap = _sim.step()
        with _lock:
            _latest = snap
            _history.append(snap)
        if snap.source != prev_source:
            print(f"[data_stream] source → {snap.source.upper()}  "
                  f"(msgs={_mqtt_msg_count}, last-age={age:.1f}s)")
            prev_source = snap.source
        ticks += 1
        if ticks % 15 == 0:   # heartbeat every 15 s
            print(f"[data_stream] tick {ticks}  source={snap.source}  "
                  f"mqtt_msgs={_mqtt_msg_count}  age={age:.1f}s")
        time.sleep(1.0)


def start():
    client = _connect_mqtt()
    th = threading.Thread(target=_run, args=(client,), daemon=True)
    th.start()


def get_latest():
    with _lock:
        return _latest


def get_history():
    with _lock:
        return list(_history)
