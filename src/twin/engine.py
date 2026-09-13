"""Digital Twin state-estimation engine (PDF Section 8).

This module wires:
    raw readings -> circular buffers -> feature extractor -> ML models
                                      -> physics model      -> residual
                                      -> state snapshot     -> (optional) MQTT

No Raspberry Pi hardware required — feed it readings via `DigitalTwin.step()`
from a recorded CSV, a simulator, or the real sensor async tasks when the
Pi is wired up.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable
import json
import numpy as np
import joblib

from .buffer import CircularBuffer
from . import physics

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / 'models'


@dataclass
class SensorReading:
    """Single tick of sensor data pushed into the twin."""
    temperature_c: float          # DHT22
    vib_h_g: float                # MPU6050 horizontal accel
    vib_v_g: float                # MPU6050 vertical accel
    airflow_pressure_kpa: float   # MPX5010
    current_a: float              # ACS712
    pressure_pa: float            # HX710B
    flow_lps: float = 0.0         # derived from airflow delta (user provides)
    tidal_volume_l: float = 0.5   # current breath integral (user provides)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TwinState:
    """Snapshot of the virtual ventilator at one instant."""
    timestamp: float
    # physics predictions
    predicted_airway_pressure_cmH2O: float
    predicted_motor_power_w: float
    predicted_temp_c: float
    bearing_wear_index: float
    # residuals = measured - predicted
    pressure_residual_pa: float
    # ML outputs (None until models are loaded AND enough data buffered)
    anomaly_score: float | None = None
    rul_fraction: float | None = None
    fault_class: str | None = None
    fault_probability: float | None = None


class DigitalTwin:
    """Orchestrates physics + ML + state tracking.

    Designed to be stepped at ~1 Hz. Hz-accurate sensor streams (200 Hz vibration,
    50 Hz pressure) should be decimated or aggregated into one SensorReading per
    tick before calling `step()`.
    """

    def __init__(self,
                 nominal: physics.VentilatorNominal | None = None,
                 window_size: int = 1000,
                 iforest_path: Path | None = None,
                 femto_rul_path: Path | None = None,
                 fault_rf_path: Path | None = None,
                 mqtt_publisher=None,
                 influx_sink=None,
                 alert_rul_threshold: float = 0.2,
                 alert_fault_prob_threshold: float = 0.8):
        self.nominal = nominal or physics.VentilatorNominal()
        self.buf_temp = CircularBuffer(window_size, 1)
        self.buf_vib = CircularBuffer(window_size, 2)   # h, v
        self.buf_pressure = CircularBuffer(window_size, 1)
        self.buf_current = CircularBuffer(window_size, 1)
        self.buf_airflow = CircularBuffer(window_size, 1)
        self._temp_state = 25.0          # internal temp estimate (°C)
        self._bearing_wear = 0.0
        self._last_tick: float | None = None
        self.history: list[TwinState] = []

        self.iforest = self._load(iforest_path or MODELS / 'iforest_fd001.joblib')
        self.femto_rul = self._load(femto_rul_path or MODELS / 'xgb_rul_femto.joblib')
        self.fault_rf = self._load(fault_rf_path or MODELS / 'rf_multiclass_ai4i.joblib')

        self.mqtt = mqtt_publisher
        self.influx = influx_sink
        self.alert_rul_threshold = alert_rul_threshold
        self.alert_fault_prob_threshold = alert_fault_prob_threshold

    @staticmethod
    def _load(p: Path):
        return joblib.load(p) if p.exists() else None

    # ------------------------------------------------------------------
    def step(self, r: SensorReading) -> TwinState:
        """Push one reading in, pop one TwinState out."""
        dt = (r.timestamp - self._last_tick) if self._last_tick else 0.1
        self._last_tick = r.timestamp

        # update buffers
        self.buf_temp.append([r.temperature_c])
        self.buf_vib.append([r.vib_h_g, r.vib_v_g])
        self.buf_pressure.append([r.pressure_pa])
        self.buf_current.append([r.current_a])
        self.buf_airflow.append([r.airflow_pressure_kpa])

        # --- Physics predictions -----------------------------------
        pred_paw = physics.airway_pressure(
            r.tidal_volume_l, self.nominal.compliance,
            self.nominal.resistance, r.flow_lps,
        ) * 10.197  # cmH2O → kPa? actually pressure_pa is in Pa. Keep cmH2O as-is.
        pred_paw /= 10.197  # revert — kept explicit to flag the unit choice
        pred_motor = physics.motor_power(
            self.nominal.motor_voltage, r.current_a, self.nominal.motor_efficiency,
        )
        self._temp_state = physics.thermal_step(
            self._temp_state, pred_motor * 0.15,  # ~15 % dissipated as heat
            r.temperature_c,
            self.nominal.thermal_mass_kg, self.nominal.thermal_cp,
            self.nominal.thermal_resistance, dt_s=dt,
        )

        vib_window = self.buf_vib.window(min(200, len(self.buf_vib)))
        rms_vib = float(np.sqrt(np.mean(vib_window ** 2))) if len(vib_window) else 0.0
        self._bearing_wear = physics.bearing_wear_integral(
            self._bearing_wear, rms_vib, freq_shift_factor=1.0, dt_s=dt,
        )

        # residual: measured vs predicted pressure (convert cmH2O→Pa: 1 cmH2O ≈ 98.0665 Pa)
        pred_paw_pa = pred_paw * 98.0665
        pressure_residual = r.pressure_pa - pred_paw_pa

        state = TwinState(
            timestamp=r.timestamp,
            predicted_airway_pressure_cmH2O=pred_paw,
            predicted_motor_power_w=pred_motor,
            predicted_temp_c=self._temp_state,
            bearing_wear_index=self._bearing_wear,
            pressure_residual_pa=pressure_residual,
        )

        # --- ML inference (best-effort; skip if models or features missing) --
        self._run_ml(r, state)
        self.history.append(state)

        # --- Side effects: publish + persist + alerts -----------------
        if self.mqtt is not None:
            try:
                self.mqtt.publish_state(state)
                self.mqtt.publish_sensor('temperature',
                                         {'value': r.temperature_c, 'unit': 'C',
                                          'ts': r.timestamp})
                self.mqtt.publish_sensor('vibration',
                                         {'ax': r.vib_h_g, 'ay': r.vib_v_g,
                                          'az': 0.0, 'ts': r.timestamp})
                self.mqtt.publish_sensor('current',
                                         {'current_a': r.current_a, 'ts': r.timestamp})
                self.mqtt.publish_sensor('airflow',
                                         {'pressure_kpa': r.airflow_pressure_kpa,
                                          'ts': r.timestamp})
                self.mqtt.publish_sensor('pressure',
                                         {'pressure_pa': r.pressure_pa,
                                          'ts': r.timestamp})
                if state.rul_fraction is not None and state.rul_fraction < self.alert_rul_threshold:
                    self.mqtt.publish_alert('rul_low', 'vibration',
                                            score=state.rul_fraction)
                if (state.fault_probability is not None
                        and state.fault_class and state.fault_class != 'Normal'
                        and state.fault_probability > self.alert_fault_prob_threshold):
                    self.mqtt.publish_alert('fault_detected',
                                            state.fault_class, score=state.fault_probability)
            except Exception as exc:
                # Publish failures must never crash the twin loop
                import logging
                logging.getLogger(__name__).warning('MQTT publish failed: %s', exc)

        if self.influx is not None:
            try:
                self.influx.write_state(state)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning('Influx write failed: %s', exc)

        return state

    # ------------------------------------------------------------------
    def _run_ml(self, r: SensorReading, state: TwinState):
        """Lightweight feature vector → models. Uses a dictionary with only
        the features our models actually need; missing keys fall through as 0.
        """
        # FEMTO RUL uses vibration FFT / stats — compute from the vib buffer
        if self.femto_rul is not None and len(self.buf_vib) >= 32:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from features import extract_window_features  # noqa
            win = self.buf_vib.window(256)
            feats = extract_window_features(win, fs=200.0, channel_names=['h', 'v'])
            import pandas as pd
            row = {c: float(feats.get(c, 0.0)) for c in self.femto_rul['features']}
            state.rul_fraction = float(self.femto_rul['model'].predict(pd.DataFrame([row]))[0])

        # Fault classifier — AI4I features (scale + predict)
        if self.fault_rf is not None:
            import pandas as pd
            # Map live sensors to AI4I-style features (rough proxies)
            ai4i_row = {
                'Air temperature [K]': r.temperature_c + 273.15,
                'Process temperature [K]': state.predicted_temp_c + 273.15,
                'Rotational speed [rpm]': 1500.0,  # not measured in our sensor set
                'Torque [Nm]': state.predicted_motor_power_w / (1500.0 * 2 * np.pi / 60 + 1e-6),
                'Tool wear [min]': state.bearing_wear_index * 10.0,
                'type_H': 0, 'type_L': 1, 'type_M': 0,
            }
            bundle = self.fault_rf
            X = pd.DataFrame([{c: float(ai4i_row.get(c, 0.0)) for c in bundle['features']}])
            X_s = bundle['scaler'].transform(X)
            probs = bundle['model'].predict_proba(X_s)[0]
            i_top = int(np.argmax(probs))
            state.fault_class = bundle['classes'][i_top]
            state.fault_probability = float(probs[i_top])

    # ------------------------------------------------------------------
    def to_json(self, state: TwinState) -> str:
        d = asdict(state)
        return json.dumps(d, default=float)


# ----------------------------------------------------------------------
# CLI demo: synthetic ventilator cycle with progressive degradation
# ----------------------------------------------------------------------
def _demo(n_ticks: int = 60, seed: int = 42):
    """Run a short demo showing the twin react to a degrading vibration signal."""
    rng = np.random.default_rng(seed)
    twin = DigitalTwin()
    now = time.time()
    for i in range(n_ticks):
        degrade = i / n_ticks  # 0 → 1
        r = SensorReading(
            temperature_c=36.5 + rng.normal(0, 0.2) + 4 * degrade,
            vib_h_g=rng.normal(0, 0.05 + 0.3 * degrade),
            vib_v_g=rng.normal(0, 0.05 + 0.3 * degrade),
            airflow_pressure_kpa=2.4 + rng.normal(0, 0.05),
            current_a=1.35 + rng.normal(0, 0.02) + 0.3 * degrade,
            pressure_pa=1000 + rng.normal(0, 20),
            flow_lps=1.0,
            tidal_volume_l=0.5,
            timestamp=now + i * 1.0,
        )
        s = twin.step(r)
        if i in (0, n_ticks // 2, n_ticks - 1):
            print(f'tick {i:3d}  degrade={degrade:.2f}  '
                  f'paw={s.predicted_airway_pressure_cmH2O:.2f} cmH2O  '
                  f'motor={s.predicted_motor_power_w:.1f} W  '
                  f'temp={s.predicted_temp_c:.2f}°C  '
                  f'wear={s.bearing_wear_index:.3f}  '
                  f'RUL_frac={s.rul_fraction}  fault={s.fault_class}')

    # Final snapshot as JSON
    print('\nFinal state (JSON):')
    print(twin.to_json(twin.history[-1]))


if __name__ == '__main__':
    _demo()
