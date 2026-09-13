"""MQTT publisher for digital-twin state and sensor readings (PDF 8.4).

Topics (from PDF Table 3):
    ventilator/temperature   0.5 Hz
    ventilator/vibration     200 Hz
    ventilator/airflow       50 Hz
    ventilator/current       50 Hz
    ventilator/pressure      10 Hz
    ventilator/alerts        on event
    ventilator/twin/state    1 Hz (this module's primary output)

Usage:
    pub = MQTTPublisher(host='localhost')
    pub.connect()
    pub.publish_state(twin_state)
    pub.publish_sensor('temperature', {'value': 36.5, 'unit':'C', 'ts': ts})
"""
from __future__ import annotations
import json
import time
import logging
from dataclasses import asdict
from typing import Any

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except Exception:
    HAS_MQTT = False

log = logging.getLogger(__name__)


class MQTTPublisher:
    def __init__(self, host: str = 'localhost', port: int = 1883,
                 client_id: str = 'vent-twin', keepalive: int = 60):
        if not HAS_MQTT:
            raise RuntimeError('paho-mqtt not installed. pip install paho-mqtt')
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.host, self.port, self.keepalive = host, port, keepalive
        self._connected = False

    def connect(self):
        self.client.on_connect = lambda c, u, f, rc, p: self._mark_connected(rc)
        self.client.on_disconnect = lambda c, u, f, rc, p: self._mark_disconnected()
        self.client.connect(self.host, self.port, self.keepalive)
        self.client.loop_start()

    def _mark_connected(self, rc):
        self._connected = (rc == 0)
        log.info('MQTT connected rc=%s', rc)

    def _mark_disconnected(self):
        self._connected = False
        log.warning('MQTT disconnected')

    def publish_sensor(self, sensor: str, payload: dict, qos: int = 0):
        """sensor ∈ {temperature, vibration, airflow, current, pressure, alerts}"""
        topic = f'ventilator/{sensor}'
        self.client.publish(topic, json.dumps(payload), qos=qos)

    def publish_state(self, state: Any, qos: int = 0):
        """Publish a TwinState dataclass (or dict) to ventilator/twin/state."""
        body = asdict(state) if hasattr(state, '__dataclass_fields__') else dict(state)
        self.client.publish('ventilator/twin/state', json.dumps(body, default=float), qos=qos)

    def publish_alert(self, alert_type: str, sensor: str, score: float, **extra):
        payload = {'type': alert_type, 'sensor': sensor, 'score': score,
                   'ts': time.time(), **extra}
        self.client.publish('ventilator/alerts', json.dumps(payload), qos=1)

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
