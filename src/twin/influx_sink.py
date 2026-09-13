"""InfluxDB 2.x sink for twin state snapshots (PDF 8.5).

Persists a snapshot every `snapshot_interval_s` seconds.

Expected env:
    INFLUX_URL    e.g. http://localhost:8086
    INFLUX_TOKEN  InfluxDB 2.x API token
    INFLUX_ORG    organisation name
    INFLUX_BUCKET bucket name (default: ventilator)
"""
from __future__ import annotations
import os
import time
import logging
from dataclasses import asdict

try:
    from influxdb_client import InfluxDBClient, Point, WriteOptions
    from influxdb_client.client.write_api import SYNCHRONOUS
    HAS_INFLUX = True
except Exception:
    HAS_INFLUX = False

log = logging.getLogger(__name__)


class InfluxSink:
    def __init__(self, url: str | None = None, token: str | None = None,
                 org: str | None = None, bucket: str = 'ventilator',
                 snapshot_interval_s: float = 10.0):
        if not HAS_INFLUX:
            raise RuntimeError('influxdb-client not installed')
        self.url = url or os.environ.get('INFLUX_URL', 'http://localhost:8086')
        self.token = token or os.environ.get('INFLUX_TOKEN')
        self.org = org or os.environ.get('INFLUX_ORG', 'icu')
        self.bucket = bucket
        self.interval = snapshot_interval_s
        self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._last_ts: float = 0.0

    def write_state(self, state, force: bool = False):
        """Write if enough time has passed since the last write."""
        now = time.time()
        if not force and (now - self._last_ts) < self.interval:
            return False
        self._last_ts = now

        body = asdict(state) if hasattr(state, '__dataclass_fields__') else dict(state)
        point = Point('twin_state').time(int(body.get('timestamp', now) * 1e9))
        for k, v in body.items():
            if v is None or k == 'timestamp':
                continue
            if isinstance(v, (int, float)):
                point = point.field(k, float(v))
            else:
                point = point.tag(k, str(v))
        self._write_api.write(bucket=self.bucket, org=self.org, record=point)
        return True

    def close(self):
        try:
            self._write_api.close()
            self._client.close()
        except Exception:
            pass
