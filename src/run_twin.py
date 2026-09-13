"""Run the full digital twin on the Raspberry Pi.

Wires: hardware sensors -> engine -> MQTT -> InfluxDB.

Use --simulate to run without hardware (feeds synthetic data).

Examples:
    python src/run_twin.py --simulate
    python src/run_twin.py --mqtt-host localhost --influx-url http://localhost:8086
"""
from __future__ import annotations
import argparse
import asyncio
import time
import logging
import os

log = logging.getLogger('twin_runner')


async def _hz_loop(name, fn, hz):
    period = 1.0 / hz
    while True:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:
            log.warning('%s read failed: %s', name, exc)
        wait = period - (time.time() - t0)
        if wait > 0:
            await asyncio.sleep(wait)


async def run(args):
    from twin.engine import DigitalTwin, SensorReading

    mqtt = None
    if args.mqtt_host:
        from twin.mqtt_publisher import MQTTPublisher
        mqtt = MQTTPublisher(host=args.mqtt_host, port=args.mqtt_port)
        mqtt.connect()

    influx = None
    if args.influx_url and os.environ.get('INFLUX_TOKEN'):
        from twin.influx_sink import InfluxSink
        influx = InfluxSink(url=args.influx_url,
                            bucket=args.influx_bucket,
                            snapshot_interval_s=args.influx_interval)

    twin = DigitalTwin(mqtt_publisher=mqtt, influx_sink=influx)

    if args.simulate:
        import numpy as np
        rng = np.random.default_rng()
        tick = 0
        while True:
            degrade = min(1.0, tick / 600.0)
            r = SensorReading(
                temperature_c=36.5 + rng.normal(0, 0.2) + 4 * degrade,
                vib_h_g=rng.normal(0, 0.05 + 0.3 * degrade),
                vib_v_g=rng.normal(0, 0.05 + 0.3 * degrade),
                airflow_pressure_kpa=2.4 + rng.normal(0, 0.05),
                current_a=1.35 + rng.normal(0, 0.02) + 0.3 * degrade,
                pressure_pa=1000 + rng.normal(0, 20),
                flow_lps=1.0, tidal_volume_l=0.5,
                timestamp=time.time(),
            )
            state = twin.step(r)
            if tick % 10 == 0:
                log.info('tick=%d paw=%.1f RUL=%s fault=%s',
                         tick, state.predicted_airway_pressure_cmH2O,
                         state.rul_fraction, state.fault_class)
            tick += 1
            await asyncio.sleep(1.0)
    else:
        # Real Pi mode: async tasks per sensor frequency -> push to engine at 1 Hz
        from sensors.readers import (DHT22Reader, MPU6050Reader,
                                     MCP3008AnalogReader, MPX5010Reader,
                                     ACS712Reader, HX710BReader)
        dht = DHT22Reader(gpio_pin=4)
        imu = MPU6050Reader()
        adc = MCP3008AnalogReader()
        mpx = MPX5010Reader(adc, channel=0)
        acs = ACS712Reader(adc, channel=1)
        hx = HX710BReader(dout_pin=17, sck_pin=27)

        latest = {'temp_c': 36.5, 'vib_h': 0.0, 'vib_v': 0.0,
                  'air_kpa': 2.4, 'cur_a': 1.35, 'press_pa': 1000.0}

        def tick_dht():
            latest['temp_c'] = dht.read()['temperature_c']
        def tick_imu():
            d = imu.read()
            latest['vib_h'], latest['vib_v'] = d['ax'], d['ay']
        def tick_mpx():
            latest['air_kpa'] = mpx.read()['pressure_kpa']
        def tick_acs():
            latest['cur_a'] = acs.read()['current_a']
        def tick_hx():
            latest['press_pa'] = hx.read()['pressure_pa']

        async def publish_loop():
            while True:
                r = SensorReading(
                    temperature_c=latest['temp_c'],
                    vib_h_g=latest['vib_h'], vib_v_g=latest['vib_v'],
                    airflow_pressure_kpa=latest['air_kpa'],
                    current_a=latest['cur_a'], pressure_pa=latest['press_pa'],
                    flow_lps=0.0, tidal_volume_l=0.5, timestamp=time.time(),
                )
                twin.step(r)
                await asyncio.sleep(1.0)

        await asyncio.gather(
            _hz_loop('dht', tick_dht, 0.5),
            _hz_loop('imu', tick_imu, 200),
            _hz_loop('mpx', tick_mpx, 50),
            _hz_loop('acs', tick_acs, 50),
            _hz_loop('hx', tick_hx, 10),
            publish_loop(),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simulate', action='store_true')
    ap.add_argument('--mqtt-host', default=os.environ.get('MQTT_HOST'))
    ap.add_argument('--mqtt-port', type=int, default=1883)
    ap.add_argument('--influx-url', default=os.environ.get('INFLUX_URL'))
    ap.add_argument('--influx-bucket', default='ventilator')
    ap.add_argument('--influx-interval', type=float, default=10.0)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
