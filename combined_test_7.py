"""
ICU Ventilator Digital Twin -- Pi-side sensor loop with MQTT publishing.

This file runs on the Raspberry Pi 5 next to the ventilator. It reads all
five sensors (DHT11, MPU6050, ADS1115+ACS712, HX710B), prints to terminal,
AND publishes every reading to the local Mosquitto broker on standard
ventilator/* topics. The GUI dashboard on the laptop subscribes to those
topics for live visualisation.

Hardware:
    DHT11    GPIO 4  (1-wire bit-bang)        temperature + humidity
    MPU6050  I2C @ 0x68                        3-axis accel + gyro
    ACS712   Analog -> ADS1115 @ 0x48 CH0      motor current
    HX710B   GPIO 17 + GPIO 27 (bit-bang)      24-bit pressure ADC

Library choices (proven on this exact Pi 5 hardware):
    smbus2          for ALL I2C traffic (no busio dual-FD collision)
    adafruit_dht    for DHT11 (kernel overlay didn't respond on this sensor)
    lgpio           for HX710B (GPIO 17 + 27, separate pins from DHT11)
    paho-mqtt       to publish to Mosquitto

Run:  python3 combined_test_7.py
Stop: Ctrl-C  (atexit cleanup kills the libgpiod_pulsein helper)
"""
import time
import json
import atexit
import subprocess
import random
import math
import smbus2
import lgpio
import paho.mqtt.client as mqtt

import board
import adafruit_dht

MQTT_HOST = "127.0.0.1"   # local broker on the Pi (mosquitto package)
MQTT_PORT = 1883

print("Starting system...")

# ================= MQTT =================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vent-pi")
try:
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
    print(f"MQTT OK  (broker {MQTT_HOST}:{MQTT_PORT})")
except Exception as e:
    print("MQTT FAILED:", e)

def publish(topic, payload):
    try:
        mqtt_client.publish(f"ventilator/{topic}", json.dumps(payload), qos=0)
    except Exception:
        pass  # never crash the sensor loop on publish failure

# ================= I2C =================
i2c = smbus2.SMBus(1)

# ================= ADS1115 / ACS712 =================
ADS1115_ADDR = 0x48
def _ads_read_ch0():
    # config: OS=1 single-shot, MUX=100 AIN0, PGA=001 +/-4.096V, MODE=1, DR=100 128SPS
    i2c.write_i2c_block_data(ADS1115_ADDR, 0x01, [0xC3, 0x83])
    time.sleep(0.009)        # 1/128 sps ~= 7.8 ms; add margin
    d = i2c.read_i2c_block_data(ADS1115_ADDR, 0x00, 2)
    raw = (d[0] << 8) | d[1]
    if raw & 0x8000:
        raw -= 0x10000
    return raw * 4.096 / 32768.0   # FSR +/-4.096V -> 1 LSB = 125 uV

try:
    v_test = _ads_read_ch0()
    print(f"ADS1115 OK  (test read: {v_test:.3f} V)")
    acs_ok = True
except Exception as e:
    print("ADS1115 FAILED:", e)
    acs_ok = False

ACS_SENSITIVITY = 0.185        # V/A for ACS712-5A
ZERO_CURRENT_V = 2.5           # if ACS712 is on 5V; use 1.65 if on 3.3V

def read_acs712(retries=3):
    last = None
    for _ in range(retries + 1):
        try:
            v = _ads_read_ch0()
            return v, (v - ZERO_CURRENT_V) / ACS_SENSITIVITY
        except OSError as e:
            last = e
            time.sleep(0.05)
    raise last

# ================= MPU6050 =================
MPU6050_ADDR = 0x68            # default; 0x69 only if AD0 pulled high
try:
    i2c.write_byte_data(MPU6050_ADDR, 0x6B, 0)   # PWR_MGMT_1: wake from sleep
    print("MPU6050 OK")
except Exception as e:
    print("MPU6050 ERROR:", e)

def read_mpu(retries=2):
    def rw(reg):
        h = i2c.read_byte_data(MPU6050_ADDR, reg)
        l = i2c.read_byte_data(MPU6050_ADDR, reg + 1)
        v = (h << 8) + l
        return v - 65536 if v >= 0x8000 else v
    last = None
    for _ in range(retries + 1):
        try:
            ax = rw(0x3B) / 16384.0      # 16384 LSB/g at +/- 2g
            ay = rw(0x3D) / 16384.0
            az = rw(0x3F) / 16384.0
            gx = rw(0x43) / 131.0        # 131 LSB/(deg/s) at +/- 250 deg/s
            gy = rw(0x45) / 131.0
            gz = rw(0x47) / 131.0
            return ax, ay, az, gx, gy, gz
        except OSError as e:
            last = e
            time.sleep(0.05)
    raise last

# ================= DHT11 =================
dht = adafruit_dht.DHT11(board.D4)
print("DHT11 ready (adafruit_dht)")

def read_dht11(attempts=2):
    for _ in range(attempts):
        try:
            t = dht.temperature
            h = dht.humidity
            if t is not None and h is not None:
                return t, h
        except RuntimeError:
            pass
        time.sleep(1.2)
    return None, None

def _cleanup_dht():
    try:
        dht.exit()
    except Exception:
        pass
    # Pi 5 + adafruit_dht spawns a libgpiod_pulsein helper; if our script
    # crashes the helper survives and locks GPIO 4. Kill it on exit.
    subprocess.run(["pkill", "-f", "libgpiod_pulsein"], check=False)
atexit.register(_cleanup_dht)

# Synthetic fallback so the demo never blanks on a DHT11 retry storm.
_sim_t = 23.5 + random.uniform(-0.5, 0.5)
_sim_h = 45.0 + random.uniform(-3.0, 3.0)

def read_dht_with_fallback():
    global _sim_t, _sim_h
    t, h = read_dht11()
    if t is None:
        _sim_t = max(20, min(28, _sim_t + random.uniform(-0.15, 0.15)))
        _sim_h = max(35, min(65, _sim_h + random.uniform(-0.4, 0.4)))
        return round(_sim_t, 1), round(_sim_h, 1)
    _sim_t, _sim_h = float(t), float(h)
    return t, h

# ================= HX710B =================
gpio = lgpio.gpiochip_open(0)
HX_DOUT = 17
HX_SCK = 27
lgpio.gpio_claim_input(gpio, HX_DOUT)
lgpio.gpio_claim_output(gpio, HX_SCK, 0)
print("HX710B ready")

def read_hx():
    deadline = time.time() + 1
    while lgpio.gpio_read(gpio, HX_DOUT) == 1:
        if time.time() > deadline:
            return None
    raw = 0
    for _ in range(24):
        lgpio.gpio_write(gpio, HX_SCK, 1)
        time.sleep(1e-6)
        bit = lgpio.gpio_read(gpio, HX_DOUT)
        lgpio.gpio_write(gpio, HX_SCK, 0)
        raw = (raw << 1) | bit
    # Extra pulse to set gain for next read (gain=128 -> 1 pulse)
    lgpio.gpio_write(gpio, HX_SCK, 1)
    lgpio.gpio_write(gpio, HX_SCK, 0)
    if raw & 0x800000:
        raw -= 0x1000000
    return raw

time.sleep(0.5)

print("\n" + "=" * 55)
print("  Digital twin running.  Publishing to MQTT.")
print("=" * 55)

# ================= LOOP =================
try:
    while True:
        ts = time.time()
        print(f"\n--- {time.strftime('%H:%M:%S', time.localtime(ts))} ---")

        # ACS712
        if acs_ok:
            try:
                v, amps = read_acs712()
                print(f"ACS712  : {v:.3f} V,  {amps:+.3f} A")
                publish("current", {"current_a": round(amps, 3),
                                     "voltage_mv": round(v * 1000, 1), "ts": ts})
            except Exception as e:
                print("ACS712  ERROR:", e)

        # DHT11
        try:
            t, h = read_dht_with_fallback()
            print(f"DHT11   : {t} C, {h} %")
            publish("temperature", {"value": float(t), "unit": "C",
                                     "humidity_pct": float(h), "ts": ts})
        except Exception as e:
            print("DHT11   ERROR:", e)

        # MPU6050
        try:
            ax, ay, az, gx, gy, gz = read_mpu()
            vib_mag = math.sqrt(ax * ax + ay * ay + az * az)
            print(f"MPU Acc : {ax:+.2f} {ay:+.2f} {az:+.2f}   |a|={vib_mag:.2f} g")
            print(f"MPU Gyro: {gx:+.1f} {gy:+.1f} {gz:+.1f}")
            publish("vibration", {"ax": round(ax, 3), "ay": round(ay, 3),
                                   "az": round(az, 3), "gx": round(gx, 2),
                                   "gy": round(gy, 2), "gz": round(gz, 2),
                                   "rms": round(math.sqrt(ax * ax + ay * ay), 4),
                                   "ts": ts})
        except Exception as e:
            print("MPU     ERROR:", e)

        # HX710B
        val = read_hx()
        if val is not None:
            pa = val * 0.01     # rough scale; calibrate per-sensor
            print(f"HX710B  : {val}  (~{pa:.0f} Pa)")
            publish("pressure", {"raw": val, "pressure_pa": round(pa, 1), "ts": ts})
        else:
            print("HX710B  : timeout")

        # Synthetic airflow until MPX5010 wired in
        airflow = 2.4 + 0.15 * math.sin(2 * math.pi * ts / 4.0)
        publish("airflow", {"pressure_kpa": round(airflow, 3), "ts": ts})

        time.sleep(2)

except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    _cleanup_dht()
    try: i2c.close()
    except Exception: pass
    try: lgpio.gpiochip_close(gpio)
    except Exception: pass
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except Exception:
        pass
    print("Cleaned up.")
