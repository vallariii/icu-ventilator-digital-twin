"""Raspberry Pi 5 sensor readers (PDF Section 3, 4, 10).

Each class exposes a `read()` method returning the primary measurement.
All hardware libraries are imported lazily so this module can be imported
on a dev laptop — only the actual sensor classes will fail until you run
them on the Pi with the drivers installed.

Wiring summary (PDF Table 2):
    DHT22    GPIO4  (pin 7)  — temperature / humidity, 0.5 Hz
    MPU6050  I²C (pins 3,5)  — 3-axis accel + gyro, addr 0x68, 200 Hz
    MPX5010  MCP3008 CH0     — differential airflow pressure, 50 Hz (SPI)
    ACS712   MCP3008 CH1     — motor current, 50 Hz (SPI)
    HX710B   GPIO17/GPIO27   — 24-bit ADC for precision pressure, 10 Hz
"""
from __future__ import annotations
import time
from dataclasses import dataclass


# ----------------------------------------------------------------------
@dataclass
class DHT22Reader:
    """DHT22 temperature + humidity sensor, bit-bang on GPIO4.

    Install: pip install adafruit-circuitpython-dht
    Wiring:  VCC→3.3V, GND→GND, DATA→GPIO4 (+4.7kΩ pull-up to 3.3V)
    """
    gpio_pin: int = 4

    def __post_init__(self):
        import board, adafruit_dht
        self._pin = getattr(board, f'D{self.gpio_pin}')
        self._dht = adafruit_dht.DHT22(self._pin, use_pulseio=False)

    def read(self) -> dict:
        """Return {'temperature_c', 'humidity_pct', 'ts'}.

        DHT22 occasionally raises; caller should retry at most 2–3 times.
        """
        return {
            'temperature_c': float(self._dht.temperature),
            'humidity_pct': float(self._dht.humidity),
            'ts': time.time(),
        }


# ----------------------------------------------------------------------
class MPU6050Reader:
    """MPU6050 6-axis IMU over I²C. Accel range ±2g, 1000 Hz internal."""

    I2C_ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    ACCEL_SCALE = 16384.0  # LSB/g at ±2g
    GYRO_SCALE = 131.0     # LSB/(°/s) at ±250°/s

    def __init__(self, bus_num: int = 1):
        from smbus2 import SMBus
        self._bus = SMBus(bus_num)
        # Wake up the MPU (bit 6 of PWR_MGMT_1)
        self._bus.write_byte_data(self.I2C_ADDR, self.PWR_MGMT_1, 0)

    def _read_word(self, reg: int) -> int:
        hi = self._bus.read_byte_data(self.I2C_ADDR, reg)
        lo = self._bus.read_byte_data(self.I2C_ADDR, reg + 1)
        val = (hi << 8) | lo
        return val - 65536 if val >= 0x8000 else val

    def read(self) -> dict:
        """Return accel (g) and gyro (°/s) for all 3 axes."""
        ax = self._read_word(0x3B) / self.ACCEL_SCALE
        ay = self._read_word(0x3D) / self.ACCEL_SCALE
        az = self._read_word(0x3F) / self.ACCEL_SCALE
        gx = self._read_word(0x43) / self.GYRO_SCALE
        gy = self._read_word(0x45) / self.GYRO_SCALE
        gz = self._read_word(0x47) / self.GYRO_SCALE
        return {'ax': ax, 'ay': ay, 'az': az,
                'gx': gx, 'gy': gy, 'gz': gz, 'ts': time.time()}


# ----------------------------------------------------------------------
class MCP3008AnalogReader:
    """Helper for analog sensors via MCP3008 SPI ADC.

    MPX5010 and ACS712 share one MCP3008 on CE0, different channels.
    """

    def __init__(self, spi_bus: int = 0, spi_device: int = 0, vref: float = 5.0):
        import spidev
        self._spi = spidev.SpiDev()
        self._spi.open(spi_bus, spi_device)
        self._spi.max_speed_hz = 1_000_000
        self.vref = vref

    def read_channel(self, channel: int) -> float:
        """Return voltage (V) for channel 0..7."""
        assert 0 <= channel <= 7
        cmd = [1, (8 + channel) << 4, 0]
        adc = self._spi.xfer2(cmd)
        raw = ((adc[1] & 0x3) << 8) | adc[2]
        return raw * self.vref / 1023.0


class MPX5010Reader:
    """Differential airflow pressure, 0–10 kPa.

    Datasheet: Vout ≈ Vs × (0.09 × P + 0.04), so P_kPa = (Vout/Vs − 0.04) / 0.09
    """
    def __init__(self, adc: MCP3008AnalogReader, channel: int = 0, vs: float = 5.0):
        self.adc = adc
        self.channel = channel
        self.vs = vs

    def read(self) -> dict:
        v = self.adc.read_channel(self.channel)
        p_kpa = (v / self.vs - 0.04) / 0.09
        return {'pressure_kpa': p_kpa, 'voltage': v, 'ts': time.time()}


class ACS712Reader:
    """Current sensor, bidirectional. 185 mV/A for 5A module (default).

    Zero-current output sits at Vcc/2 (~2.5V). Positive current pushes voltage
    above Vcc/2; negative pulls it below.
    """
    def __init__(self, adc: MCP3008AnalogReader, channel: int = 1,
                 sensitivity_v_per_a: float = 0.185, vref: float = 5.0):
        self.adc = adc
        self.channel = channel
        self.sensitivity = sensitivity_v_per_a
        self.vref = vref

    def read(self) -> dict:
        v = self.adc.read_channel(self.channel)
        current_a = (v - self.vref / 2.0) / self.sensitivity
        return {'current_a': current_a, 'voltage_mv': v * 1000.0,
                'ts': time.time()}


# ----------------------------------------------------------------------
class HX710BReader:
    """HX710B 24-bit ADC for precision pressure sensing.

    Bit-bang protocol on DOUT + SCK. Requires `RPi.GPIO`.
    """
    def __init__(self, dout_pin: int = 17, sck_pin: int = 27,
                 gain: int = 128, scale: float = 1.0, tare_offset: int = 0):
        import RPi.GPIO as GPIO
        self._GPIO = GPIO
        self.dout, self.sck = dout_pin, sck_pin
        self.gain_pulses = {128: 1, 64: 3, 32: 2}[gain]
        self.scale = scale
        self.tare = tare_offset
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.dout, GPIO.IN)
        GPIO.setup(self.sck, GPIO.OUT, initial=GPIO.LOW)

    def _read_raw(self) -> int:
        GPIO = self._GPIO
        while GPIO.input(self.dout) == 1:
            time.sleep(0.001)
        value = 0
        for _ in range(24):
            GPIO.output(self.sck, True)
            value = (value << 1) | GPIO.input(self.dout)
            GPIO.output(self.sck, False)
        for _ in range(self.gain_pulses):
            GPIO.output(self.sck, True)
            GPIO.output(self.sck, False)
        if value & 0x800000:
            value -= 0x1000000
        return value

    def read(self) -> dict:
        raw = self._read_raw()
        pa = (raw - self.tare) * self.scale
        return {'raw': raw, 'pressure_pa': pa, 'ts': time.time()}
