"""Physics sub-models for the ventilator digital twin (PDF Section 8).

Pure functions — no state. The simulation engine calls these with the
current sensor readings and an evolving internal state dict.
"""
from __future__ import annotations
from dataclasses import dataclass


def airway_pressure(tidal_volume_l: float, compliance_l_per_cmH2O: float,
                    resistance_cmH2O_per_lps: float, flow_lps: float) -> float:
    """Mead-Whittenberger equation.

    P_aw = (V_t / C) + R * V̇

    tidal_volume_l:           inspired volume (L)
    compliance_l_per_cmH2O:   lung compliance (L / cmH2O)
    resistance_cmH2O_per_lps: airway resistance (cmH2O / (L/s))
    flow_lps:                 instantaneous flow (L/s)

    Returns airway pressure (cmH2O).
    """
    if compliance_l_per_cmH2O <= 0:
        return float('inf')
    return tidal_volume_l / compliance_l_per_cmH2O + resistance_cmH2O_per_lps * flow_lps


def motor_power(voltage_v: float, current_a: float, efficiency: float = 0.85) -> float:
    """Electrical → mechanical power.

    P_mech = V * I * η   (watts)
    """
    return voltage_v * current_a * efficiency


def thermal_step(T_current_c: float, P_dissipated_w: float,
                 T_ambient_c: float, mass_kg: float,
                 cp_j_per_kgk: float, R_thermal_k_per_w: float,
                 dt_s: float) -> float:
    """Euler step of a first-order thermal model.

    dT/dt = P_diss / (m * Cp) - (T - T_amb) / (m * Cp * R_thermal)

    Returns new temperature (°C) after dt_s seconds.
    """
    heat_in = P_dissipated_w / (mass_kg * cp_j_per_kgk)
    heat_loss = (T_current_c - T_ambient_c) / (mass_kg * cp_j_per_kgk * R_thermal_k_per_w)
    return T_current_c + dt_s * (heat_in - heat_loss)


def bearing_wear_integral(prev_wear: float, rms_vibration_g: float,
                          freq_shift_factor: float, dt_s: float) -> float:
    """Integrate RMS vibration × frequency-shift penalty over time.

    Higher vibration or larger frequency shift from baseline → faster wear.
    This is a qualitative index, not a calibrated measurement.
    """
    return prev_wear + dt_s * rms_vibration_g * max(0.0, freq_shift_factor)


# ---- Nominal (healthy) constants used for residual computation ----

@dataclass
class VentilatorNominal:
    """Healthy-operating-point constants for a generic ICU ventilator."""
    compliance: float = 0.05          # L/cmH2O (adult typical 0.04-0.07)
    resistance: float = 5.0           # cmH2O/(L/s) (normal 2-5, intubated 6-10)
    motor_voltage: float = 24.0       # V
    motor_efficiency: float = 0.85
    thermal_mass_kg: float = 1.2
    thermal_cp: float = 900.0         # J/(kg·K) aluminium chassis-ish
    thermal_resistance: float = 0.4   # K/W (cabinet to ambient)
