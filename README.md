# ICU Ventilator Digital Twin

An IoT-enabled digital twin framework for monitoring an ICU ventilator using multi-sensor data, physics-based modelling and machine-learning-based predictive maintenance.

## Overview

This project develops a digital representation of an ICU mechanical ventilator that combines real-time sensor measurements, physics-based calculations and machine-learning models.

The system collects temperature, vibration, pressure, airflow and motor-current information and processes these measurements to estimate the operating condition of the ventilator.

The project explores:

* Real-time sensor monitoring
* Digital-twin state estimation
* Physics-based modelling
* Anomaly detection
* Fault classification
* Remaining Useful Life (RUL) estimation
* MQTT-based communication
* Live dashboard visualisation

---

## System Architecture

```text
              ICU Ventilator
                    |
                    v
          +--------------------+
          |      Sensors       |
          |--------------------|
          | Temperature        |
          | Vibration          |
          | Airflow/Pressure   |
          | Motor Current      |
          +--------------------+
                    |
                    v
             Raspberry Pi
                    |
                    v
          Sensor Data Pipeline
                    |
          +---------+---------+
          |                   |
          v                   v
   Physics Models       ML Models
          |                   |
          |          +--------+--------+
          |          |        |        |
          |       Anomaly   Fault     RUL
          |       Detection Classification
          |                   |
          +---------+---------+
                    |
                    v
             Digital Twin
                    |
                    v
                   MQTT
                    |
                    v
             Live Dashboard
                    |
                    v
        Monitoring & Visualisation
```

---

## Hardware

The prototype uses a Raspberry Pi-based sensing system with multiple sensors:

* Raspberry Pi
* DHT11/DHT22 temperature and humidity sensor
* MPU6050 IMU for vibration measurements
* MPX5010 pressure sensor
* ACS712 current sensor
* HX710B pressure ADC

The exact sensor configuration may vary depending on the hardware setup used during testing.

---

## Software and Technologies

* Python
* Raspberry Pi
* MQTT
* Mosquitto
* InfluxDB
* Telegraf
* Grafana
* Plotly Dash
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* TensorFlow/Keras
* Docker

---

## Digital Twin

The digital-twin engine maintains a software representation of the ventilator's operating state.

Sensor readings are converted into a common state representation and processed by the twin engine.

The system combines:

1. Sensor measurements
2. Physics-based calculations
3. Feature extraction
4. Machine-learning predictions
5. Historical/temporal information

The resulting state is used for monitoring and predictive-maintenance analysis.

---

## Physics-Based Modelling

The project incorporates simplified physical relationships related to ventilator operation and equipment condition.

The physics layer provides interpretable quantities that complement the data-driven machine-learning models.

This creates a hybrid approach:

```text
Sensor Data
    |
    +------> Physics Model
    |
    +------> Feature Extraction
                   |
                   v
             ML Prediction
                   |
                   v
             Twin State
```

---

## Machine Learning

Multiple machine-learning approaches are explored for different predictive-maintenance tasks.

### 1. Anomaly Detection

Two approaches are implemented:

* Isolation Forest
* LSTM Autoencoder

These models identify patterns that differ from expected operating behaviour.

### 2. Fault Classification

Classification models are used to identify equipment fault conditions.

Implemented approaches include:

* Support Vector Machine
* Random Forest

### 3. Remaining Useful Life

XGBoost-based regression models are used to estimate Remaining Useful Life using predictive-maintenance datasets.

Models are trained using:

* NASA C-MAPSS data
* FEMTO bearing data

---

## Sensor Data Pipeline

The Raspberry Pi sensor program collects measurements from the connected sensors.

The data flow is:

```text
Sensors
   |
   v
Raspberry Pi
   |
   v
Python Sensor Interface
   |
   v
MQTT Publisher
   |
   v
Mosquitto Broker
   |
   +----------> Digital Twin
   |
   +----------> Dashboard
   |
   +----------> Time-Series Storage
```

MQTT topics are organised under the `ventilator/` namespace.

---

## Live Dashboard

A Plotly Dash dashboard provides a visual representation of the digital-twin state.

The dashboard includes:

* Temperature monitoring
* Vibration monitoring
* Current monitoring
* Airflow/pressure monitoring
* Anomaly information
* RUL estimation
* Fault classification
* Model performance metrics
* Live time-series plots
* 3D ventilator visualisation

The GUI can operate using either live Raspberry Pi data or simulated data for development and demonstration.

---

## Running the Project

### Simulation Mode

The digital-twin engine supports a simulated sensor-data mode.

```bash
python src/run_twin.py --simulate
```

This allows the software pipeline to be tested without requiring the physical sensor hardware.

### Dashboard

The dashboard can be started using the provided launcher.

On Windows:

```text
run_gui.bat
```

On Linux/macOS:

```bash
./run_gui.sh
```

The dashboard can then be opened locally in a web browser.

### Docker Services

The project also includes a Docker Compose configuration for services such as:

* Mosquitto
* InfluxDB
* Telegraf
* Grafana

Start the services with:

```bash
docker compose up -d
```

---

## Testing and Validation

The project combines several forms of testing:

* Sensor communication testing
* MQTT communication testing
* Simulated-data testing
* Digital-twin state updates
* Machine-learning model evaluation
* Dashboard validation
* Inference-latency measurement

The sensor-side program also includes handling for sensor read failures and fallback behaviour for selected measurements.

---

