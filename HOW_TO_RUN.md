# How to Run the ICU Ventilator Digital Twin

## 1. Requirements

### Hardware

* Raspberry Pi
* Temperature sensor
* Vibration/IMU sensor
* Pressure sensor
* Airflow sensing setup
* Current sensor

### Software

* Python 3.10+
* MQTT / Mosquitto
* Docker and Docker Compose
* Required Python packages

---

## 2. Install Python Dependencies

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```powershell
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install the required Python packages according to the imports used by the project.

---

## 3. Start the Data Services

From the project directory:

```bash
docker compose up -d
```

This starts the supporting services including:

* Mosquitto
* InfluxDB
* Telegraf
* Grafana

---

## 4. Run the Raspberry Pi Sensor Publisher

On the Raspberry Pi:

```bash
python3 combined_test_7.py
```

The program reads the connected sensors and publishes measurements through MQTT.

The MQTT topics include:

```text
ventilator/temperature
ventilator/vibration
ventilator/airflow
ventilator/current
ventilator/pressure
```

---

## 5. Run the Digital Twin

The digital-twin software processes sensor information and updates the estimated system state.

A simulation mode can also be used when physical sensor hardware is unavailable.

---

## 6. Run the Dashboard

Start the dashboard using the provided launcher:

### Windows

```text
run_gui.bat
```

### Linux/macOS

```bash
./run_gui.sh
```

The dashboard can then be opened in a web browser using the local dashboard address.

---

## 7. Simulation Mode

The simulator allows the software pipeline to be tested without physical sensor hardware.

This is useful for:

* Software testing
* Dashboard development
* Model testing
* Demonstrations

---

## 8. Machine-Learning Models

Training scripts are provided in `src/`.

Examples include:

```bash
python src/train_iforest.py
python src/train_lstm_autoencoder.py
python src/train_fault_classifier.py
python src/train_xgb_rul.py
python src/train_xgb_femto_rul.py
```

Large trained model artifacts are not included in the GitHub repository.

---

## 9. Network Configuration

The Raspberry Pi and computer running the dashboard should be connected to the same network when using live MQTT data.

Configure the Raspberry Pi address locally instead of storing a fixed personal IP address in the repository.

---

## 10. Troubleshooting

### Dashboard shows simulated data

Check that:

* Raspberry Pi is powered on
* Both devices are on the same network
* MQTT broker is running
* The correct Raspberry Pi IP address is configured

### Sensor communication errors

Check:

* Sensor wiring
* I2C connections
* SPI connections where applicable
* Sensor power supply
* Raspberry Pi permissions

### MQTT connection problems

Check that Mosquitto is running and that the configured MQTT address and port are correct.

---



