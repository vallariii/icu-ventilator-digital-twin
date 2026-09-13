# Dataset Information

This project uses sensor data and predictive-maintenance datasets for digital-twin modelling, anomaly detection, fault classification and Remaining Useful Life (RUL) estimation.

## Data Categories

### Raw Data

Raw sensor and external dataset files were used as inputs for preprocessing and feature extraction.

The complete raw datasets are not included in this repository because of their size.

### Processed Data

Processed datasets were generated from the raw data for use by the machine-learning models.

The complete processed datasets are also excluded from this repository to keep the GitHub repository lightweight.

## Datasets Used

The project works with data associated with:

* AI4I-style fault classification
* NASA C-MAPSS for Remaining Useful Life estimation
* FEMTO bearing data for additional RUL modelling

## Data Processing

The repository contains the scripts used to load and process the datasets:

* `src/cmapss_loader.py`
* `src/femto_loader.py`
* `src/features.py`
* `src/femto_features.py`

These scripts prepare the data for the machine-learning pipeline.

## Reproducing the Data Pipeline

After obtaining the required datasets, the corresponding training scripts in the `src/` directory can be used to regenerate the model inputs and train the models.


