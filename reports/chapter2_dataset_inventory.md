# Chapter 2 Dataset Inventory

This file tracks the datasets required for Chapter 2 and the evidence needed for later chapters.

## Minimum Requirement

- Group A, tabular: at least 3 datasets, each with more than 5 features and more than 100 samples.
- Group B, time series: at least 3 datasets, each with more than 500 samples.
- Group C, multimedia: at least 2 datasets selected from image, audio, video, and text.
- Total: at least 8 datasets.

## Proposed Dataset Set

| ID | Group | Dataset | Source | Local status | Main task | Notes |
|---|---|---|---|---|---|---|
| A1 | A - Tabular | Heart Disease | UCI 45 | Available: `datasets/heart+disease/` | Classification, clustering, descriptive statistics | First analysis script implemented in `src/analyze_heart_disease.py`. |
| A2 | A - Tabular | CDC Diabetes Health Indicators | UCI 891 | Available: `datasets/uci_891_cdc_diabetes_health_indicators/` | Classification, clustering | 253,680 samples, 21 features, target `Diabetes_binary`; downloaded by `src/download_uci_datasets.py`. |
| A3 | A - Tabular | Breast Cancer Wisconsin Diagnostic | UCI 17 | Available: `datasets/uci_17_breast_cancer_wisconsin_diagnostic/` | Classification, clustering | 569 samples, 30 features, target `Diagnosis`; downloaded by `src/download_uci_datasets.py`. |
| B1 | B - Time series | Appliances Energy Prediction | UCI 374 | Available: `datasets/uci_374_appliances_energy_prediction/` | Regression/forecasting | 19,735 samples, 28 features, target `Appliances`; downloaded by `src/download_uci_datasets.py`. |
| B2 | B - Time series/sensor | Human Activity Recognition Using Smartphones | UCI 240 | Available: `datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/` | Classification/clustering; may be treated as sensor time-series features | 10,299 samples, 561 features, 6 activity labels; extracted and inspected by `src/inspect_local_datasets.py`. |
| B3 | B - Time series/sensor | MHEALTH Dataset | UCI 319 | Available: `datasets/MHEALTHDATASET/MHEALTHDATASET/` | Classification/clustering; possible time-series analysis | 1,215,745 rows, 23 sensor features plus activity label; downloaded manually from UCI archive and inspected. |
| C1 | C - Image | Skin Cancer MNIST HAM10000 | Kaggle | Available: `datasets/ham10000/`, images in KaggleHub cache path from `datasets/ham10000/source_path.txt` | Image feature extraction, classification, clustering | 10,015 images, 7 diagnosis classes; metadata and sampled image features analyzed by `src/analyze_ham10000.py`. |
| C2 | C - Text | Reuters-21578 Text Categorization | UCI 137 / UCI KDD Archive | Available: `datasets/reuters21578/` | Text feature extraction, classification, clustering | 21,578 documents in 22 SGML files; downloaded manually and inspected. |

Optional alternative for C1:

| ID | Group | Dataset | Source | Local status | Main task | Notes |
|---|---|---|---|---|---|---|
| C1-alt | C - Image | FairFace | Kaggle | Missing | Image metadata and feature extraction | Requires Kaggle access and storage planning. |

## Per-Dataset Chapter 2 Fields

For each dataset, fill these fields before writing the LaTeX chapter:

- Dataset ID and official name.
- Original source URL.
- Local raw data path.
- License or access note.
- Number of samples.
- Number of features or extracted features.
- Target/label columns.
- Feature dictionary: column name, type, meaning, unit where available.
- Missing-value conventions.
- Class distribution or target distribution.
- Related studies: citation key, algorithm/method, metrics, reported results, strengths, limitations.
- Planned use in Chapter 3 and Chapter 4.

## Missing Data and Decisions

- Dataset acquisition minimum is complete: 8 datasets are now available across groups A, B, and C.
- Parse Reuters SGML into a tabular text feature file for Chapter 3/4.
- Build MHEALTH/HAR cleaned feature matrices for statistics and clustering.
- Collect at least one related-study citation per dataset; ideally 2-3 for the most important datasets.
- Decide final citation style and add BibTeX entries to `latex/bibliographie.bib`.
- Create Excel comparison workbook for one group A dataset and one group B dataset.

Completed data acquisition:

- A2 CDC Diabetes Health Indicators: downloaded to `datasets/uci_891_cdc_diabetes_health_indicators/`.
- A3 Breast Cancer Wisconsin Diagnostic: downloaded to `datasets/uci_17_breast_cancer_wisconsin_diagnostic/`.
- B1 Appliances Energy Prediction: downloaded to `datasets/uci_374_appliances_energy_prediction/`.
- B2 Human Activity Recognition: extracted to `datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/`.
- B3 MHEALTH: downloaded and extracted to `datasets/MHEALTHDATASET/MHEALTHDATASET/`.
- C1 HAM10000: downloaded through KaggleHub; metadata copied to `datasets/ham10000/`, images remain in KaggleHub cache to avoid duplicate 5GB storage.
- C2 Reuters-21578: downloaded and extracted to `datasets/reuters21578/`.

## Evidence Convention

Every generated result should keep a path that can be cited in the appendix:

- Python scripts: `src/`
- Excel workbooks: `excel/`
- Figures: `outputs/figures/`
- Tables: `outputs/tables/`
- Logs: `outputs/logs/`
- Cleaned datasets and feature matrices: `outputs/cleaned/`

## First Implementation Target

Start with A1 Heart Disease because the raw data is already local. Required outputs:

- `outputs/cleaned/heart_disease_cleaned.csv`
- `outputs/tables/heart_disease_summary.csv`
- `outputs/tables/heart_disease_missing_values.csv`
- `outputs/tables/heart_disease_frequency_*.csv`
- `outputs/figures/heart_disease_*.png`
- `outputs/logs/heart_disease_analysis.log`
