# Make Clinical Dataset

Generate clinical dataset from EHR data of cancer patients treated at Princess Margaret Cancer Center (PMCC). Create cleaned, transformed, and engineered features from laboratory tests, symptom scores, treatment and medications, and demographics. Features can be aligned on treatment dates, clinical visit dates, weekly timepoints, daily timepoints, etc.

The main aim of this repository is to have a central pipeline for generating the same clinical dataset to be used for multiple projects, from recommending treatments/medications, estimating cancer progression, to predicting various undesirable cancer events (e.g. venous thromboembolism, cytopenia, acute care use, nephrotoxicity, symptom deterioration, death).

HISTORY:
Prior to 2022, PMCC was under the EPR system. Now, it migrated to the EPIC system. Thus, there are two pipelines. 

# Data Location
The external data is located in the Google Drive folder [ml4o/projects/aim2reduce/data](https://drive.google.com/drive/folders/1DcUbnKlEmj0wObx1VMOnPWheBp0szp8r?usp=drive_link). The raw data is located in the HPC4Health cluster hosted by University Health Network. 

# Getting Started
```bash
git clone https://github.com/ml4oncology/make-clinical-dataset
pip install -e .

# optional
nbstripout --install --keep-output
```

# Instructions
EPR Flow
```bash
python scripts/EPR/csv_to_parquet.py
python scripts/EPR/build.py
python scripts/EPR/unify.py [OPTIONAL args]
```

EPIC Flow
```bash
python scripts/EPIC/curate_procs.py
python scripts/EPIC/separate.py
python scripts/EPIC/build.py
```

# Project Organization
```
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final unified data sets
│   └── raw            <- The original immutable data dump
├── notebooks               <- Jupyter notebooks
├── pyproject.toml          <- Build configuration
├── scripts                 <- Python scripts
└── src                     <- Python package where the main functionality goes
```