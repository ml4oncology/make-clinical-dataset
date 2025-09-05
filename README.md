# Make Clinical Dataset

Generate clinical dataset from EHR data of cancer patients treated at Princess Margaret Cancer Center (PMCC). Create cleaned, transformed, and engineered features from laboratory tests, symptom scores, treatment and medications, and demographics. Features can be aligned on treatment dates, clinical visit dates, weekly timepoints, daily timepoints, etc.

The main aim of this repository is to have a central pipeline for generating the same clinical dataset to be used for multiple projects, from recommending treatments/medications, estimating cancer progression, to predicting various undesirable cancer events (e.g. venous thromboembolism, cytopenia, acute care use, nephrotoxicity, symptom deterioration, death).

HISTORY:
Prior to 2022, PMCC was under the EPR system. Now, it migrated to the EPIC system. Thus, there are two pipelines. 

# Data Location
All data is located in the HPC4Health cluster hosted by University Health Network. 

# Getting Started
```bash
git clone https://github.com/ml4oncology/make-clinical-dataset
pip install -e ".[dev]"

# optional
nbstripout --install --keep-output
```

# Instructions
EPR Flow
```bash
# convert each dataset from csv to parquet format
python epr/scripts/csv_to_parquet.py
# build out the features from each dataset
python epr/scripts/build.py
# unify the features and targets into a central dataset
python epr/scripts/unify.py [OPTIONAL args]
```

EPIC Flow
```bash
# curate procedure names/codes into central files
python epic/scripts/curate_procs.py
# separate the data dump into individual datasets
python epic/scripts/separate.py
# process the independent cancer therapies datasets
python epic/scripts/process_therapies.py
# normalize unstructured text fields with groq
python epic/scripts/ask_groq.py --normalize drugs
# build out the features from each dataset
python epic/scripts/build.py
# unify the features and targets into a central dataset
python epic/scripts/unify.py [OPTIONAL args]
```

# Project Organization
```
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final unified data sets
│   └── raw            <- The original immutable data dump
├── epic
│   ├── notebooks      <- Jupyter notebooks
│   ├── scripts        <- Python scripts
│   └── src            <- Python package where the main functionality goes
├── epr
│   ├── notebooks
│   ├── scripts
│   └── src
├── shared             <- shared modules between epic and epr
├── config             <- configuration files
├── pyproject.toml          <- Build configuration
├── .env                    <- Environment variables (i.e. personal keys)
```
