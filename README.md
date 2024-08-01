# Make Clinical Dataset

Generate clinical dataset from EMR data of cancer patients treated at Princess Margaret hospital. Create cleaned, transformed, and engineered features from laboratory tests, symptom scores, treatment and medications, and demographics. Features can be aligned on treatment dates, clinical visit dates, weekly timepoints, daily timepoints, etc.

The main aim of this repository is to have a central pipeline for generating the same clinical dataset to be used for multiple projects, from recommending treatments/medications, estimating cancer progression, to predicting various undesirable cancer events (e.g. venous thromboembolism, cytopenia, acute care use, nephrotoxicity, symptom deterioration, death).

# Data Location
The external data is located in the Google Drive folder [ml4o/projects/aim2reduce/data](https://drive.google.com/drive/folders/1DcUbnKlEmj0wObx1VMOnPWheBp0szp8r?usp=drive_link). The raw data is located in the HPC4Health cluster hosted by University Health Network. 

# Getting Started
```bash
git clone https://github.com/ml4oncology/make-clinical-dataset
pip install -e .
```

# Instructions
```bash
python scripts/csv_to_parquet.py
python scripts/build_features.py
python scripts/combine_features.py [OPTIONAL args]
```