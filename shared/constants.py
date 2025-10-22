from pathlib import Path

###############################################################################
# Paths
###############################################################################
ROOT_DIR = '/cluster/projects/gliugroup/2BLAST'
SRC_DIR = str(Path(__file__).parent.parent)
DEFAULT_CONFIG_PATH = f"{SRC_DIR}/config/default.yaml"
INFO_DIR = f'{ROOT_DIR}/data/info/'

###############################################################################
# Column Mappings - TODO: deprecate this in favor of model.py
###############################################################################
OPIS_COL_MAP = { # 2023-02-21
    'Hosp_Chart': 'mrn',
    'Regimen': 'regimen',
    'First_Trt_Date': 'first_treatment_date',
    'Trt_Date': 'treatment_date',
    'cycle_number': 'cycle_number',
    'Drug_name': 'drug_name',
    'REGIMEN_DOSE': 'regimen_dose',
    'Dose_Given': 'given_dose',
    'Height': 'height',
    'Weight': 'weight',
    'Body_Surface_Area': 'body_surface_area',
    'Intent': 'intent',
    'Change_Reason_Desc': 'change_reason_desc',
    'Route': 'route',
    'Dose_Ord': 'dose_ordered',
    'CHEMO_FLAG': 'chemo_flag'
}
CHEMO_EPIC_COL_MAP = { # 2025-07-02
    'PATIENT_RESEARCH_ID': 'patient_id',
    'protocol_name': 'regimen',
    'cycle_start_date': 'treatment_date',
    'cycle_number': 'cycle_number',
    'med_Epic_id': 'uhn_drug_code',
    'medication_order_name': 'drug_name',
    'medication_generic_name': 'drug_name_ext',
    'minimum_dose': 'given_dose',
    'dose_unit': 'given_dose_unit',
    'medcation_volume': 'diluent_volume',
    'strength': 'drug_dose',
    'current_dose_percentage_of_original_dose': 'percentage_of_ideal_dose',
    'dosing_weight': 'weight',
    'dosing_height': 'height',
    'dosing_bsa': 'body_surface_area',
    'treatment_intent': 'intent',
    'route': 'route',
    'frequency': 'frequency',
    'treatment_plan_start_date_as_schedueled': 'first_scheduled_treatment_date',
    'original_planned_cycle_start_date': 'scheduled_treatment_date',
    'day_status': 'status',
    'cycle_name': 'cycle_desc',
    'order_category': 'treatment_category'
}
CHEMO_PRE_EPIC_COL_MAP = { # 2025-07-02
    'PATIENT_RESEARCH_ID': 'patient_id',
    'treatment_plan': 'regimen', # 'treatment_plan',
    'fist_treatment_date': 'first_treatment_date',
    'treatment_date': 'treatment_date',
    'cycle_num': 'cycle_number',
    'DIN': 'drug_id', # DIN - drug identification number - assigned by Health Canada
    'fdb_generic_code': 'fdb_drug_code', # FDB - First Databank - world's drug database leader
    'medication_name': 'drug_name', # 'medication_name',
    'medication_dose': 'drug_dose', # 'medication_dose', # beware some have "nan mg" and need to be combined with dose_ordered
    'medication_dose_ordered': 'drug_dose_ordered',
    'height': 'height',
    'weight': 'weight',
    'BSA': 'body_surface_area',
    'treatment_intent': 'intent',
    'route': 'route',
    'regimen_link': 'regimen_normalized', # very useful! we want all regimen to follow this format
    'treatment_type': 'treatment_category'
}
RAD_COL_MAP = { # 2025-07-02
    'PATIENT_RESEARCH_ID': 'patient_id',
    'TX_Start_Date': 'treatment_start_date',
    'TX_End_Date': 'treatment_end_date',
    'Site_Treated': 'site_treated',
    'Dose_Prescribed': 'dose_prescribed',
    'Fractions_Prescribed': 'fractions_prescribed',
    'Dose_Delivered': 'dose_given',
    'Fractions_Delivered': 'fractions_given',
    'TX_Intent': 'intent',
    'Diagnosis_ICD_Code': 'diagnosis_icd_code',
    'Diagnosis_Description': 'diagnosis_desc',
    'Diagnosis_Category': 'diagnosis_category',
    'Morphology': 'morphology'
}

###############################################################################
# Lab Observations
###############################################################################
OBS_MAP = {
    'Hematology': {
        'Basophils': 'basophil',
        # 'Basos': 'basophil', # distribution seems odd, not using this one
        'Eosinophils': 'eosinophil',
        'Eosin': 'eosinophil',
        'Eos': 'eosinophil',
        'Lymphocytes': 'lymphocyte',
        'Lymphs': 'lymphocyte',
        'Lymph': 'lymphocyte',
        'Monocytes': 'monocyte', 
        'Monos': 'monocyte',
        'Mono': 'monocyte', 
        'Neutrophils': 'neutrophil',
        'NeutroA': 'neutrophil',
        
        # NOTE: Hb and Hemoglobin also exists in Biochemistry, but the procedure is different (arterial blood gas test instead of CBC)
        'Hb': 'hemoglobin',
        'Hct': 'hematocrit',
        'Plt': 'platelet',
        
        'MCH': 'mean_corpuscular_hemoglobin',
        'MCHC': 'mean_corpuscular_hemoglobin_concentration',
        'MCV': 'mean_corpuscular_volume',
        'MPV': 'mean_platelet_volume',
        'PCV': 'hematocrit', # packed cell volume, same as hematocrit test
        'RBC': 'red_blood_cell',
        'RDW': 'red_cell_distribution_width',
        'WBC': 'white_blood_cell',

        'INR': 'prothrombin_time_international_normalized_ratio',
        'aPTT': 'activated_partial_thromboplastin_time',

        'Mean Platelet Volume': 'mean_platelet_volume',
    },
    'Biochemistry': {
        'Albumin': 'albumin',
        'Bicarbonate': 'bicarbonate',
        'Calcium': 'calcium',
        'Carcinoembryonic Antigen': 'carcinoembryonic_antigen',
        'Chloride': 'chloride',
        'Creatinine': 'creatinine',
        'Estimated GFR': 'eGFR', # estimated glomerular filtration rate'
        'Glucose': 'glucose',
        'Magnesium': 'magnesium',
        'Phosphate': 'phosphate',
        'Potassium': 'potassium',
        'Sodium': 'sodium',
        'Total Bilirubin': 'total_bilirubin',
        'Tot Bilirubin': 'total_bilirubin',

        'AST': 'aspartate_aminotransferase',
        'ALT': 'alanine_aminotransferase',
        'ALP': 'alkaline_phosphatase',
        'CA 19-9': 'carbohydrate_antigen_19-9',
        'LDH': 'lactate_dehydrogenase',
        'RDW-CV': 'red_cell_distribution_width',

        # Other possible useful features
        # 'Urea Plasma': 'urea', # per Rob, not a common test, might cause label leakage - this test is only taken when clinician believes something is wrong
        # 'Amylase': 'amylase',
        # 'Anion Gap': 'anion_gap',
        # 'Lactate': 'lactate',
        # 'pH': 'pH',
        # 'pCO2': 'pCO2',
        # 'pO2': 'pO2',
        # 'HCO3': 'HCO3',
        # 'O2 Saturation': 'oxygen_saturation',

        # EPIC specific names
        'Alkaline Phosphatase': 'alkaline_phosphatase',
        'Bilirubin,Total': 'total_bilirubin',
        'Glucose, Random': 'glucose',
        'Lactate Dehydrogenase': 'lactate_dehydrogenase',
    }
}

UNIT_MAP = {
    "L/L": ["hematocrit"],
    "U/L": [
        "alanine_aminotransferase",
        "alkaline_phosphatase",
        "aspartate_aminotransferase",
        "lactate_dehydrogenase",
    ],
    "cm": ["height"],
    "fL": ["mean_corpuscular_volume", "mean_platelet_volume"],
    "g/L": ["albumin", "hemoglobin", "mean_corpuscular_hemoglobin_concentration"],
    "kg": ["weight"],
    "kU/L": ["carbohydrate_antigen_19-9"],
    "mL/min/1.73m2": ["eGFR"],
    "mmol/L": [
        "bicarbonate",
        "calcium",
        "chloride",
        "glucose",
        "magnesium",
        "phosphate",
        "potassium",
        "sodium",
    ],
    "m^2": ["body_surface_area"],
    "pg": ["mean_corpuscular_hemoglobin"],
    "s": ["activated_partial_thromboplastin_time"],
    "umol/L": ["creatinine", "total_bilirubin"],
    "x10e9/L": [
        "basophil",
        "eosinophil",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
        "white_blood_cell",
    ],
    "x10e12/L": ["red_blood_cell"],
    "years": ["age"],
    "%CV": ["red_cell_distribution_width"],
}


###############################################################################
# Symptom Surveys
###############################################################################
ESAS_MAP = {
    'Anxiety': 'anxiety',
    'Appetite': 'lack_of_appetite', 
    'Depression': 'depression',
    'Drowsiness': 'drowsiness',
    'ECOG (Patient reported)': 'ecog',
    'Feeling of Well-being': 'well_being',
    'Lack of Appetite': 'lack_of_appetite',
    'Nausea': 'nausea',
    'Pain': 'pain', 
    'Shortness of breath': 'shortness_of_breath',
    'Tiredness': 'tiredness',
    'Wellbeing': 'well_being',
}


###############################################################################
# Columns
###############################################################################
LAB_COLS = sorted(set([*OBS_MAP['Hematology'].values(), *OBS_MAP['Biochemistry'].values()]))
SYMP_COLS = sorted(set(ESAS_MAP.values()))

# Specifically for EPR
# TODO: move somewhere else?
LAB_CHANGE_COLS = [f"{col}_change" for col in LAB_COLS]
SYMP_CHANGE_COLS = [f"{col}_change" for col in SYMP_COLS]
EPR_DRUG_COLS = [
    '%_ideal_dose_given_GEMCITABINE',
    '%_ideal_dose_given_CISPLATIN',
    '%_ideal_dose_given_FLUOROURACIL',
    '%_ideal_dose_given_ETOPOSIDE',
    '%_ideal_dose_given_CARBOPLATIN',
    '%_ideal_dose_given_PEMETREXED',
    '%_ideal_dose_given_OXALIPLATIN',
    '%_ideal_dose_given_VINORELBINE',
    '%_ideal_dose_given_IRINOTECAN',
    '%_ideal_dose_given_PACL',
    '%_ideal_dose_given_NAB-PACL',
    '%_ideal_dose_given_PEMBROLIZUMAB',
    '%_ideal_dose_given_NIVOLUMAB',
    '%_ideal_dose_given_DOCETAXEL',
    '%_ideal_dose_given_EPIRUBICIN',
    '%_ideal_dose_given_TRASTUZUMAB',
    '%_ideal_dose_given_CYCLOPHOSPHAMIDE',
    '%_ideal_dose_given_DOXORUBICIN',
    '%_ideal_dose_given_DURVALUMAB',
    '%_ideal_dose_given_BEVACIZUMAB',
    '%_ideal_dose_given_CETUXIMAB',
    '%_ideal_dose_given_RAMUCIRUMAB',
    '%_ideal_dose_given_RALTITREXED',
    '%_ideal_dose_given_TREMELIMUMAB',
    '%_ideal_dose_given_PERTUZUMAB',
    '%_ideal_dose_given_TOPOTECAN',
    '%_ideal_dose_given_LIPOSOMAL IRINOTECAN',
    '%_ideal_dose_given_ATEZOLIZUMAB',
    '%_ideal_dose_given_IPILIMUMAB',
    '%_ideal_dose_given_MITOMYCIN',
    '%_ideal_dose_given_PANITUMUMAB',
    '%_ideal_dose_given_CAPECITABINE',
    '%_ideal_dose_given_ERLOTINIB',
    '%_ideal_dose_given_LENVATINIB',
    '%_ideal_dose_given_OLAPARIB',
]
###############################################################################
# Other Mappings
###############################################################################
# CTCAE: Common Terminology Criteria for Adverse Events
# CTCAE v5.0: https://ctep.cancer.gov/protocoldevelopment/electronic_applications/docs/CTCAE_v5_Quick_Reference_8.5x11.pdf
# Constants for CTCAE thresholds
# ULN = upper limit of normal
CTCAE_CONSTANTS = {
    'hemoglobin': {
        'grade2plus': 100, # <100 - 80 g/L
        'grade3plus': 80 # <80 g/L
    },
    'neutrophil': {
        'grade2plus': 1.5, # <1.5 - 1.0 x 10e9 /L
        'grade3plus': 1.0 # <1.0 - 0.5 x 10e9 /L
    },
    'platelet': {
        'grade2plus': 75, # <75 - 50 x 10e9 /L
        'grade3plus': 50 # <50 - 25 x 10e9 /L
    },
    'bilirubin': {
        'grade2plus': 1.5, # >1.5 - 3.0 x ULN if baseline was normal, >1.5 - 3.0 x baseline if baseline was abnormal
        'grade3plus': 3.0, # >3.0 - 10.0 x ULN if baseline was normal; >3.0 - 10.0 x baseline if baseline was abnormal
        'ULN': 22.0
    },
    'AKI': {
        'grade2plus': 1.5, # >1.5 - 3.0 x baseline; >1.5 - 3.0 x ULN
        'grade3plus': 3.0, # >3.0 x baseline; >3.0 - 6.0 x ULN
        'ULN': 353.68
    },
    'ALT': {
        'grade2plus': 3.0, # >3.0 - 5.0 x ULN if baseline was normal; >3.0 - 5.0 x baseline if baseline was abnormal
        'grade3plus': 5.0, # >5.0 - 20.0 x ULN if baseline was normal; >5.0 - 20.0 x baseline if baseline was abnormal
        'ULN': 40.0
    },
    'AST': {
        'grade2plus': 3.0, # >3.0 - 5.0 x ULN if baseline was normal; >3.0 - 5.0 x baseline if baseline was abnormal
        'grade3plus': 5.0, # >5.0 - 20.0 x ULN if baseline was normal; >5.0 - 20.0 x baseline if baseline was abnormal
        'ULN': 34.0
    },
}

# Mapping from CTCAE names to dataframe lab names
MAP_CTCAE_LAB = {
    'AKI': 'creatinine', # Acute kidney injury / creatinine increase
    'ALT': 'alanine_aminotransferase', # Alanine aminotransferase increase
    'AST': 'aspartate_aminotransferase', # Aspartate aminotransferase increase
    'bilirubin': 'total_bilirubin', # Blood bilirubin increase
    'hemoglobin': 'hemoglobin', # Hemoglobin decrease
    'neutrophil': 'neutrophil', # Neutrophil count decrease
    'platelet': 'platelet' # Platelet count decrease
}

# RECIST: Response Evaluation Criteria in Solid Tumors
# RECIST v1.1: https://ctep.cancer.gov/protocoldevelopment/docs/recist_guideline.pdf
RECIST_RANKING = {
    'CR': 5, # complete response
    'PR': 4, # partial response
    'SD': 3, # stable disease
    'PD': 2, # progressive disease
    'NE': 1 # not evaluable
}

# Intent of treatment
TRT_INTENT = {
    'P': 'palliative',
    'C': 'curative',
    'A': 'adjuvant',
    'N': 'neoadjuvant',
}