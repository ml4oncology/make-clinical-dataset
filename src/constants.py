# lab observations
OBS_MAP = {
    'Hematology': {
        'Basophils': 'basophil',
        'Basos': 'basophil',
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
        
        # NOTE: Hb also exists in Biochemistry, but the procedure is different (arterial blood gas test instead of CBC)
        'Hb': 'hemoglobin',
        'Hct': 'hematocrit',
        'Plt': 'platelet',
        
        'MCH': 'mean_corpuscular_hemoglobin',
        'MCHC': 'mean_corpuscular_hemoglobin_concentration',
        'MCV': 'mean_corpuscular_volume',
        'MPV': 'mean_platelet_volume',
        'RBC': 'red_blood_cell',
        'RDW': 'red_cell_distribution_width',
        'WBC': 'white_blood_cell',

        'INR': 'prothrombin_time_international_normalized_ratio',
        'aPTT': 'activated_partial_thromboplastin_time',

        'Mean Platelet Volume': 'mean_platelet_volume',
    },
    'Biochemistry': {
        'Albumin': 'albumin',
        'Alkaline Phosphatase': 'alkaline_phosphatase',
        'Bicarbonate': 'bicarbonate',
        'Bilirubin,Total': 'total_bilirubin',
        'Carcinoembryonic Antigen': 'carcinoembryonic_antigen',
        'Chloride': 'chloride',
        'Creatinine': 'creatinine',
        'Glucose': 'glucose',
        'Glucose, Random': 'glucose',
        'Lactate Dehydrogenase': 'lactate_dehydrogenase',
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
    }
}

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