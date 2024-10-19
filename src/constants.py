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
    },
    'Biochemistry': {
        'Albumin': 'albumin',
        'Bicarbonate': 'bicarbonate',
        'Carcinoembryonic Antigen': 'carcinoembryonic_antigen',
        'Chloride': 'chloride',
        'Creatinine': 'creatinine',
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
    }
}