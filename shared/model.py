"""
Centralized place for data schemas

TODO: decide whether to keep this or not
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class OPISRecord(BaseModel): # 2023-02-21
    mrn: str = Field(alias='Hosp_Chart')
    treatment_date: date = Field(alias='Trt_Date')
    regimen: str = Field(None, alias='Regimen')
    first_treatment_date: date = Field(None, alias='First_Trt_Date')
    cycle_number: int = Field(None, alias='cycle_number')
    drug_name: str = Field(None, alias='Drug_name')
    regimen_dose: float = Field(None, alias='REGIMEN_DOSE')
    given_dose: float = Field(None, alias='Dose_Given')
    dose_ordered: float = Field(None, alias='Dose_Ord')
    height: float = Field(None, alias='Height')
    weight: float = Field(None, alias='Weight')
    body_surface_area: float = Field(None, alias='Body_Surface_Area')
    intent: str = Field(None, alias='Intent')
    change_reason_desc: str = Field(None, alias='Change_Reason_Desc')
    route: str = Field(None, alias='Route')
    chemo_flag: str = Field(None, alias='CHEMO_FLAG')


class ChemoEpicRecord(BaseModel): # 2025-07-02
    patient_id: str = Field(alias='PATIENT_RESEARCH_ID')
    treatment_date: date = Field(alias='cycle_start_date')
    regimen: str = Field(None, alias='protocol_name')
    cycle_number: float = Field(None, alias='cycle_number')
    uhn_drug_code: float = Field(None, alias='med_Epic_id') # special UHN drug code
    drug_name: str = Field(None, alias='medication_order_name')
    drug_name_ext: str = Field(None, alias='medication_generic_name') # need to merge the drug names together
    given_dose: float = Field(None, alias='minimum_dose')
    given_dose_unit: str = Field(None, alias='dose_unit')
    diluent_volume: float = Field(None, alias='medcation_volume') # the solution in which drug is dissolved with
    # diluent_volume_unit: str = Field(None, alias='medication_volume_unit') # Literal['mL', '*Unspecified']
    drug_dose: str = Field(None, alias='strength') # can be useful when given dose is missing
    percentage_of_ideal_dose: float = Field(None, alias="current_dose_percentage_of_original_dose")
    weight: float = Field(None, alias='dosing_weight')
    height: float = Field(None, alias='dosing_height')
    body_surface_area: float = Field(None, alias='dosing_bsa')
    intent: str = Field(None, alias='treatment_intent')
    route: str = Field(None, alias='route')
    frequency: str = Field(None, alias='frequency')
    # infusion_duration: float = Field(None, alias="minimum_infusion_duration")
    # infusion_duration_unit: str = Field(None, alias="infusion_duration_unit") # Literal['Minutes', 'Hours', 'Days', '*Unspecified']
    # infusion_rate: float = Field(None, alias="minimum_infusion_rate")
    # infusion_rate_unit: str = Field(None, alias="infusion_rate_unit") # Literal['mL/hr', '*Unspecified']]
    first_scheduled_treatment_date: date = Field(None, alias='treatment_plan_start_date_as_schedueled')
    scheduled_treatment_date: date = Field(None, alias='original_planned_cycle_start_date')
    status: str = Field(alias='day_status') # Literal['Deleted', 'Canceled', 'Completed', 'Deferred', 'Given Externally']
    cycle_desc: str = Field(None, alias="cycle_name")
    treatment_category: str = Field(None, alias='order_category')
    # "treamtment_plan_name" - same as protocol_name
    # "cycle_status" - all "Completed"
    # "maximum_infusion_duration" - only 330 records
    # "maximum_infusion_rate" - only 44 records
    # "maximum_dose" - <3000 records


class ChemoPreEpicRecord(BaseModel): # 2025-07-02
    patient_id: str = Field(alias='PATIENT_RESEARCH_ID')
    treatment_date: date = Field(None, alias='treatment_date')
    regimen: str = Field(None, alias='treatment_plan')
    first_treatment_date: date = Field(None, alias='fist_treatment_date') 
    cycle_number: float = Field(None, alias='cyclc_num')
    drug_id: float = Field(None, alias='DIN') # DIN - drug identification number - assigned by Health Canada
    fdb_drug_code: float = Field(None, alias='fdb_generic_code') # FDB - First Databank - world's drug database leader
    drug_name: str = Field(None, alias='medication_name')
    drug_dose: str = Field(None, alias='medication_dose') # beware some have "nan mg" and need to be combined with drug_dose_ordered
    drug_dose_ordered: float = Field(None, alias='medication_dose_ordered')
    height: float = Field(None, alias='height')
    weight: float = Field(None, alias='weight')
    body_surface_area: float = Field(None, alias='BSA')
    intent: str = Field(None, alias='treatment_intent')
    route: str = Field(None, alias='route')
    cco_regimen: str = Field(None, alias='regimen_link')  # very useful! we want all regimen to follow the Cancer Care Ontario format
    treatment_category: str = Field(None, alias='treatment_type')  # Literal['Pre', 'Post', 'Chemo']


class RadiationRecord(BaseModel):
    patient_id: str = Field(alias='PATIENT_RESEARCH_ID')
    treatment_start_date: date = Field(alias='TX_Start_Date')
    treatment_end_date: date = Field(None, alias='TX_End_Date')
    site_treated: str = Field(None, alias='Site_Treated')
    dose_prescribed: float = Field(None, alias='Dose_Prescribed')
    fractions_prescribed: float = Field(None, alias='Fractions_Prescribed')
    dose_given: float = Field(None, alias='Dose_Delivered')
    fractions_given: float = Field(None, alias='Fractions_Delivered')
    intent: str = Field(None, alias='TX_Intent')
    diagnosis_icd_code: str = Field(None, alias='Diagnosis_ICD_Code')
    diagnosis_desc: str = Field(None, alias='Diagnosis_Description')
    diagnosis_category: str = Field(None, alias='Diagnosis_Category')
    morphology: str = Field(None, alias='Morphology')


OPIS_COL_MAP = {field.alias: name for name, field in OPISRecord.model_fields.items()}
CHEMO_EPIC_COL_MAP = {field.alias: name for name, field in ChemoEpicRecord.model_fields.items()}
CHEMO_PRE_EPIC_COL_MAP = {field.alias: name for name, field in ChemoPreEpicRecord.model_fields.items()}
RAD_COL_MAP = {field.alias: name for name, field in RadiationRecord.model_fields.items()}