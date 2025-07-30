"""
Centralized place for data schemas

TODO: decide whether to keep this or not
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class OPISRecord(BaseModel): # 2023-02-21
    mrn: str = Field(alias='Hosp_Chart')
    regimen: str = Field(alias='Regimen')
    first_treatment_date: Optional[date] = Field(alias='First_Trt_Date')
    treatment_date: Optional[date] = Field(alias='Trt_Date')
    cycle_number: Optional[int] = Field(alias='cycle_number')
    drug_name: str = Field(alias='Drug_name')
    regimen_dose: Optional[float] = Field(alias='REGIMEN_DOSE')
    given_dose: Optional[float] = Field(alias='Dose_Given')
    height: Optional[float] = Field(alias='Height')
    weight: Optional[float] = Field(alias='Weight')
    body_surface_area: Optional[float] = Field(alias='Body_Surface_Area')
    intent: Optional[str] = Field(alias='Intent')
    change_reason_desc: Optional[str] = Field(alias='Change_Reason_Desc')
    route: Optional[str] = Field(alias='Route')
    dose_ordered: Optional[float] = Field(alias='Dose_Ord')
    chemo_flag: Optional[str] = Field(alias='CHEMO_FLAG')


class ChemoEpicRecord(BaseModel): # 2025-07-02
    patient_id: str = Field(alias='PATIENT_RESEARCH_ID')
    regimen: str = Field(alias='treamtment_plan_name')
    first_treatment_date: Optional[date] = Field(alias='plan_start_date')
    treatment_date: Optional[date] = Field(alias='cycle_start_date')
    cycle_number: Optional[int] = Field(alias='cycle_number')
    uhn_drug_code: Optional[int] = Field(alias='med_id') # special UHN drug code
    drug_name: str = Field(alias='medication_name')
    given_dose: Optional[str] = Field(alias='medication_dose')
    weight: Optional[float] = Field(alias='dosing_weight')
    height: Optional[float] = Field(alias='dosing_height')
    body_surface_area: Optional[float] = Field(alias='dosing_BSA')
    intent: Optional[str] = Field(alias='treatment_intent')


class ChemoPreEpicRecord(BaseModel): # 2025-07-02
    patient_id: str = Field(alias='PATIENT_RESEARCH_ID')
    regimen: str = Field(alias='treatment_plan')
    first_treatment_date: Optional[date] = Field(alias='fist_treatment_date') 
    treatment_date: Optional[date] = Field(alias='treatment_date')
    cycle_number: Optional[int] = Field(alias='cyclc_num')
    drug_id: Optional[int] = Field(alias='DIN') # DIN - drug identification number - assigned by Health Canada
    fdb_drug_code: Optional[int] = Field(alias='fdb_generic_code') # FDB - First Databank - world's drug database leader
    drug_name: str = Field(alias='medication_name')
    given_dose: Optional[str] = Field(alias='medication_dose') # beware some have "nan mg" and need to be combined with dose_ordered
    height: Optional[float] = Field(alias='height')
    weight: Optional[float] = Field(alias='weight')
    body_surface_area: Optional[float] = Field(alias='BSA')
    intent: Optional[str] = Field(alias='treatment_intent')
    dose_ordered: Optional[float] = Field(alias='medication_dose_ordered')
    route: Optional[str] = Field(alias='route')
    regimen_normalized: Optional[str] = Field(alias='regimen_link')  # very useful! we want all regimen to follow this format


class RadiationRecord(BaseModel):
    patient_id: str = Field(alias='PATIENT_RESEARCH_ID')
    treatment_start_date: Optional[date] = Field(alias='TX_Start_Date')
    treatment_end_date: Optional[date] = Field(alias='TX_End_Date')
    site_treated: Optional[str] = Field(alias='Site_Treated')
    dose_prescribed: Optional[float] = Field(alias='Dose_Prescribed')
    fractions_prescribed: Optional[int] = Field(alias='Fractions_Prescribed')
    dose_given: Optional[float] = Field(alias='Dose_Delivered')
    fractions_given: Optional[int] = Field(alias='Fractions_Delivered')
    intent: Optional[str] = Field(alias='TX_Intent')
    diagnosis_icd_code: Optional[str] = Field(alias='Diagnosis_ICD_Code')
    diagnosis_desc: Optional[str] = Field(alias='Diagnosis_Description')
    diagnosis_category: Optional[str] = Field(alias='Diagnosis_Category')
    morphology: Optional[str] = Field(alias='Morphology')


OPIS_COL_MAP = {field.alias: name for name, field in OPISRecord.model_fields.items()}
CHEMO_EPIC_COL_MAP = {field.alias: name for name, field in ChemoEpicRecord.model_fields.items()}
CHEMO_PRE_EPIC_COL_MAP = {field.alias: name for name, field in ChemoPreEpicRecord.model_fields.items()}
RAD_COL_MAP = {field.alias: name for name, field in RadiationRecord.model_fields.items()}