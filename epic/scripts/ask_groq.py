"""
Module to extract/normalize/collapse information from non-sensitive text via LLMs.
We will specificially use Groq for it's generous API request quota.

Current use case is to normalize/collapse the drug name, regimen name, and primary cancer site 
from the raw treatment and diagnosis data text. 

In addition, we want to extract whether each drug is supportive or a direct anticancer drug,
and its prescribed dosage and unit.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import polars as pl
from dotenv import load_dotenv
from groq import Groq
from make_clinical_dataset.shared.constants import INFO_DIR, ROOT_DIR
from ml_common.util import load_pickle, save_pickle, save_table
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PRE_EPIC_CHEMO_PATH = f'{ROOT_DIR}/data/processed/treatment/chemo_2025-07-02.parquet'
EPIC_CHEMO_PATH = f'{ROOT_DIR}/data/processed/treatment/chemo_2025-11-03.parquet'
DIAG_PATH = f'{INFO_DIR}/cancer_diag.csv'


class DrugFormat(BaseModel):
    drug_name_normalized: str = Field(..., description="Normalized drug name as per INN (International Nonproprietary Names)")
    type: Literal["supportive", "direct"] = Field(..., description="Classification of the drug based on its purpose")
    dose: Optional[float] = Field(default=None, description="Prescribed dosage amount (e.g., 4.0 for '4MG')")
    unit: Optional[str] = Field(default=None, description="Prescribed dosage unit (e.g., 'mg' for '4MG')")


class GroqPrompter():
    """
    TODO: Move this to LLM-info-extractor?
    """
    def __init__(self, system_instr: str):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.system_instr = system_instr


    def generate_responses(
        self, 
        dataset: list[str], 
        save_path: str, 
        model_name: str = "llama-3.3-70b-versatile"
    ):
        """
        Args:
            dataset: A list of user input strings
            save_path: path to save results
        """
        save_dir, filename = Path(save_path).parent, Path(save_path).stem
        
        # resume from checkpoint if exists
        if os.path.exists(f'{save_dir}/checkpoint_{filename}.pkl'):
            results = load_pickle(save_dir, f'checkpoint_{filename}')
            dataset = dataset[len(results):]
        else:
            results = []
                    
        for i, text in tqdm(enumerate(dataset)):
            generated_text = self.generate_response(user_input=text, model_name=model_name)
            try:
                result = json.loads(generated_text)
                if isinstance(result, list):
                    raise ValueError("Received a list instead of a dict")
            except (json.JSONDecodeError, ValueError):
                result = {'failed_output': generated_text}
            result['drug_name'] = text
            results.append(result)

            # save checkpoints at every 10th data point
            if i % 10 == 0 and i != 0:
                print(f"Saving checkpoint at i = {i}")
                save_pickle(results, save_dir, f'checkpoint_{filename}')
                
        results = pd.DataFrame(results)

        return results

    
    def generate_response(self, user_input: str, model_name: str, response_format: BaseModel | None = None):
        """
        Args:
            user_input: The raw input string
            model_name: The name of the model to use for inference. Must be supported by Groq.
            response_format: The Pydantic model that defines the expected JSON response format
                If not provided, does not systematically enforce structured JSON output.
        """
        kwargs = {}
        if response_format is not None:
            kwargs['response_format'] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "drug",
                    "schema": response_format.model_json_schema()
                }
            }
            
        chat_completion = self.client.chat.completions.create(
            messages=self.construct_msgs(user_input),
            model=model_name,
            **kwargs
        )

        return chat_completion.choices[0].message.content
    
    
    def construct_msgs(self, content: str):
        return [
            {"role": "system", "content": self.system_instr},
            {"role": "user", "content": content}
        ]


def get_all_drugs(save_path: str):
    pre_epic_drugs = pl.read_parquet(PRE_EPIC_CHEMO_PATH, columns='drug_name')
    epic_drugs = pl.read_parquet(EPIC_CHEMO_PATH, columns='drug_name').drop_nulls()
    drugs = pl.concat([pre_epic_drugs, epic_drugs])['drug_name'].value_counts().sort('count', descending=True)

    # include what's already been processed
    drug_map = pl.read_excel(f'{INFO_DIR}/drug_names_normalized_reviewed_v1.xlsx')
    drug_map = drug_map.rename({'orig_text': 'drug_name', 'drug_name': 'drug_name_normalized'})
    drugs = drugs.join(drug_map, on='drug_name', how='left')

    drugs.write_csv(save_path)


def get_all_regimens(save_path: str):
    pre_epic_regimens = pl.read_parquet(PRE_EPIC_CHEMO_PATH, columns=['regimen', 'cco_regimen'])
    epic_regimens = pl.read_parquet(EPIC_CHEMO_PATH, columns='regimen').drop_nulls()
    regimens = pl.concat([pre_epic_regimens, epic_regimens], how='diagonal')
    regimens = (
        regimens
        .group_by('regimen')
        .agg(pl.len(), pl.col('cco_regimen').unique().drop_nulls().str.concat(delimiter=', '))
        .sort('len', descending=True)
    )
    regimens.write_csv(save_path)


def get_all_sites(save_path: str):
    diag = pl.read_csv(DIAG_PATH)
    sites = diag['PRIMARY_SITE_DESC'].value_counts().sort('count', descending=True)
    sites.write_csv(save_path)


def normalize_drugs(data_dir: str):
    save_path = f'{data_dir}/interim/drugs/drug_names_normalized.csv'
    if not os.path.exists(save_path):
        get_all_drugs(save_path)
    drugs = pd.read_csv(save_path)

    mask = drugs['drug_name_normalized'].isna()
    processed_drugs, unprocessed_drugs = drugs[~mask], drugs[mask]

    # INN = International Nonproprietary Names (INN)
    # globally recognized, unique names assigned to pharmaceutical substances
    # Ref: https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/tools-to-assist-with-the-classification-in-the-hs/hs_classification-decisions/inn-table.aspx
    # inn = pd.read_excel(f'{data_dir}/external/INN.xlsx', header=[2]) # k, this dataset is not helpful...there are missing/incorrect drugs
    # inn = inn['Product (En)'].str.lower()

    system_instr = """
You are a professional natural language processing assistant specialized in medical text.
Your task is to extract and normalize drug information from unstructured text.

Only return a structured JSON object matching the following schema:

{
  "drug_name_normalized": "<string>",
  "type": "supportive | direct",
  "dose": <float or null>,
  "unit": "<string or null>"
}

Ensure 'drug_name_normalized' is in INN (International Nonproprietary Names) format and lowercase. 
Note, some text may only contain the generic brand name of the drug. 
You must convert to INN. Please double check using the internet. 

Use the internet to determine if the drug is supportive or direct cancer treatment drug.

If the dose or unit is missing or unclear, set them to null. 

Only include information present in the text. Do not guess or infer missing values.

Here are some examples:
Input: 'dexamethasone TABLET 4MG'
Output: {"drug_name": "dexamethasone", "type": "supportive", "dose": 4, "unit": "mg"}

Input: 'PEMBROLIZUMAB IN 50 ML NS IV INTERMITTENT MIXTURE'
Output: {"drug_name": "pembrolizumab", "type": "direct", "dose": 50, "unit": "ml"}
"""
    prompter = GroqPrompter(system_instr)
    results = prompter.generate_responses(
        dataset=unprocessed_drugs['drug_name'].tolist(), 
        save_path=save_path, 
        model_name="openai/gpt-oss-120b", # "llama-3.3-70b-versatile"
    )

    # set all drugs used in trial and studies as study_drug
    mask = results['drug_name'].str.contains('study|trial|placebo') | results['drug_name'].str.startswith('INV ')
    results.loc[mask, 'drug_name_normalized'] = 'study_drug'

    # resave the results
    results = pd.concat([processed_drugs, results])
    save_table(results, save_path, index=False)


def normalize_regimens(data_dir: str):
    system_instr = """
You are a professional natural language processing assistant specialized in medical text.
Your task is to extract and normalize regimen information from unstructured text.

Only return a structured JSON object matching the following schema:

{
  "regimen_normalized": "<string>",
  "schedule": "<string or null>",
  "radiation_therapy": <bool>,
  "additional_notes": "<string or null>"
}

'regimen_normalized' must follow the Cancer Care Ontario regimen taxonomy.
Please double check using the internet. 

If the schedule is missing or unclear, set them to null. 

Only include information present in the text. Do not guess or infer missing values.

Here are some examples:
Input: 'GI-GEM D1,8,15'
Output: {"regimen_normalized": "GEMC", "schedule": "D1,8,15", "radiation_therapy": false, "additional_notes": null}

Input: 'LU-ETOPCISP 3 DAY'
Output: {"regimen_normalized": "CISPETOP(3D)", "schedule": "3 DAY", "radiation_therapy": false, "additional_notes": null}

Input: 'LU-ETOPCISP-RT'
Output: {"regimen_normalized": "CISPETOP(RT)", "schedule": null, "radiation_therapy": yes, "additional_notes": null}
"""


def normalize_sites(data_dir: str):
    save_path = f'{data_dir}/interim/site_names_normalized.csv'
    if not os.path.exists(save_path):
        get_all_sites(save_path)
    sites = pd.read_csv(save_path)
    system_instr = """
You are a professional natural language processing assistant specialized in clinical oncology terminology.
Your task is to normalize unstructured cancer site descriptions into ICD-10 cancer codes.

Only return a structured JSON object matching the following schema:

{
  "cancer_code_ICD10": "<string or null>",
  "cancer_site_normalized": "<string or null>",
}

Rules:
1. ONLY provide the first three letters of the ICD-10 code
2. Select the BEST and MOST SPECIFIC ICD-10 code that matches the input text.
3. If the description does not clearly match any code, set "cancer_code_ICD10" to null.
4. For "ncancer_site_normalized":
   - Return a brief, standardized anatomical site name.
   - Do **not** just copy the input text; normalize to the canonical anatomical term.
5. Do NOT infer or guess beyond what is explicitly stated.
6. Use the internet if needed to confirm anatomical terminology or definitions.
7. Only output the JSON. No explanations.

Here are some examples:

Input: "Central portion of breast"
Output:
{
  "cancer_code_ICD10": "C50",
  "cancer_site_normalized": "Breast",
}

Input: "Submandibular gland"
Output:
{
  "cancer_code_ICD10": "C08",
  "cancer_site_normalized": "Other and unspecified major salivary glands",
}
"""
    prompter = GroqPrompter(system_instr)
    results = prompter.generate_responses(
        dataset=sites['PRIMARY_SITE_DESC'].tolist(), 
        save_path=save_path, 
        model_name="openai/gpt-oss-120b",
    )
    results.index = sites.index
    results = pd.concat([sites, results], axis=1)
    save_table(results, save_path, index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--normalize', type=str, choices=['drugs', 'regimens', 'sites'])
    args = parser.parse_args()

    if args.normalize == 'drugs':
        normalize_drugs(data_dir=args.data_dir)
    elif args.normalize == 'regimens':
        normalize_regimens(data_dir=args.data_dir)
    elif args.normalize == 'sites':
        normalize_sites(data_dir=args.data_dir)