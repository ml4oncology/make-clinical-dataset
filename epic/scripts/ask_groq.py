"""
Module to extract/normalize/collapse information from non-sensitive text via LLMs.
We will specificially use Groq for it's generous API request quota.

Current use case is to normalize/collapse the drug name and regimen name from the raw treatment data text. 
In addition, we want to extract whether each drug is supportive or a direct anticancer drug,
and its prescribed dosage and unit.
"""
import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from make_clinical_dataset.shared.constants import ROOT_DIR
from ml_common.util import load_pickle, save_pickle, save_table
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATE = '2025-07-02'
DATA_DIR = f'{ROOT_DIR}/data/raw/data_pull_{DATE}'


class DrugFormat(BaseModel):
    drug_name: str = Field(..., description="Normalized drug name as per INN (International Nonproprietary Names)")
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
            except json.JSONDecodeError:
                result = {'failed_output': generated_text}
            result['orig_text'] = text
            results.append(result)

            # save checkpoints at every 10th data point
            if i % 10 == 0 and i != 0:
                print(f"Saving checkpoint at i = {i}")
                save_pickle(results, save_dir, f'checkpoint_{filename}')
                
        results = pd.DataFrame(results)

        # save the results
        save_table(results, save_path)

        # delete the checkpoint
        os.remove(f'{save_dir}/checkpoint_{filename}.pkl')

        return results

    
    def generate_response(self, user_input: str, model_name: str, response_format: BaseModel | None = None):
        """
        Args:
            user_input: The raw input string
            model_name: The name of the model to use for inference. Must be supported by Groq.
            response_format: The Pydantic model that defines the expected JSON response format
                If not provided, does not systematically enforce structured JSON output.

        NOTE: response format is only currently supported by 
            moonshotai/kimi-k2-instruct  - 1k requests per day
            meta-llama/llama-4-maverick-17b-128e-instruct  - 1k requests per day
            meta-llama/llama-4-scout-17b-16e-instruct  - 1k requests per day

        API Rate limit quota for the other models:
            llama3-70b-8192 - 14.4k requests per day
            llama-3.3-70b-versatile - 1k requests per day

        NOTE: llama3-70b-8192 does not support tool or function calling, including internet lookups or browsing
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


def get_all_drugs(output_path: str):
    paths = sorted(glob(f'{DATA_DIR}/chemo_pre_epic_csv/*.csv'))
    ct_pre_epic = pd.concat([pd.read_csv(path, encoding='cp1252') for path in paths], ignore_index=True)
    paths = sorted(glob(f'{DATA_DIR}/chemo_epic_csv/*.csv'))
    ct_epic = pd.concat([pd.read_csv(path, encoding='cp1252') for path in paths], ignore_index=True)
    ct_epic['medication_name'] = ct_epic['medication_name'] + (' - ' + ct_epic['generic_name']).fillna('')
    drugs = pd.concat([ct_pre_epic['medication_name'], ct_epic['medication_name']]).value_counts()
    drugs.to_csv(output_path)


def normalize_drugs(data_dir: str):
    drug_filepath = f'{data_dir}/interim/drugs/drug_names.csv'
    if not os.path.exists(drug_filepath):
        get_all_drugs(drug_filepath)
    drugs = pd.read_csv(drug_filepath)

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
  "drug_name": "<string>",
  "type": "supportive | direct",
  "dose": <float or null>,
  "unit": "<string or null>"
}

Ensure 'drug_name' is in INN (International Nonproprietary Names) format and lowercase. 
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
    save_path = f'{data_dir}/interim/drugs/drug_names_normalized.xlsx'
    results = prompter.generate_responses(
        dataset=drugs['medication_name'].tolist(), 
        save_path=save_path, 
        model_name="llama-3.3-70b-versatile"
    )

    # set all drugs used in trial and studies as study_drug
    mask = results['orig_text'].str.contains('study|trial|placebo')
    results.loc[mask, 'drug_name'] = 'study_drug'

    # resave the results
    save_table(results, save_path)


def normalize_regimens():
    pass

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--normalize', type=str, choices=['drugs', 'regimens'])
    args = parser.parse_args()

    if args.normalize == 'drugs':
        normalize_drugs(data_dir=args.data_dir)
    elif args.normalize == 'regimens':
        normalize_regimens(data_dir=args.data_dir)