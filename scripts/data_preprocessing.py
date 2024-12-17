import pandas as pd
import numpy as np


def data_preprocessing_function(data):
    columns_to_drop = ['encounter_id', 'patient_nbr', 'payer_code', 'medical_specialty']
    data.drop(columns=columns_to_drop, inplace=True)
    threshold = 0.4
    missing_percentages = data.isnull().mean()
    columns_to_drop_due_to_missing = missing_percentages[missing_percentages > threshold].index
    data.drop(columns=columns_to_drop_due_to_missing, inplace=True)
    # data['readmitted'] = data['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
    data['readmitted'] = data['readmitted'].apply(lambda x: 1 if x == '<30' or x == '>30' else 0)

    ordinal_mappings =  {
    'age': {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, 
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }}

    data['age'] = data['age'].map(ordinal_mappings['age'])
    data["gender"] = data["gender"].apply(lambda x: 1 if x=="Male" else 0)
    data["change"] = data["change"].apply(lambda x: 1 if x=='Ch' else 0)
    new_data = data[['age', 
      'gender', 
      'time_in_hospital', 
      'num_lab_procedures', 
      'num_procedures', 
      'num_medications', 
      'number_outpatient', 
      'number_emergency', 
      'number_inpatient', 
      'number_diagnoses',
     'change',
    'readmitted']]
    
    return new_data
