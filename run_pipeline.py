import pandas as pd
import numpy as np
from scripts import data_ingestion
from scripts import data_preprocessing
from scripts import model_dev


def run_pipeline(data_path):
    df = data_ingestion.load_data(file_path=data_path)
    cleaned_data = data_preprocessing.data_preprocessing_function(data=df)
    model_dev.model_dovelopment(data=cleaned_data)




if __name__ == '__main__':
    data_path = 'D:\FREELANCE_PROJECTS\diabetes-client-readmit-prediction\data\diabetic_data.csv'
    run_pipeline(data_path)