import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sys


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion(DataIngestionConfig):
    def __init__(self):
        super().__init__()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            file_loc = os.path.join("notebook", "data", "StudentsPerformance.csv")
            df = pd.read_csv(file_loc)
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)
            return (self.train_data_path, self.test_data_path)

        except Exception as error:
            raise CustomException(error, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
