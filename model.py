import os
import sys
import pandas as pd
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

sys.path.append("src")
from logger import logging
from exception import CustomException

from dataclasses import dataclass
import pickle

print("Dagshub Tutorial")
print('-'*80)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data',"Iris.csv")
    print(f"Inside DataIngestionConfig {train_data_path}")
    print('-'*80)
    

class ModelTrainer:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # print(f"Inside __Init__ method {self.ingestion_config}")

    #function to log the model parameters
    def model_run(self,model_type):
        
        # print(f"Inside model_run {self.ingestion_config.train_data_path}")  
      
        #  Reading the data
        df = pd.read_csv(self.ingestion_config.train_data_path)  
        # print(df.head())
        # print('-'*80)
        
        #splitting in features and labels    
        X  = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm']]    
        y = df['Species']

        #test train split    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
        
        # models = {
        #         "RandomForestClassifier": RandomForestClassifier(),
        #         "DecisionTreeClassifier": DecisionTreeClassifier(),
        #         "LogisticRegression": LogisticRegression(),
        #         "SGDClassifier": SGDClassifier(),
        #     }

        with dagshub.dagshub_logger() as logger:
            try:
                
                    #model defnition
                model = model_type(random_state=42)
                logging.info(f"Model {model}/n")
                logging.info('-'*80)
                    
                    #log the model parameters
                logger.log_hyperparams(model_class=type(model).__name__)
                logger.log_hyperparams({'model': model.get_params()})
                    
                    #training the model
                rfc = model.fit(X_train, y_train)
                logging.info(f"rfc {rfc}/n")
                logging.info('-'*80)
                    
                    #predictions
                y_pred = rfc.predict(X_test)
                # logging.info(f"y_pred {y_pred}\n")
                # logging.info('-'*80)
                    
                #log the model's performances
                logger.log_metrics({f'accuracy':round(accuracy_score(y_test, y_pred),3)})
                logging.info({f'accuracy':round(accuracy_score(y_test, y_pred),3)})
                logging.info('-'*80)
                
                    #saving the model
                file_name = model_type.__name__ + '_model.sav'
                pickle.dump(model, open('models/'+ file_name, 'wb'))
                
                return round(accuracy_score(y_test, y_pred),3)
            
            except Exception as e:
                logging.info(f"Indside Model_run exception!! {e}")
                CustomException(e, sys)

if __name__ == "__main__":
    
     models = ['RandomForestClassifier',
               'DecisionTreeClassifier',
               'LogisticRegression',
               'SGDClassifier'
            ]
    
     obj = ModelTrainer()
     accuracy = obj.model_run(SGDClassifier)
    #  result = [obj.model_run(model_name) for model_name in models]
     logging.info(f'Accuracy: {accuracy}')
    
    #  result = []
    #  for x in models:
    #      result.append(obj.model_run(x))
    #      logging.info(f'Result: {result}')
     
    
#running an experiment
# model_run(RandomForestClassifier)


