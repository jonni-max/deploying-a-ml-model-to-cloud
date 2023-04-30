"""
Put the code for your API here.

"""

# from fastapi import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel, Field
# from typing import Union
# from joblib import load
from builtins import str, int

# import json
# import numpy  as np
import pandas as pd

from model.ml_pipeline import model_prediction

# from model.ml_pipeline import model_prediction

# from pandas._libs.tslibs.timezones import num
# from celery.bin.control import status

# Data model for the model input data
# class XData(BaseModel):
#     input_data: pd.DataFrame = None
#
#     class Config:
#         arbitrary_types_allowed = True


class XData(BaseModel):
    age: str
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 76,
                "workclass": "private",
                "fnlgt": 124191,
                "education": "Masters",
                "education-num": 14,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


# Entry point to the API server
app = FastAPI()


# A GET to print 'Hello World' on the browser
@app.get("/")
async def print_welcome():
    return {"Welcome to the udacity-mldevops project 3 " +
            "salary prediction REST API!"}


# A POST to do inference with our trained model
@app.post("/predict/")
async def model_predict(input: XData):
    # Check input conformity, eg. is empty?
    #      if len(input) == 0:
    #          raise HTTPException(status_code=401, detail="X data is empty")

    # Convert XData to data frame with column naming as in training data
    input_df = pd.DataFrame(input.dict(), index=[0])
    input_df.columns = [cn.replace('_', '-') for cn in input_df.columns.values]

#     print(f"\ninput_df\n{input_df}")

    # Call function for model inference based on trained model
    prediction = model_prediction(input_df)

    return {"prediction": prediction.tolist()}
