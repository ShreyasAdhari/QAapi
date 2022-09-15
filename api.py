from fastapi import FastAPI, Depends
from pydantic import BaseModel
from .answer import get_model
from typing import List

class ModelInput( BaseModel ):
    text: str

class ModelOutput( BaseModel ):
    ans: List

app = FastAPI()

@app.post('/predict', response_model=ModelOutput)
def predict( input_text:ModelInput, model=Depends( get_model ) ):

    ans = model.ans( input_text.text )

    return ModelOutput(
        ans = ans,
    )