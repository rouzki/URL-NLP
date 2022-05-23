## depoloiement dependcies
import os
import numpy as np
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

## model dependencies
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer


### for cleaning
import re
from setuptools.namespaces import flatten
from urllib.parse import urlparse, unquote_plus



MODEL_DIR = "models"

MODEL_NAME = "model_finetuned.h5"
MLB_NAME = "mlb.pickle"
TOKENIZER_NAME = "camembert-base"

model = torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location='cpu')
model.eval()

MultiLabelBinarizer_ = joblib.load(os.path.join(MODEL_DIR, MLB_NAME))

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME, map_location=torch.device("cpu")
)


def preprocess_url(url):
    ## convert to urlparse with quoted
    url_parsed = urlparse(unquote_plus(url))
    ## join all url attributes
    url_text = ''.join(x for x in [url_parsed.netloc, url_parsed.path, url_parsed.params, url_parsed.query])
    ## split url to tokens ie: words
    tokens = re.split('[- _ % : , / \. \+ = ]', url_text)
    ## spliting by upper case
    tokens = list(flatten([re.split(r'(?<![A-Z\W])(?=[A-Z])', s) for s in tokens]))
    ## delete token with digits with len < 2
    tokens = [token for token in tokens if (not any(c.isdigit() for c in token)) and (not len(token) <=2)]
    tokens = [token for token in tokens if token not in ['www', 'html', 'com', 'net', 'org']]
    return ' '.join(token for token in tokens)



app = FastAPI(title="URL Classification - Streamlit APP", description="API to predict class of urls")

class Data(BaseModel):
    urls: List[str] = []

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict")
def predict(data:Data):


    cleaned_urls = [preprocess_url(url) for url in data.urls]

    inputs = tokenizer(
        cleaned_urls,
        truncation=True,
        add_special_tokens=True,
        max_length=40,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    print(inputs['input_ids'].shape)
    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    pred_probs = torch.sigmoid(out).detach().numpy()
    pred_bools = np.where(pred_probs > 0.5, 1, 0)

    return {
        
        "urls" : data.urls,
        "predictions": MultiLabelBinarizer_.inverse_transform(pred_bools)

    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)