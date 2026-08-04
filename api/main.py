import os

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

RUTA_MODELO = os.environ.get("MODEL_PATH", "model_champion")

app = FastAPI(title="Netflix Rating Classifier API")

# Kubernetes setea HOSTNAME automaticamente al nombre del pod
POD_NAME = os.environ.get("HOSTNAME", "local")

modelo = mlflow.pyfunc.load_model(RUTA_MODELO)

with open(os.path.join(RUTA_MODELO, "registered_model_meta")) as f:
    metadata_modelo = yaml.safe_load(f)


class PrediccionInput(BaseModel):
    type: str
    release_year: int
    listed_in: str
    country: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "pod": POD_NAME,
        "modelo": metadata_modelo["model_name"],
        "version": metadata_modelo["model_version"],
        "alias": "champion",
    }


@app.post("/predict")
def predict(entrada: PrediccionInput):
    df = pd.DataFrame([entrada.model_dump()])
    prediccion = modelo.predict(df)
    return {"rating_predicho": prediccion[0], "pod": POD_NAME}
