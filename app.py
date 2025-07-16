import sys
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from uvicorn import run as app_run
from fastapi.responses import FileResponse

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipelines.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Initialize FastAPI app
app = FastAPI()
origins = ["*"]

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates and static files
templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["home"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training completed successfully.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_batch(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Predict
        y_pred = network_model.predict(df)

        # Convert predictions to int (in case they are float) and map to readable labels
        df["predicted_column"] = pd.Series(y_pred).astype(int).map({1: "Phishing", 0: "Legitimate"})
        os.makedirs("prediction_output", exist_ok=True)

        # Save and display
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict-single")
async def predict_single(request: Request, features: str = Form(...)):
    try:
        feature_values = list(map(float, features.strip().split(",")))
        df = pd.DataFrame([feature_values])

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        prediction = int(y_pred[0])

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "features": features, "prediction": prediction}
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

@app.get("/download")
async def download_result():
    output_path = "prediction_output/output.csv"
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="text/csv", filename="prediction_result.csv")
    return Response("No result available to download.", status_code=404)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
