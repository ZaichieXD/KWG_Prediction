from fastapi import FastAPI

import kwago_model_generator as kwago_model_gen
from kwago_model_prediction import KwagoPredictionModel

app = FastAPI()
kwago_model = KwagoPredictionModel()

@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/kwago_predict")
def kwago_predict():
    kwago_data = kwago_model.run_predictions('2024-')
    kwago_data['date'].dt.strftime("%Y-%m-%d")
    json_data = kwago_data.to_dict(orient="records")
    return json_data




