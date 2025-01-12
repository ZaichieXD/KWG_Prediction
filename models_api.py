from fastapi import FastAPI
import pandas as pd
from kwago_model_prediction import KuwagoPredictionModel

app = FastAPI()
kuwago_model = KuwagoPredictionModel()

@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/kwago_update")
def kuwago_update():
    kuwago_model.update_predicitons()
    return {'update-predictions': 'complete'}

@app.get("/kwago_predict")
def kuwago_predict():
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv('future_predictions.csv')
    # Convert the DataFrame to a dictionary
    json_data = df.to_dict(orient='records')
    return json_data




