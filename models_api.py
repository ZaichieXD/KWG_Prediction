from fastapi import FastAPI

import weather_model_generator as weather_model_gen
import kwago_model_generator as kwago_model_gen

app = FastAPI()
weather_model = weather_model_gen.WeatherPredictionModel()
kwago_model = kwago_model_gen.KwagoPredictionModel()

weather_training_data = weather_model.load_training_data()
weather_final_dataset = weather_model.process_data(weather_training_data)
weather_input_features = weather_model.split_data(weather_final_dataset)
weather_model.model_predict('weather_prediction_model.keras', weather_input_features)

kwago_training_data = kwago_model.load_training_data()
kwago_final_dataset = kwago_model.process_dataset(kwago_training_data)
kwago_input_features = kwago_model.prepare_dataset(kwago_final_dataset, kwago_training_data)
kwago_model.model_predict('kwago_prediction_model.keras', kwago_input_features)

@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/weather_predict")
def weather_predict():
    weather_data = weather_model.predict_df
    weather_data['date'].dt.strftime("%Y-%m-%d")
    json_data = weather_data.to_dict(orient="records")
    return json_data

@app.get("/kwago_predict")
def kwago_predict():
    kwago_data = kwago_model.predict_df
    kwago_data['date'].dt.strftime("%Y-%m-%d")
    json_data = kwago_data.to_dict(orient="records")
    return json_data

