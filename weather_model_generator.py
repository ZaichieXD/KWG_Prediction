from keras.api.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import load_model


class WeatherPredictionModel:
    # initialize the variables
    def __init__(self):
        self.scaler = None
        # variables for testing
        self.act_temp = None
        self.actual_data = None
        self.temp_date = None
        self.predict_df = None


    def load_training_data(self):
        """
            Loads the default training data, use this for presentation
        :return:
        """
        # Gets the data from the folder
        weather_data_directory = "Weather_Data/"

        # And convert it to a pandas dataframe
        weather_csv = [file for file in os.listdir(weather_data_directory) if file.endswith('.csv')]
        weather_df = pd.concat(
            (pd.read_csv(os.path.join(weather_data_directory, file)) for file in weather_csv),
            ignore_index=True
        )

        # Fix the time of the dataset only get the date and don't include time
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.date

        # rename the column from datetime to date
        weather_df = weather_df.rename(columns={"datetime": "date"})

        # Drop useless columns that won't help with prediction
        weather_df = weather_df.drop(
            ["visibility", "coord.lon", "coord.lat", "sys.sunset", "weather.id", "weather.icon", "extraction_date_time",
             "sys.type", "sys.id", "rain.1h", "clouds.all", "wind.gust", "main.grnd_level", "main.sea_level",
             "weather.main", "sys.sunrise", "weather.description"], axis=1)

        # filters the data to only Dagupan
        dagupan_weather_df = weather_df[weather_df['city_name'] == 'Dagupan'].reset_index(drop=True)

        # fix date again to make sure its a datetime dtype
        dagupan_weather_df['date'] = pd.to_datetime(dagupan_weather_df['date'])

        # Drop duplicates, N/A and the rest of the useless columns
        dagupan_weather_df = dagupan_weather_df.drop_duplicates(subset='date').reset_index(drop=True)
        dagupan_weather_df = dagupan_weather_df.dropna().reset_index(drop=True)
        dagupan_weather_df = dagupan_weather_df.drop(["main.feels_like", "main.temp_min", "main.temp_max", "city_name"],
                                                     axis=1)

        # Renames the main values that will be used by the model
        dagupan_weather_df = dagupan_weather_df.rename(
            {"main.temp": "temperature", "main.pressure": "pressure", "main.humidity": "humidity",
             "wind.speed": "wind_speed", "wind.deg": "wind_direction"}, axis=1)

        return dagupan_weather_df

    def load_new_prediction_data(self, path):
        """
            For loading new data for prediction
        :param path: Path to the csv
        :return:
        """
        new_prediction_set = pd.read_csv(path)

        # check if the required columns are in the dataset
        required_columns = ['date', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']

        missing_columns = [col for col in required_columns if col not in new_prediction_set.columns]

        if missing_columns:
            # if there are missing columns exit
            print(f"Missing columns: {', '.join(missing_columns)}")
            exit(0)
        else:
            # If all required columns are present continue
            print("All required columns are present")
            if new_prediction_set['date'].dtype != 'datetime64[ns]':
                new_prediction_set['datetime'] = pd.to_datetime(new_prediction_set['datetime']).dt.date
            return new_prediction_set

    def process_data(self, input_data):
        """
           Get the Difference of the dataset and transform the data to a scaled version
        :param input_data:
        :return:
        """

        # Assigns actual data to values
        self.act_temp = input_data['temperature'][-13:].to_list()
        self.actual_data = input_data.drop(['pressure', 'humidity', 'wind_speed', 'wind_direction'], axis=1)
        self.actual_data = self.actual_data[-13:]
        self.actual_data = self.actual_data.reset_index(drop=True)
        self.temp_date = input_data['date'][-12:].reset_index(drop=True)
        self.predict_df = pd.DataFrame(self.temp_date)

        # Gets the difference of columns in the dataset
        input_data['temperature_diff'] = input_data['temperature'].diff()
        input_data['pressure_diff'] = input_data['pressure'].diff()
        input_data['humidity_diff'] = input_data['humidity'].diff()
        input_data['wind_speed_diff'] = input_data['wind_speed'].diff()
        input_data['wind_direction_x'] = np.sin(np.radians(input_data['wind_direction']))
        input_data['wind_direction_y'] = np.cos(np.radians(input_data['wind_direction']))

        # Get Diff
        input_data['wind_direction_x_diff'] = input_data['wind_direction_x'].diff()
        input_data['wind_direction_y_diff'] = input_data['wind_direction_y'].diff()
        refined_dg_df = input_data.drop(['temperature',	'pressure',	'humidity',	'wind_speed', 'wind_direction', 'wind_direction_x', 'wind_direction_y'], axis=1)
        refined_dg_df = refined_dg_df.dropna().reset_index(drop=True)
        for i in range(1, 25):
            # Shifts the data to add lag values
            refined_dg_df[f'temperature_lag_{i}'] = refined_dg_df['temperature_diff'].shift(i)
            refined_dg_df[f'pressure_lag_{i}'] = refined_dg_df['pressure_diff'].shift(i)
            refined_dg_df[f'humidity_lag_{i}'] = refined_dg_df['humidity_diff'].shift(i)
            refined_dg_df[f'wind_speed_lag_{i}'] = refined_dg_df['wind_speed_diff'].shift(i)
            refined_dg_df[f'wind_direction_x_lag_{i}'] = refined_dg_df['wind_direction_x_diff'].shift(i)
            refined_dg_df[f'wind_direction_y_lag_{i}'] = refined_dg_df['wind_direction_y_diff'].shift(i)

        # drop all N/A and reset the index
        refined_dg_df = refined_dg_df.dropna().reset_index(drop=True)

        # drop the diff after the shift
        refined_dg_df = refined_dg_df.drop(
            ['pressure_diff', 'humidity_diff', 'wind_speed_diff', 'wind_direction_x_diff', 'wind_direction_y_diff',
             'date'], axis=1)

        return refined_dg_df

    def split_data(self, dataset):
        """
            Splits the data to train, validation, and test
        :param dataset:
        :return:
        """
        train_data = dataset[:-24] # 80% of the data
        val_data = dataset[-24:-12] # 10% of the data
        test_data = dataset[-12:] # 10% of the data

        # Scales the values to prevent features with high values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        val_data = self.scaler.transform(val_data)
        test_data = self.scaler.transform(test_data)

        # splits the data to x and y
        x_train, y_train = train_data[:, 1:], train_data[:, 0]
        x_val, y_val = val_data[:, 1:], val_data[:, 0]
        x_test, y_test = test_data[:, 1:], test_data[:, 0]

        # Reshape the data to match the model requirements
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Creates a dictionary to distribute the data
        dataset_dict = dict()
        dataset_dict['x_train'] = x_train
        dataset_dict['y_train'] = y_train
        dataset_dict['x_val'] = x_val
        dataset_dict['y_val'] = y_val
        dataset_dict['x_test'] = x_test
        dataset_dict['y_test'] = y_test

        return dataset_dict

    def train_model(self, input_features):
        """
            Trains the model and generates a Model.keras file for later use
        :param input_features: the input parameters to train the model with
        :return:
        """
        # Define the model
        model = Sequential([
            LSTM(128, activation='tanh', input_shape=(input_features['x_train'].shape[1], input_features['x_train'].shape[2])),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        # Assigns a variable to get the history model data
        model_history = model.fit(input_features['x_train'], input_features['y_train'], validation_data=(input_features['x_val'], input_features['y_val']), epochs=100,
                                callbacks=[early_stopping])

        model.save('weather_prediction_model.keras')

        return model_history

    def model_predict(self, model_name: str, dataset):
        """
            Predict values using the saved model
        :param model_name: The name of the model file
        :param dataset: The input features the model is going to use
        :return:
        """

        weather_model = load_model(model_name)

        # Gets only the test set from the dataset
        x_test = dataset['x_test']

        # assigns the prediction to a variable
        predictions_test = weather_model.predict(x_test)

        # Reshape the model to its original form
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
        predictions_test = predictions_test.reshape(x_test.shape[0], -1)

        # Removes the scaling
        lr_pre_test_set = np.concatenate([predictions_test, x_test], axis=1)
        lr_pre_test_set = self.scaler.inverse_transform(lr_pre_test_set)

        results = []

        # Adds the prediction to the last known actual value
        for index in range(0, len(lr_pre_test_set)):
            results.append(lr_pre_test_set[0][index] + self.act_temp[index])

        # Combines the dates and the prediction into one dataframe
        temperature_data = pd.Series(results, name="Temperature Prediction")
        self.predict_df = self.predict_df.merge(temperature_data, left_index=True, right_index=True)

    def plot_history(self, model_history):
        """
         Plot the history and get the loss to see if the prediction is accurate
        :param model_history:
        :return:
        """
        # Extract values
        loss = model_history.history.history['loss']
        val_loss = model_history.history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_results(self):
        """
            Plot the results
        :return:
        """
        plt.figure(figsize=(16, 5))
        plt.plot(self.actual_data['date'], self.actual_data['temperature'], label='Actual Temperature')
        plt.plot(self.predict_df['date'], self.predict_df['Temperature Prediction'], label='Temperature Prediction')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Actual Temperature')
        plt.title('Temperature Prediction')
        plt.show()
