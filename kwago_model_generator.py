import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import SimpleRNN, Dense, Dropout, LSTM, Bidirectional
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.models import load_model

class KwagoPredictionModel:

    def __init__(self):
        self.sales_dates = None
        self.act_sales = None
        self.predict_df = None
        self.scaler = None
        self.actual_data = None


    def load_training_data(self):
        """
            Loads the dataset that is required for this project, clean it and prepare it for merging and processing
        :return:
        """
        # Load Reports and read the CSV using pandas
        reports_df = pd.read_csv('kwago_csv/reports.csv')

        # Drops unwanted columns
        reports_df = reports_df.drop(
            ['id', 'gcash', 'cash', 'total_discount', 'total_sales', 'remitted', 'total_unpaid', 'late_payments',
             'total_remittance', 'spoilages', 'unpaid', 'late', 'purchases', 'created_at', 'updated_at'], axis=1)

        # Renames the column to total_expenses to prevent confusions
        reports_df.rename(columns={'total_purchases': 'total_expenses'}, inplace=True)

        # Load Order Details and read CSV using pandas
        order_details_df = pd.read_csv('kwago_csv/order_details.csv')

        # Drops unwanted columns
        order_details_df = order_details_df.drop(
            ['id', 'order_id', 'price_per_piece', 'dish_id', 'note', 'discount_type', 'discount', 'orig_price',
             'printed',
             'deleted_at', 'updated_at'], axis=1)

        # Renames the column to prevent confusion
        order_details_df.rename(columns={'created_at': 'date'}, inplace=True)
        order_details_df.rename(columns={'pcs': 'number_of_sales'}, inplace=True)
        order_details_df.rename(columns={'price': 'total_sales'}, inplace=True)

        # Changes object datatype to datetime
        reports_df['date'] = pd.to_datetime(reports_df['date'])
        order_details_df['date'] = pd.to_datetime(order_details_df['date'])

        # Add the values of each day for every month to match the reports_df (e.g all values in 2022-11-20 will be added)
        sales_df = order_details_df.groupby(order_details_df['date'].dt.date).agg({
            'number_of_sales': 'sum',
            'total_sales': 'sum'
        }).reset_index()  # Reset index to prevent overlapping index

        # Fixes date again to date in not an object datatype
        sales_df['date'] = pd.to_datetime(sales_df['date'])

        # Performs inner merge removing overlapping dates and combining it through the date column
        main_df = pd.merge(sales_df, reports_df, on='date', how='inner')

        # Fill all N/A with 0.0 to prevent problems during training
        main_df['total_expenses'] = main_df['total_expenses'].fillna(0.0)

        # Subtracts total_profit from total_sales to get the total_expenses (just a workaround because there is no expenses in database)
        main_df['total_profit'] = main_df['total_sales'] - main_df['total_expenses']

        # Drop duplicates and resets index
        main_df = main_df.drop_duplicates(subset=['date'])
        main_df.reset_index(drop=True)

        return main_df

    def load_new_prediction_data(self, path):
        new_prediction_set = pd.read_csv(path)
        required_columns = ['date', 'number_of_sales', 'total_sales', 'total_expenses', 'total_profit']

        missing_columns = [col for col in required_columns if col not in new_prediction_set.columns]

        if missing_columns:
            print(f"Missing columns: {', '.join(missing_columns)}")
            exit(0)
        else:
            # If all required columns are present continue
            print("All required columns are present")
            return new_prediction_set

    def process_dataset(self, sales_dataset):
        """
            Process dataset to get ready for training
        :return:
        """

        # gets the difference in each columns
        columns_to_diff = ['number_of_sales', 'total_sales', 'total_expenses', 'total_profit']
        for col in columns_to_diff:
            sales_dataset[col + '_diff'] = sales_dataset[col].diff()
        sales_dataset.dropna(inplace=True)

        self.actual_data = sales_dataset.drop(
            ['number_of_sales_diff', 'total_sales_diff', 'total_expenses_diff', 'total_profit_diff'],
            axis=1)
        self.actual_data = self.actual_data[-13:]
        self.actual_data = self.actual_data.reset_index(drop=True)

        # shift it according to the diff
        main_df_data = sales_dataset.drop(['date', 'number_of_sales', 'total_sales', 'total_expenses', 'total_profit'],
                                    axis=1)

        columns_to_shift = ['number_of_sales_diff', 'total_sales_diff', 'total_expenses_diff', 'total_profit_diff']
        for col in columns_to_shift:
            for i in range(1, 13):
                main_df_data[col + '_shift_' + str(i)] = main_df_data[col].shift(i)

        main_df_data.dropna(inplace=True) # drops the n/a

        return main_df_data

    def prepare_dataset(self, processed_dataset, original_dataset):
        """
        Split the data into training, validation and test and use a scaler to flatten the data
        :return:
        """
        # Separates the data to 80, 10, 10
        train_data = processed_dataset[:-24] # 80% training
        val_data = processed_dataset[-24:-12] # 10% validation
        test_data = processed_dataset[-12:]# 10% testing

        # Scales the values to ensure a balanced dataset
        self.scaler = MinMaxScaler(feature_range=(0, 0.16))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        val_data = self.scaler.transform(val_data)
        test_data = self.scaler.transform(test_data)

        # Splits the data to x and y
        x_train, y_train = train_data[:, 4:], train_data[:, 0:4]
        x_val, y_val = val_data[:, 4:], val_data[:, 0:4]
        x_test, y_test = test_data[:, 4:], test_data[:, 0:4]

        # prepares the dates for the actual values
        self.sales_dates = original_dataset['date'][-12:].reset_index(drop=True)
        self.predict_df = pd.DataFrame(self.sales_dates)

        # Reshape the input features (x) to match the model requirements
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        # Gets the actual sales at the very last of the dataset
        self.act_sales = [original_dataset['number_of_sales'][-13:].to_list(),
                          original_dataset['total_sales'][-13:].to_list(),
                          original_dataset['total_expenses'][-13:].to_list(),
                          original_dataset['total_profit'][-13:].to_list()]

        dataset_dict = dict()
        dataset_dict['x_train'] = x_train
        dataset_dict['y_train'] = y_train
        dataset_dict['x_val'] = x_val
        dataset_dict['y_val'] = y_val
        dataset_dict['x_test'] = x_test
        dataset_dict['y_test'] = y_test

        return dataset_dict



    # Train the model using the dataset
    def train_model(self, input_features):
        # Creates a Sequential Model
        kwago_prediction_model = Sequential([
            # Use LSTM on top of Bidirectional to learn dependencies from both past and future context of the data
            Bidirectional(LSTM(256, activation='tanh', return_sequences=True),
                          input_shape=(input_features['x_train'].shape[1], input_features['x_train'].shape[2])),

            # Drop neurons to prevent overfitting
            Dropout(0.3),
            # Adds another LSTM in the hidden layer, enables the model to learn more complex pattern
            LSTM(128, activation='tanh'),
            # Allows the model to learn data that is non-linear
            Dense(64, activation='relu'),
            Dropout(0.2),
            # Same but decreases the units by 32 to distill the important features for the next layers
            Dense(32, activation='relu'),
            Dropout(0.1),
            # Generates 4 output features the one specified in the y_train
            Dense(4, activation='linear')
        ])

        # Compiles the model using the adam optimizer, mean squared error as the loss function
        # and mean absolute error as the metrics
        # this is for calculation of the loss or how 'off' the model is from the dataset
        kwago_prediction_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Prevent overfitting by stopping the training process if the training is enough
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=13,
            restore_best_weights=True
        )

        # Prevent overfitting by scheduling and adjustment to the learning rate
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Runs the model with every thing in the top combined and use 100 epochs (How many times the data is feed to the model)
        model_history = kwago_prediction_model.fit(input_features['x_train'], input_features['y_train'], epochs=100, validation_data=(input_features['x_val'], input_features['y_val']),
                  callbacks=[early_stopping, lr_scheduler])

        kwago_prediction_model.save('kwago_prediction_model.keras')

        return model_history

    def plot_history(self, model_history):
        """
            Plot the loss function of the fitted model
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

    def model_predict(self, model_name: str, dataset):
        """
            Uses the saved model to make predictions
        :param model_name: name or the path of the model
        :param dataset: the input features the model is going to use
        :return:
        """
        kwago_prediction_model = load_model(model_name)

        x_test = dataset['x_test']

        # Get predictions for test data
        predictions = kwago_prediction_model.predict(x_test)

        x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
        predictions = predictions.reshape(x_test.shape[0], -1)

        lr_pre_test_set = np.concatenate([predictions, x_test], axis=1)
        lr_pre_test_set = self.scaler.inverse_transform(lr_pre_test_set)

        number_of_sales_result = []
        total_sales_result = []
        total_expenses_result = []
        total_profit_result = []

        for index in range(0, len(lr_pre_test_set)):
            number_of_sales_result.append(lr_pre_test_set[0][index] + self.act_sales[0][index])
            total_sales_result.append(lr_pre_test_set[1][index] + self.act_sales[1][index])
            total_expenses_result.append(lr_pre_test_set[2][index] + self.act_sales[2][index])
            total_profit_result.append(lr_pre_test_set[3][index] + self.act_sales[3][index])

        lr_pre_series_number_of_sales = pd.Series(number_of_sales_result, name="Number of Sales Prediction")
        lr_pre_series_total_sales = pd.Series(total_sales_result, name="Total Sales Prediction")
        lr_pre_series_total_expenses = pd.Series(total_expenses_result, name="Total Expenses Prediction")
        lr_pre_series_total_profit = pd.Series(total_profit_result, name="Total Profit Prediction")

        combined_series = pd.concat(
            [lr_pre_series_number_of_sales, lr_pre_series_total_sales, lr_pre_series_total_expenses,
             lr_pre_series_total_profit], axis=1)

        self.predict_df = self.predict_df.merge(combined_series, left_index=True, right_index=True)
        
    def plot_all_predictions(self):
        plt.figure(figsize=(16, 5))
        plt.plot(self.actual_data['date'], self.actual_data['number_of_sales'], label='Actual Number of Sales')
        plt.plot(self.predict_df['date'], self.predict_df['Number of Sales Prediction'], label='Number of Sales Prediction')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Number of Sales')
        plt.title('Number of Sales Prediction')
        plt.show()

        plt.figure(figsize=(16, 5))
        plt.plot(self.actual_data['date'], self.actual_data['total_sales'], label='Actual Sales')
        plt.plot(self.predict_df['date'], self.predict_df['Total Sales Prediction'], label='Total Sales Prediction')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.title('Total Sales Prediction')
        plt.show()

        plt.figure(figsize=(16, 5))
        plt.plot(self.actual_data['date'], self.actual_data['total_expenses'], label='Actual Expenses')
        plt.plot(self.predict_df['date'], self.predict_df['Total Expenses Prediction'], label='Total Expenses Prediction')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Total Expenses')
        plt.title('Total Expenses Prediction')
        plt.show()

        plt.figure(figsize=(16, 5))
        plt.plot(self.actual_data['date'], self.actual_data['total_profit'], label='Actual Profit')
        plt.plot(self.predict_df['date'], self.predict_df['Total Profit Prediction'], label='Total Profit Prediction')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Total Profit')
        plt.title('Total Profit Prediction')
        plt.show()
        

        