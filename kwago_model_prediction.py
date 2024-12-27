import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from keras.api.models import Sequential
from keras.api.layers import SimpleRNN, Dense, Dropout, LSTM, Bidirectional
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.models import load_model

class KwagoPredictionModel:

    def run_predictions(self, target_date):

        # Get reports
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

        # gets the difference in each columns
        columns_to_diff = ['number_of_sales', 'total_sales', 'total_expenses', 'total_profit']
        for col in columns_to_diff:
            main_df[col + '_diff'] = main_df[col].diff()
        main_df.dropna(inplace=True)

        actual_data = main_df.drop(
            ['number_of_sales_diff', 'total_sales_diff', 'total_expenses_diff', 'total_profit_diff'],
            axis=1)
        actual_data = actual_data[-13:]
        actual_data = actual_data.reset_index(drop=True)

        # shift it according to the diff
        main_df_data = main_df.drop(
            ['date', 'number_of_sales', 'total_sales', 'total_expenses', 'total_profit'],
            axis=1)

        columns_to_shift = ['number_of_sales_diff', 'total_sales_diff', 'total_expenses_diff', 'total_profit_diff']
        for col in columns_to_shift:
            for i in range(1, 13):
                main_df_data[col + '_shift_' + str(i)] = main_df_data[col].shift(i)

        main_df_data.dropna(inplace=True)  # drops the n/a

        input_features = main_df_data[-12:]  # 10% testing

        # Scales the values to ensure a balanced dataset
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(main_df_data)
        scaled_input = scaler.transform(input_features)

        # Splits the data to x and y
        splitted_input = scaled_input[:, 4:]

        # Reshape the input features (x) to match the model requirements
        splitted_input.reshape((splitted_input.shape[0], 1, splitted_input.shape[1]))

        # prepares the dates for the actual values
        sales_dates = main_df['date'][-12:].reset_index(drop=True)

        kuwago_prediction_model = load_model('kwago_prediction_model.keras')

        target_date = pd.to_datetime(target_date)
        last_known_date = sales_dates.iloc[-1]

        # Generate additional dates to predict up to the target date
        new_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), end=target_date, freq='D')
        new_dates = pd.DataFrame(new_dates, columns=['date'])

        last_known_data = splitted_input[-1:]

        current_input = last_known_data.copy()
        predictions = []

        for _ in range(len(new_dates)):
            model_input = np.expand_dims(current_input, axis=0)

            next_step = kuwago_prediction_model.predict(model_input)
            next_step = next_step[0]

            predictions.append(next_step)

            updated_steps = current_input[-1].copy()
            updated_steps[-4:] = next_step

            current_input = np.append(current_input[1:], [updated_steps], axis=0)

        predictions = np.array(predictions)
        n_shape, num_features = predictions.shape

        # Pad predictions to match original shape
        padded_predictions = np.zeros((n_shape, 52))
        padded_predictions[:, -num_features:] = predictions  # Place predictions in the correct columns

        # Inverse transform the padded data
        rescaled_padded = scaler.inverse_transform(padded_predictions)

        # Extract only the rescaled predictions (last 4 columns)
        rescaled_predictions = rescaled_padded[:, -num_features:]

        prediction_df = pd.DataFrame(rescaled_predictions, columns=['number_of_sales', 'total_sales',
                                                                  'total_expenses', 'total_profit'])
        prediction_df['date'] = new_dates['date'].values

        print(prediction_df)
        return prediction_df

kwago = KwagoPredictionModel()

kwago.run_predictions('2024-12-30')