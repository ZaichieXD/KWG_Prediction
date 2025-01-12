import calendar
from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras.api.models import load_model

class KuwagoPredictionModel:
    def update_predicitons(self):
        # Import all the data
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
        order_details_df.rename(columns={'pcs': 'total_orders'}, inplace=True)
        order_details_df.rename(columns={'price': 'total_sales'}, inplace=True)

        # Changes object datatype to datetime
        reports_df['date'] = pd.to_datetime(reports_df['date'])
        order_details_df['date'] = pd.to_datetime(order_details_df['date'])

        # Add the values of each day for every month to match the reports_df (e.g all values in 2022-11-20 will be added)
        sales_df = order_details_df.groupby(order_details_df['date'].dt.date).agg({
            'total_orders': 'sum',
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

        # Prepare Dataset for training
        # Prepare Date for extra features

        # Transfer main_df data to a new variable to keep original values
        main_df_data = main_df

        main_df_data['day_of_the_week'] = main_df_data['date'].dt.dayofweek
        main_df_data['week_of_year'] = main_df_data['date'].dt.isocalendar().week
        main_df_data['month'] = main_df_data['date'].dt.month
        main_df_data['is_weekend'] = main_df_data['day_of_the_week'].isin([5, 6]).astype(int)

        # Fix missing dates
        main_df_data.set_index('date', inplace=True)
        main_df_data = main_df_data.asfreq('D', fill_value=0)  # fill empty dates with 0
        main_df_data.reset_index(inplace=True)

        # Apply Rolling windows on the data
        main_df_data['rolling_orders_7'] = main_df_data['total_orders'].rolling(window=7).mean()
        main_df_data['rolling_sales_7'] = main_df_data['total_sales'].rolling(window=7).mean()
        main_df_data['rolling_profit_7'] = main_df_data['total_profit'].rolling(window=7).mean()
        main_df_data['rolling_expenses_7'] = main_df_data['total_expenses'].rolling(window=7).mean()

        # Apply shifts on the data
        columns_to_shift = ['total_orders', 'total_sales', 'total_profit', 'total_expenses']
        for i in range(1, 4):
            for col in columns_to_shift:
                main_df_data[f'{col}_{i}'] = main_df_data[col].shift(i)

        prediction_dates = pd.DataFrame(main_df_data['date'][-80:].reset_index(drop=True))

        actual_data = main_df_data[['date', 'total_orders', 'total_sales', 'total_profit', 'total_expenses']]
        actual_data = actual_data[-80:].reset_index(drop=True)

        # Drop the uneccesarry columns
        main_df_data = main_df_data.drop(['date'], axis=1)
        main_df_data = main_df_data.dropna().reset_index(drop=True)

        # Split the data into train, validation, and test sets
        train_data = main_df_data[:-160]
        val_data = main_df_data[-160:-80]
        test_data = main_df_data[-80:]

        # Check the split
        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        scaler = RobustScaler()

        train_scaled = scaler.fit_transform(train_data)

        # Transform the validation and test data (use the same scaler)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)

        # Function to create sequences
        def create_sequences(data, target_columns, time_steps):
            x, y = [], []
            for index in range(len(data) - time_steps):
                # Features: all columns are scaled
                x.append(data[index:index + time_steps])  # All input columns are scaled
                # Targets: only the target columns, already scaled
                y.append(data[index + time_steps, target_columns])  # Targets in their scaled form
            return np.array(x), np.array(y)

        # Example parameters
        time_steps = 7
        target_columns = [0, 1, 2, 3]  # Adjust this based on your actual target columns

        # Generate sequences
        x_train, y_train = create_sequences(train_scaled, target_columns, time_steps)
        x_val, y_val = create_sequences(val_scaled, target_columns, time_steps)
        x_test, y_test = create_sequences(test_scaled, target_columns, time_steps)

        # Print shapes for verification
        print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
        print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

        # Load the trained model
        model = load_model('kuwago_prediction_model.keras')

        # Future steps match the time_steps
        future_steps = time_steps

        # Start with the last sequence from the test set
        last_sequence = x_test[-1]  # Shape: (time_steps, features)

        # Placeholder for future predictions
        future_predictions = []
        last_date = None
        current_date = None
        target_stop_date = None

        # Check if the file exists
        file_exists = os.path.isfile('future_predictions.csv')

        try:
            # Try loading the saved predictions file
            future_data = pd.read_csv('future_predictions.csv')
            last_date = pd.to_datetime(future_data['date'].iloc[-1])
            current_date = pd.to_datetime(future_data['date'].iloc[-1])  # Get the last date
        except FileNotFoundError:
            print("File not found")
            # Initialize the current date and iteration counter
            current_date = pd.to_datetime(actual_data['date'].iloc[-1])  # Get the last date
            last_date = pd.to_datetime(actual_data['date'].iloc[-1])

        # Function to check if a date is in or past the middle of the month
        def is_middle_of_month(date):
            """
            Check if a given date is at or after the middle of the month.
            """
            month = date.month
            year = date.year

            # Get the number of days in the month
            days_in_month = pd.Period(f"{year}-{month}").days_in_month

            # Calculate the middle of the month
            middle_day = days_in_month // 2

            # Return True if the date is on or after the middle day
            return date.day >= middle_day

        def get_end_of_next_month(date):
            """ Get the last day of the next month from the given date dynamically """
            # Add one month (next month)
            if date.month == 12:  # Handle December case (next year)
                next_month = date.replace(year=date.year + 1, month=1, day=1)
            else:
                next_month = date.replace(month=date.month + 1, day=1)

            # Get the last day of the next month using calendar
            # This will get the number of days in the next month
            last_day_of_next_month = calendar.monthrange(next_month.year, next_month.month)[1]

            # Return the last day of the next month
            return next_month.replace(day=last_day_of_next_month)

        # Reference today's date
        current_date_today = datetime.today()

        if is_middle_of_month(current_date_today):
            # Iteratively predict future steps, stop based on condition
            target_stop_date = get_end_of_next_month(current_date_today)

            while True:  # Or use condition on prediction values
                if current_date >= target_stop_date:
                    break

                # Reshape to match model input
                last_sequence_reshaped = np.expand_dims(last_sequence, axis=0)

                # Predict the next step
                next_prediction = model.predict(last_sequence_reshaped, verbose=0)

                # Append the prediction
                future_predictions.append(next_prediction[0])

                # Update the sequence for the next iteration
                new_row = last_sequence[-1].copy()
                new_row[target_columns] = next_prediction[0] + np.random.normal(0, 0.01, size=next_prediction[0].shape)
                last_sequence = np.vstack([last_sequence[1:], new_row])

                # Increment the date
                current_date += pd.Timedelta(days=1)

            # Convert future predictions to a NumPy array
            future_predictions = np.array(future_predictions)

            # Handle empty or mismatched future_predictions
            if future_predictions.size == 0:
                print("Predictions is already updated")
            else:
                # Reverse scaling to get original scale
                inverse_scaling_input = np.zeros((len(future_predictions), last_sequence.shape[1]))
                inverse_scaling_input[:, :-len(target_columns)] = last_sequence[-1, :-len(target_columns)]
                inverse_scaling_input[:, -len(target_columns):] = future_predictions

                # Reverse scaling
                future_predictions_original_scale = scaler.inverse_transform(inverse_scaling_input)
                future_predictions_original_scale = future_predictions_original_scale[:, -len(target_columns):]

                # Generate future dates
                future_dates = pd.date_range(start=last_date, periods=len(future_predictions) + 1, freq='D')[1:]

                # Create DataFrame for future predictions
                future_series = pd.DataFrame({
                    'date': future_dates,
                    'Total Orders Prediction': future_predictions_original_scale[:, 0],
                    'Total Sales Prediction': future_predictions_original_scale[:, 1],
                    'Total Expenses Prediction': future_predictions_original_scale[:, 2],
                    'Total Profit Prediction': future_predictions_original_scale[:, 3]
                })

                # Append to the existing CSV file instead of overwriting
                future_series.to_csv('future_predictions.csv', mode='a', header=not file_exists, index=False)
        else:
            print('There is an error')