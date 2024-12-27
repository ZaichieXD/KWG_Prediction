import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, GRU
from keras.api.callbacks import EarlyStopping, LearningRateScheduler
from keras.api.models import load_model

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
main_df_data = main_df_data.asfreq('D', fill_value=0) # fill empty dates with 0
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

# Drop the uneccesarry columns
main_df_data = main_df_data.drop(['date'], axis=1)
main_df_data = main_df_data.dropna().reset_index(drop=True)

# Define the split ratios
train_size = int(len(main_df_data) * 0.7)  # 70% for training
val_size = int(len(main_df_data) * 0.15)  # 15% for validation
test_size = len(main_df_data) - train_size - val_size  # 15% for testing

# Split the data into train, validation, and test sets
train_data = main_df_data[:train_size]
val_data = main_df_data[train_size:train_size+val_size]
test_data = main_df_data[train_size+val_size:]

# Check the split
print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")

scaler = RobustScaler()

# Fit and transform on the training data (scale all columns)
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

model = load_model('kuwago_prediction_model.keras')

predictions = model.predict(x_test)

print(y_test.shape)
print(predictions.shape)

print(test_scaled.shape)

def reconstruct_original_shape(x_test, y_predictions, original_data, target_columns_ref):
    # Number of time_steps used during sequence creation
    time_steps = original_data.shape[0] - x_test.shape[0]

    # Step 1: Take the first `time_steps` rows from the original data
    missing_rows = original_data[:time_steps]

    # Step 2: Align the predictions with the last row of each sequence in x_test
    reconstructed_rows = []
    for i in range(len(y_predictions)):
        # Take the last row of the sequence
        last_row = x_test[i, -1]  # Shape: (24,)

        # Remove target columns from the last row
        filtered_last_row = np.delete(last_row, target_columns_ref)  # Shape: (20,)

        # Create a full row with predictions inserted at the target column positions
        full_row = np.insert(filtered_last_row, target_columns_ref, y_predictions[i])  # Shape: (24,)

        reconstructed_rows.append(full_row)

    # Convert reconstructed rows to numpy array
    reconstructed_rows = np.array(reconstructed_rows)

    # Step 3: Concatenate the missing rows (original rows) and the reconstructed rows
    full_data = np.vstack([missing_rows, reconstructed_rows])  # Shape: (120, 24)

    return full_data

original_scaled_data = reconstruct_original_shape(x_test, predictions, test_scaled, target_columns)

# Suppress scientific notation for better readability
np.set_printoptions(suppress=True)

# Reverse the scaling transformation
reconstructed_original_scale = scaler.inverse_transform(original_scaled_data)
reversed_y_data = scaler.inverse_transform(test_scaled)

# Print the original scale data
print(reconstructed_original_scale[:, :4])
print(reversed_y_data[:, :4])