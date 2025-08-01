#LSTM Code for Leave-One-Subject-Out (LOSO) Across All 4 Cases
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plts
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Load data (columns: 'SubjectID', 'EMG', 'GRF', 'Knee_Angle', 'ANKLE')
data = pd.read_csv('LSTM_main.csv')

# Define the 4 cases
cases = {
    'Case I': ['EMG', 'GRF'],
    'Case II': ['EMG', 'Knee_Angle'],
    'Case III': ['GRF', 'Knee_Angle'],
    'Case IV': ['EMG', 'GRF', 'Knee_Angle']
}

subjects = data['SubjectID'].unique()

results_summary = []

for case_name, features in cases.items():
    print(f"\n--- Running {case_name} ---")
    rmse_list = []
    r2_list = []

    for test_subject in subjects:
        # Split data into training and testing sets
        train_data = data[data['SubjectID'] != test_subject]
        test_data = data[data['SubjectID'] == test_subject]

        X_train = train_data[features]
        y_train = train_data['ANKLE']
        X_test = test_data[features]
        y_test = test_data['ANKLE']

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reshape for LSTM: (samples, timesteps, features)
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        # Define LSTM model
        model = Sequential([
            LSTM(50, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train
        history = model.fit(X_train_scaled, y_train, 
                            epochs=400, batch_size=32, 
                            validation_split=0.2, 
                            callbacks=[early_stopping], verbose=0)

        # Predict
        y_pred = model.predict(X_test_scaled).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rmse_list.append(rmse)
        r2_list.append(r2)
        print(f"Subject {test_subject}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")

    # Aggregate results
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    mean_r2 = np.mean(r2_list)

    results_summary.append({
        'Case': case_name,
        'Mean RMSE': mean_rmse,
        'Std RMSE': std_rmse,
        'Mean R2': mean_r2
    })

# Convert results to DataFrame and display
summary_df = pd.DataFrame(results_summary)
print("\n--- LOSO Evaluation Summary ---")
print(summary_df)

# Optional: Save results to CSV
summary_df.to_csv('LOSO_LSTM_AnkleAngle_Summary.csv', index=False)
