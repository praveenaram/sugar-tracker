import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import datetime
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import gspread
from oauth2client.service_account import ServiceAccountCredentials


app = Flask(__name__)

# Load SENIOR_SPREADSHEET_MAPPING from environment variables
spreadsheet_mapping_json = os.getenv("SENIOR_SPREADSHEET_MAPPING", "{}")
SENIOR_SPREADSHEET_MAPPING = json.loads(spreadsheet_mapping_json)

# Load Google credentials from environment variables
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


credentials_json = {
    "type": os.environ.get("TYPE"),
    "project_id": os.environ.get("PROJECT_ID"),
    "private_key_id": os.environ.get("PRIVATE_KEY_ID"),
    "private_key": os.environ.get("PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.environ.get("CLIENT_EMAIL"),
    "client_id": os.environ.get("CLIENT_ID"),
    "auth_uri": os.environ.get("AUTH_URI"),
    "token_uri": os.environ.get("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.environ.get("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.environ.get("CLIENT_X509_CERT_URL"),
}

creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_json, scope)
client = gspread.authorize(creds)


# Get data from Google Sheets
def get_sugar_data(spreadsheet_id):
    try:
        sheet = client.open_by_key(spreadsheet_id).worksheet('Main')
        data = sheet.get_all_records()
        if not data:
            raise ValueError("No data found in the Google Sheet.")
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            raise ValueError("Invalid or missing dates in the 'Date' column.")
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Error accessing Google Sheet: {str(e)}")


def get_model_paths(spreadsheet_id):
    model_path = f'saved_models/lstm_model_{spreadsheet_id}.h5'
    scaler_path = f'saved_models/scaler_{spreadsheet_id}.pkl'
    return model_path, scaler_path

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Sugar Level'].values.reshape(-1, 1))

    X, y = [], []
    sequence_length = 60

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    if len(scaled_data) < sequence_length:
        for i in range(1, len(scaled_data)):
            sequence = scaled_data[:i, 0]
            sequence = np.pad(sequence, (sequence_length - len(sequence), 0), 'constant')
            X.append(sequence)
            y.append(scaled_data[i, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X = X.reshape(-1, sequence_length, 1)

    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_or_load_lstm_model(df, spreadsheet_id):
    model_path, scaler_path = get_model_paths(spreadsheet_id)
    X, y, scaler = preprocess_data(df)

    if os.path.exists(model_path):
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        model.save(model_path)

    else:
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=20, batch_size=32, verbose=1)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    return model, scaler

def predict_future_sugar_levels(model, scaler, df, days_to_predict=30):
    if len(df) < 60:  # Threshold for sufficient data
        return None  # Indicate not enough data

    last_60_days = df['Sugar Level'].values[-min(60, len(df)):]

    if len(last_60_days) < 60:
        last_60_days = np.pad(last_60_days, (60 - len(last_60_days), 0), 'constant')

    scaled_last_60_days = scaler.transform(last_60_days.reshape(-1, 1))
    prediction_list = list(scaled_last_60_days)

    for _ in range(days_to_predict):
        x_input = np.array(prediction_list[-60:]).reshape(1, 60, 1)
        pred = model.predict(x_input)
        prediction_list.append(np.array([pred[0][0]]))

    predictions = np.array(prediction_list[60:]).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions.flatten()

def plot_trends(df, predictions, future_dates):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Sugar Level'], marker='o', label='Actual Readings')
    if predictions is not None:
        plt.plot(future_dates, predictions, linestyle='--', color='blue', label='Predicted Readings')

    plt.axhline(y=140, color='r', linestyle='-', label='Diabetic Threshold (140 mg/dL)')
    plt.axhline(y=100, color='y', linestyle='-', label='Pre-Diabetic Threshold (100 mg/dL)')

    plt.title('Blood Sugar Level Trends and Predictions')
    plt.xlabel('Date')
    plt.ylabel('Blood Sugar Level (mg/dL)')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

def analyze_trends(df):
        last_reading = df['Sugar Level'].iloc[-1]
        min_reading = df['Sugar Level'].min()
        max_reading = df['Sugar Level'].max()

        if 80 <= min_reading <= 100 and 80 <= max_reading <= 100:
            # Normal range
            return (
                "<div style='color: #4CAF50;'>"
                "<img src='https://img.icons8.com/color/48/000000/checked.png' alt='Normal'>"
                "<p>Great job! Your blood sugar levels are normal.</p>"
                "<p>Keep up the good work. Continue eating healthy and staying active.</p>"
                "<p>Try gentle exercises like walking for 30 minutes each day.</p>"
                "</div>",
                ["Walking for 30 minutes a day", "Light stretching", "Gentle yoga"]
            )
        elif 101 <= last_reading <= 125:
            # Prediabetic range
            return (
                "<div style='color: #ff9800;'>"
                "<img src='https://img.icons8.com/color/48/000000/warning-shield.png' alt='Pre-diabetic'>"
                "<p>Your blood sugar levels are in the prediabetic range.</p>"
                "<p>This means your levels are higher than normal, but not yet in the diabetic range.</p>"
                "<p>Increase your physical activity. Try stretching, resistance training, and regular walks.</p>"
                "</div>",
                ["Stretching for 10 minutes every morning", "Resistance training with bands", "Walking for 30 minutes a day"]
            )
        elif last_reading >= 126:
            # Diabetic range
            return (
                "<div style='color: #f44336;'>"
                "<img src='https://img.icons8.com/color/48/000000/cancel.png' alt='Diabetic'>"
                "<p>Your blood sugar levels are in the diabetic range and are rising.</p>"
                "<p>This is urgent. Please see your doctor immediately.</p>"
                "<p>Regular, gentle exercises can help manage your sugar levels.</p>"
                "<p>Start with walking or seated leg lifts, and do them daily.</p>"
                "</div>",
                ["Walking for 30 minutes a day", "Seated leg lifts", "Senior yoga"]
            )
        else:
            # Default case
            return (
                "<div style='color: #607D8B;'>"
                "<img src='https://img.icons8.com/color/48/000000/info.png' alt='Monitor'>"
                "<p>Keep checking your blood sugar regularly.</p>"
                "<p>Stay active with simple exercises like walking or light stretching.</p>"
                "</div>",
                ["Walking for 20-30 minutes a day", "Light stretching", "Gentle yoga"]
            )


@app.route('/', methods=['GET', 'POST'])
def welcome():
    """
    Handles the welcome page where seniors select their name from a dropdown.
    """
    if request.method == 'POST':
        senior_name = request.form.get('senior_name')  # Get the selected name
        spreadsheet_id = SENIOR_SPREADSHEET_MAPPING.get(senior_name)  # Get corresponding spreadsheet ID
        if spreadsheet_id:
            # Redirect to the trends page with the spreadsheet ID
            return redirect(url_for('trends', spreadsheet_id=spreadsheet_id))
        else:
            # Invalid selection: re-render the welcome page with an error message
            return render_template(
                'welcome.html',
                message="Invalid selection. Please try again.",
                senior_names=list(SENIOR_SPREADSHEET_MAPPING.keys())
            )

    # Render the welcome page with dropdown for senior names
    return render_template(
        'welcome.html',
        message="Select your name to proceed.",
        senior_names=list(SENIOR_SPREADSHEET_MAPPING.keys())
    )


@app.route('/trends')
def trends():
    spreadsheet_id = request.args.get('spreadsheet_id')
    df = get_sugar_data(spreadsheet_id)

    # Train or load the LSTM model
    model, scaler = train_or_load_lstm_model(df, spreadsheet_id)

    # Predict future sugar levels
    predictions = predict_future_sugar_levels(model, scaler, df)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

    # Plot trends
    chart = plot_trends(df, predictions, future_dates)

    # Analyze actual sugar trends
    actual_message, recommended_exercises = analyze_trends(df)

    # Analyze current sugar trends for trend_detail
    last_reading = df['Sugar Level'].iloc[-1]
    min_reading = df['Sugar Level'].min()
    max_reading = df['Sugar Level'].max()

    if max_reading - min_reading > 30:
        trend_detail = (
            "Your sugar levels have been fluctuating significantly. "
            "This may indicate instability. Please consult your doctor for guidance."
        )
    elif 80 <= last_reading <= 100:
        trend_detail = (
            "Your sugar levels are stable and within the normal range. "
            "Keep up your healthy lifestyle!"
        )
    elif 101 <= last_reading <= 125:
        trend_detail = (
            "Your sugar levels are in the prediabetic range. "
            "Consider making dietary adjustments and increasing physical activity."
        )
    elif last_reading >= 126:
        trend_detail = (
            "Your sugar levels are in the diabetic range. "
            "It's crucial to consult your doctor for personalized advice."
        )
    else:
        trend_detail = (
            "No significant trends detected. Keep monitoring your sugar levels regularly."
        )

    # Predictive insights
    if len(df) < 60:
        predictive_message = "Predictive analysis is only available after collecting at least 60 readings."
        next_prediction = None
        trend_analysis = None
    else:
        next_prediction = round(predictions[0], 2)  # Show the first predicted reading
        if next_prediction > last_reading:
            trend_analysis = "The next predicted reading is higher than the previous reading. This trend may be concerning."
        elif next_prediction < last_reading:
            trend_analysis = "The next predicted reading is lower than the previous reading. This trend is a good sign."
        else:
            trend_analysis = "The next predicted reading is the same as the previous reading. No significant trend change detected."
        predictive_message = f"The model predicts your next sugar reading could be approximately {next_prediction} mg/dL."

    return render_template(
        'trends.html',
        chart=chart,
        alert=actual_message,
        trend_detail=trend_detail,  # Pass the defined trend_detail
        exercises=recommended_exercises,
        predictive_message=predictive_message,
        trend_analysis=trend_analysis,
        data=df.tail(10).to_html(classes='table table-striped', index=True)
    )



if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080)


