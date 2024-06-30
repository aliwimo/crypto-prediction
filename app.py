from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

app = Flask(__name__)

# Initialize an empty model variable
model = None

day_offset = 2
datasets_dir = 'datasets'  # Update this with the actual directory path
models_dir = 'models'  # Update this with the actual directory path


def load_dataset(dataset_name):
    folder_path = Path(__file__).parent.resolve()
    dataset_path = folder_path / f'{datasets_dir}/{dataset_name}.csv'

    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']].shift(day_offset)
    df = df.dropna()
    return df


def ensure_models_directory():
    folder_path = Path(__file__).parent.resolve()
    models = folder_path / f'{models_dir}'
    if not os.path.exists(models):
        os.makedirs(models)


def load_or_train_model(dataset_name, test_size=0.25, random_state=42):
    global model

    # Check if the model exists for the dataset
    folder_path = Path(__file__).parent.resolve()
    model_filename = folder_path / f'{models_dir}/{dataset_name}.joblib'

    if os.path.exists(model_filename):
        # Load the existing model
        model = joblib.load(model_filename)
    else:
        # Train a new model if it doesn't exist
        ensure_models_directory()
        df = load_dataset(dataset_name)
        X = df[['Open', 'High', 'Low']]
        y = df['Close']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, model_filename)


def fetch_yahoo_finance_data(ticker_symbol, start_date, end_date):
    # Fetch historical stock data from Yahoo Finance API
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data.reset_index()


@app.route('/', methods=['GET'])
def index():
    return jsonify({'response': 'Crypto API'})


@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get training data from the POST request
        data = request.json

        # Get the dataset name from the request
        dataset_name = data.get('dataset_name')
        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'})

        # Ensure the 'models' directory exists
        ensure_models_directory()

        # Load the dataset
        df_train = load_dataset(dataset_name)

        # Feature engineering and preprocessing (if needed)
        # ...

        # Split the dataset into features (X) and target variable (y)
        X_train = df_train[['Open', 'High', 'Low']]
        y_train = df_train['Close']

        # Create and train a new linear regression model
        new_model = LinearRegression()
        new_model.fit(X_train, y_train)

        # Save the trained model to the 'models' directory
        model_filename = os.path.join(models_dir, f'{dataset_name}.joblib')
        joblib.dump(new_model, model_filename)

        global model
        model = new_model  # Update the global model variable

        return jsonify({'message': f'Model trained and saved successfully for dataset {dataset_name}'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.json

        # Get the dataset name from the request
        dataset_name = data.get('dataset_name')
        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'})

        # Load or train the model for the specified dataset
        load_or_train_model(dataset_name)

        # Prepare the data for prediction
        new_data = [[data['Open'], data['High'], data['Low']]]

        # Make a prediction using the loaded or trained model
        if model is None:
            return jsonify({'error': 'Model not loaded or trained'})

        prediction = model.predict(new_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    try:
        # Get data from the POST request
        data = request.json

        # Get the ticker symbol from the request
        ticker_symbol = data.get('ticker_symbol')
        if not ticker_symbol:
            return jsonify({'error': 'Ticker symbol is required'})

        # Calculate start and end dates for the latest year
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        # Fetch historical stock data from Yahoo Finance
        yahoo_data = fetch_yahoo_finance_data(ticker_symbol, start_date, end_date)

        # Save the fetched data to a CSV file
        ensure_models_directory()
        folder_path = Path(__file__).parent.resolve()
        dataset_filename = folder_path / f'{datasets_dir}/{ticker_symbol}.csv'
        yahoo_data.to_csv(dataset_filename, index=False)

        return jsonify({'message': f'Dataset fetched and saved successfully for {ticker_symbol}'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/closing_prices_chart_data', methods=['GET'])
def closing_prices_chart_data():
    try:
        # Get the dataset name from the request
        dataset_name = request.args.get('dataset_name')
        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'})

        # Ensure the 'models' directory exists
        ensure_models_directory()

        # Load or train the model for the specified dataset
        load_or_train_model(dataset_name)

        # Fetch historical stock data from Yahoo Finance for the latest year
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        stock_data = yf.download(dataset_name, start=start_date, end=end_date)
        stock_data = stock_data.reset_index()

        # Use the trained model to predict closing prices
        X_new = stock_data[['Open', 'High', 'Low']]
        stock_data['Predicted_Close'] = model.predict(X_new)

        # Prepare data for the chart
        chart_data = {
            'Date': stock_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'Actual_Close': stock_data['Close'].tolist(),
            'Predicted_Close': stock_data['Predicted_Close'].tolist()
        }

        return jsonify(chart_data)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/last_day_data', methods=['POST'])
def last_day_data():
    try:
        # Get data from the POST request
        data = request.json

        # Get the ticker symbol from the request
        ticker_symbol = data.get('ticker_symbol')
        if not ticker_symbol:
            return jsonify({'error': 'Ticker symbol is required'})

        # Fetch intraday stock data from Yahoo Finance for the past 24 hours
        intraday_data = yf.download(ticker_symbol, period='1d', interval='1m')

        # Extract the relevant columns (Open, High, Low) from the last row
        last_row = intraday_data.iloc[-1]
        open_price = last_row['Open']
        high_price = last_row['High']
        low_price = last_row['Low']

        return jsonify({
            'ticker_symbol': ticker_symbol,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/scores', methods=['GET'])
def get_scores(test_size=0.25, random_state=42):
    try:
        # Get the dataset name from the request
        dataset_name = request.args.get('dataset_name')
        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'})

        # Ensure the 'models' directory exists
        ensure_models_directory()

        # Load or train the model for the specified dataset
        load_or_train_model(dataset_name)

        # prepare dataset
        df = load_dataset(dataset_name)
        X = df[['Open', 'High', 'Low']]
        y = df['Close']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        y_pred = model.predict(X_test)

        r2_test_score = r2_score(y_test, y_pred)
        rms_test_error = root_mean_squared_error(y_test, y_pred)

        return jsonify({
            'r2_score': r2_test_score,
            'rms_error': rms_test_error,
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
