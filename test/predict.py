import joblib
import pandas as pd

# Load the saved model from the file
loaded_model = joblib.load('linear_regression_model.joblib')

def predict(model, data):
    # 2022-09-24,0.507200,0.515015,0.474019,0.488716,0.488716,3512953969
    # Assume `new_data` is a pandas DataFrame with the same structure as your training data
    new_data = pd.DataFrame({'Open': [data[0]], 'High': [data[1]], 'Low': [data[2]]})

    # Use the trained model for prediction
    new_prediction = model.predict(new_data)

    # Print or use the prediction as needed
    print(f'Predicted Close Price: {new_prediction[0]}')


predict(loaded_model, [0.507200, 0.515015, 0.474019])