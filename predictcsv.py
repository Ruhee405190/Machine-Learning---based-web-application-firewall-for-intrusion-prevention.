import pandas as pd
import joblib
import numpy as np

# Load the saved model, scaler, and label encoders for each column
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_switch_id = joblib.load('label_encoder_switch_id.pkl')  # Separate encoder for Switch ID
label_encoder_port_number = joblib.load('label_encoder_port_number.pkl')  # Separate encoder for Port Number
label_encoder_connection_point = joblib.load('label_encoder_connection_point.pkl')  # Separate encoder for Connection Point

# Function to predict the label based on the input data
def predict(input_data):
    try:
        # Encode categorical features (Switch ID, Port Number, and Connection Point)
        switch_id_encoded = label_encoder_switch_id.transform([input_data['Switch ID']])[0]
        port_number_encoded = label_encoder_port_number.transform([input_data['Port Number']])[0]
        connection_point_encoded = label_encoder_connection_point.transform([input_data['Connection Point']])[0]

        # Creating DataFrame for the input data
        input_df = pd.DataFrame([{
            'Received Packets': float(input_data['Received Packets']),
            'Received Bytes': float(input_data['Received Bytes']),
            'Sent Packets': float(input_data['Sent Packets']),
            'Sent Bytes': float(input_data['Sent Bytes']),
            'Port alive Duration (S)': float(input_data['Port alive Duration (S)']),
            'Packets Rx Dropped': float(input_data['Packets Rx Dropped']),
            'Packets Tx Dropped': float(input_data['Packets Tx Dropped']),
            'Packets Rx Errors': float(input_data['Packets Rx Errors']),
            'Packets Tx Errors': float(input_data['Packets Tx Errors']),
            'Delta Received Packets': float(input_data['Delta Received Packets']),
            'Delta Received Bytes': float(input_data['Delta Received Bytes']),
            'Delta Sent Bytes': float(input_data['Delta Sent Bytes']),
            'Delta Sent Packets': float(input_data['Delta Sent Packets']),
            'Total Load/Rate': float(input_data['Total Load/Rate']),
            'Switch ID': switch_id_encoded,
            'Port Number': port_number_encoded,
            'Connection Point': connection_point_encoded,
            'Max Size': float(input_data['Max Size'])
        }])

        # Scale the features using the pre-fitted scaler
        input_data_scaled = scaler.transform(input_df)

        # Make the prediction
        prediction = svm_model.predict(input_data_scaled)

        # Return the prediction result
        return prediction[0]

    except Exception as e:
        print(f"Error: {e}")
        return None

# Load your dataset
df = pd.read_csv('dataset.csv')

# Create an empty list to store the predictions
predictions = []

# Iterate through each row in the dataset to predict the label
for index, row in df.iterrows():
    input_data = {
        'Received Packets': row['Received Packets'],
        'Received Bytes': row['Received Bytes'],
        'Sent Packets': row['Sent Packets'],
        'Sent Bytes': row['Sent Bytes'],
        'Port alive Duration (S)': row['Port alive Duration (S)'],
        'Packets Rx Dropped': row['Packets Rx Dropped'],
        'Packets Tx Dropped': row['Packets Tx Dropped'],
        'Packets Rx Errors': row['Packets Rx Errors'],
        'Packets Tx Errors': row['Packets Tx Errors'],
        'Delta Received Packets': row['Delta Received Packets'],
        'Delta Received Bytes': row['Delta Received Bytes'],
        'Delta Sent Bytes': row['Delta Sent Bytes'],
        'Delta Sent Packets': row['Delta Sent Packets'],
        'Total Load/Rate': row['Total Load/Rate'],
        'Switch ID': row['Switch ID'],
        'Port Number': row['Port Number'],
        'Connection Point': row['Connection Point'],
        'Max Size': row['Max Size']
    }

    # Predict the label for the current row
    predicted_label = predict(input_data)

    # Append the predicted label to the predictions list
    predictions.append(predicted_label)

# Add the predictions as a new column to the dataframe
df['Predicted Label'] = predictions

# Save the dataframe with predictions to a new CSV file
df.to_csv('dataset_with_predictions.csv', index=False)

print("Predictions have been added to the new column 'Predicted Label' and saved to 'dataset_with_predictions.csv'.")
