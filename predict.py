import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the saved model, scaler, and label encoders for each column
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_switch_id = joblib.load('label_encoder_switch_id.pkl')  # Separate encoder for Switch ID
label_encoder_port_number = joblib.load('label_encoder_port_number.pkl')  # Separate encoder for Port Number
label_encoder_connection_point = joblib.load('label_encoder_connection_point.pkl')  # Separate encoder for Connection Point

# Load dataset for charts and graphs
df = pd.read_csv('dataset.csv')  # Replace with your dataset file


# Generate 10 meaningful charts/graphs
def generate_charts(dataframe):
    try:
        # 1. Correlation heatmap
        numeric_columns = dataframe.select_dtypes(include=['number'])  # Select numerical columns
        if not numeric_columns.empty:
            plt.figure(figsize=(12, 10))
            corr = numeric_columns.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Correlation Heatmap")
            plt.show()

        # 2. Distribution of 'Received Packets'
        if 'Received Packets' in dataframe:
            sns.histplot(dataframe['Received Packets'], kde=True, color='blue')
            plt.title("Distribution of Received Packets")
            plt.xlabel("Received Packets")
            plt.ylabel("Frequency")
            plt.show()

        # 3. Distribution of 'Sent Packets'
        if 'Sent Packets' in dataframe:
            sns.histplot(dataframe['Sent Packets'], kde=True, color='green')
            plt.title("Distribution of Sent Packets")
            plt.xlabel("Sent Packets")
            plt.ylabel("Frequency")
            plt.show()

        # 4. Boxplot for 'Received Bytes'
        if 'Received Bytes' in dataframe:
            sns.boxplot(x=dataframe['Received Bytes'], color='orange')
            plt.title("Boxplot of Received Bytes")
            plt.xlabel("Received Bytes")
            plt.show()

        # 5. Boxplot for 'Sent Bytes'
        if 'Sent Bytes' in dataframe:
            sns.boxplot(x=dataframe['Sent Bytes'], color='purple')
            plt.title("Boxplot of Sent Bytes")
            plt.xlabel("Sent Bytes")
            plt.show()

        # 6. Scatter plot: 'Received Packets' vs 'Sent Packets'
        if 'Received Packets' in dataframe and 'Sent Packets' in dataframe:
            sns.scatterplot(x='Received Packets', y='Sent Packets', data=dataframe, color='red')
            plt.title("Scatter Plot: Received Packets vs Sent Packets")
            plt.xlabel("Received Packets")
            plt.ylabel("Sent Packets")
            plt.show()

        # 7. Bar plot for 'Connection Point' distribution
        if 'Connection Point' in dataframe:
            sns.countplot(x='Connection Point', data=dataframe, palette='coolwarm')
            plt.title("Connection Point Distribution")
            plt.xlabel("Connection Point")
            plt.ylabel("Count")
            plt.show()

        # 8. Bar plot for 'Switch ID' distribution
        if 'Switch ID' in dataframe:
            sns.countplot(x='Switch ID', data=dataframe, palette='coolwarm')
            plt.title("Switch ID Distribution")
            plt.xlabel("Switch ID")
            plt.ylabel("Count")
            plt.show()

        # 9. Line plot: 'Port alive Duration (S)' over index
        if 'Port alive Duration (S)' in dataframe:
            plt.plot(dataframe.index, dataframe['Port alive Duration (S)'], color='cyan', marker='o')
            plt.title("Port alive Duration (S) Over Index")
            plt.xlabel("Index")
            plt.ylabel("Port alive Duration (S)")
            plt.show()

        # 10. Pie chart for categorical column 'Connection Point'
        if 'Connection Point' in dataframe:
            connection_point_counts = dataframe['Connection Point'].value_counts()
            connection_point_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
            plt.title("Connection Point Proportion")
            plt.ylabel("")
            plt.show()

    except Exception as e:
        print(f"Error in generate_charts: {e}")

# Call the chart generation function
generate_charts(df)

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

# Example input data (single row of data to predict)
input_data = {
    'Received Packets': '2335',
    'Received Bytes': '56779872',
    'Sent Packets': '3898',
    'Sent Bytes': '44420382',
    'Port alive Duration (S)': '2537',
    'Packets Rx Dropped': '0',
    'Packets Tx Dropped': '0',
    'Packets Rx Errors': '0',
    'Packets Tx Errors': '0',
    'Delta Received Packets': '121',
    'Delta Received Bytes': '5119794',
    'Delta Sent Bytes': '4780',
    'Delta Sent Packets': '68',
    'Total Load/Rate': '5.0',
    'Switch ID': 'of:0000000000000001',  # Categorical value (encoded)
    'Port Number': 'Port#:2',            # Categorical value (encoded)
    'Connection Point': '3',             # Categorical value (encoded)
    'Max Size': '-1'
}

# Call the prediction function
result = predict(input_data)
print(f"Predicted Label: {result}")
if(result == 'TCP-SYN'):
    print('Recommended Firewall: Radware Cloud WAF')
elif(result=='Diversion'):
    print('Recommended Firewall: Akamai App & API Protector')
elif(result=='Blackhole'):
    print('Recommended Firewall: Amazon Web Services (AWS) WAF')
elif(result=='Overflow'):
    print('Recommended Firewall: Barracuda Web Application Firewall')
elif(result=='PortScan'):
    print('Recommended Firewall: Cloudflare WAF')
else:
    print('Connection is normal. No firewall needed')
