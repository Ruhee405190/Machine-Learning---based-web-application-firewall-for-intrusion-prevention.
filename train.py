import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load your dataset (replace 'dataset.csv' with your actual file path)
df = pd.read_csv('dataset.csv')

# Features: Select relevant columns
X = df[['Received Packets', 'Received Bytes', 'Sent Packets', 'Sent Bytes', 
        'Port alive Duration (S)', 'Packets Rx Dropped', 'Packets Tx Dropped',
        'Packets Rx Errors', 'Packets Tx Errors', 'Delta Received Packets', 
        'Delta Received Bytes', 'Delta Sent Bytes', 'Delta Sent Packets', 
        'Total Load/Rate', 'Switch ID', 'Port Number', 'Connection Point', 'Max Size']]

# Target: 'Label' column
y = df['Label']

# Preprocessing: Encoding categorical variables (Switch ID, Port Number, Connection Point)
label_encoder_switch_id = LabelEncoder()
X['Switch ID'] = label_encoder_switch_id.fit_transform(X['Switch ID'])

label_encoder_port_number = LabelEncoder()
X['Port Number'] = label_encoder_port_number.fit_transform(X['Port Number'])

label_encoder_connection_point = LabelEncoder()
X['Connection Point'] = label_encoder_connection_point.fit_transform(X['Connection Point'])

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model using Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')  # Linear kernel, you can experiment with others like 'rbf'
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model, scaler, and label encoders
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder_switch_id, 'label_encoder_switch_id.pkl')
joblib.dump(label_encoder_port_number, 'label_encoder_port_number.pkl')
joblib.dump(label_encoder_connection_point, 'label_encoder_connection_point.pkl')

print("Model and necessary files saved.")

# Visualization: Generate meaningful graphs and charts

# 1. Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Label', data=df, palette='viridis')
plt.title('Distribution of Target Variable (Label)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


# 3. Boxplot of Total Load/Rate by Label
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Total Load/Rate', data=df, palette='Set2')
plt.title('Boxplot of Total Load/Rate by Label')
plt.xlabel('Label')
plt.ylabel('Total Load/Rate')
plt.show()

# 4. Distribution of Port alive Duration (S)
plt.figure(figsize=(8, 6))
sns.histplot(df['Port alive Duration (S)'], kde=True, bins=30, color='blue')
plt.title('Distribution of Port Alive Duration (S)')
plt.xlabel('Port Alive Duration (S)')
plt.ylabel('Frequency')
plt.show()

# 5. Pairplot of Selected Features
selected_features = ['Received Packets', 'Sent Packets', 'Total Load/Rate', 'Max Size']
sns.pairplot(df[selected_features + ['Label']], hue='Label', palette='viridis')
plt.title('Pairplot of Selected Features')
plt.show()

# 6. Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 7. Feature importance based on coefficients
feature_importance = pd.Series(svm_model.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='teal')
plt.title('Feature Importance Based on SVM Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()

# 8. Distribution of Delta Sent Bytes by Label
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Delta Sent Bytes', data=df, palette='cool')
plt.title('Distribution of Delta Sent Bytes by Label')
plt.xlabel('Label')
plt.ylabel('Delta Sent Bytes')
plt.show()

# 9. Countplot of Connection Point
plt.figure(figsize=(8, 6))
sns.countplot(x='Connection Point', data=df, palette='Set1', order=df['Connection Point'].value_counts().index)
plt.title('Distribution of Connection Point')
plt.xlabel('Connection Point')
plt.ylabel('Count')
plt.show()

# 10. Violin plot of Received Bytes by Label
plt.figure(figsize=(8, 6))
sns.violinplot(x='Label', y='Received Bytes', data=df, palette='muted')
plt.title('Violin Plot of Received Bytes by Label')
plt.xlabel('Label')
plt.ylabel('Received Bytes')
plt.show()
