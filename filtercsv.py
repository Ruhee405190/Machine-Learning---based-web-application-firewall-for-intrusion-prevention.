import pandas as pd

# Load the dataset with predictions
df = pd.read_csv('dataset_with_predictions.csv')

# Compare the 'Label' and 'Predicted Label' columns and filter out rows where they do not match
filtered_df = df[df['Label'] == df['Predicted Label']]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('dataset_filtered.csv', index=False)

print("Rows where 'Label' and 'Predicted Label' match have been saved to 'dataset_filtered.csv'.")
