import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

# Shuffle the dataset
data = data.sample(frac=1)

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2)

# Save the split datasets
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
