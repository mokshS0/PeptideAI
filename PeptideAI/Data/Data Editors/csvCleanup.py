import pandas as pd

# Load data
df = pd.read_csv("cleaned_amp_data.csv")

# Drop index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Drop duplicate sequences
df = df.drop_duplicates(subset='sequence')

# Save cleaned data
df.to_csv("2cleaned_amp_data.csv", index=False)
