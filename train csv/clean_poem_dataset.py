import pandas as pd

# Load the dataset (make sure PoetryFoundationData.csv is in the same folder as this script)
input_file = 'PoetryFoundationData.csv'
output_file = 'Modified_PoetryData.csv'

# Read the CSV
df = pd.read_csv(input_file)

# Drop rows where 'Tags' (i.e., Genre) is missing or empty
df_filtered = df[df['Tags'].notna()]
df_filtered = df_filtered[df_filtered['Tags'].str.strip() != ""]

# Keep only 'Tags' and 'Poem' columns and rename 'Tags' to 'Genre'
df_final = df_filtered[['Tags', 'Poem']].rename(columns={'Tags': 'Genre'})

# Save the cleaned dataset
df_final.to_csv(output_file, index=False, quoting=1)

print(f"✅ Cleaned CSV saved as '{output_file}'!")
