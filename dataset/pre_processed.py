import pandas as pd

# Read dataset
df = pd.read_csv("dataset\Combined_data.csv")

# Keep only required columns
df = df[["statement", "status"]]

# Rename columns
df.columns = ["text", "label"]

# Remove empty rows
df.dropna(inplace=True)

# Save processed file
df.to_csv(
    "dataset/processed_mental_health.csv",
    index=False
)

print("Done!")
print("Total Records:", len(df))
print("\nLabels:\n")
print(df["label"].value_counts())