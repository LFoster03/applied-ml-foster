import pandas as pd

# Download the CSV from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Load into a DataFrame (semicolon-separated!)
df = pd.read_csv(url, sep=";")

# Save to your data folder
save_path = r"C:\Repos\applied-ml-foster\notebooks\project05\data\winequality-red.csv"
df.to_csv(save_path, index=False)

df.head()
