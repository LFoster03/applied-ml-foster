import numpy as np
import pandas as pd
import os

# Path to your data folder
save_path = r"C:\Repos\applied-ml-foster\notebooks\project05\data\spiral.csv"


# Function to generate a spiral dataset
def make_spiral(n_points_per_class=500, noise=0.2):
    n = np.arange(0, n_points_per_class)
    theta = np.sqrt(n) * 0.5

    # Spiral 1
    x1 = np.cos(theta) * n
    y1 = np.sin(theta) * n
    c1 = np.zeros(n_points_per_class)

    # Spiral 2 (rotated)
    x2 = np.cos(theta + np.pi) * n
    y2 = np.sin(theta + np.pi) * n
    c2 = np.ones(n_points_per_class)

    # Combine
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    labels = np.concatenate([c1, c2])

    # Add noise
    x = x + np.random.normal(scale=noise, size=x.shape)
    y = y + np.random.normal(scale=noise, size=y.shape)

    return pd.DataFrame({"x": x, "y": y, "label": labels})


# Generate the dataset
spiral = make_spiral()

# Save to CSV
os.makedirs(os.path.dirname(save_path), exist_ok=True)
spiral.to_csv(save_path, index=False)

print(f"spiral.csv saved to:\n{save_path}")
