import pandas as pd
import numpy as np

# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('./dataset/star_classification.csv')
print(dataset)
print(f"Shape: {dataset.shape}")

