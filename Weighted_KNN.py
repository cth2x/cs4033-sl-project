import pandas as pd
import math
import numpy as np
import sys

df = pd.read_csv("heart+disease/processed.cleveland.data", header=None)

df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

def euclidean_distance(row1, row2):
    difference_vector = subtract_vectors(row1, row2)
    sum = 0
    for i in range(difference_vector.count):
        sum += i * i
    return math.sqrt(sum)

def subtract_vectors(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")
    
    result = [v1 - v2 for v1, v2 in zip(vector1, vector2)]
    return result

def get_rows_as_array(df):
    return df.to_numpy().tolist()

rows_array = get_rows_as_array(df)
print(rows_array[:5]) 