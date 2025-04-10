# import pandas as pd
# import math
# import numpy as np
# import sys

# df = pd.read_csv("heart+disease/processed.cleveland.data", header=None)

# df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
#               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# def euclidean_distance(row1, row2):
#     difference_vector = subtract_vectors(row1, row2)
#     sum = 0
#     for i in range(difference_vector.count):
#         sum += i * i
#     return math.sqrt(sum)

# def subtract_vectors(vector1, vector2):
#     if len(vector1) != len(vector2):
#         raise ValueError("Vectors must have the same dimension")
    
#     result = [v1 - v2 for v1, v2 in zip(vector1, vector2)]
#     return result

# def get_rows_as_array(df):
#     return df.to_numpy().tolist()

# rows_array = get_rows_as_array(df)
# print(rows_array[:5])

# euclidean_distance(rows_array[0], rows_array[1])

import pandas as pd
import math

df = pd.read_csv("heart+disease/processed.cleveland.data", header=None)

df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

df = df.apply(pd.to_numeric, errors='coerce')

def get_rows_as_array(dataframe):
    return dataframe.to_numpy().tolist()

rows_array = get_rows_as_array(df)
print("First five rows:")
print(rows_array[:5])

def subtract_vectors(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")
    return [v1 - v2 for v1, v2 in zip(vector1, vector2)]

def euclidean_distance(row1, row2):
    diff_vector = subtract_vectors(row1, row2)
    sum_squared = sum(d ** 2 for d in diff_vector)
    return math.sqrt(sum_squared)

def knn_predict(training_data, test_instance, k):
    distances = []
    for row in training_data:
        dist = euclidean_distance(test_instance[:-1], row[:-1])
        distances.append((row, dist))
    
    distances.sort(key=lambda x: x[1])
    
    nearest_neighbors = distances[:k]
    
    votes = {}
    for neighbor, _ in nearest_neighbors:
        label = neighbor[-1]
        votes[label] = votes.get(label, 0) + 1
    
    predicted_label = max(votes.items(), key=lambda x: x[1])[0]
    return predicted_label

for i in range(len(rows_array) - 1):
    test_instance = rows_array[i]
    training_data = rows_array[i+1:]
    k = 5

    prediction = knn_predict(training_data, test_instance, k)
    print("Predicted label for the test instance:", prediction)
