import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("heart+disease/processed.cleveland.data", header=None)
df = pd.read_csv("heart+disease/combined.data", header=None)

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

def weighted_knn_predict(training_data, test_instance, k):
    distances = []
    for row in training_data:
        dist = euclidean_distance(test_instance[:-1], row[:-1])
        distances.append((row, dist))
    
    distances.sort(key=lambda x: x[1])
    nearest_neighbors = distances[:k]
    
    votes = {}
    for neighbor, dist in nearest_neighbors:
        label = neighbor[-1]
        weight = 1 / (dist + 1e-5)
        votes[label] = votes.get(label, 0) + weight
    
    predicted_label = max(votes.items(), key=lambda x: x[1])[0]
    return predicted_label

correctPercentagePlotable = []
timePlotable = []

for k in range(1,len(rows_array)):
    correctPercentage = 0.0
    start_time = time.time()
    for i in range(len(rows_array) - 1):
        test_instance = rows_array[i]
        training_data = rows_array[i+1:]

        prediction = weighted_knn_predict(training_data, test_instance, k)
        # prediction = knn_predict(training_data, test_instance, k)
        # print("Predicted label for the test instance:", prediction)
        isCorrect = test_instance[13] > 0 and prediction > 0
        if test_instance[13] > 0 and prediction > 0 or test_instance[13] == 0 and prediction == 0 :
            correctPercentage += 1
        # print(f"is prediction correct? {isCorrect}")
    correctPercentage = correctPercentage / len(rows_array)
    end_time = time.time()
    elapsed_time = end_time - start_time
    timePlotable.append(elapsed_time)
    correctPercentagePlotable.append(correctPercentage)
    print("------------------------------------------")
    print(f"    Elapsed time: {elapsed_time} seconds")
    print(f"    k = {k}")
    print(f"    correctPercentage = {correctPercentage}")

print("------------------------------------------")

plt.plot(range(1,len(rows_array)), correctPercentagePlotable)

plt.xlabel("Numbers of Neighbors")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Numbers of Neighbors")
plt.show()

plt.plot(range(1,len(rows_array)), timePlotable)

plt.xlabel("Numbers of Neighbors")
plt.ylabel("Time (s)")
plt.title("Time vs Numbers of Neighbors")
plt.show()