import pickle

path = "spacegraph/data_collection/Place2Vec_center/pointset.pkl"

with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print(type(data))
print(len(data))
print(len(data[1]))
print(data[0])

print(type(data[0]))
print(type(data[1]))

print("\ndata[1][0]:")
print(data[1][0])

print("\ndata[1][1]:")
print(data[1][1])

print("\ndata[1][-1]:")
print(data[1][-1])

print(type(data[1][0]))

from collections import Counter

points = data[1]

print("POI type数:", data[0])
print("地点数:", len(points))

split_counts = Counter(p[3] for p in points)
print(split_counts)

import numpy as np

coords = np.array([p[1] for p in points])

print("x:", coords[:, 0].min(), coords[:, 0].max())
print("y:", coords[:, 1].min(), coords[:, 1].max())