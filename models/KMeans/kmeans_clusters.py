import pandas as pd
from sklearn.metrics._dist_metrics import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import requests
import os

class Circle:
    def __init__(self, radius):
        self.radius = radius
        self.area = None

    def calculate_area(self):
        self.area = 3.14 * self.radius * self.radius


circle = Circle(5)
circle.calculate_area()
print("Area of the circle:", circle.area)
