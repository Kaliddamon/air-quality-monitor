import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class SpatialMarkov:
    def __init__(self):
        self.sensors = {}
        self.dist_matrix = None
        self.corr_matrix = None
        self.spatial_transition = None

    def load_sensors(self, records):
        for r in records:
            loc = r.get("sensorLocation")
            pm = r.get("airQualityData", {}).get("PM25")
            if loc not in self.sensors:
                self.sensors[loc] = []
            self.sensors[loc].append(pm)
        self._compute_distances()
        self._compute_correlations()

    def _compute_distances(self):
        locs = np.array([[hash(k) % 1000, (hash(k) // 1000) % 1000] for k in self.sensors.keys()])
        self.dist_matrix = cdist(locs, locs, metric="euclidean")

    def _compute_correlations(self):
        keys = list(self.sensors.keys())
        n = len(keys)
        self.corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a = np.array(self.sensors[keys[i]])
                b = np.array(self.sensors[keys[j]])
                if len(a) == len(b) and len(a) > 1:
                    self.corr_matrix[i, j] = np.corrcoef(a, b)[0, 1]
        self.corr_matrix = np.nan_to_num(self.corr_matrix)

    def build_spatial_transition(self):
        inv_dist = 1 / (self.dist_matrix + 1)
        raw = inv_dist * (self.corr_matrix + 1)
        row_sums = raw.sum(axis=1, keepdims=True)
        self.spatial_transition = raw / row_sums
        return self.spatial_transition

    def simulate_diffusion(self, initial_vector, steps=5):
        x = np.array(initial_vector, dtype=float)
        for _ in range(steps):
            x = x @ self.spatial_transition
        return x

    def export_transition(self):
        return self.spatial_transition.tolist() if self.spatial_transition is not None else None