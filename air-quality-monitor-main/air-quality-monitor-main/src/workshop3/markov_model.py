import numpy as np
import pandas as pd

class MarkovModel:
    def __init__(self, states=None):
        self.states = states or ["bueno", "moderado", "insalubre", "peligroso"]
        self.n = len(self.states)
        self.transition_matrix = np.zeros((self.n, self.n))
        self.weather_adjustment = np.eye(self.n)
        self.stationary_distribution = None

    def fit(self, records):
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp")
        df["state"] = df["PM25"].apply(self._classify)
        transitions = np.zeros((self.n, self.n))
        for i in range(len(df) - 1):
            a = self.states.index(df["state"].iloc[i])
            b = self.states.index(df["state"].iloc[i + 1])
            transitions[a, b] += 1
        row_sums = transitions.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

    def _classify(self, pm):
        if pm < 12: return "bueno"
        if pm < 35: return "moderado"
        if pm < 55: return "insalubre"
        return "peligroso"

    def update_with_weather(self, humidity, wind_speed):
        h_factor = min(1.5, 1 + humidity / 200)
        w_factor = max(0.5, 1 - wind_speed / 100)
        adj = np.eye(self.n)
        adj[2, 3] *= h_factor
        adj[1, 2] *= h_factor
        adj[3, 2] *= w_factor
        self.weather_adjustment = adj
        self.transition_matrix = self.transition_matrix * adj
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)

    def compute_stationary(self):
        P = np.copy(self.transition_matrix)
        A = np.transpose(P) - np.eye(self.n)
        A = np.vstack([A, np.ones(self.n)])
        b = np.zeros(self.n + 1)
        b[-1] = 1
        self.stationary_distribution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return self.stationary_distribution

    def export(self):
        return {
            "states": self.states,
            "transition_matrix": self.transition_matrix.tolist(),
            "stationary_distribution": self.stationary_distribution.tolist() if self.stationary_distribution is not None else None
        }