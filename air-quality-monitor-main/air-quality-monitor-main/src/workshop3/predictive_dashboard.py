import numpy as np
import streamlit as st
import pandas as pd
from markov_model import MarkovModel
from spatial_markov import SpatialMarkov

class PredictiveDashboard:
    def __init__(self, records):
        self.records = records
        self.markov = MarkovModel()
        self.spatial = SpatialMarkov()

    def run_models(self):
        self.markov.fit(self.records)
        self.markov.compute_stationary()
        self.spatial.load_sensors(self.records)
        self.spatial.build_spatial_transition()

    def render_transition_visualization(self):
        st.subheader("Matriz de transición (Markov)")
        df = pd.DataFrame(self.markov.transition_matrix, columns=self.markov.states, index=self.markov.states)
        st.dataframe(df)

    def render_stationary_visualization(self):
        st.subheader("Distribución estacionaria")
        d = self.markov.stationary_distribution
        st.bar_chart(pd.DataFrame({"prob": d}, index=self.markov.states))

    def render_spatial_map(self):
        st.subheader("Transición espacial entre sensores")
        df = pd.DataFrame(
            self.spatial.spatial_transition,
            columns=list(self.sensors.keys()),
            index=list(self.sensors.keys())
        )
        st.dataframe(df)

    def render_predictions(self):
        st.subheader("Probabilidad de estado peligroso (próximos pasos)")
        P = self.markov.transition_matrix
        final = np.linalg.matrix_power(P, 5)[0]
        st.metric("Probabilidad estimada", f"{final[self.markov.states.index('peligroso')]:.2f}")

    def render_alerts(self):
        st.subheader("Alertas de riesgo")
        d = self.markov.stationary_distribution
        idx = self.markov.states.index("peligroso")
        if d[idx] > 0.3:
            st.error("Alta probabilidad de alcanzar estado peligroso")
        else:
            st.success("Riesgo bajo")

    def display(self):
        self.run_models()
        self.render_transition_visualization()
        self.render_stationary_visualization()
        self.render_predictions()
        self.render_alerts()