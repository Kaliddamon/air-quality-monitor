# streamlit_app.py
import os
from datetime import datetime

import streamlit as st
import pandas as pd

from hash_table import HashTable
from prediction import run_all_and_alert
from w2_pipeline import run_w2_from_records
from w3_pipeline import run_w3_from_records, plot_markov_chain
from w4_pipeline import run_w4_from_records
from w6_pipeline import run_w6_from_records

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Real-Time Air Quality Monitoring", layout="wide")

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))
DATA_PATH = os.path.join(project_root, "data", "sensor_data_clean.json")

# ---------------------------------------------------------
# UTIL: caché de carga de datos y wrappers para tareas pesadas
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_hashtable(path: str, last_modified: float):
    """Construye y cachea HashTable a partir del archivo (keyed por last_modified)."""
    ht = HashTable()
    ht.load_json(path)
    return ht

def get_file_mtime(path: str):
    return os.path.getmtime(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def cached_run_w2(mtime):
    ht = build_hashtable(DATA_PATH, mtime)
    return run_w2_from_records(ht.all_records)

@st.cache_data(show_spinner=False)
def cached_run_w3(mtime):
    ht = build_hashtable(DATA_PATH, mtime)
    return run_w3_from_records(ht.all_records)

@st.cache_data(show_spinner=False)
def cached_run_w4(mtime, workers_config):
    ht = build_hashtable(DATA_PATH, mtime)
    return run_w4_from_records(ht.all_records, workers_config)

@st.cache_data(show_spinner=False)
def cached_run_w6(mtime, sensor_id, k):
    ht = build_hashtable(DATA_PATH, mtime)
    return run_w6_from_records(ht.all_records, sensor_id, k)

@st.cache_data(show_spinner=False)
def cached_run_predictions(mtime):
    ht = build_hashtable(DATA_PATH, mtime)
    return run_all_and_alert(ht)

# ---------------------------------------------------------
# ESTILOS (igual que tu original)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-weight: 600; letter-spacing: 0.02em; }
    .stDataFrame tbody tr th { position: sticky; left: 0; z-index: 1; }
    .stDataFrame thead tr th { position: sticky; top: 0; z-index: 2; background-color: #f4f4f4; }
    section[data-testid="stSidebar"] > div { max-height: 100vh; overflow-y: auto; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CARGA DE DATOS (cacheada)
# ---------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"No se encontró el archivo de datos en {DATA_PATH}")
    st.stop()

file_mtime = get_file_mtime(DATA_PATH)
ht = build_hashtable(DATA_PATH, file_mtime)  # cacheado
records = ht.all_records

# ---------------------------------------------------------
# Conversor a DataFrame
# ---------------------------------------------------------
def records_to_dataframe(records_list):
    rows = []
    for r in records_list:
        aq = r.get("airQualityData", {}) or {}
        rows.append({
            "_id": r.get("_id"),
            "sensorLocation": r.get("sensorLocation"),
            "PM25": aq.get("PM25"),
            "NO2": aq.get("NO2"),
            "timestamp": r.get("timestamp"),
        })
    df_local = pd.DataFrame(rows)
    if "timestamp" in df_local.columns:
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], errors="coerce")
    return df_local

df = records_to_dataframe(records)

# ---------------------------------------------------------
# SIDEBAR: filtros en formulario para reducir reruns
# ---------------------------------------------------------
st.sidebar.title("Filtros")

all_cities = sorted([c for c in df["sensorLocation"].dropna().unique()]) if not df.empty else []

# session_state defaults
if "city_filter" not in st.session_state:
    st.session_state["city_filter"] = []
if "run_w2" not in st.session_state:
    st.session_state["run_w2"] = False
if "run_w3" not in st.session_state:
    st.session_state["run_w3"] = False
if "run_w4" not in st.session_state:
    st.session_state["run_w4"] = False
if "run_w6" not in st.session_state:
    st.session_state["run_w6"] = False
if "run_predictions" not in st.session_state:
    st.session_state["run_predictions"] = False

# quick select buttons outside form (they mutate session_state)
btn_col1, btn_col2 = st.sidebar.columns(2)
with btn_col1:
    if st.button("Todo"):
        st.session_state["city_filter"] = all_cities
with btn_col2:
    if st.button("Limpiar"):
        st.session_state["city_filter"] = []

# Use a form so the user configures filters and submits once
with st.sidebar.form("filters_form"):
    city_multiselect = st.multiselect("Ciudades", options=all_cities, key="city_filter")
    pollutant = st.selectbox("Contaminante principal", ["PM25", "NO2"])
    if not df.empty and pollutant in df.columns:
        min_val = float(df[pollutant].min()) if pd.notna(df[pollutant].min()) else 0.0
        max_val = float(df[pollutant].max()) if pd.notna(df[pollutant].max()) else min_val + 1.0
        low_high = st.slider(
            f"Rango de {pollutant}",
            min_value=float(min_val),
            max_value=float(max_val if max_val > min_val else min_val + 1),
            value=(float(min_val), float(max_val if max_val > min_val else min_val + 1)),
            step=0.5
        )
    else:
        low_high = (0.0, 0.0)
    show_only_outliers = st.checkbox("Mostrar solo registros fuera de umbral", value=False)
    st.form_submit_button("Aplicar filtros")

low, high = low_high

# ---------------------------------------------------------
# LAYOUT PRINCIPAL: placeholders para actualizaciones puntuales
# ---------------------------------------------------------
st.title("Real-Time Air Quality Monitoring Dashboard")

# summary stats (siempre se muestran, son rápidos)
stats = ht.stats()
global_stats = ht.global_stats()
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Resumen de registros")
    st.metric("Total de lecturas", stats.get("total_records", 0))
    st.metric("Sensores únicos", stats.get("unique_sensors", 0))
with col2:
    st.subheader("PM2.5 (global)")
    pm = global_stats.get("PM25", {})
    st.metric("Promedio", f"{pm.get('avg'):.2f}" if pm.get("avg") is not None else "N/D")
    st.metric("Máximo", f"{pm.get('max'):.2f}" if pm.get("max") is not None else "N/D")
with col3:
    st.subheader("NO2 (global)")
    no2 = global_stats.get("NO2", {})
    st.metric("Promedio", f"{no2.get('avg'):.2f}" if no2.get("avg") is not None else "N/D")
    st.metric("Máximo", f"{no2.get('max'):.2f}" if no2.get("max") is not None else "N/D")

st.markdown("---")

# Ranking (placeh.)
ranking_placeholder = st.empty()
with ranking_placeholder.container():
    st.subheader("Ranking de ciudades por contaminante")
    ranking = ht.rank_cities_by_pollutant(pollutant=pollutant)
    rank_df = pd.DataFrame(ranking, columns=["Ciudad", f"Promedio {pollutant}"])
    st.dataframe(rank_df, use_container_width=True, height=250)

st.markdown("---")

# Tabla de lecturas (placeholder)
table_placeholder = st.empty()

def render_filtered_table():
    filtered_df = df.copy()
    if st.session_state.get("city_filter"):
        filtered_df = filtered_df[filtered_df["sensorLocation"].isin(st.session_state["city_filter"])]
    if pollutant in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df[pollutant] >= low) & (filtered_df[pollutant] <= high)]
    if show_only_outliers:
        outliers = ht.detect_outliers()
        ids_outliers = {rec.get("_id") for rec, info in outliers if pollutant in info}
        filtered_df = filtered_df[filtered_df["_id"].isin(ids_outliers)]
    if "timestamp" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("timestamp", ascending=False)
    return filtered_df

with table_placeholder.container():
    st.subheader("Lecturas de calidad del aire")
    filtered_df = render_filtered_table()
    st.dataframe(filtered_df, use_container_width=True, height=400)

st.markdown("---")

# ---------------------------------------------------------
# PREDICCIONES (W1) - usar botón que sólo cambia session_state
# ---------------------------------------------------------
pred_placeholder = st.empty()
with pred_placeholder.container():
    st.subheader("Predicciones y alertas")
    if st.button("Ejecutar predicciones y generar alertas", key="btn_preds"):
        st.session_state["run_predictions"] = True

    if st.session_state.get("run_predictions"):
        with st.spinner("Ejecutando predicciones..."):
            alerts = cached_run_predictions(file_mtime)
        if alerts:
            st.write("Alertas generadas:")
            st.dataframe(pd.DataFrame(alerts), use_container_width=True, height=250)
        else:
            st.write("No se generaron alertas con las condiciones actuales.")

st.markdown("---")

# ---------------------------------------------------------
# W2 - botón + placeholder
# ---------------------------------------------------------
w2_placeholder = st.empty()
with w2_placeholder.container():
    st.subheader("Análisis probabilístico W2")
    if st.button("Ejecutar análisis avanzado (W2)", key="btn_w2"):
        st.session_state["run_w2"] = True

    if st.session_state.get("run_w2"):
        with st.spinner("Ejecutando W2..."):
            w2 = cached_run_w2(file_mtime)
        st.write(f"Estimación de eventos distintos: {w2.get('distinct_events_estimate', 0)}")

        moment1 = w2.get("moment1_by_city", {})
        if moment1:
            rows = []
            for city, info in moment1.items():
                if isinstance(info, dict):
                    crit = info.get("count_critical") or info.get("critical") or info.get("frequency") or info.get("moment1") or 0
                else:
                    crit = info
                rows.append({"Ciudad": city, "Alertas críticas": crit})
            st.write("Frecuencia de alertas críticas por ciudad (momento 1):")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=250)

        top = w2.get("top_zones", [])
        if top:
            rows = []
            for item in top:
                if isinstance(item, (tuple, list)):
                    rows.append({"Zona": item[0], "Frecuencia crítica": item[1]})
                elif isinstance(item, dict):
                    rows.append({"Zona": item.get("zone"), "Frecuencia crítica": item.get("frequency")})
            st.write("Zonas con mayor frecuencia de eventos críticos:")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=250)

        st.write("DGIM - estimación últimas 100 posiciones:")
        st.write(f"Aproximado: {w2.get('dgim_last_100')} | Exacto: {w2.get('dgim_exact_last_100')}")
        st.write("Tendencia DGIM:")
        st.write(w2.get("dgim_trend"))
        st.write("Predicción próxima ventana:")
        st.write(w2.get("dgim_prediction"))
        st.write("Resumen de muestreo adaptativo:")
        st.json(w2.get("sampling_stats", {}))

st.markdown("---")

# ---------------------------------------------------------
# W3 - Markov (botón + placeholder + gráfico cacheado)
# ---------------------------------------------------------
w3_placeholder = st.empty()
with w3_placeholder.container():
    st.subheader("Análisis predictivo W3 (Modelos de Markov)")
    if st.button("Ejecutar análisis W3", key="btn_w3"):
        st.session_state["run_w3"] = True

    if st.session_state.get("run_w3"):
        with st.spinner("Ejecutando W3..."):
            w3 = cached_run_w3(file_mtime)
        st.write("Matriz de transición:")
        st.json(w3.get("transition_matrix"))
        st.write("Secuencia de estados PM2.5:")
        st.write(w3.get("states_sequence"))
        st.write("Predicción del próximo estado PM2.5:")
        st.write(w3.get("predicted_state"))
        st.write("Probabilidades espaciales por ciudad:")
        st.json(w3.get("spatial_probabilities"))

        st.write("Diagrama de la cadena de Markov:")
        try:
            fig = plot_markov_chain(w3.get("states"), w3.get("transition_matrix"))
            st.pyplot(fig)
        except Exception as e:
            st.error(f"No se pudo generar el gráfico: {e}")

st.markdown("---")

# ---------------------------------------------------------
# W4 - MapReduce (usar formulario para parámetros)
# ---------------------------------------------------------
w4_placeholder = st.empty()
with w4_placeholder.container():
    st.subheader("Análisis avanzado W4 (MapReduce)")
    with st.form("w4_form"):
        st.write("Escoge la cantidad de workers para el Map Reduce Paralelo")
        m = st.number_input("Map Workers", value=3, min_value=1, max_value=15)
        r = st.number_input("Reduce Workers", value=3, min_value=1, max_value=15)
        s = st.number_input("Shuffle Workers", value=3, min_value=1, max_value=15)
        submit_w4 = st.form_submit_button("Ejecutar W4")
        if submit_w4:
            st.session_state["run_w4"] = True
            st.session_state["w4_workers"] = (m, r, s)

    if st.session_state.get("run_w4"):
        workers_config = st.session_state.get("w4_workers", (3, 3, 3))
        with st.spinner("Ejecutando W4..."):
            w4_data = cached_run_w4(file_mtime, workers_config)

        st.write("Drones por ciudad")
        counts = w4_data.get("counts")
        if isinstance(counts, (list, dict, pd.DataFrame)):
            st.dataframe(counts, use_container_width=True, height=250)
        else:
            st.write(counts)

        st.write("Promedio de AQI por ciudad")
        averages = w4_data.get("averages")
        if isinstance(averages, (list, dict, pd.DataFrame)):
            st.dataframe(averages, use_container_width=True, height=250)
        else:
            st.write(averages)

        col1, col2 = st.columns(2)
        max_city = w4_data.get("max_city", ("N/D", None))
        min_city = w4_data.get("min_city", ("N/D", None))
        with col1:
            st.subheader("Mayor")
            st.metric(max_city[0], f"{max_city[1]:.2f}" if (max_city[1] is not None) else "N/D")
        with col2:
            st.subheader("Menor")
            st.metric(min_city[0], f"{min_city[1]:.2f}" if (min_city[1] is not None) else "N/D")

        st.write("Gráfico de barras (Comparación Map Reduce Serial y Paralelo):")
        chart_data = w4_data.get("chart_data")
        if chart_data is not None:
            try:
                st.bar_chart(chart_data)
            except Exception:
                st.write(chart_data)

st.markdown("---")

# ---------------------------------------------------------
# W6 - KNN (selects + ejecución controlada)
# ---------------------------------------------------------
w6_placeholder = st.empty()
with w6_placeholder.container():
    st.subheader("Análisis avanzado W6 (KNN)")
    city = st.selectbox("Selecciona una ciudad a analizar", all_cities, index=0 if all_cities else None)
    df_knn = df[df["sensorLocation"].isin([city])] if not df.empty else pd.DataFrame()
    sensor_choices = list(df_knn["_id"].unique()) if not df_knn.empty else []
    _id = st.selectbox(f"Selecciona uno de los sensores de la ciudad {city}", sensor_choices, index=0 if sensor_choices else None)
    k = st.number_input("Hallar k vecinos", value=3, min_value=1, max_value=15)

    if st.button("Ejecutar W6 (KNN)", key="btn_w6"):
        st.session_state["run_w6"] = True
        st.session_state["w6_params"] = {"_id": _id, "k": k}

    if st.session_state.get("run_w6"):
        params = st.session_state.get("w6_params", {"_id": _id, "k": k})
        if not params.get("_id"):
            st.warning("Selecciona un sensor válido para ejecutar KNN.")
        else:
            with st.spinner("Ejecutando W6..."):
                w6_data = cached_run_w6(file_mtime, params["_id"], params["k"])

            st.write("KNN Pollution")
            knn_pollution = w6_data.get("simple_knn")
            st.dataframe(knn_pollution, use_container_width=True, height=250)

            st.write("KNN Geographic")
            geographic_knn = w6_data.get("geographic_knn")
            st.dataframe(geographic_knn, use_container_width=True, height=250)

            st.write("KNN Alert")
            alert_knn = w6_data.get("alert_knn")
            st.dataframe(alert_knn, use_container_width=True, height=250)

            st.write("KNN Performance")
            perf = w6_data.get("perf")
            st.json(perf)

            st.write("Similarity by geographic distance")
            geo_effect = w6_data.get("geo_effect")
            st.json(geo_effect)

# ---------------------------------------------------------
# Footer: reset botones (opcional)
# ---------------------------------------------------------
st.markdown("---")
reset_col1, reset_col2 = st.columns(2)
with reset_col1:
    if st.button("Reset W2/W3/W4/W6 flags"):
        st.session_state["run_w2"] = False
        st.session_state["run_w3"] = False
        st.session_state["run_w4"] = False
        st.session_state["run_w6"] = False
        st.session_state["run_predictions"] = False
with reset_col2:
    if st.button("Re-cargar datos (invalidate cache)"):
        # invalidar cache forzado: cambiando last_modified param isn't possible here —
        # simpler approach: clear cache (only available on some Streamlit versions)
        try:
            st.cache_data.clear()
        except Exception:
            # fallback: instruct user to restart app if clear not available
            st.warning("No se pudo invalidar programáticamente la cache. Reinicia la app si necesitas recargar datos.")
        st.experimental_rerun()
