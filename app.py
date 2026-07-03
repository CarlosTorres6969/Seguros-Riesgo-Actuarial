import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

st.set_page_config(
    page_title="Riesgo Actuarial - KMeans",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "models/kmeans_riesgo_actuarial.pkl"
METADATA_PATH = "models/model_metadata.json"
CLUSTER_DATA_PATH = "outputs/insurance_con_clusters.csv"

MAPA_RIESGO = {1: "Bajo", 2: "Medio", 0: "Alto"}
COLORES = {"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_cluster_data():
    return pd.read_csv(CLUSTER_DATA_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

modelo = load_model()
df_clusters = load_cluster_data()
metadata = load_metadata()

st.title("🛡️ Clasificador de Riesgo Actuarial")
st.markdown("""
Esta aplicación predice el **nivel de riesgo actuarial** de un cliente utilizando un modelo 
**K-Means** entrenado sobre datos históricos de seguros médicos.
""")

with st.sidebar:
    st.header("📋 Datos del Cliente")

    age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
    sex = st.selectbox("Sexo", ["male", "female"], format_func=lambda x: "Masculino" if x == "male" else "Femenino")
    bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, value=26.0, step=0.1, format="%.1f")
    children = st.number_input("Número de hijos", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Fumador", ["no", "yes"], format_func=lambda x: "No" if x == "no" else "Sí")
    region = st.selectbox(
        "Región",
        ["southwest", "southeast", "northwest", "northeast"],
        format_func=lambda x: x.capitalize()
    )
    charges = st.number_input(
        "Cargos médicos (USD)",
        min_value=100.0, max_value=100000.0, value=5000.0, step=100.0, format="%.2f"
    )

    predecir = st.button("🔍 Predecir Riesgo", type="primary", use_container_width=True)

if predecir:
    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    cluster = int(modelo.predict(cliente)[0])
    riesgo = MAPA_RIESGO[cluster]

    silueta = metadata.get("silhouette_score", "N/A")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("📌 Cluster Asignado")
        st.markdown(
            f"<div style='background:#f0f2f6;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h1 style='color:#4A90E2;margin:0;'>Cluster {cluster}</h1>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("⚖️ Nivel de Riesgo")
        color = COLORES[riesgo]
        st.markdown(
            f"<div style='background:{color}20;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h1 style='color:{color};margin:0;'>{riesgo}</h1>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col3:
        st.subheader("📊 Confianza del Modelo")
        st.markdown(
            f"<div style='background:#f0f2f6;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='margin:0;'>Silhouette: {silueta}</h2>"
            f"<p style='margin:5px 0 0;color:#666;'>n_clusters = {metadata.get('n_clusters', 3)}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    if riesgo == "Bajo":
        explicacion = (
            "Este perfil presenta **cargos médicos por debajo del promedio general**. "
            "Suele corresponder a personas **no fumadoras** con BMI moderado y sin condiciones de alto costo. "
            "El riesgo actuarial es **bajo**."
        )
    elif riesgo == "Medio":
        explicacion = (
            "Este perfil se encuentra en un **rango intermedio de cargos médicos**. "
            "Puede incluir personas de **mayor edad** o con **BMI elevado**, "
            "pero sin el factor fumador activo. El riesgo actuarial es **medio**."
        )
    else:
        explicacion = (
            "Este perfil presenta los **cargos médicos más altos**, asociados fuertemente con "
            "**personas fumadoras**. El riesgo actuarial es **alto** y representa el grupo de "
            "mayor costo para la aseguradora."
        )

    st.subheader("📝 Explicación del Resultado")
    st.info(explicacion)

    st.markdown("---")
    st.subheader("📊 Visualización de Clusters")

    tab1, tab2 = st.tabs(["Cargos vs Edad", "Distribución de Clusters"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = sns.scatterplot(
            data=df_clusters,
            x="age", y="charges",
            hue="riesgo_actuarial",
            palette=COLORES,
            alpha=0.6, s=30, ax=ax
        )
        ax.scatter(age, charges, c="black", s=150, marker="X", edgecolors="white",
                   linewidth=2, zorder=5, label="Tu cliente")
        ax.set_title("Cargos Médicos vs Edad por Nivel de Riesgo", fontsize=14)
        ax.set_xlabel("Edad")
        ax.set_ylabel("Cargos Médicos (USD)")
        ax.legend(title="Riesgo", bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)

    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            counts = df_clusters["riesgo_actuarial"].value_counts()
            ax2.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=[COLORES[c] for c in counts.index],
                startangle=90
            )
            ax2.set_title("Distribución de Clientes por Riesgo")
            st.pyplot(fig2)

        with col_b:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            promedios = df_clusters.groupby("riesgo_actuarial")["charges"].mean().reindex(["Bajo", "Medio", "Alto"])
            ax3.bar(promedios.index, promedios.values, color=[COLORES[c] for c in promedios.index])
            ax3.set_title("Cargos Promedio por Nivel de Riesgo")
            ax3.set_ylabel("Cargos Promedio (USD)")
            ax3.ticklabel_format(style="plain", axis="y")
            st.pyplot(fig3)

else:
    st.info("👈 Ingresa los datos del cliente en el panel lateral y presiona **Predecir Riesgo**.")

    st.markdown("---")
    st.subheader("📈 Vista General del Modelo")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        scatter = sns.scatterplot(
            data=df_clusters,
            x="age", y="charges",
            hue="riesgo_actuarial",
            palette=COLORES,
            alpha=0.6, s=20, ax=ax
        )
        ax.set_title("Distribución de Clusters (Cargos vs Edad)")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Cargos Médicos (USD)")
        ax.legend(title="Riesgo")
        st.pyplot(fig)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        counts = df_clusters["riesgo_actuarial"].value_counts()
        ax2.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[COLORES[c] for c in counts.index],
            startangle=90
        )
        ax2.set_title("Distribución por Nivel de Riesgo")
        st.pyplot(fig2)
