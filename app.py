import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar la página de Streamlit
st.set_page_config(page_title="Big Five Personality Predictor", layout="centered")

# Título de la aplicación
st.title("🧠 Predicción de Personalidad - Big Five (OCEAN)")
st.write("¡Ingresa tus puntuaciones para descubrir tu clúster de personalidad, compararte con famosos y visualizar tu perfil!")

# Cargar datos originales para mostrar a los famosos
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df_data = pd.read_csv(url)
    # Mapear profesiones para mejor visualización
    profesiones = {
        1: 'Actor/actriz', 2: 'Cantante', 3: 'Modelo', 4: 'Tv, series',
        5: 'Radio', 6: 'Tecnología', 7: 'Deportes', 8: 'Politica', 9: 'Escritor'
    }
    df_data['profesion'] = df_data['categoria'].map(profesiones)
    return df_data

df_original = load_data()

# Cargar modelo y escalador
@st.cache_resource
def load_model_and_scaler():
    kmeans = joblib.load('kmeans_bigFive_model.joblib')
    scaler = joblib.load('scaler_bigFive.joblib')
    return kmeans, scaler

try:
    kmeans, scaler = load_model_and_scaler()
except Exception as e:
    st.error("Error al cargar el modelo o el escalador. Asegúrate de tener 'kmeans_bigFive_model.joblib' y 'scaler_bigFive.joblib' en el mismo directorio.")
    st.stop()

# Configurar barra lateral con valores limpios por defecto (0.0)
st.sidebar.header("Tus Puntuaciones (0 - 100)")
op = st.sidebar.slider("Apertura (op)", 0.0, 100.0, 80.9, 0.1)
co = st.sidebar.slider("Conciencia (co)", 0.0, 100.0, 90.2, 0.1)
ex = st.sidebar.slider("Extroversión (ex)", 0.0, 100.0, 90.4, 0.1)
ag = st.sidebar.slider("Amabilidad (ag)", 0.0, 100.0, 90.1, 0.1)
ne = st.sidebar.slider("Neuroticismo (ne)", 0.0, 100.0, 20.0, 0.1)

# Botón para calcular
if st.sidebar.button("Analizar Mi Personalidad"):
    # Preparar los datos de entrada en el mismo orden exacto
    input_data = pd.DataFrame([[op, co, ex, ag, ne]], columns=['op', 'co', 'ex', 'ag', 'ne'])

    # Escalar y Predecir usando el escalador cargado
    input_scaled = scaler.transform(input_data)
    pred_cluster = kmeans.predict(input_scaled)[0]

    # Asignar clúster en los datos de famosos para visualización
    df_original_scaled = scaler.transform(df_original[['op', 'co', 'ex', 'ag', 'ne']])
    df_original['cluster'] = kmeans.predict(df_original_scaled)

    # Mostrar el Clúster Asignado con emojis
    st.balloons()
    st.success(f"🎉 ¡Felicidades! Has sido clasificado en el **Clúster: {pred_cluster}** 🧠✨")

    # Obtener famosos de ese clúster
    famosos_cluster = df_original[df_original['cluster'] == pred_cluster]
    total_famosos = len(famosos_cluster)

    # Mostrar Famosos con el total acumulado y limitado exactamente al TOP 5
    st.subheader(f"🌟 Top 5 Famosos en tu mismo Clúster {pred_cluster} (Total: {total_famosos} famosos)")
    if not famosos_cluster.empty:
        st.dataframe(famosos_cluster[['usuario', 'profesion', 'op', 'co', 'ex', 'ag', 'ne']].head(5))
    else:
        st.write("No se encontraron famosos en este clúster.")

    # Gráfico de Radar de perfil promedio de personalidad
    st.subheader("📊 Perfil de Personalidad (Radar): Tu Perfil vs. Promedio del Clúster")

    categories = ['Apertura (op)', 'Conciencia (co)', 'Extroversión (ex)', 'Amabilidad (ag)', 'Neuroticismo (ne)']

    # Promedio del clúster
    cluster_avg_series = famosos_cluster[['op', 'co', 'ex', 'ag', 'ne']].mean()
    cluster_avg = cluster_avg_series.values.tolist()
    cluster_avg += cluster_avg[:1]

    # Valores del usuario
    user_vals = [op, co, ex, ag, ne]
    user_vals += user_vals[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Dibujar promedio de clúster
    ax.plot(angles, cluster_avg, color='#4ba3a5', linewidth=2, label=f'Promedio Clúster {pred_cluster}')
    ax.fill(angles, cluster_avg, color='#4ba3a5', alpha=0.25)

    # Dibujar perfil del usuario
    ax.plot(angles, user_vals, color='#ff5722', linewidth=2, linestyle='dashed', label='Tu Perfil Personal')
    ax.fill(angles, user_vals, color='#ff5722', alpha=0.15)

    # Configuración de ejes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')

    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=9)
    ax.set_ylim(0, 100)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)

    # Gráfica de Barras adicional de rasgos predominantes
    st.subheader(f"📊 Rasgos de Personalidad Predominantes del Clúster {pred_cluster}")

    promedios_df = pd.DataFrame({
        'Rasgo': ['Apertura (op)', 'Conciencia (co)', 'Extroversión (ex)', 'Amabilidad (ag)', 'Neuroticismo (ne)'],
        'Promedio': cluster_avg_series.values
    }).sort_values(by='Promedio', ascending=False)

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=promedios_df,
        x='Promedio',
        y='Rasgo',
        palette='viridis',
        hue='Rasgo',
        legend=False,
        ax=ax_bar
    )
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Puntuación Promedio', fontsize=11)
    ax_bar.set_ylabel('Rasgo', fontsize=11)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

    for idx, row in enumerate(promedios_df.itertuples()):
        ax_bar.text(row.Promedio + 1, idx, f"{row.Promedio:.2f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig_bar)
