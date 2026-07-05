import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import shap
import scipy.stats as stats
from collections import Counter
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predictor Mundial 2026", page_icon="🏆", layout="centered")

# --- CACHE DEL MODELO ---
# Usamos cache para que los modelos no se recarguen cada vez que pulsamos un botón
@st.cache_resource
def cargar_oraculo():
    ruta = "modelos_guardados"
    try:
        elo_dict = joblib.load(f"{ruta}/elo_dict.pkl")
        elo_history = joblib.load(f"{ruta}/elo_history.pkl")
        poisson_model = joblib.load(f"{ruta}/poisson_model.pkl")
        
        base_model = CatBoostClassifier()
        base_model.load_model(f"{ruta}/catboost_base.cbm")
        calibrated_model = joblib.load(f"{ruta}/calibrated_model.pkl")
        explainer = shap.TreeExplainer(base_model)
        
        market_values_m = {
            'France': 1520.0, 'England': 1360.0, 'Spain': 1220.0, 'Portugal': 1010.0,
            'Germany': 947.0, 'Brazil': 928.2, 'Argentina': 807.5, 'Netherlands': 754.2,
            'Norway': 589.9, 'Belgium': 547.5, 'Ivory Coast': 522.1, 'Senegal': 478.1,
            'Turkey': 473.7, 'Morocco': 447.7, 'Sweden': 409.08, 'Croatia': 387.3,
            'United States': 385.65, 'Ecuador': 368.7, 'Denmark': 365.0, 'Uruguay': 359.3,
            'Switzerland': 332.5, 'Italy': 319.0, 'Colombia': 302.35, 'Mexico': 191.85,
            'Japan': 270.85, 'Canada': 198.65, 'Senegal': 478.1, 'Serbia': 209.5,
            'Poland': 232.1, 'Australia': 77.45, 'Nigeria': 172.3, 'Egypt': 116.48,
            'Costa Rica': 28.2, 'Austria': 245.2, 'South Korea': 139.05, 'Ukraine': 247.0
        }
        return elo_dict, elo_history, poisson_model, base_model, calibrated_model, explainer, market_values_m
    except Exception as e:
        return None, str(e)

elo_dict, elo_history, poisson_model, base_model, calibrated_model, explainer, market_values_m = cargar_oraculo()

if elo_dict is None:
    st.error(f"❌ Error al cargar los modelos. ¿Subiste la carpeta 'modelos_guardados'? Detalle: {elo_history}")
    st.stop()

# --- FUNCIONES CORE ---
def obtener_vector_partido(equipo_a, equipo_b, es_mundial=True):
    elo_a = elo_dict.get(equipo_a, 1500)
    elo_b = elo_dict.get(equipo_b, 1500)
    mom_a = elo_a - elo_history.get(equipo_a, [1500]*5)[0]
    mom_b = elo_b - elo_history.get(equipo_b, [1500]*5)[0]
    mv_a = market_values_m.get(equipo_a, max(5, 10 ** ((elo_a - 1000) / 300)))
    mv_b = market_values_m.get(equipo_b, max(5, 10 ** ((elo_b - 1000) / 300)))
    
    try:
        lambda_a = poisson_model.predict(pd.DataFrame({'team': [equipo_a], 'opponent': [equipo_b], 'is_home': [0]})).values[0]
        mu_b = poisson_model.predict(pd.DataFrame({'team': [equipo_b], 'opponent': [equipo_a], 'is_home': [0]})).values[0]
    except:
        lambda_a, mu_b = 1.2, 1.2 
        
    match_data = pd.DataFrame([{
        'elo_home': elo_a, 'elo_away': elo_b, 'elo_diff': elo_a - elo_b,
        'poisson_lambda_home': lambda_a, 'poisson_mu_away': mu_b, 'xg_diff': lambda_a - mu_b,
        'momentum_home': mom_a, 'momentum_away': mom_b,
        'market_value_home': mv_a, 'market_value_away': mv_b,
        'tournament_weight': 2.0 if es_mundial else 1.0, 'neutral': True
    }])
    return match_data, lambda_a, mu_b, elo_a, elo_b

def simular_partido_eliminatoria(equipo_a, equipo_b):
    match_data, _, _, _, _ = obtener_vector_partido(equipo_a, equipo_b)
    probs = calibrated_model.predict_proba(match_data)[0]
    resultado = np.random.choice([0, 1, 2], p=[probs[0], probs[1], probs[2]])
    if resultado == 1: return equipo_a
    elif resultado == 2: return equipo_b
    else: return np.random.choice([equipo_a, equipo_b])

# --- INTERFAZ DE USUARIO ---
st.title("🏆 Predictor Mundial 2026")
st.markdown("Motor V3.0")

tab1, tab2 = st.tabs(["⚽ Predecir Partido", "🌍 Simular Mundial (Montecarlo)"])

equipos_disponibles = sorted(list(market_values_m.keys()) + ["Mali", "Tunisia", "Panama", "Chile", "Saudi Arabia"])

with tab1:
    st.header("Análisis de Cruce Directo")
    col1, col2 = st.columns(2)
    with col1: equipo_a = st.selectbox("Selecciona Equipo A", equipos_disponibles, index=equipos_disponibles.index("Spain") if "Spain" in equipos_disponibles else 0)
    with col2: equipo_b = st.selectbox("Selecciona Equipo B", equipos_disponibles, index=equipos_disponibles.index("Brazil") if "Brazil" in equipos_disponibles else 1)

    if st.button("🚀 Lanzar Predicción", use_container_width=True):
        if equipo_a == equipo_b:
            st.warning("Selecciona dos equipos distintos.")
        else:
            with st.spinner('Analizando variables...'):
                match_data, lambda_a, mu_b, elo_a, elo_b = obtener_vector_partido(equipo_a, equipo_b)
                probs = calibrated_model.predict_proba(match_data)[0]
                
                rho = -0.15 
                max_prob, score_a, score_b = 0, 0, 0
                for i in range(6):
                    for j in range(6):
                        p_base = stats.poisson.pmf(i, lambda_a) * stats.poisson.pmf(j, mu_b)
                        if i==0 and j==0: p_base *= (1 - (lambda_a*mu_b*rho))
                        elif i==0 and j==1: p_base *= (1 + (lambda_a*rho))
                        elif i==1 and j==0: p_base *= (1 + (mu_b*rho))
                        elif i==1 and j==1: p_base *= (1 - rho)
                        if p_base > max_prob: max_prob, score_a, score_b = p_base, i, j

                prob_over_2_5 = (1 - stats.poisson.cdf(2, lambda_a + mu_b)) * 100
                
                st.subheader(f"{equipo_a} {score_a} - {score_b} {equipo_b}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Victoria {equipo_a}", f"{probs[1]*100:.1f}%")
                c2.metric("Empate", f"{probs[0]*100:.1f}%")
                c3.metric(f"Victoria {equipo_b}", f"{probs[2]*100:.1f}%")
                
                st.markdown("---")
                st.markdown("### 📊 Estadísticas Subyacentes")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Goles Esperados (xG)", f"{lambda_a:.2f} - {mu_b:.2f}")
                sc2.metric("Probabilidad +2.5 Goles", f"{prob_over_2_5:.1f}%")
                sc3.metric("Valor Mercado (M€)", f"{match_data['market_value_home'][0]:.0f} - {match_data['market_value_away'][0]:.0f}")

with tab2:
    st.header("Simulador de Torneo")
    n_simulaciones = st.slider("Número de universos paralelos a simular", min_value=100, max_value=2000, value=500, step=100)
    
    if st.button("🧬 Ejecutar Motor Montecarlo", use_container_width=True):
        enfrentamientos_1_16 = [
            ("Argentina", "Canada"), ("Italy", "Switzerland"), ("Spain", "Egypt"), ("Croatia", "Ecuador"),
            ("Brazil", "Costa Rica"), ("Uruguay", "Austria"), ("Netherlands", "South Korea"), ("USA", "Senegal"),
            ("France", "Australia"), ("Belgium", "Denmark"), ("England", "Nigeria"), ("Colombia", "Ukraine"),
            ("Portugal", "Ivory Coast"), ("Mexico", "Serbia"), ("Germany", "Poland"), ("Morocco", "Japan")
        ]
        
        campeones = []
        barra_progreso = st.progress(0)
        texto_estado = st.empty()
        
        for i in range(n_simulaciones):
            octavos = [simular_partido_eliminatoria(a, b) for a, b in enfrentamientos_1_16]
            cuartos = [simular_partido_eliminatoria(octavos[j], octavos[j+1]) for j in range(0, 16, 2)]
            semis = [simular_partido_eliminatoria(cuartos[j], cuartos[j+1]) for j in range(0, 8, 2)]
            finalistas = [simular_partido_eliminatoria(semis[0], semis[1]), simular_partido_eliminatoria(semis[2], semis[3])]
            campeones.append(simular_partido_eliminatoria(finalistas[0], finalistas[1]))
            
            if (i+1) % 50 == 0:
                barra_progreso.progress((i+1)/n_simulaciones)
                texto_estado.text(f"Simulando universo {i+1} de {n_simulaciones}...")
        
        texto_estado.text("¡Simulación completada!")
        conteo = Counter(campeones)
        
        # Formatear datos para gráfico
        df_resultados = pd.DataFrame([{"Equipo": eq, "Probabilidad (%)": (vic/n_simulaciones)*100} for eq, vic in conteo.most_common(10)])
        st.bar_chart(df_resultados.set_index("Equipo"))
        st.dataframe(df_resultados, use_container_width=True)