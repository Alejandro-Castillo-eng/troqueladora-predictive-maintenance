"""
app.py
======
Tablero Andon / Lean de Monitoreo en Tiempo Real y Predicción de Fallas (RUL)
para Troqueladora Industrial.

Soporta:
- Dataset de Validación Out-of-Time: 'telemetria_troqueladora_test_2026.csv'
- Dataset Histórico Base: 'telemetria_troqueladora.csv'
- Bitácora de Paradas: 'log_fallas_mantenimiento.csv'
- Modelo Entrenado: 'modelo_rul.joblib'
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="Troqueladora Industrial - Tablero Andon & RUL",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    
    .andon-header {
        background: linear-gradient(90deg, #131c31 0%, #0d1424 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 26px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .andon-title { color: #f8fafc; font-size: 23px; font-weight: 800; margin: 0; }
    .andon-subtitle { color: #94a3b8; font-size: 13px; margin-top: 4px; }
    
    .status-banner-normal {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.25);
        margin-bottom: 20px;
    }
    .status-banner-warning {
        background: linear-gradient(135deg, #78350f 0%, #b45309 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
        margin-bottom: 20px;
    }
    .status-banner-critical {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 25px rgba(239, 68, 68, 0.5);
        margin-bottom: 20px;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .status-title { color: #ffffff; font-size: 26px; font-weight: 900; margin: 0; letter-spacing: 0.5px; }
    .status-desc { color: #e2e8f0; font-size: 14.5px; margin-top: 6px; }
    
    .rul-container {
        background: #151d2f;
        border: 1px solid #2b3a55;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .rul-label { color: #94a3b8; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; }
    .rul-value-normal { color: #34d399; font-size: 46px; font-weight: 900; margin: 6px 0; }
    .rul-value-warning { color: #fbbf24; font-size: 46px; font-weight: 900; margin: 6px 0; }
    .rul-value-critical { color: #f87171; font-size: 46px; font-weight: 900; margin: 6px 0; }
    .rul-subtext { color: #cbd5e1; font-size: 14px; }
    
    .kpi-card {
        background: #151d2f;
        border: 1px solid #2b3a55;
        border-radius: 10px;
        padding: 14px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-header { font-size: 12.5px; font-weight: 700; color: #94a3b8; margin-bottom: 6px; }
    .kpi-value { font-size: 26px; font-weight: 800; color: #f8fafc; margin: 4px 0; }
    .kpi-unit { font-size: 14px; color: #64748b; font-weight: 600; }
    .kpi-badge-normal { display: inline-block; background: #064e3b; color: #6ee7b7; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
    .kpi-badge-warning { display: inline-block; background: #78350f; color: #fcd34d; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
    .kpi-badge-critical { display: inline-block; background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
    .kpi-meta { font-size: 11.5px; color: #94a3b8; margin-top: 8px; border-top: 1px solid #2b3a55; padding-top: 6px; }
    
    .root-cause-box {
        background: #151d2f;
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .root-cause-warning { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.08); }
    .root-cause-critical { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.12); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo(path_modelo: str = "modelo_rul.joblib"):
    if not os.path.exists(path_modelo):
        return None
    try:
        paquete = joblib.load(path_modelo)
        return paquete
    except Exception as e:
        st.error(f"Error al cargar el modelo '{path_modelo}': {e}")
        return None


@st.cache_data
def cargar_telemetria_y_procesar(path_telemetria: str):
    if not os.path.exists(path_telemetria):
        return None
        
    df = pd.read_csv(path_telemetria)
    df['dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('dt').reset_index(drop=True)
    
    sensores = ["temp_motor_c", "presion_hidraulica_bar", "vibracion_rms_mms", "corriente_motor_a"]
    
    for s in sensores:
        df[f"{s}_mean_15"] = df[s].rolling(15, min_periods=1).mean()
        df[f"{s}_mean_60"] = df[s].rolling(60, min_periods=1).mean()
        df[f"{s}_std_60"] = df[s].rolling(60, min_periods=1).std().fillna(0.0)
        df[f"{s}_trend_60"] = df[s] - df[s].shift(60).bfill()
        df[f"{s}_trend_15"] = df[s] - df[s].shift(15).bfill()
        
    nom_vals = {
        "temp_motor_c": (52.5, 2.0),
        "presion_hidraulica_bar": (214.0, 3.0),
        "vibracion_rms_mms": (1.35, 0.25),
        "corriente_motor_a": (21.0, 1.2)
    }
    
    for s, (m, sd) in nom_vals.items():
        df[f"{s}_zscore_nom"] = (df[f"{s}_mean_60"] - m) / sd
        
    df["indice_estres_termomecanico"] = (
        np.maximum(0, df["temp_motor_c_zscore_nom"]) * 0.30 +
        np.maximum(0, df["vibracion_rms_mms_zscore_nom"]) * 0.35 +
        np.maximum(0, -df["presion_hidraulica_bar_zscore_nom"]) * 0.20 +
        np.maximum(0, df["corriente_motor_a_zscore_nom"]) * 0.15
    )
    
    return df


@st.cache_data
def cargar_bitacora_fallas(path_fallas: str = "log_fallas_mantenimiento.csv"):
    if not os.path.exists(path_fallas):
        return None
    try:
        return pd.read_csv(path_fallas)
    except Exception:
        return None


def evaluar_estado_y_causa_raiz(row: pd.Series, pred_rul: float, bundle: dict):
    temp_m60 = row["temp_motor_c_mean_60"]
    vib_m60 = row["vibracion_rms_mms_mean_60"]
    pres_m60 = row["presion_hidraulica_bar_mean_60"]
    corr_m60 = row["corriente_motor_a_mean_60"]
    estres = row["indice_estres_termomecanico"]
    
    # 1. Filtro de picos transitorios aislados (valor bruto alto pero media móvil normal)
    es_pico_aislado = False
    if (row["temp_motor_c"] > 60.0 and temp_m60 < 54.0) or \
       (row["vibracion_rms_mms"] > 2.5 and vib_m60 < 1.6) or \
       (row["corriente_motor_a"] > 25.0 and corr_m60 < 22.5):
        es_pico_aislado = True

    # 2. Criterios de desgaste sostenido
    es_critico = (
        temp_m60 >= 68.0 or
        vib_m60 >= 3.80 or
        pres_m60 <= 185.0 or
        corr_m60 >= 28.5 or
        estres >= 6.0 or
        pred_rul <= 1.0
    )
    
    es_advertencia = (
        not es_critico and (
            temp_m60 >= 57.5 or
            vib_m60 >= 2.20 or
            pres_m60 <= 204.0 or
            corr_m60 >= 23.5 or
            estres >= 1.5 or
            pred_rul <= 4.0
        )
    )
    
    if es_critico and not es_pico_aislado:
        estado = "CRITICO"
        color = "🔴"
        mensaje = "PARADA CRÍTICA INMINENTE"
    elif es_advertencia and not es_pico_aislado:
        estado = "ADVERTENCIA"
        color = "🟡"
        mensaje = "DEGRADACIÓN DETECTADA"
    else:
        estado = "NORMAL"
        color = "🟢"
        mensaje = "ESTADO: OPERACIÓN NORMAL"
        
    desviaciones = {
        "Temperatura Motor": (temp_m60 - 52.5) / 2.0,
        "Vibración RMS del Troquel": (vib_m60 - 1.35) / 0.25,
        "Presión Hidráulica": (214.0 - pres_m60) / 3.0,
        "Corriente Eléctrica": (corr_m60 - 21.0) / 1.2
    }
    
    sensor_critico, z_max = max(desviaciones.items(), key=lambda x: x[1])
    
    return {
        "estado": estado,
        "color": color,
        "mensaje": mensaje,
        "es_pico_aislado": es_pico_aislado,
        "sensor_critico": sensor_critico,
        "desviacion_z": z_max,
        "indice_estres": estres
    }


def main():
    bundle = cargar_modelo("modelo_rul.joblib")
    df_fallas = cargar_bitacora_fallas("log_fallas_mantenimiento.csv")
    
    if bundle is None:
        st.error("⚠️ No se encontró 'modelo_rul.joblib'. Ejecuta primero: `python train_model.py`")
        st.stop()

    # --- BARRA LATERAL: FUENTE DE DATOS Y CONTROL ---
    st.sidebar.image("https://img.icons8.com/fluency/96/pressure-gauge.png", width=64)
    st.sidebar.title("Tablero Andon 4.0")
    st.sidebar.markdown("**Línea de Estampado Automotriz - Prensa P-04**")
    st.sidebar.markdown("---")
    
    # Selector de Dataset
    st.sidebar.subheader("📁 Fuente de Telemetría")
    datasets_disponibles = {}
    if os.path.exists("telemetria_troqueladora_test_2026.csv"):
        datasets_disponibles["2026 (Dataset de Test / Validación)"] = "telemetria_troqueladora_test_2026.csv"
    if os.path.exists("telemetria_troqueladora.csv"):
        datasets_disponibles["2025 (Dataset Base / Entrenamiento)"] = "telemetria_troqueladora.csv"

    selected_dataset_label = st.sidebar.selectbox(
        "Año a Monitorear:",
        options=list(datasets_disponibles.keys()),
        index=0  # 2026 por defecto
    )
    selected_path = datasets_disponibles[selected_dataset_label]
    es_2026 = "2026" in selected_dataset_label
    
    df_tel = cargar_telemetria_y_procesar(selected_path)
    if df_tel is None:
        st.error(f"No se pudo cargar el archivo: {selected_path}")
        st.stop()
        
    total_filas = len(df_tel)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🕹️ Escenarios de Demostración Rápida")
    
    # Presets dinámicos según el año
    if es_2026:
        presets = {
            "normal": 1000,
            "pico": 2113,      # Vibración 3.89 mm/s transitoria en normal
            "desgaste": 33200, # Vibración 5.08 mm/s sostenida
            "critico": 81263   # Temp 75.15 °C, Vib 6.76 mm/s, RUL < 0.2d
        }
    else:
        presets = {
            "normal": 2000,
            "pico": 903,
            "desgaste": 15800,
            "critico": 16750
        }

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = presets["normal"]
        
    with col_btn1:
        if st.button("🟢 1. Normal", use_container_width=True, help="Máquina en régimen nominal óptimo."):
            st.session_state.selected_index = presets["normal"]
        if st.button("🟡 3. Desgaste", use_container_width=True, help="Inicio de degradación sostenida."):
            st.session_state.selected_index = presets["desgaste"]

    with col_btn2:
        if st.button("⚡ 2. Pico Ruido", use_container_width=True, help="Pico de ruido aislado que el filtro de media móvil absorbe sin disparar alarma."):
            st.session_state.selected_index = presets["pico"]
        if st.button("🔴 4. Crítico", use_container_width=True, help="Fase crítica inminente con alarma roja."):
            st.session_state.selected_index = presets["critico"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Línea Temporal de Operación")
    
    selected_idx = st.sidebar.slider(
        "Muestra temporal (Minuto a minuto):",
        min_value=0,
        max_value=total_filas - 1,
        value=min(total_filas - 1, st.session_state.selected_index),
        step=1
    )
    st.session_state.selected_index = selected_idx
    
    c_sub1, c_sub2, c_sub3 = st.sidebar.columns(3)
    with c_sub1:
        if st.button("◀ -1 Hora"):
            st.session_state.selected_index = max(0, selected_idx - 60)
            st.rerun()
    with c_sub2:
        if st.button("+1 Hora ▶"):
            st.session_state.selected_index = min(total_filas - 1, selected_idx + 60)
            st.rerun()
    with c_sub3:
        if st.button("+1 Día ⏩"):
            st.session_state.selected_index = min(total_filas - 1, selected_idx + 600)
            st.rerun()

    current_row = df_tel.iloc[selected_idx]
    current_time_str = current_row["dt"].strftime("%Y-%m-%d %H:%M")
    st.sidebar.info(f"📅 **Marca de tiempo:** `{current_time_str}`\n\n📌 **Muestra:** `{selected_idx:,}` / `{total_filas:,}`\n\n📁 **Archivo:** `{selected_path}`")
    
    # Inferencia con el modelo entrenado
    feature_vals = df_tel.loc[[selected_idx], bundle["feature_cols"]]
    pred_rul = float(bundle["modelo"].predict(feature_vals)[0])
    pred_rul = np.clip(pred_rul, 0.0, bundle.get("max_rul_dias", 10.0))
    
    # Evaluación con la lógica de 2 etapas
    diag = evaluar_estado_y_causa_raiz(current_row, pred_rul, bundle)
    
    # --- ENCABEZADO ANDON ---
    badge_year = "TEST 2026 (VALIDACIÓN OUT-OF-TIME)" if es_2026 else "BASE 2025 (ENTRENAMIENTO)"
    badge_year_color = "#38bdf8" if es_2026 else "#94a3b8"
    
    st.markdown(f"""
    <div class="andon-header">
        <div>
            <h1 class="andon-title">🏭 TABLERO ANDON - SISTEMA PREDICTIVO RUL</h1>
            <div class="andon-subtitle">Prensa Troqueladora Hidráulica 500T | Estación de Estampado Automotriz</div>
        </div>
        <div style="text-align: right;">
            <div style="background: rgba(56, 189, 248, 0.15); color: {badge_year_color}; padding: 3px 10px; border-radius: 6px; font-weight: 800; font-size: 12px; display: inline-block; margin-bottom: 4px;">DATASET: {badge_year}</div>
            <div style="color: #94a3b8; font-size: 13px;">Fecha / Hora: <b>{current_time_str}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- SEMÁFORO DE ESTADO ---
    if diag["estado"] == "NORMAL":
        st.markdown(f"""
        <div class="status-banner-normal">
            <h2 class="status-title">🟢 {diag['mensaje']}</h2>
            <div class="status-desc">Todos los sensores se encuentran operando dentro de la ventana nominal. Equipo estable.</div>
        </div>
        """, unsafe_allow_html=True)
    elif diag["estado"] == "ADVERTENCIA":
        st.markdown(f"""
        <div class="status-banner-warning">
            <h2 class="status-title">🟡 {diag['mensaje']} (ALERTA PREVENTIVA)</h2>
            <div class="status-desc">Tendencia sostenida de desgaste termomecánico detectada. Planificar inspección antes de alcanzar el umbral crítico.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner-critical">
            <h2 class="status-title">🔴 {diag['mensaje']}</h2>
            <div class="status-desc">¡PELIGRO DE AVERÍA SEVERA! Condición crítica de desgaste detectada. Intervención correctiva inmediata requerida.</div>
        </div>
        """, unsafe_allow_html=True)
        
    # --- MÉTRICA GIGANTE RUL + SALUD ---
    col_rul, col_health = st.columns([2, 1])
    
    with col_rul:
        if diag["estado"] == "NORMAL":
            st.markdown("""
            <div class="rul-container">
                <div class="rul-label">Vida Útil Remanente Estimada (RUL)</div>
                <div class="rul-value-normal">> 10.0 DÍAS / RÉGIMEN ÓPTIMO</div>
                <div class="rul-subtext">Operación estable sin desgaste sostenido. No se proyecta parada no programada a corto plazo.</div>
            </div>
            """, unsafe_allow_html=True)
        elif diag["estado"] == "ADVERTENCIA":
            horas_op = pred_rul * 10.0
            st.markdown(f"""
            <div class="rul-container">
                <div class="rul-label">Vida Útil Remanente Estimada (RUL) - Cuenta Regresiva</div>
                <div class="rul-value-warning">FALTAN {pred_rul:.1f} DÍAS PARA LA FALLA</div>
                <div class="rul-subtext">Equivalente a aproximadamente <b>{horas_op:.1f} horas operativas</b> restantes antes del umbral de parada.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            horas_op = pred_rul * 10.0
            st.markdown(f"""
            <div class="rul-container">
                <div class="rul-label">Vida Útil Remanente Estimada (RUL) - Crítico</div>
                <div class="rul-value-critical">🚨 FALTAN {pred_rul:.2f} DÍAS ({horas_op:.1f} HORAS)</div>
                <div class="rul-subtext">Desgaste acelerado crítico. Detener troqueladora de forma segura al terminar el ciclo actual.</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_health:
        estres = diag["indice_estres"]
        health_index = max(5.0, min(100.0, 100.0 - (estres * 8.5)))
        if diag["estado"] == "NORMAL":
            health_color = "#10b981"
            health_status = "Excelente (98-100%)"
        elif diag["estado"] == "ADVERTENCIA":
            health_color = "#f59e0b"
            health_status = "Degradado (60-80%)"
        else:
            health_color = "#ef4444"
            health_status = "Crítico (< 40%)"
            
        st.markdown(f"""
        <div class="rul-container" style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
            <div class="rul-label">Índice de Salud de Máquina</div>
            <div style="font-size: 44px; font-weight: 900; color: {health_color}; margin: 8px 0;">{health_index:.1f}%</div>
            <div style="color: #94a3b8; font-size: 13px;">Condición actual: <b>{health_status}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # --- DIAGNÓSTICO DE CAUSA RAÍZ ---
    if diag["estado"] == "NORMAL":
        if diag["es_pico_aislado"]:
            st.markdown("""
            <div class="root-cause-box" style="border-left-color: #10b981;">
                <b style="color: #10b981; font-size: 15px;">🛡️ FILTRO DE RUIDO TRANSITORIO ACTIVO:</b><br>
                <span style="color: #cbd5e1; font-size: 13.5px;">
                Se detectó una fluctuación transitoria en una muestra aislada, pero las <b>medias móviles a 15 y 60 min permanecen en rango nominal</b>. 
                El sistema descarta la perturbación como ruido y evita falsas alarmas operativas.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="root-cause-box" style="border-left-color: #10b981;">
                <b style="color: #10b981; font-size: 15px;">✓ DIAGNÓSTICO DE CONDICIÓN:</b><br>
                <span style="color: #cbd5e1; font-size: 13.5px;">
                Todos los subsistemas mecánicos, hidráulicos y eléctricos operan dentro de los parámetros de diseño. Desgaste acumulado despreciable.
                </span>
            </div>
            """, unsafe_allow_html=True)
    elif diag["estado"] == "ADVERTENCIA":
        st.markdown(f"""
        <div class="root-cause-box root-cause-warning">
            <b style="color: #f59e0b; font-size: 16px;">⚠️ DIAGNÓSTICO DE CAUSA RAÍZ (DESGASTE SOSTENIDO DETECTADO):</b><br>
            <span style="color: #e2e8f0; font-size: 14px;">
            El subsistema principal que origina la alerta es: <b>{diag['sensor_critico']}</b>.<br>
            • Recomendación Lean: Programar inspección preventiva del subsistema señalado para el cambio de turno.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="root-cause-box root-cause-critical">
            <b style="color: #ef4444; font-size: 16px;">🚨 DIAGNÓSTICO DE CAUSA RAÍZ - FALLA INMINENTE:</b><br>
            <span style="color: #fecaca; font-size: 14px;">
            Sobreesfuerzo severo liderado por: <b>{diag['sensor_critico']}</b> con sobrecalentamiento y desbalance mecánico.<br>
            • Acción Inmediata: Detención segura del ciclo de troquelado y convocatoria al equipo de mantenimiento electromecánico.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # --- 4 TARJETAS KPI ---
    st.markdown(f"### 📊 Telemetría de Sensores en Vivo ({'Año 2026' if es_2026 else 'Año 2025'})")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        val = current_row["temp_motor_c"]
        m60 = current_row["temp_motor_c_mean_60"]
        delta = val - 52.5
        badge_cls = "kpi-badge-normal" if m60 < 58.0 else ("kpi-badge-warning" if m60 < 68.0 else "kpi-badge-critical")
        badge_txt = "NOMINAL" if m60 < 58.0 else ("ELEVADA" if m60 < 68.0 else "CRÍTICA")
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-header">🌡️ TEMPERATURA MOTOR</div>
                <div class="kpi-value">{val:.1f} <span class="kpi-unit">°C</span></div>
                <span class="{badge_cls}">{badge_txt}</span>
            </div>
            <div class="kpi-meta">
                Media 60 min: <b>{m60:.1f} °C</b><br>
                Nominal: 50.0 - 55.0 °C ({delta:+.1f} °C)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        val = current_row["presion_hidraulica_bar"]
        m60 = current_row["presion_hidraulica_bar_mean_60"]
        delta = val - 214.0
        badge_cls = "kpi-badge-normal" if m60 > 204.0 else ("kpi-badge-warning" if m60 > 185.0 else "kpi-badge-critical")
        badge_txt = "NOMINAL" if m60 > 204.0 else ("BAJA" if m60 > 185.0 else "CRÍTICA")
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-header">💧 PRESIÓN HIDRÁULICA</div>
                <div class="kpi-value">{val:.1f} <span class="kpi-unit">bar</span></div>
                <span class="{badge_cls}">{badge_txt}</span>
            </div>
            <div class="kpi-meta">
                Media 60 min: <b>{m60:.1f} bar</b><br>
                Nominal: 210 - 220 bar ({delta:+.1f} bar)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        val = current_row["vibracion_rms_mms"]
        m60 = current_row["vibracion_rms_mms_mean_60"]
        delta = val - 1.35
        badge_cls = "kpi-badge-normal" if m60 < 2.20 else ("kpi-badge-warning" if m60 < 3.80 else "kpi-badge-critical")
        badge_txt = "NOMINAL" if m60 < 2.20 else ("ALERTA" if m60 < 3.80 else "CRÍTICA")
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-header">📳 VIBRACIÓN RMS</div>
                <div class="kpi-value">{val:.2f} <span class="kpi-unit">mm/s</span></div>
                <span class="{badge_cls}">{badge_txt}</span>
            </div>
            <div class="kpi-meta">
                Media 60 min: <b>{m60:.2f} mm/s</b><br>
                Nominal: 1.0 - 1.6 mm/s ({delta:+.2f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        val = current_row["corriente_motor_a"]
        m60 = current_row["corriente_motor_a_mean_60"]
        delta = val - 21.0
        badge_cls = "kpi-badge-normal" if m60 < 23.5 else ("kpi-badge-warning" if m60 < 28.5 else "kpi-badge-critical")
        badge_txt = "NOMINAL" if m60 < 23.5 else ("SOBRECARGA" if m60 < 28.5 else "CRÍTICA")
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-header">⚡ CORRIENTE MOTOR</div>
                <div class="kpi-value">{val:.1f} <span class="kpi-unit">A</span></div>
                <span class="{badge_cls}">{badge_txt}</span>
            </div>
            <div class="kpi-meta">
                Media 60 min: <b>{m60:.1f} A</b><br>
                Nominal: 20.0 - 22.5 A ({delta:+.1f} A)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- PESTAÑAS: TENDENCIAS Y BITÁCORA ---
    tab_graficos, tab_fallas, tab_modelo = st.tabs([
        "📈 Evolución Temporal y Tendencia Anual",
        "📋 Bitácora Histórica de Paradas",
        "🧠 Validación del Modelo ML"
    ])
    
    with tab_graficos:
        st.subheader(f"Evolución Temporal Anual (Primer a Último Registro: {'Año 2026' if es_2026 else 'Año 2025'})")
        ts_primero = df_tel['dt'].iloc[0].strftime('%Y-%m-%d %H:%M')
        ts_ultimo = df_tel['dt'].iloc[-1].strftime('%Y-%m-%d %H:%M')
        st.caption(f"Mostrando la serie temporal completa del año: desde el **primer registro ({ts_primero})** hasta el **último registro ({ts_ultimo})**. Total lecturas: **{total_filas:,}**.")
        
        col_graf_s1, col_graf_s2 = st.columns([3, 2])
        with col_graf_s1:
            sensor_seleccionado = st.selectbox(
                "Selecciona sensor para graficar:",
                options=["vibracion_rms_mms", "temp_motor_c", "presion_hidraulica_bar", "corriente_motor_a"],
                format_func=lambda x: {
                    "temp_motor_c": "Temperatura Motor (°C)",
                    "vibracion_rms_mms": "Vibración RMS (mm/s)",
                    "presion_hidraulica_bar": "Presión Hidráulica (bar)",
                    "corriente_motor_a": "Corriente Motor (A)"
                }[x]
            )
        with col_graf_s2:
            modo_lapso = st.selectbox(
                "Secuencia temporal en gráfica:",
                options=[
                    "⏱️ Secuencia por Toma de Datos (Fila a Fila del Sensor)",
                    "🔍 Ventana Detallada (±24 Horas)",
                    "📅 Panorama Anual (Primer al Último Registro)"
                ],
                index=0
            )

        if modo_lapso.startswith("⏱️ Secuencia"):
            start_w = max(0, selected_idx - 30)
            end_w = min(total_filas, selected_idx + 30)
            sub_df = df_tel.iloc[start_w:end_w].copy()
            x_axis_title = "Secuencia por Toma de Datos (+1 fila creada por cada lectura del sensor)"
        elif modo_lapso.startswith("📅 Año Completo"):
            # Submuestreo acelerado por WebGL garantizando estrictamente el primer y último registro
            n_puntos = min(6000, total_filas)
            indices_anuales = np.linspace(0, total_filas - 1, n_puntos, dtype=int)
            indices_anuales[0] = 0
            indices_anuales[-1] = total_filas - 1
            sub_df = df_tel.iloc[indices_anuales].copy()
            x_axis_title = "Línea de Tiempo Anual (Desliza los controles del rango inferior para hacer zoom)"
        else:
            start_w = max(0, selected_idx - 1440)
            end_w = min(total_filas, selected_idx + 1440)
            sub_df = df_tel.iloc[start_w:end_w].copy()
            x_axis_title = "Ventana de 24 Horas (Lecturas Minuto a Minuto)"
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Scattergl(
                x=sub_df["dt"],
                y=sub_df[sensor_seleccionado],
                mode="lines",
                name="Lectura Instantánea (con ruido)",
                line=dict(color="#64748b", width=1.0)
            ))
            fig.add_trace(go.Scattergl(
                x=sub_df["dt"],
                y=sub_df[f"{sensor_seleccionado}_mean_15"],
                mode="lines",
                name="Media Móvil 15 min (Filtro Suave)",
                line=dict(color="#38bdf8", width=1.8)
            ))
            fig.add_trace(go.Scattergl(
                x=sub_df["dt"],
                y=sub_df[f"{sensor_seleccionado}_mean_60"],
                mode="lines",
                name="Media Móvil 60 min (Tendencia de Desgaste)",
                line=dict(color="#f59e0b", width=2.8)
            ))
            fig.add_trace(go.Scatter(
                x=[current_row["dt"]],
                y=[current_row[sensor_seleccionado]],
                mode="markers",
                name="Muestra Actual Seleccionada",
                marker=dict(
                    size=16,
                    color="#ef4444" if diag["estado"] == "CRITICO" else ("#f59e0b" if diag["estado"] == "ADVERTENCIA" else "#10b981"),
                    line=dict(color="#ffffff", width=2.5),
                    symbol="circle"
                )
            ))

            fig.add_vline(
                x=current_row["dt"],
                line_width=2,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"📍 Muestra Actual ({current_row['dt'].strftime('%b %d %H:%M')})",
                annotation_position="top left",
                annotation_font=dict(color="#38bdf8", size=11)
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=460,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(
                    title=x_axis_title,
                    gridcolor="#1e293b",
                    rangeslider=dict(visible=True if modo_lapso.startswith("📅 Año Completo") else False, thickness=0.08, bgcolor="#0f172a")
                ),
                yaxis=dict(title=f"Valor ({bundle['rangos_nominales'][sensor_seleccionado]['unidad']})", gridcolor="#1e293b")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart_data = sub_df.set_index("dt")[[sensor_seleccionado, f"{sensor_seleccionado}_mean_60"]]
            st.line_chart(chart_data)
            
        st.caption("ℹ️ La gráfica muestra cómo las medias móviles absorben los picos transitorios aislados y detectan subidas sostenidas de desgaste real a lo largo de todo el ciclo anual.")

    with tab_fallas:
        st.subheader("Registro de Paradas de Mantenimiento Históricas")
        if df_fallas is not None:
            st.markdown(f"Total de eventos de parada registrados en la bitácora: **{len(df_fallas)} paradas**.")
            st.dataframe(
                df_fallas,
                use_container_width=True,
                column_config={
                    "timestamp_falla": "Fecha/Hora de Parada",
                    "id_parada": "ID Evento",
                    "tipo_falla": "Tipo de Falla",
                    "componente_afectado": "Componente Afectado",
                    "duracion_parada_min": "Duración (min)",
                    "accion_correctiva": "Acción Correctiva Ejecutada",
                    "tecnico_responsable": "Técnico Responsable"
                },
                hide_index=True
            )
        else:
            st.warning("No se encontró el archivo 'log_fallas_mantenimiento.csv'.")

    with tab_modelo:
        st.subheader("Información del Modelo de Machine Learning")
        st.markdown(f"""
        - **Algoritmo Entrenado:** `{bundle.get('nombre_modelo', 'XGBRegressor')}`
        - **Características de Entrada:** `{len(bundle.get('feature_cols', []))}` variables móviles y de tendencia.
        - **Tope Máximo de RUL (Capping):** `{bundle.get('max_rul_dias', 10.0)} días` (régimen nominal).
        - **Dataset Actual en Pantalla:** `{selected_dataset_label}` ({len(df_tel):,} registros).
        - **Métricas de Evaluación en el Último 20 % de 2025:**
          * **MAE Global:** `{bundle['metricas'].get('mae_global', 0.0):.3f} días`
          * **RMSE Global:** `{bundle['metricas'].get('rmse_global', 0.0):.3f} días`
          * **MAE en Fase Crítica (RUL ≤ 3 días):** `{bundle['metricas'].get('mae_critico', 0.0):.3f} días` ({bundle['metricas'].get('mae_critico', 0.0) * 600:.1f} minutos operativos)
          * **Coeficiente R²:** `{bundle['metricas'].get('r2_global', 0.0):.4f}`
        """)


if __name__ == "__main__":
    main()
