"""
build_html.py
=============
Script constructor que genera 'index.html', un tablero web 100% autónomo (standalone)
listo para ser alojado en GitHub Pages o abierto con doble clic en cualquier navegador.
Incluye:
1. Botón para ver TODA LA SECUENCIA COMPLETA de inicio a fin (2026-01-01 a 2026-12-26).
2. Botones para los 4 escenarios de secuencias operativas específicas (80 tomas consecutivas cada una).
3. Modos de visualización: Secuencia Completa y Adquisición Progresiva dato a dato.
"""

import json
import os
import pandas as pd
import numpy as np
import joblib

def generate_telemetry_tomas():
    """Genera la secuencia completa de inicio a fin y los 4 escenarios específicos."""
    path_2026 = "telemetria_troqueladora_test_2026.csv"
    if not os.path.exists(path_2026):
        path_2026 = "telemetria_troqueladora.csv"
        
    df = pd.read_csv(path_2026)
    sensores = ["temp_motor_c", "presion_hidraulica_bar", "vibracion_rms_mms", "velocidad_cpm", "corriente_motor_a"]
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
        "velocidad_cpm": (42.0, 0.8),
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

    bundle = joblib.load("modelo_rul.joblib")
    model = bundle["modelo"]
    feature_cols = bundle["feature_cols"]

    # 1. Secuencia Completa de Inicio a Fin (600 tomas distribuidas uniformemente)
    N_COMPLETA = 600
    idx_completa = np.linspace(0, len(df) - 1, N_COMPLETA, dtype=int)
    idx_completa[0] = 0
    idx_completa[-1] = len(df) - 1
    df_completa = df.iloc[idx_completa].copy()

    # 2. Cuatro segmentos de 80 filas estrictamente consecutivas
    raw_segments = {
        "completa": df_completa,
        "normal": df.iloc[100:180].copy(),       # Jan 01 09:40 a 10:59
        "pico": df.iloc[2075:2155].copy(),        # Jan 04 12:35 a 13:54 (Pico en toma #38 / 13:12)
        "desgaste": df.iloc[3873:3953].copy(),    # Jan 07 12:33 a 13:52 (Desgaste sostenido)
        "critico": df.iloc[81220:81300].copy()    # May 20 09:25 a 10:44 (Falla crítica inminente RUL < 0.2d)
    }

    scenarios = {}
    for name, seg in raw_segments.items():
        preds = model.predict(seg[feature_cols])
        preds = np.clip(preds, 0.05, 10.0)
        seg["pred_rul"] = preds
        tomas = []
        for i, (_, r) in enumerate(seg.iterrows()):
            ts_str = str(r["timestamp"])
            tomas.append({
                "toma": i + 1,
                "t": ts_str[:19],
                "hm": ts_str[11:16],
                "date": ts_str[:10],
                "temp": round(float(r["temp_motor_c"]), 1),
                "pres": round(float(r["presion_hidraulica_bar"]), 1),
                "vib": round(float(r["vibracion_rms_mms"]), 2),
                "vel": round(float(r["velocidad_cpm"]), 1),
                "corr": round(float(r["corriente_motor_a"]), 1),
                "temp_m": round(float(r["temp_motor_c_mean_60"]), 1),
                "pres_m": round(float(r["presion_hidraulica_bar_mean_60"]), 1),
                "vib_m": round(float(r["vibracion_rms_mms_mean_60"]), 2),
                "vel_m": round(float(r["velocidad_cpm_mean_60"]), 1),
                "corr_m": round(float(r["corriente_motor_a_mean_60"]), 1),
                "rul": round(float(r["pred_rul"]), 2),
                "estres": round(float(r["indice_estres_termomecanico"]), 2)
            })
        scenarios[name] = tomas

    return scenarios


def build():
    scenarios = generate_telemetry_tomas()
    scenarios_json = json.dumps(scenarios)

    with open('fallas_embedded.json', 'r', encoding='utf-8') as f:
        fallas_json = f.read()

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tablero Andon 4.0 - Secuencia Completa de Telemetría (Inicio a Fin)</title>
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #151d2f;
            --bg-card-sub: #1e293b;
            --border-color: #2b3a55;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --green-main: #10b981;
            --green-glow: rgba(16, 185, 129, 0.25);
            --yellow-main: #f59e0b;
            --yellow-glow: rgba(245, 158, 11, 0.3);
            --red-main: #ef4444;
            --red-glow: rgba(239, 68, 68, 0.4);
            --blue-accent: #38bdf8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header Andon */
        .andon-header {{
            background: linear-gradient(90deg, #131c31 0%, #0d1424 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px 26px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}

        .header-title-box h1 {{
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .header-title-box p {{
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 4px;
        }}

        .header-meta {{
            text-align: right;
            font-size: 13px;
        }}

        .shift-badge {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--blue-accent);
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            display: inline-block;
            margin-bottom: 4px;
        }}

        /* Semáforo de Planta Andon */
        .status-banner {{
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.4s ease;
        }}

        .status-banner.normal {{
            background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
            border: 2px solid var(--green-main);
            box-shadow: 0 0 25px var(--green-glow);
        }}

        .status-banner.warning {{
            background: linear-gradient(135deg, #78350f 0%, #b45309 100%);
            border: 2px solid var(--yellow-main);
            box-shadow: 0 0 25px var(--yellow-glow);
        }}

        .status-banner.critical {{
            background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
            border: 2px solid var(--red-main);
            box-shadow: 0 0 35px var(--red-glow);
            animation: pulse-border 1.8s infinite;
        }}

        @keyframes pulse-border {{
            0% {{ box-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }}
            50% {{ box-shadow: 0 0 30px rgba(239, 68, 68, 0.9); }}
            100% {{ box-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }}
        }}

        .status-title {{
            font-size: 26px;
            font-weight: 900;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}

        .status-desc {{
            font-size: 14px;
            margin-top: 6px;
            color: #e2e8f0;
        }}

        /* Grid Superior: Métrica RUL + Salud */
        .top-metrics-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 18px;
            margin-bottom: 20px;
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}

        .rul-card {{
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        .card-label {{
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
        }}

        .rul-val {{
            font-size: 44px;
            font-weight: 900;
            margin: 8px 0;
            letter-spacing: 0.5px;
        }}

        .rul-val.normal {{ color: #34d399; }}
        .rul-val.warning {{ color: #fbbf24; }}
        .rul-val.critical {{ color: #f87171; }}

        .rul-sub {{
            font-size: 14px;
            color: #cbd5e1;
        }}

        /* Health Card */
        .health-card {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}

        .health-val {{
            font-size: 44px;
            font-weight: 900;
            margin: 6px 0;
        }}

        .health-bar-bg {{
            width: 100%;
            height: 12px;
            background: var(--bg-card-sub);
            border-radius: 6px;
            overflow: hidden;
            margin-top: 8px;
        }}

        .health-bar-fill {{
            height: 100%;
            width: 98%;
            background: var(--green-main);
            transition: width 0.4s ease, background 0.4s ease;
        }}

        /* Caja de Diagnóstico / Causa Raíz */
        .diagnosis-box {{
            background: var(--bg-card);
            border-left: 5px solid var(--blue-accent);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.5;
        }}

        .diagnosis-box.warning {{
            border-left-color: var(--yellow-main);
            background: rgba(245, 158, 11, 0.08);
        }}

        .diagnosis-box.critical {{
            border-left-color: var(--red-main);
            background: rgba(239, 68, 68, 0.12);
        }}

        /* Grid de los 4 Sensores */
        .sensors-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 24px;
        }}

        .sensor-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease;
        }}

        .sensor-card:hover {{
            transform: translateY(-2px);
        }}

        .sensor-title {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .sensor-reading {{
            font-size: 26px;
            font-weight: 800;
            color: #ffffff;
            margin: 6px 0 4px;
        }}

        .sensor-unit {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 10.5px;
            font-weight: 800;
            letter-spacing: 0.5px;
            width: fit-content;
        }}

        .badge.normal {{ background: #064e3b; color: #6ee7b7; }}
        .badge.warning {{ background: #78350f; color: #fcd34d; }}
        .badge.critical {{ background: #7f1d1d; color: #fca5a5; }}

        .sensor-sub {{
            margin-top: 10px;
            border-top: 1px solid var(--border-color);
            padding-top: 6px;
            font-size: 11.5px;
            color: var(--text-muted);
            line-height: 1.4;
        }}

        /* Panel de Simulación y Controles */
        .control-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 24px;
        }}

        .control-panel h3 {{
            font-size: 16px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .presets-row {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}

        .btn-preset {{
            flex: 1;
            min-width: 170px;
            padding: 11px 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-card-sub);
            color: #ffffff;
            font-size: 12.5px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
        }}

        .btn-preset:hover {{
            border-color: var(--blue-accent);
            background: #253349;
        }}

        .btn-preset.active {{
            border-color: var(--blue-accent);
            background: rgba(56, 189, 248, 0.18);
            color: var(--blue-accent);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
        }}

        .btn-preset.highlight-all {{
            border-color: #38bdf8;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(30, 41, 59, 0.8) 100%);
        }}

        .timeline-box {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-card-sub);
            padding: 12px 18px;
            border-radius: 8px;
            flex-wrap: wrap;
        }}

        .timeline-slider {{
            flex: 1;
            min-width: 200px;
            height: 6px;
            border-radius: 3px;
            accent-color: var(--blue-accent);
            cursor: pointer;
        }}

        .btn-action {{
            background: var(--bg-card);
            color: var(--blue-accent);
            border: 1px solid var(--blue-accent);
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            font-size: 13px;
            white-space: nowrap;
            transition: all 0.2s ease;
        }}

        .btn-action:hover {{
            background: rgba(56, 189, 248, 0.15);
        }}

        .btn-play {{
            background: var(--blue-accent);
            color: #0b1120;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 800;
            cursor: pointer;
            font-size: 13px;
            white-space: nowrap;
        }}

        .btn-play:hover {{
            background: #7dd3fc;
        }}

        /* Gráfico Canvas */
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .chart-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .chart-select {{
            background: var(--bg-card-sub);
            color: #ffffff;
            border: 1px solid var(--border-color);
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }}

        canvas#trendChart {{
            width: 100%;
            height: 310px;
            background: #0f1627;
            border-radius: 8px;
            display: block;
        }}

        /* Tabla de Paradas */
        .table-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}

        .table-wrap {{
            overflow-x: auto;
            max-height: 280px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            text-align: left;
        }}

        th {{
            background: var(--bg-card-sub);
            color: var(--text-muted);
            padding: 10px 14px;
            font-weight: 700;
            position: sticky;
            top: 0;
        }}

        td {{
            padding: 9px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        /* Footer */
        footer {{
            text-align: center;
            margin-top: 30px;
            padding: 16px;
            color: var(--text-muted);
            font-size: 12px;
            border-top: 1px solid var(--border-color);
        }}

        @media (max-width: 1050px) {{
            .sensors-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .top-metrics-grid {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 700px) {{
            .sensors-grid {{ grid-template-columns: 1fr; }}
            .andon-header {{ flex-direction: column; text-align: center; gap: 10px; }}
            .header-meta {{ text-align: center; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER ANDON -->
        <header class="andon-header">
            <div class="header-title-box">
                <h1>⚙️ TABLERO ANDON 4.0 · PRENSA TROQUELADORA</h1>
                <p>Monitoreo Continuo por Cada Toma de Datos (Fila a Fila) & Estimación de RUL</p>
            </div>
            <div class="header-meta">
                <span class="shift-badge" id="sequenceBadge">SECUENCIA COMPLETA (INICIO A FIN)</span>
                <div style="color: #94a3b8;">Estampa de la Toma: <b id="clockDisplay">2026-01-01 08:00:00</b></div>
            </div>
        </header>

        <!-- SEMÁFORO DE ESTADO ANDON -->
        <div id="statusBanner" class="status-banner normal">
            <div class="status-title" id="statusTitle">🟢 ESTADO: OPERACIÓN NORMAL</div>
            <div class="status-desc" id="statusDesc">Todos los sensores se encuentran operando dentro de la tolerancia nominal. Régimen estable.</div>
        </div>

        <!-- MÉTRICA GIGANTE RUL + SALUD -->
        <div class="top-metrics-grid">
            <div class="card rul-card">
                <div class="card-label">Vida Útil Remanente Estimada (RUL)</div>
                <div class="rul-val normal" id="rulValue">> 10.0 DÍAS / ÓPTIMO</div>
                <div class="rul-sub" id="rulSubtext">Operación estable. No se proyecta parada no programada a corto plazo.</div>
            </div>

            <div class="card health-card">
                <div class="card-label">Índice de Salud de Máquina</div>
                <div class="health-val" id="healthValue" style="color: var(--green-main);">98.5%</div>
                <div style="font-size: 13px; color: var(--text-muted);" id="healthDesc">Condición: <b>Excelente</b></div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" id="healthBarFill"></div>
                </div>
            </div>
        </div>

        <!-- DIAGNÓSTICO DE CAUSA RAÍZ -->
        <div id="diagnosisBox" class="diagnosis-box">
            <b style="color: var(--green-main); font-size: 15px;" id="diagTitle">✓ DIAGNÓSTICO DE CONDICIÓN:</b><br>
            <span id="diagMessage" style="color: #cbd5e1;">
                Todos los subsistemas mecánicos, hidráulicos y eléctricos reportan valores nominales. El desgaste acumulado es mínimo.
            </span>
        </div>

        <!-- 4 TARJETAS KPI DE SENSORES -->
        <div class="sensors-grid">
            <!-- 1. Temperatura -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">🌡️ Temperatura Motor</div>
                    <div class="sensor-reading"><span id="valTemp">52.4</span> <span class="sensor-unit">°C</span></div>
                    <span class="badge normal" id="badgeTemp">NOMINAL</span>
                </div>
                <div class="sensor-sub">
                    Media 60 min: <b id="meanTemp">52.2 °C</b><br>
                    Nominal: 50.0 - 55.0 °C
                </div>
            </div>

            <!-- 2. Presión Hidráulica -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">💧 Presión Hidráulica</div>
                    <div class="sensor-reading"><span id="valPres">214.8</span> <span class="sensor-unit">bar</span></div>
                    <span class="badge normal" id="badgePres">NOMINAL</span>
                </div>
                <div class="sensor-sub">
                    Media 60 min: <b id="meanPres">214.5 bar</b><br>
                    Nominal: 210 - 220 bar
                </div>
            </div>

            <!-- 3. Vibración RMS -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">📳 Vibración RMS</div>
                    <div class="sensor-reading"><span id="valVib">1.32</span> <span class="sensor-unit">mm/s</span></div>
                    <span class="badge normal" id="badgeVib">NOMINAL</span>
                </div>
                <div class="sensor-sub">
                    Media 60 min: <b id="meanVib">1.34 mm/s</b><br>
                    Nominal: 1.0 - 1.6 mm/s
                </div>
            </div>

            <!-- 4. Corriente Motor -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">⚡ Corriente Motor</div>
                    <div class="sensor-reading"><span id="valCorr">20.9</span> <span class="sensor-unit">A</span></div>
                    <span class="badge normal" id="badgeCorr">NOMINAL</span>
                </div>
                <div class="sensor-sub">
                    Media 60 min: <b id="meanCorr">21.1 A</b><br>
                    Nominal: 20.0 - 22.5 A
                </div>
            </div>
        </div>

        <!-- PANEL DE SIMULACIÓN Y PRESETS -->
        <div class="control-panel">
            <h3>🎮 Centro de Telemetría: Secuencia de Tomas de Datos (Fila a Fila)</h3>
            <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
                Selecciona <b>Toda la Secuencia Completa (Inicio a Fin)</b> para recorrer el ciclo entero o elige cualquiera de los 4 escenarios para inspeccionar la respuesta del modelo toma a toma:
            </p>

            <div class="presets-row">
                <button class="btn-preset active highlight-all" id="btnPresetCompleta" onclick="selectScenario('completa')">🌐 Toda la Secuencia Completa (Inicio a Fin)</button>
                <button class="btn-preset" id="btnPreset1" onclick="selectScenario('normal')">🟢 1. Secuencia Normal (80 Tomas)</button>
                <button class="btn-preset" id="btnPreset2" onclick="selectScenario('pico')">⚡ 2. Secuencia Pico Transitorio (80 Tomas)</button>
                <button class="btn-preset" id="btnPreset3" onclick="selectScenario('desgaste')">🟡 3. Secuencia Desgaste Sostenido (80 Tomas)</button>
                <button class="btn-preset" id="btnPreset4" onclick="selectScenario('critico')">🔴 4. Secuencia Falla Inminente (80 Tomas)</button>
            </div>

            <div class="timeline-box">
                <button class="btn-play" id="playBtn" onclick="togglePlay()">▶ Iniciar Reproducción (Toma a Toma)</button>
                <button class="btn-action" onclick="stepToma(-1)">⏮ -1 Toma</button>
                <button class="btn-action" onclick="stepToma(1)">⏭ +1 Toma (Nueva Fila)</button>
                <button class="btn-action" onclick="stepToma(10)">⏩ +10 Tomas</button>
                <input type="range" id="timeSlider" class="timeline-slider" min="0" max="599" value="0" oninput="onSliderChange(this.value)">
                <span id="sliderTimeText" style="font-size: 13px; font-weight: 700; min-width: 270px; text-align: right; color: var(--blue-accent);">Toma #1 de 600 · 2026-01-01 08:00:00</span>
            </div>
        </div>

        <!-- GRÁFICA DE TENDENCIAS EN TIEMPO REAL -->
        <div class="chart-card">
            <div class="chart-header">
                <div>
                    <h3 style="font-size: 16px;" id="chartTitle">📈 Toda la Secuencia Completa de Tomas (Inicio a Fin: 2026)</h3>
                    <p style="font-size: 12.5px; color: var(--text-muted);" id="chartSubtitle">
                        Mostrando toda la secuencia cronológica de inicio a fin (desde la primera toma 2026-01-01 hasta la última 2026-12-26).
                    </p>
                </div>
                <div class="chart-controls">
                    <select id="drawModeSelect" class="chart-select" onchange="updateChart()">
                        <option value="full" selected>📊 Secuencia Completa (Trazo Total con Cursor)</option>
                        <option value="progressive">📈 Adquisición Progresiva (Aparece cada nuevo dato)</option>
                    </select>
                    <select id="sensorChartSelect" class="chart-select" onchange="updateChart()">
                        <option value="vib">Vibración RMS (mm/s)</option>
                        <option value="temp">Temperatura Motor (°C)</option>
                        <option value="pres">Presión Hidráulica (bar)</option>
                        <option value="corr">Corriente Motor (A)</option>
                    </select>
                </div>
            </div>
            <canvas id="trendChart"></canvas>
            <div style="display: flex; gap: 20px; font-size: 12px; margin-top: 10px; justify-content: flex-end; flex-wrap: wrap;">
                <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 14px; height: 2px; background: #64748b;"></span> Lectura Instantánea (con ruido en cada toma)</span>
                <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 14px; height: 3px; background: #f59e0b;"></span> Media Móvil 60 min (Filtro de Desgaste)</span>
                <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 50%; background: #38bdf8; border: 2px solid #ffffff;"></span> Toma / Fila Actual</span>
            </div>
        </div>

        <!-- BITÁCORA HISTÓRICA DE PARADAS -->
        <div class="table-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="font-size: 16px;">📋 Registro Histórico de Paradas de Mantenimiento (Ground Truth)</h3>
                <span style="font-size: 12px; color: var(--text-muted);">37 eventos históricos documentados</span>
            </div>
            <div class="table-wrap">
                <table id="fallasTable">
                    <thead>
                        <tr>
                            <th>ID Parada</th>
                            <th>Fecha y Hora</th>
                            <th>Tipo de Falla</th>
                            <th>Componente Afectado</th>
                            <th>Duración</th>
                            <th>Acción Correctiva Ejecutada</th>
                            <th>Técnico Responsable</th>
                        </tr>
                    </thead>
                    <tbody id="fallasBody"></tbody>
                </table>
            </div>
        </div>

        <!-- FOOTER -->
        <footer>
            Tablero de Mantenimiento Predictivo Andon · Prensa Troqueladora 500T · Secuencia Completa de Telemetría (Inicio a Fin)
        </footer>
    </div>

    <!-- SCRIPT DE INTELIGENCIA PREDICTIVA Y DINÁMICA -->
    <script>
        const SCENARIOS = {scenarios_json};
        const FALLAS = {fallas_json};

        let activeScenarioKey = 'completa';
        let currentTomaIdx = 0;
        let isPlaying = false;
        let playInterval = null;

        function getActiveList() {{
            return SCENARIOS[activeScenarioKey];
        }}

        function renderTable() {{
            const tbody = document.getElementById('fallasBody');
            tbody.innerHTML = FALLAS.map(f => `
                <tr>
                    <td><b>${{f.id_parada}}</b></td>
                    <td>${{f.timestamp_falla}}</td>
                    <td>${{f.tipo_falla}}</td>
                    <td>${{f.componente_afectado}}</td>
                    <td>${{f.duracion_parada_min}} min</td>
                    <td>${{f.accion_correctiva}}</td>
                    <td>${{f.tecnico_responsable}}</td>
                </tr>
            `).join('');
        }}

        function calculateRULAndState(sample) {{
            const tempM60 = sample.temp_m;
            const vibM60 = sample.vib_m;
            const presM60 = sample.pres_m;
            const corrM60 = sample.corr_m;
            const estres = sample.estres;
            const predRul = sample.rul;

            // Filtro de ruido transitorio aislado: valor bruto alto pero media móvil normal
            const esPicoAislado = (sample.temp > 60 && tempM60 < 54) ||
                                  (sample.vib > 2.5 && vibM60 < 1.6) ||
                                  (sample.corr > 25 && corrM60 < 22.5);

            // Criterios de desgaste sostenido
            const esCritico = (tempM60 >= 68.0 || vibM60 >= 3.80 || presM60 <= 185.0 || corrM60 >= 28.5 || estres >= 6.0 || predRul <= 1.0);
            const esAdvertencia = !esCritico && (tempM60 >= 57.5 || vibM60 >= 2.20 || presM60 <= 204.0 || corrM60 >= 23.5 || estres >= 1.5 || predRul <= 4.0);

            let estado = "NORMAL";
            let rulDias = predRul;

            if (esCritico && !esPicoAislado) {{
                estado = "CRITICO";
                rulDias = Math.min(predRul, 0.85);
            }} else if (esAdvertencia && !esPicoAislado) {{
                estado = "ADVERTENCIA";
                rulDias = Math.max(1.1, Math.min(predRul, 4.5));
            }} else {{
                estado = "NORMAL";
                rulDias = Math.max(predRul, 7.5);
            }}

            const sensorDevs = [
                {{ name: "Vibración RMS Rodamientos", z: Math.max(0, (vibM60 - 1.35) / 0.25), comp: "Rodamiento Principal de Apoyo" }},
                {{ name: "Temperatura Motor", z: Math.max(0, (tempM60 - 52.5) / 2.0), comp: "Bobinado y Sistema de Ventilación" }},
                {{ name: "Presión Hidráulica", z: Math.max(0, (214.0 - presM60) / 3.0), comp: "Bomba Hidráulica y Válvula de Alivio" }},
                {{ name: "Corriente Eléctrica", z: Math.max(0, (corrM60 - 21.0) / 1.2), comp: "Estator y Fricción de Guías" }}
            ];
            sensorDevs.sort((a, b) => b.z - a.z);

            return {{
                estado,
                rulDias,
                esPicoAislado,
                estres,
                sensorCritico: sensorDevs[0].name,
                componenteCritico: sensorDevs[0].comp,
                maxZ: sensorDevs[0].z
            }};
        }}

        function updateUI(idx) {{
            const list = getActiveList();
            if (idx >= list.length) idx = list.length - 1;
            if (idx < 0) idx = 0;
            currentTomaIdx = idx;
            const sample = list[idx];
            const diag = calculateRULAndState(sample);

            document.getElementById('clockDisplay').innerText = sample.t;
            document.getElementById('sliderTimeText').innerText = `Secuencia: Toma #${{sample.toma}} de ${{list.length}} · ${{sample.t}}`;
            document.getElementById('timeSlider').value = idx;

            document.getElementById('valTemp').innerText = sample.temp.toFixed(1);
            document.getElementById('meanTemp').innerText = sample.temp_m.toFixed(1) + ' °C';

            document.getElementById('valPres').innerText = sample.pres.toFixed(1);
            document.getElementById('meanPres').innerText = sample.pres_m.toFixed(1) + ' bar';

            document.getElementById('valVib').innerText = sample.vib.toFixed(2);
            document.getElementById('meanVib').innerText = sample.vib_m.toFixed(2) + ' mm/s';

            document.getElementById('valCorr').innerText = sample.corr.toFixed(1);
            document.getElementById('meanCorr').innerText = sample.corr_m.toFixed(1) + ' A';

            setBadge('badgeTemp', sample.temp_m < 58 ? 'normal' : (sample.temp_m < 68 ? 'warning' : 'critical'), sample.temp_m < 58 ? 'NOMINAL' : (sample.temp_m < 68 ? 'ELEVADA' : 'CRÍTICA'));
            setBadge('badgePres', sample.pres_m > 204 ? 'normal' : (sample.pres_m > 185 ? 'warning' : 'critical'), sample.pres_m > 204 ? 'NOMINAL' : (sample.pres_m > 185 ? 'BAJA' : 'CRÍTICA'));
            setBadge('badgeVib', sample.vib_m < 2.2 ? 'normal' : (sample.vib_m < 3.8 ? 'warning' : 'critical'), sample.vib_m < 2.2 ? 'NOMINAL' : (sample.vib_m < 3.8 ? 'ALERTA' : 'CRÍTICA'));
            setBadge('badgeCorr', sample.corr_m < 23.5 ? 'normal' : (sample.corr_m < 28.5 ? 'warning' : 'critical'), sample.corr_m < 23.5 ? 'NOMINAL' : (sample.corr_m < 28.5 ? 'SOBRECARGA' : 'CRÍTICA'));

            const banner = document.getElementById('statusBanner');
            const title = document.getElementById('statusTitle');
            const desc = document.getElementById('statusDesc');
            const rulVal = document.getElementById('rulValue');
            const rulSub = document.getElementById('rulSubtext');
            const diagBox = document.getElementById('diagnosisBox');
            const diagTitle = document.getElementById('diagTitle');
            const diagMsg = document.getElementById('diagMessage');

            const health = Math.max(5, Math.min(100, 100 - (diag.estres * 8.5)));
            document.getElementById('healthValue').innerText = health.toFixed(1) + '%';
            document.getElementById('healthBarFill').style.width = health + '%';

            if (diag.estado === 'NORMAL') {{
                banner.className = 'status-banner normal';
                title.innerText = '🟢 ESTADO: OPERACIÓN NORMAL';
                desc.innerText = 'Todos los sensores operan dentro del régimen nominal. Equipo estable.';
                rulVal.className = 'rul-val normal';
                rulVal.innerText = '> 10.0 DÍAS / ÓPTIMO';
                rulSub.innerText = 'Operación estable. No se proyecta parada no programada a corto plazo.';

                document.getElementById('healthValue').style.color = 'var(--green-main)';
                document.getElementById('healthBarFill').style.background = 'var(--green-main)';
                document.getElementById('healthDesc').innerHTML = 'Condición: <b>Excelente</b>';

                diagBox.className = 'diagnosis-box';
                if (diag.esPicoAislado) {{
                    diagTitle.style.color = 'var(--green-main)';
                    diagTitle.innerText = '🛡️ FILTRO DE RUIDO TRANSITORIO ACTIVO (PICO AISLADO EN ESTA FILA):';
                    diagMsg.innerHTML = `En esta toma de datos (<b>Toma #${{sample.toma}}</b>) se registró un pico transitorio aislado de vibración (<b>${{sample.vib.toFixed(2)}} mm/s</b>), pero la <b>media móvil a 60 min permanece normal (${{sample.vib_m.toFixed(2)}} mm/s)</b>. El sistema descarta la fluctuación como ruido transitorio y previene una falsa alarma.`;
                }} else {{
                    diagTitle.style.color = 'var(--green-main)';
                    diagTitle.innerText = '✓ DIAGNÓSTICO DE CONDICIÓN:';
                    diagMsg.innerText = 'Todos los subsistemas mecánicos, hidráulicos y eléctricos operan dentro de especificaciones nominales. Desgaste acumulado mínimo.';
                }}
            }} else if (diag.estado === 'ADVERTENCIA') {{
                banner.className = 'status-banner warning';
                title.innerText = '🟡 DEGRADACIÓN DETECTADA (ALERTA PREVENTIVA)';
                desc.innerText = 'Tendencia sostenida de desgaste termomecánico detectada en las últimas tomas de datos. Planificar inspección.';
                rulVal.className = 'rul-val warning';
                rulVal.innerText = `FALTAN ${{diag.rulDias.toFixed(1)}} DÍAS PARA LA FALLA`;
                rulSub.innerText = `Equivalente a aproximadamente ${{(diag.rulDias * 10).toFixed(1)}} horas operativas restantes antes de parada.`;

                document.getElementById('healthValue').style.color = 'var(--yellow-main)';
                document.getElementById('healthBarFill').style.background = 'var(--yellow-main)';
                document.getElementById('healthDesc').innerHTML = 'Condición: <b>Degradación Sostenida</b>';

                diagBox.className = 'diagnosis-box warning';
                diagTitle.style.color = 'var(--yellow-main)';
                diagTitle.innerText = '⚠️ DIAGNÓSTICO DE CAUSA RAÍZ (DESGASTE DETECTADO):';
                diagMsg.innerHTML = `Disparo liderado por: <b>${{diag.sensorCritico}}</b> (+${{diag.maxZ.toFixed(1)}}σ sobre media nominal).<br>• Componente bajo estrés: <b>${{diag.componenteCritico}}</b>.<br>• Acción recomendada: Inspección y lubricación preventiva en el cambio de turno.`;
            }} else {{
                banner.className = 'status-banner critical';
                title.innerText = '🔴 PARADA CRÍTICA INMINENTE';
                desc.innerText = '¡PELIGRO DE AVERÍA SEVERA! Umbral crítico superado en la lectura actual. Intervención correctiva inmediata.';
                rulVal.className = 'rul-val critical';
                rulVal.innerText = `🚨 FALTAN ${{diag.rulDias.toFixed(2)}} DÍAS (${{(diag.rulDias * 10).toFixed(1)}} HORAS)`;
                rulSub.innerText = 'Desgaste acelerado crítico. Detener troqueladora de forma segura al terminar el ciclo actual.';

                document.getElementById('healthValue').style.color = 'var(--red-main)';
                document.getElementById('healthBarFill').style.background = 'var(--red-main)';
                document.getElementById('healthDesc').innerHTML = 'Condición: <b>Peligro Crítico</b>';

                diagBox.className = 'diagnosis-box critical';
                diagTitle.style.color = 'var(--red-main)';
                diagTitle.innerText = '🚨 DIAGNÓSTICO DE CAUSA RAÍZ - FALLA INMINENTE:';
                diagMsg.innerHTML = `Sobreesfuerzo severo en: <b>${{diag.sensorCritico}}</b> (+${{diag.maxZ.toFixed(1)}}σ sostenido).<br>• Componente crítico: <b>${{diag.componenteCritico}}</b>.<br>• Protocolo: Detención inmediata y convocatoria al equipo de mantenimiento electromecánico.`;
            }}

            updateChart();
        }}

        function setBadge(id, cls, text) {{
            const el = document.getElementById(id);
            el.className = 'badge ' + cls;
            el.innerText = text;
        }}

        function selectScenario(scenarioKey) {{
            activeScenarioKey = scenarioKey;
            const buttons = {{
                'completa': 'btnPresetCompleta',
                'normal': 'btnPreset1',
                'pico': 'btnPreset2',
                'desgaste': 'btnPreset3',
                'critico': 'btnPreset4'
            }};
            for (let k in buttons) {{
                const el = document.getElementById(buttons[k]);
                if (el) el.classList.remove('active');
            }}
            if (buttons[scenarioKey]) {{
                const el = document.getElementById(buttons[scenarioKey]);
                if (el) el.classList.add('active');
            }}

            const list = getActiveList();
            const slider = document.getElementById('timeSlider');
            slider.max = list.length - 1;

            if (scenarioKey === 'completa') {{
                document.getElementById('sequenceBadge').innerText = 'SECUENCIA COMPLETA (INICIO A FIN: 212,476 FILAS)';
                document.getElementById('chartTitle').innerText = '📈 Toda la Secuencia Completa de Tomas (Inicio a Fin: 2026)';
                document.getElementById('chartSubtitle').innerText = 'Mostrando toda la secuencia cronológica de inicio a fin (desde la primera toma 2026-01-01 hasta la última 2026-12-26).';
                // En secuencia completa, seleccionar por defecto el modo de trazo total para verla completa de inmediato
                document.getElementById('drawModeSelect').value = 'full';
                slider.value = 0;
                updateUI(0);
            }} else {{
                document.getElementById('sequenceBadge').innerText = `ESCENARIO: ${{scenarioKey.toUpperCase()}} (80 TOMAS CONSECUTIVAS)`;
                document.getElementById('chartTitle').innerText = `📈 Secuencia de Tomas: ${{scenarioKey.toUpperCase()}} (Fila a Fila)`;
                document.getElementById('chartSubtitle').innerText = 'El eje horizontal incrementa toma por toma (+1 fila generada por el sensor).';
                
                let startIdx = 0;
                if (scenarioKey === 'pico') startIdx = 37;
                slider.value = startIdx;
                updateUI(startIdx);
            }}
        }}

        function stepToma(delta) {{
            const list = getActiveList();
            let next = currentTomaIdx + delta;
            if (next < 0) next = 0;
            if (next >= list.length) next = list.length - 1;
            updateUI(next);
        }}

        function onSliderChange(val) {{
            updateUI(parseInt(val, 10));
        }}

        function togglePlay() {{
            const btn = document.getElementById('playBtn');
            const list = getActiveList();
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                btn.innerText = '▶ Iniciar Reproducción (Toma a Toma)';
            }} else {{
                isPlaying = true;
                btn.innerText = '⏸ Pausar Reproducción';
                playInterval = setInterval(() => {{
                    let next = currentTomaIdx + 1;
                    if (next >= list.length) next = 0;
                    updateUI(next);
                }}, 250);
            }}
        }}

        function updateChart() {{
            const canvas = document.getElementById('trendChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const sensorKey = document.getElementById('sensorChartSelect').value;
            const drawMode = document.getElementById('drawModeSelect').value; // 'full' o 'progressive'

            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const w = rect.width;
            const h = rect.height;
            const padLeft = 48;
            const padRight = 24;
            const padTop = 24;
            const padBottom = 44;

            ctx.clearRect(0, 0, w, h);

            const fullList = getActiveList();
            if (!fullList || fullList.length < 2) return;

            const totalT = fullList.length;
            const rawKey = sensorKey;
            const meanKey = sensorKey + '_m';

            // Escala Y
            const allRaw = fullList.map(s => s[rawKey]);
            const allMean = fullList.map(s => s[meanKey] || s[rawKey]);
            const allVals = allRaw.concat(allMean);

            let minVal = Math.min(...allVals) * 0.95;
            let maxVal = Math.max(...allVals) * 1.05;
            if (maxVal === minVal) {{ maxVal += 1; minVal -= 1; }}

            const getX = (i) => padLeft + (i / (totalT - 1)) * (w - padLeft - padRight);
            const getY = (val) => h - padBottom - ((val - minVal) / (maxVal - minVal)) * (h - padTop - padBottom);

            // 1. Cuadrícula horizontal y valores Y
            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 1;
            for (let step = 0; step <= 4; step++) {{
                const y = padTop + (step / 4) * (h - padTop - padBottom);
                ctx.beginPath();
                ctx.moveTo(padLeft, y);
                ctx.lineTo(w - padRight, y);
                ctx.stroke();

                const v = maxVal - (step / 4) * (maxVal - minVal);
                ctx.fillStyle = '#64748b';
                ctx.font = '10.5px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(v.toFixed(1), padLeft - 6, y + 4);
            }}

            // 2. Eje X: Secuencia de Tomas
            
            // Micro-ticks a lo largo del eje
            const microStep = totalT > 100 ? 5 : 1;
            ctx.beginPath();
            ctx.strokeStyle = '#334155';
            ctx.lineWidth = 1;
            for (let i = 0; i < totalT; i += microStep) {{
                const x = getX(i);
                ctx.moveTo(x, h - padBottom);
                ctx.lineTo(x, h - padBottom + 4);
            }}
            ctx.stroke();

            // Intervalo de marcas principales según longitud de la secuencia
            const stepTomas = totalT > 100 ? Math.floor(totalT / 8) : 10;
            for (let i = 0; i < totalT; i += stepTomas) {{
                const x = getX(i);
                
                // Línea vertical punteada
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
                ctx.setLineDash([3, 3]);
                ctx.moveTo(x, padTop);
                ctx.lineTo(x, h - padBottom);
                ctx.stroke();
                ctx.setLineDash([]);

                // Tick principal
                ctx.beginPath();
                ctx.strokeStyle = '#64748b';
                ctx.lineWidth = 1.2;
                ctx.moveTo(x, h - padBottom);
                ctx.lineTo(x, h - padBottom + 6);
                ctx.stroke();

                // Etiqueta de número de toma
                ctx.fillStyle = '#38bdf8';
                ctx.font = 'bold 10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`Toma #${{i + 1}}`, x, h - padBottom + 17);

                // Fecha/hora de la toma
                ctx.fillStyle = '#94a3b8';
                ctx.font = '9px sans-serif';
                if (totalT > 100) {{
                    ctx.fillText(`${{fullList[i].date.slice(5)}} ${{fullList[i].hm}}`, x, h - padBottom + 28);
                }} else {{
                    ctx.fillText(fullList[i].hm, x, h - padBottom + 28);
                }}
            }}

            // Asegurar etiqueta en la última toma
            const lastIdx = totalT - 1;
            if (lastIdx % stepTomas !== 0) {{
                const lx = getX(lastIdx);
                ctx.beginPath();
                ctx.strokeStyle = '#64748b';
                ctx.moveTo(lx, h - padBottom);
                ctx.lineTo(lx, h - padBottom + 6);
                ctx.stroke();

                ctx.fillStyle = '#38bdf8';
                ctx.font = 'bold 10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`Toma #${{lastIdx + 1}}`, lx, h - padBottom + 17);

                ctx.fillStyle = '#94a3b8';
                ctx.font = '9px sans-serif';
                if (totalT > 100) {{
                    ctx.fillText(`${{fullList[lastIdx].date.slice(5)}} ${{fullList[lastIdx].hm}}`, lx, h - padBottom + 28);
                }} else {{
                    ctx.fillText(fullList[lastIdx].hm, lx, h - padBottom + 28);
                }}
            }}

            // Pie descriptivo del eje
            ctx.fillStyle = '#64748b';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            if (activeScenarioKey === 'completa') {{
                ctx.fillText(`🌐 Secuencia Completa: Desde Toma #1 (${{fullList[0].date}} ${{fullList[0].hm}}) hasta Toma #${{totalT}} (${{fullList[lastIdx].date}} ${{fullList[lastIdx].hm}})`, padLeft, h - 4);
            }} else {{
                ctx.fillText(`📅 ${{fullList[0].date}} | Eje X: 1 paso = 1 fila de sensor (Δt = 1 min)`, padLeft, h - 4);
            }}

            ctx.textAlign = 'right';
            ctx.fillText(`Total: ${{totalT}} tomas en secuencia`, w - padRight, h - 4);

            // Cantidad de puntos a dibujar
            const drawCount = drawMode === 'progressive' ? (currentTomaIdx + 1) : totalT;

            // 3. Trazo Instantáneo
            if (drawCount > 1) {{
                ctx.beginPath();
                ctx.strokeStyle = '#64748b';
                ctx.lineWidth = totalT > 100 ? 1.2 : 1.6;
                for (let i = 0; i < drawCount; i++) {{
                    const x = getX(i);
                    const y = getY(fullList[i][rawKey]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();

                // 4. Trazo Media Móvil
                ctx.beginPath();
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = totalT > 100 ? 2.4 : 2.8;
                for (let i = 0; i < drawCount; i++) {{
                    const x = getX(i);
                    const mVal = fullList[i][meanKey] || fullList[i][rawKey];
                    if (i === 0) ctx.moveTo(x, getY(mVal));
                    else ctx.lineTo(x, getY(mVal));
                }}
                ctx.stroke();
            }}

            // 5. Cursor y marcador de la toma activa
            if (currentTomaIdx >= 0 && currentTomaIdx < totalT) {{
                const curX = getX(currentTomaIdx);
                const curY = getY(fullList[currentTomaIdx][rawKey]);
                const curSample = fullList[currentTomaIdx];
                const curDiag = calculateRULAndState(curSample);

                const markerColor = curDiag.estado === 'CRITICO' ? '#ef4444' : (curDiag.estado === 'ADVERTENCIA' ? '#f59e0b' : '#38bdf8');

                // Línea vertical punteada
                ctx.beginPath();
                ctx.strokeStyle = markerColor;
                ctx.lineWidth = 1.6;
                ctx.setLineDash([4, 4]);
                ctx.moveTo(curX, padTop);
                ctx.lineTo(curX, h - padBottom);
                ctx.stroke();
                ctx.setLineDash([]);

                // Halo
                ctx.beginPath();
                ctx.arc(curX, curY, 10, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
                ctx.fill();

                // Círculo
                ctx.beginPath();
                ctx.arc(curX, curY, 6, 0, 2 * Math.PI);
                ctx.fillStyle = markerColor;
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();

                // Placa indicadora
                const label = `Toma #${{curSample.toma}} (${{curSample.hm}}): ${{curSample[rawKey].toFixed(2)}}`;
                ctx.font = 'bold 11px sans-serif';
                const textWidth = ctx.measureText(label).width;
                const boxX = Math.max(padLeft, Math.min(w - padRight - textWidth - 12, curX - textWidth / 2 - 6));
                
                ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
                ctx.fillRect(boxX, padTop + 2, textWidth + 12, 19);
                ctx.strokeStyle = markerColor;
                ctx.lineWidth = 1;
                ctx.strokeRect(boxX, padTop + 2, textWidth + 12, 19);

                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'left';
                ctx.fillText(label, boxX + 6, padTop + 16);
            }}
        }}

        window.addEventListener('load', () => {{
            renderTable();
            selectScenario('completa');
        }});

        window.addEventListener('resize', updateChart);
    </script>
</body>
</html>"""

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    file_size_kb = os.path.getsize('index.html') / 1024.0
    print(f"Generated 'index.html' successfully! Size: {file_size_kb:.1f} KB")

if __name__ == '__main__':
    build()
