"""
build_html.py
=============
Script constructor que genera 'index.html', un tablero web 100% autónomo (standalone)
listo para ser alojado en GitHub Pages o abierto con doble clic en cualquier navegador.
Incluye:
1. Calendario para recorrer todas las lecturas por día, de inicio a fin del dataset.
2. Botones para los 4 escenarios de secuencias operativas específicas (80 tomas consecutivas cada una).
3. Modos de visualización: Día Completo y Adquisición Progresiva dato a dato.
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
        raise FileNotFoundError("Se requiere el dataset de test de 2026; no se sustituye por entrenamiento.")
        
    df = pd.read_csv(path_2026)
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

    bundle = joblib.load("modelo_rul.joblib")
    model = bundle["modelo"]
    feature_cols = bundle["feature_cols"]

    # Actualizar las exportaciones auxiliares conservando sus fechas y formato.
    for json_path in ["samples_telemetria.json", "scenarios_telemetria.json", "scenarios_tomas.json"]:
        if not os.path.exists(json_path):
            continue
        with open(json_path, encoding="utf-8") as f:
            exported = json.load(f)
        groups = exported.values() if isinstance(exported, dict) else [exported]
        records = [record for group in groups for record in group]
        timestamp_index = pd.Index(pd.to_datetime(df["timestamp"]))
        indices = timestamp_index.get_indexer(pd.to_datetime([r["t"] for r in records]))
        if (indices < 0).any():
            raise ValueError(f"Hay fechas sin telemetría en {json_path}")
        predictions = model.predict(df.iloc[indices][feature_cols])
        for record, prediction, index in zip(records, predictions, indices):
            record.pop("vel", None)
            record.pop("vel_m", None)
            record["rul"] = float(prediction)
            for field, column in [("temp", "temp_motor_c"), ("pres", "presion_hidraulica_bar"), ("vib", "vibracion_rms_mms"), ("corr", "corriente_motor_a")]:
                record[field] = float(df.iloc[index][column])
                record[field + "_m"] = float(df.iloc[index][column + "_mean_60"])
            record["estres"] = float(df.iloc[index]["indice_estres_termomecanico"])
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, ensure_ascii=False)

    # Exportar las lecturas exactas del CSV; solo las características móviles son derivadas.
    predictions = model.predict(df[feature_cols])
    records = pd.DataFrame({
        "toma": np.arange(1, len(df) + 1), "fila_csv": np.arange(2, len(df) + 2),
        "t": df["timestamp"], "hm": df["timestamp"].str[11:16], "date": df["timestamp"].str[:10],
        "temp": df["temp_motor_c"], "pres": df["presion_hidraulica_bar"],
        "vib": df["vibracion_rms_mms"], "corr": df["corriente_motor_a"],
        "temp_m": df["temp_motor_c_mean_60"], "pres_m": df["presion_hidraulica_bar_mean_60"],
        "vib_m": df["vibracion_rms_mms_mean_60"], "corr_m": df["corriente_motor_a_mean_60"],
        "rul": predictions, "estres": df["indice_estres_termomecanico"]
    }).to_dict("records")
    scenarios = {"completa": records}
    metadata = {}
    timestamps = pd.to_datetime(df["timestamp"])
    # Umbrales existentes del tablero; no se ajustan para conseguir escenarios.
    configs = {
        "vib": ("vibracion_rms_mms", "Vibración RMS", "mm/s", 2.20, 3.80, 2.5, 1.6, False),
        "temp": ("temp_motor_c", "Temperatura Motor", "°C", 57.5, 68.0, 60.0, 54.0, False),
        "pres": ("presion_hidraulica_bar", "Presión Hidráulica", "bar", 204.0, 185.0, 200.0, 210.0, True),
        "corr": ("corriente_motor_a", "Corriente Motor", "A", 23.5, 28.5, 25.0, 22.5, False)
    }
    for key, (column, label, unit, warning, critical, spike, baseline, low) in configs.items():
        raw = df[column]
        mean = df[f"{column}_mean_60"]
        normal = mean > warning if low else mean < warning
        wear = (mean <= warning) & (mean > critical) if low else (mean >= warning) & (mean < critical)
        severe = mean <= critical if low else mean >= critical
        spike_mask = (raw < spike) & (mean > baseline) if low else (raw > spike) & (mean < baseline)
        # El pico debe volver al rango de alerta en la lectura siguiente.
        spike_mask &= raw.shift(-1) > spike if low else raw.shift(-1) < spike
        for kind, mask in [("normal", normal), ("pico", spike_mask), ("desgaste", wear), ("critico", severe)]:
            scenario_key = f"{key}_{kind}"
            chosen = None
            # Desgaste y crítico: 15 lecturas consecutivas cumpliendo el criterio.
            candidates = mask if kind == "pico" else mask.rolling(15, min_periods=15).sum().eq(15)
            for anchor in np.flatnonzero(candidates.to_numpy()):
                start = max(0, int(anchor) - 37)
                stop = start + 80
                if stop > len(df):
                    continue
                times = timestamps.iloc[start:stop]
                if times.dt.date.nunique() != 1 or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
                    continue
                if kind == "normal" and not mask.iloc[start:stop].all():
                    continue
                chosen = (start, stop, int(anchor))
                break
            info = {"sensor": label, "column": column, "unit": unit, "kind": kind,
                    "available": chosen is not None, "dataset": path_2026,
                    "warning": warning, "critical": critical,
                    "criterion": (f"Lectura {'<' if low else '>'} {spike}, media 60 min {'>' if low else '<'} {baseline}, retorno en la siguiente lectura" if kind == "pico" else
                        f"Media 60 min: normal {'>' if low else '<'} {warning}; desgaste entre {warning} y {critical}; crítico {'<=' if low else '>='} {critical}. Desgaste/crítico requieren 15 lecturas consecutivas.")}
            if chosen:
                start, stop, anchor = chosen
                scenarios[scenario_key] = records[start:stop]
                info.update({"start": records[start]["t"], "end": records[stop-1]["t"],
                             "first_csv_row": start + 2, "last_csv_row": stop + 1,
                             "anchor_index": anchor - start, "anchor_time": records[anchor]["t"],
                             "anchor_value": float(raw.iloc[anchor]), "anchor_mean60": float(mean.iloc[anchor])})
            else:
                scenarios[scenario_key] = []
                info["reason"] = "No se encontró una ventana de 80 lecturas consecutivas que cumpla el criterio en el dataset de test."
            metadata[scenario_key] = info
    with open("scenarios_sensores_test.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "scenarios": {k:v for k,v in scenarios.items() if k != "completa"}}, f, ensure_ascii=False)
    return scenarios, metadata


def build():
    scenarios, scenario_metadata = generate_telemetry_tomas()
    scenarios_json = json.dumps(scenarios, separators=(',', ':'))

    with open('fallas_embedded.json', 'r', encoding='utf-8') as f:
        fallas_json = f.read()

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script>
        try {{ document.documentElement.dataset.theme = localStorage.getItem('andon-theme') === 'light' ? 'light' : 'dark'; }}
        catch (_) {{ document.documentElement.dataset.theme = 'dark'; }}
    </script>
    <title>Tablero Andon 4.0 - Secuencia Completa de Telemetría (Inicio a Fin)</title>
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #151d2f;
            --bg-card-sub: #1e293b;
            --border-color: #2b3a55;
            --text-main: #f8fafc;
            --text-muted: #b6c2d3;
            --green-main: #10b981;
            --green-glow: rgba(16, 185, 129, 0.25);
            --yellow-main: #f59e0b;
            --yellow-glow: rgba(245, 158, 11, 0.3);
            --red-main: #ef4444;
            --red-glow: rgba(239, 68, 68, 0.4);
            --blue-accent: #38bdf8;
            --preset-active-text: #082f49;
            color-scheme: dark;
            --chart-bg: #0f1627;
            --chart-grid: #334155;
            --chart-raw: #64748b;
            --chart-mean: #f59e0b;
            --chart-label-bg: rgba(15, 23, 42, 0.95);
            --chart-label-text: #ffffff;
        }}

        :root[data-theme="light"] {{
            color-scheme: light;
            --bg-main: #f1f5f9;
            --bg-card: #ffffff;
            --bg-card-sub: #e8eef5;
            --border-color: #c3cedc;
            --text-main: #152238;
            --text-muted: #475569;
            --blue-accent: #0369a1;
            --preset-active-text: #ffffff;
            --green-main: #047857;
            --yellow-main: #92400e;
            --red-main: #b91c1c;
            --chart-bg: #f8fafc;
            --chart-grid: #cbd5e1;
            --chart-raw: #64748b;
            --chart-mean: #b45309;
            --chart-label-bg: rgba(255, 255, 255, 0.97);
            --chart-label-text: #152238;
        }}
        [data-theme="light"] .andon-header {{ background: linear-gradient(90deg, #ffffff, #e8eef5); }}
        [data-theme="light"] .header-title-box h1,
        [data-theme="light"] .sensor-reading,
        [data-theme="light"] .btn-preset,
        [data-theme="light"] .chart-select,
        [data-theme="light"] .sensor-chart h4,
        [data-theme="light"] td {{ color: var(--text-main); }}
        [data-theme="light"] .rul-sub,
        [data-theme="light"] #diagMessage,
        [data-theme="light"] .header-meta > div {{ color: var(--text-muted) !important; }}
        [data-theme="light"] .rul-val.normal {{ color: #047857; }}
        [data-theme="light"] .rul-val.warning {{ color: #92400e; }}
        [data-theme="light"] .rul-val.critical {{ color: #b91c1c; }}
        [data-theme="light"] .btn-preset:hover:not(.active):not(:disabled) {{ background: #dce7f3; }}
        [data-theme="light"] .btn-play {{ color: white; }}
        [data-theme="light"] .btn-play:hover {{ background: #075985; }}
        [data-theme="light"] .status-title {{ color: #ffffff; }}
        [data-theme="light"] td {{ border-bottom-color: #e2e8f0; }}
        #themeToggle {{ margin-top: 12px; }}

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
            font-size: 16px;
            line-height: 1.5;
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
            font-size: 16px;
            margin-top: 4px;
        }}

        .header-meta {{
            text-align: right;
            font-size: 16px;
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

        .header-meta {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .header-calendar {{ position: relative; text-align: left; }}
        .header-calendar > summary {{
            list-style: none;
            cursor: pointer;
            font-size: 26px;
            font-weight: 800;
            padding: 14px 18px;
            border: 2px solid var(--blue-accent);
            border-radius: 12px;
            color: #ffffff;
            background: linear-gradient(135deg, #0369a1, #075985);
            box-shadow: 0 6px 20px rgba(3, 105, 161, 0.32);
        }}
        .header-calendar > summary::-webkit-details-marker {{ display: none; }}
        .header-calendar > summary:focus-visible,
        .calendar-popover button:focus-visible {{ outline: 3px solid var(--blue-accent); outline-offset: 3px; }}
        .calendar-popover {{
            position: absolute;
            top: calc(100% + 12px);
            right: 0;
            z-index: 20;
            width: 390px;
            max-width: calc(100vw - 48px);
            padding: 18px;
            border: 2px solid var(--blue-accent);
            border-radius: 16px;
            background: var(--bg-card);
            color: var(--text-main);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.3);
        }}
        .calendar-month-nav {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 16px; }}
        .calendar-month-nav strong {{ font-size: 22px; text-transform: capitalize; text-align: center; }}
        .calendar-popover button {{ cursor: pointer; font-family: inherit; }}
        .calendar-month-nav button {{ font-size: 22px; padding: 8px; border: 0; border-radius: 8px; background: var(--bg-card-sub); color: var(--blue-accent); }}
        .calendar-weekdays, .calendar-days {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 6px; text-align: center; }}
        .calendar-weekdays {{ font-size: 18px; font-weight: 700; margin-bottom: 8px; }}
        .calendar-day {{ min-height: 44px; font-size: 22px; font-weight: 800; border: 1px solid var(--blue-accent); border-radius: 8px; color: var(--blue-accent); background: var(--bg-card-sub); }}
        .calendar-day:not(:disabled):hover {{ box-shadow: inset 0 0 0 2px var(--blue-accent); }}
        .calendar-day[aria-current="date"] {{ background: var(--blue-accent); color: var(--preset-active-text); }}
        .calendar-day:disabled, .calendar-month-nav button:disabled {{ opacity: 0.3; cursor: default; border-color: transparent; }}
        .calendar-legend {{ font-size: 18px; font-weight: 700; color: var(--blue-accent); margin-top: 14px; }}

        @media (max-width: 700px) {{
            .calendar-popover {{
                position: fixed;
                top: 120px;
                left: 24px;
                right: 24px;
                width: auto;
                max-width: none;
                max-height: calc(100dvh - 144px);
                overflow-y: auto;
            }}
        }}

        #dayCharts > summary {{
            cursor: pointer;
            padding: 18px 22px;
            margin-bottom: 20px;
            border: 2px solid var(--blue-accent);
            border-radius: 12px;
            background: var(--bg-card);
            color: var(--blue-accent);
            font-size: 24px;
            font-weight: 800;
        }}
        #dayCharts > summary:focus-visible {{ outline: 3px solid var(--blue-accent); outline-offset: 3px; }}
        #dayCharts > summary[hidden], .chart-controls[hidden], .timeline-box[hidden], #dailyAverageLabel[hidden] {{ display: none; }}
        #dailyAverageLabel {{ font-size: 24px; font-weight: 800; color: var(--text-main); margin-bottom: 16px; }}

        .day-chart-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            padding: 18px 22px;
            margin-bottom: 12px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }}
        .day-chart-toolbar h3 {{ font-size: 24px; }}
        #dayPlayBtn {{ font-size: 20px; padding: 12px 18px; }}

        #dashboard[hidden], #loginScreen[hidden], #loginError[hidden] {{ display: none; }}
        .login-screen {{ min-height: calc(100svh - 40px); display: grid; place-items: center; }}
        .login-layout {{ width: min(960px, 100%); display: grid; grid-template-columns: 1.25fr 1fr; gap: 24px; align-items: stretch; }}
        .login-card, .login-credentials {{ padding: 40px; border: 1px solid var(--border-color); border-radius: 20px; background: var(--bg-card); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15); }}
        .login-brand {{ font-size: 18px; font-weight: 800; color: var(--blue-accent); margin-bottom: 22px; }}
        .login-card h1 {{ font-size: 38px; line-height: 1.2; margin-bottom: 12px; }}
        .login-intro {{ font-size: 20px; color: var(--text-muted); margin-bottom: 30px; }}
        #loginForm {{ display: flex; flex-direction: column; gap: 12px; }}
        #loginForm label {{ font-size: 20px; font-weight: 700; }}
        #loginForm input {{ width: 100%; min-width: 0; padding: 14px 16px; border: 1px solid var(--border-color); border-radius: 10px; background: var(--bg-main); color: var(--text-main); font: inherit; font-size: 20px; margin-bottom: 10px; }}
        #loginForm input::placeholder {{ color: var(--text-muted); }}
        #loginForm input:focus-visible, .login-submit:focus-visible {{ outline: 3px solid var(--blue-accent); outline-offset: 3px; }}
        .login-submit {{ padding: 16px; border: 0; border-radius: 10px; background: var(--blue-accent); color: var(--preset-active-text); cursor: pointer; font: inherit; font-size: 22px; font-weight: 800; }}
        .login-submit:hover {{ filter: brightness(1.1); }}
        .login-error {{ font-size: 18px; color: var(--red-main); font-weight: 700; }}
        .login-credentials {{ border: 2px solid var(--blue-accent); background: var(--bg-card-sub); display: flex; flex-direction: column; justify-content: center; }}
        .credentials-icon {{ font-size: 44px; margin-bottom: 14px; }}
        .login-credentials h2 {{ font-size: 28px; line-height: 1.3; margin-bottom: 12px; }}
        .login-credentials p {{ font-size: 20px; color: var(--text-muted); }}
        .login-credentials dl {{ margin: 26px 0; }}
        .login-credentials dt {{ font-size: 18px; font-weight: 700; margin-bottom: 8px; }}
        .login-credentials dd {{ margin: 0 0 20px; padding: 12px 16px; border-radius: 10px; background: var(--bg-card); }}
        .login-credentials code {{ font-size: 26px; font-weight: 800; color: var(--blue-accent); overflow-wrap: anywhere; }}
        .login-credentials .login-demo {{ font-size: 18px; font-weight: 700; color: var(--blue-accent); }}
        @media (max-width: 700px) {{
            .login-layout {{ grid-template-columns: 1fr; }}
            .login-card, .login-credentials {{ padding: 26px; }}
            .login-card h1 {{ font-size: 32px; }}
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
            font-size: 16px;
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
            font-size: 16px;
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
            font-size: 16px;
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
            font-size: 16px;
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
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease;
        }}

        .sensor-card:hover {{
            transform: translateY(-2px);
        }}

        .sensor-title {{
            font-size: 24px;
            font-weight: 800;
            color: var(--text-main);
            line-height: 1.3;
            text-transform: uppercase;
        }}

        .sensor-reading {{
            font-size: 56px;
            font-weight: 900;
            color: #ffffff;
            margin: 10px 0 8px;
        }}

        .sensor-unit {{
            font-size: 22px;
            font-weight: 600;
            color: var(--text-main);
        }}

        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 0.5px;
            width: fit-content;
        }}

        .badge[hidden] {{ display: none; }}
        .badge.normal {{ background: #064e3b; color: #6ee7b7; }}
        .badge.warning {{ background: #78350f; color: #fcd34d; }}
        .badge.critical {{ background: #7f1d1d; color: #fca5a5; }}

        .sensor-sub {{
            margin-top: 10px;
            border-top: 1px solid var(--border-color);
            padding-top: 6px;
            font-size: 16px;
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

        .control-panel summary {{
            font-size: 20px;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            list-style: none;
        }}

        .control-panel summary::-webkit-details-marker {{ display: none; }}
        .control-panel summary::after {{
            content: '▾ Mostrar controles';
            color: var(--blue-accent);
            font-size: 16px;
            white-space: nowrap;
        }}
        .control-panel[open] summary::after {{ content: '▴ Ocultar controles'; }}
        .control-panel summary:focus-visible {{
            outline: 2px solid var(--blue-accent);
            outline-offset: 6px;
            border-radius: 4px;
        }}
        .control-panel-content {{ margin-top: 16px; }}

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
            color: var(--text-main);
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
        }}

        .btn-preset:hover:not(.active):not(:disabled) {{
            border-color: var(--text-muted);
            background: #253349;
        }}

        .btn-preset.active {{
            border-color: var(--blue-accent);
            background: var(--blue-accent);
            color: var(--preset-active-text);
            box-shadow: inset 0 0 0 1px var(--blue-accent);
        }}

        .btn-preset:focus-visible {{
            outline: 2px solid var(--text-muted);
            outline-offset: 3px;
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
            font-size: 16px;
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
            font-size: 16px;
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
            font-size: 16px;
            cursor: pointer;
        }}

        .sensor-charts {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }}
        .sensor-chart {{ min-width: 0; }}
        .sensor-chart h4 {{ margin: 0 0 10px; font-size: 16px; color: #cbd5e1; }}
        @media (max-width: 640px) {{
            .sensor-charts {{ grid-template-columns: 1fr; }}
        }}
        canvas.sensor-trend {{
            width: 100%;
            height: 280px;
            background: var(--chart-bg);
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
            font-size: 16px;
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
            font-size: 16px;
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
    <link rel="stylesheet" href="daily_summary.css?v=compact-2">
</head>
<body>
    <section class="login-screen" id="loginScreen" aria-labelledby="loginTitle">
        <div class="login-layout">
            <div class="login-card">
                <div class="login-brand">⚙️ TABLERO ANDON 4.0</div>
                <h1 id="loginTitle">Iniciar sesión</h1>
                <p class="login-intro">Mantenimiento predictivo de la prensa troqueladora</p>
                <form id="loginForm">
                    <label for="loginUsername">Usuario</label>
                    <input id="loginUsername" type="text" autocomplete="username" required placeholder="Escribe tu usuario" autofocus>
                    <label for="loginPassword">Contraseña</label>
                    <input id="loginPassword" type="password" autocomplete="current-password" required placeholder="Escribe tu contraseña">
                    <p id="loginError" class="login-error" role="alert" hidden>Usuario o contraseña incorrectos. Revisa los datos del cuadro de acceso.</p>
                    <button type="submit" class="login-submit">Entrar al tablero →</button>
                </form>
            </div>
            <aside class="login-credentials" aria-labelledby="credentialsTitle">
                <div class="credentials-icon" aria-hidden="true">🎓</div>
                <h2 id="credentialsTitle">Acceso para el profesor</h2>
                <p>Utiliza estos datos para entrar al tablero.</p>
                <dl>
                    <dt>Usuario</dt><dd><code>usuario</code></dd>
                    <dt>Contraseña</dt><dd><code>artefacto</code></dd>
                </dl>
                <p class="login-demo">Acceso de demostración</p>
            </aside>
        </div>
    </section>
    <div class="dashboard-layout" id="dashboard" hidden>
    <div class="container">
        <!-- HEADER ANDON -->
        <header class="andon-header">
            <div class="header-title-box">
                <h1>⚙️ TABLERO ANDON 4.0 · PRENSA TROQUELADORA</h1>
            </div>
            <div class="header-meta">
                <details class="header-calendar" id="headerCalendar">
                    <summary aria-label="Abrir calendario de registros">📅 <span id="headerCalendarDate">Calendario de registros</span> ▾</summary>
                    <div class="calendar-popover">
                        <div class="calendar-month-nav">
                            <button type="button" id="calendarPrev" aria-label="Mes anterior" onclick="moveCalendarMonth(-1)">◀</button>
                            <strong id="calendarMonthTitle" aria-live="polite"></strong>
                            <button type="button" id="calendarNext" aria-label="Mes siguiente" onclick="moveCalendarMonth(1)">▶</button>
                        </div>
                        <div class="calendar-weekdays" aria-hidden="true"><span>L</span><span>M</span><span>M</span><span>J</span><span>V</span><span>S</span><span>D</span></div>
                        <div class="calendar-days" id="calendarDays" aria-label="Días del mes"></div>
                        <p class="calendar-legend">● Días con registros</p>
                    </div>
                </details>
                <button type="button" class="btn-action" id="themeToggle" onclick="toggleTheme()" aria-pressed="false">☀️ Modo claro</button>
            </div>
        </header>

        <!-- SEMÁFORO DE ESTADO ANDON -->
        <div id="statusBanner" class="status-banner normal">
            <div class="status-title" id="statusTitle">🟢 ESTADO: OPERACIÓN NORMAL</div>
        </div>

        <!-- MÉTRICA GIGANTE RUL + SALUD -->
        <div class="top-metrics-grid">
            <div class="card rul-card">
                <div class="rul-val normal" id="rulValue">> 10.0 DÍAS / ÓPTIMO</div>
            </div>

            <div class="card health-card">
                <div class="card-label" id="healthLabel">Índice de Salud de Máquina</div>
                <div class="health-val" id="healthValue" style="color: var(--green-main);">98.5%</div>
                <div style="font-size: 16px; color: var(--text-muted);" id="healthDesc">Condición: <b>Excelente</b></div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" id="healthBarFill"></div>
                </div>
            </div>
        </div>

        <p id="dailyAverageLabel" hidden></p>
        <!-- 4 TARJETAS KPI DE SENSORES -->
        <div class="sensors-grid">
            <!-- 1. Temperatura -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">🌡️ Temperatura Motor</div>
                    <div class="sensor-reading"><span id="valTemp">52.4</span> <span class="sensor-unit">°C</span></div>
                    <span class="badge normal" id="badgeTemp" hidden></span>
                </div>
            </div>

            <!-- 2. Presión Hidráulica -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">💧 Presión Hidráulica</div>
                    <div class="sensor-reading"><span id="valPres">214.8</span> <span class="sensor-unit">bar</span></div>
                    <span class="badge normal" id="badgePres" hidden></span>
                </div>
            </div>

            <!-- 3. Vibración RMS -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">📳 Vibración RMS</div>
                    <div class="sensor-reading"><span id="valVib">1.32</span> <span class="sensor-unit">mm/s</span></div>
                    <span class="badge normal" id="badgeVib" hidden></span>
                </div>
            </div>

            <!-- 4. Corriente Motor -->
            <div class="sensor-card">
                <div>
                    <div class="sensor-title">⚡ Corriente Motor</div>
                    <div class="sensor-reading"><span id="valCorr">20.9</span> <span class="sensor-unit">A</span></div>
                    <span class="badge normal" id="badgeCorr" hidden></span>
                </div>
            </div>
        </div>

        <div class="day-chart-toolbar">
            <h3 id="chartTitle">📈 Tomas del día</h3>
            <button type="button" class="btn-play" id="dayPlayBtn" onclick="toggleDayPlayback()" aria-pressed="false">▶ Reproducir tomas</button>
        </div>
        <details id="dayCharts" open>
        <summary id="historicalChartToggle" hidden>📈 Ver gráficas completas del día</summary>
        <!-- GRÁFICA DE TENDENCIAS EN TIEMPO REAL -->
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-controls">
                    <select id="drawModeSelect" class="chart-select" onchange="updateChart()">
                        <option value="progressive" selected>📈 Tomas del día (Adquisición Progresiva)</option>
                        <option value="full">📊 Día completo (Trazo Total con Cursor)</option>
                    </select>
                </div>
            </div>
            <div class="sensor-charts">
                <section class="sensor-chart"><h4>Vibración RMS (mm/s)</h4><canvas id="trendVib" class="sensor-trend" aria-label="Evolución de vibración RMS"></canvas></section>
                <section class="sensor-chart"><h4>Temperatura Motor (°C)</h4><canvas id="trendTemp" class="sensor-trend" aria-label="Evolución de temperatura del motor"></canvas></section>
                <section class="sensor-chart"><h4>Presión Hidráulica (bar)</h4><canvas id="trendPres" class="sensor-trend" aria-label="Evolución de presión hidráulica"></canvas></section>
                <section class="sensor-chart"><h4>Corriente Motor (A)</h4><canvas id="trendCorr" class="sensor-trend" aria-label="Evolución de corriente del motor"></canvas></section>
            </div>
        </div>

        </details>
        <!-- BITÁCORA HISTÓRICA DE PARADAS -->
        <div class="table-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="font-size: 20px;">📋 Registro Histórico de Paradas de Mantenimiento (Ground Truth)</h3>
                <span style="font-size: 16px; color: var(--text-muted);">37 eventos históricos documentados</span>
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

        <!-- PANEL DE SIMULACIÓN Y PRESETS -->
        <details class="control-panel" id="telemetryControls">
            <summary>🎮 Centro de Telemetría: Secuencia de Tomas de Datos (Fila a Fila)</summary>
            <div class="control-panel-content">

            <div class="timeline-box" style="margin-bottom: 12px;">
                <label for="scenarioSensor">Escenarios del sensor:</label>
                <select id="scenarioSensor" class="chart-select" onchange="selectScenarioSensor(this.value)">
                    <option value="vib">Vibración RMS</option><option value="temp">Temperatura Motor</option>
                    <option value="pres">Presión Hidráulica</option><option value="corr">Corriente Motor</option>
                </select>
            </div>
            <div class="presets-row">
                <button class="btn-preset active" id="btnPresetCompleta" onclick="selectScenario('completa')">📅 Telemetría por Día</button>
                <button class="btn-preset" id="btnPreset1" onclick="selectScenario('normal')">🟢 1. Secuencia Normal (80 Tomas)</button>
                <button class="btn-preset" id="btnPreset2" onclick="selectScenario('pico')">⚡ 2. Secuencia Pico Transitorio (80 Tomas)</button>
                <button class="btn-preset" id="btnPreset3" onclick="selectScenario('desgaste')">🟡 3. Secuencia Desgaste Sostenido (80 Tomas)</button>
                <button class="btn-preset" id="btnPreset4" onclick="selectScenario('critico')">🔴 4. Secuencia Falla Inminente (80 Tomas)</button>
            </div>

            <div class="timeline-box" style="margin-bottom: 12px;">
                <label for="dayCalendar">Fecha de telemetría:</label>
                <input type="date" id="dayCalendar" class="chart-select" min="{scenarios['completa'][0]['date']}" max="{scenarios['completa'][-1]['date']}" value="{scenarios['completa'][0]['date']}" onchange="selectDay(this.value)">
                <button class="btn-action" onclick="changeDay(-1)">◀ Día anterior</button>
                <button class="btn-action" onclick="changeDay(1)">Día siguiente ▶</button>
                <span id="daySummary" style="color: var(--text-muted);"></span>
            </div>
            <div class="timeline-box">
                <button class="btn-play" id="playBtn" onclick="togglePlay()">▶ Iniciar Reproducción (Toma a Toma)</button>
                <button class="btn-action" onclick="stepToma(-1)">⏮ -1 Toma</button>
                <button class="btn-action" onclick="stepToma(1)">⏭ +1 Toma (Nueva Fila)</button>
                <input type="range" id="timeSlider" class="timeline-slider" min="0" max="{len(scenarios['completa']) - 1}" step="1" value="0" oninput="onSliderChange(this.value)">
                <span id="sliderTimeText" style="font-size: 16px; font-weight: 700; min-width: 270px; text-align: right; color: var(--blue-accent);">Toma #1 de {len(scenarios['completa'])} · {scenarios['completa'][0]['t']}</span>
            </div>
            </div>
        </details>

    </div>
    </div>

    <!-- SCRIPT DE INTELIGENCIA PREDICTIVA Y DINÁMICA -->
    <script>
        const SCENARIOS = {scenarios_json};
        const FALLAS = {fallas_json};
        const SCENARIO_METADATA = {json.dumps(scenario_metadata, ensure_ascii=False)};
        let selectedScenarioSensor = 'vib';
        const DAYS = new Map();
        for (const sample of SCENARIOS.completa) {{
            if (!DAYS.has(sample.date)) DAYS.set(sample.date, []);
            DAYS.get(sample.date).push(sample);
        }}
        const availableDays = Array.from(DAYS.keys()).sort();
        const initialDay = availableDays.find(date => date.startsWith('2026-04-')) || availableDays[0];
        let selectedDay = initialDay;
        let isHistoricalView = false;

        let activeScenarioKey = 'completa';
        let currentTomaIdx = 0;
        let isPlaying = false;
        let playInterval = null;
        const chartCaches = new Map();

        let calendarMonth = null;

        function renderHeaderCalendar() {{
            if (!calendarMonth) calendarMonth = selectedDay.slice(0, 7);
            const [year, month] = calendarMonth.split('-').map(Number);
            const start = new Date(year, month - 1, 1);
            const count = new Date(year, month, 0).getDate();
            document.getElementById('calendarMonthTitle').textContent = start.toLocaleDateString('es-CO', {{ month: 'long', year: 'numeric' }});
            document.getElementById('calendarPrev').disabled = calendarMonth <= availableDays[0].slice(0, 7);
            document.getElementById('calendarNext').disabled = calendarMonth >= availableDays[availableDays.length - 1].slice(0, 7);
            const grid = document.getElementById('calendarDays');
            grid.replaceChildren();
            for (let blank = 0; blank < (start.getDay() + 6) % 7; blank++) grid.appendChild(document.createElement('span'));
            for (let day = 1; day <= count; day++) {{
                const date = `${{calendarMonth}}-${{String(day).padStart(2, '0')}}`;
                const rows = DAYS.get(date);
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'calendar-day';
                button.textContent = day;
                button.dataset.date = date;
                button.disabled = !rows;
                button.setAttribute('aria-label', `${{date.split('-').reverse().join('/')}} · ${{rows ? rows.length + ' tomas registradas' : 'Sin registros'}}`);
                button.title = button.getAttribute('aria-label');
                if (date === selectedDay) button.setAttribute('aria-current', 'date');
                button.addEventListener('click', () => {{
                    selectDay(date);
                    document.getElementById('headerCalendar').open = false;
                    document.querySelector('#headerCalendar > summary').focus();
                }});
                grid.appendChild(button);
            }}
        }}

        function syncHeaderCalendar() {{
            document.getElementById('headerCalendarDate').textContent = selectedDay.split('-').reverse().join('/');
            calendarMonth = selectedDay.slice(0, 7);
            renderHeaderCalendar();
        }}

        function moveCalendarMonth(delta) {{
            const [year, month] = calendarMonth.split('-').map(Number);
            const next = new Date(year, month - 1 + delta, 1);
            const candidate = `${{next.getFullYear()}}-${{String(next.getMonth() + 1).padStart(2, '0')}}`;
            if (candidate < availableDays[0].slice(0, 7) || candidate > availableDays[availableDays.length - 1].slice(0, 7)) return;
            calendarMonth = candidate;
            renderHeaderCalendar();
        }}

        document.addEventListener('click', event => {{
            const calendar = document.getElementById('headerCalendar');
            if (!calendar.contains(event.target)) calendar.open = false;
        }});
        document.addEventListener('keydown', event => {{
            const calendar = document.getElementById('headerCalendar');
            if (event.key === 'Escape' && calendar.open) {{
                calendar.open = false;
                calendar.querySelector('summary').focus();
            }}
        }});

        function updateThemeButton() {{
            const light = document.documentElement.dataset.theme === 'light';
            const button = document.getElementById('themeToggle');
            button.innerText = light ? '🌙 Modo oscuro' : '☀️ Modo claro';
            button.setAttribute('aria-pressed', String(light));
            button.setAttribute('aria-label', light ? 'Cambiar a modo oscuro' : 'Cambiar a modo claro');
        }}

        function toggleTheme() {{
            const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
            document.documentElement.dataset.theme = next;
            try {{ localStorage.setItem('andon-theme', next); }} catch (_) {{}}
            updateThemeButton();
            chartCaches.clear();
            updateChart();
        }}

        function getActiveList() {{
            return activeScenarioKey === 'completa' ? (DAYS.get(selectedDay) || []) : SCENARIOS[activeScenarioKey];
        }}

        function selectDay(date, keepPlaying = false) {{
            if (!date) return;
            const historical = !keepPlaying && (date < selectedDay || (date === selectedDay && isHistoricalView));
            if (!keepPlaying && isPlaying) togglePlay();
            selectedDay = date;
            isHistoricalView = historical;
            activeScenarioKey = 'completa';
            selectScenario('completa', keepPlaying);
            if (!historical && !isPlaying && getActiveList().length) togglePlay();
        }}

        function dailyAverageSample(rows) {{
            const sample = {{ ...rows[rows.length - 1] }};
            for (const key of ['temp', 'pres', 'vib', 'corr', 'rul', 'estres']) {{
                sample[key] = rows.reduce((sum, row) => sum + row[key], 0) / rows.length;
            }}
            for (const key of ['temp', 'pres', 'vib', 'corr']) sample[key + '_m'] = sample[key];
            return sample;
        }}

        function syncDayPlaybackButton() {{
            const button = document.getElementById('dayPlayBtn');
            button.disabled = !getActiveList().length;
            button.textContent = isPlaying ? '⏸ Pausar tomas' : '▶ Reproducir tomas';
            button.setAttribute('aria-pressed', String(isPlaying));
        }}

        function toggleDayPlayback() {{
            if (!getActiveList().length) return;
            if (isHistoricalView) {{
                isHistoricalView = false;
                selectScenario('completa');
            }} else if (!isPlaying && currentTomaIdx >= getActiveList().length - 1) {{
                updateUI(0);
            }}
            togglePlay();
        }}

        function configureDayView() {{
            const charts = document.getElementById('dayCharts');
            charts.open = !isHistoricalView;
            document.getElementById('historicalChartToggle').hidden = !isHistoricalView;
            document.getElementById('historicalChartToggle').textContent = `📈 Ver gráficas completas del ${{selectedDay.split('-').reverse().join('/')}}`;
            document.getElementById('drawModeSelect').value = isHistoricalView ? 'full' : 'progressive';
            document.querySelector('.chart-controls').hidden = isHistoricalView;
            document.querySelector('#telemetryControls .timeline-box:last-child').hidden = isHistoricalView;
            const label = document.getElementById('dailyAverageLabel');
            label.hidden = !isHistoricalView;
            label.textContent = `Promedios del ${{selectedDay.split('-').reverse().join('/')}} · ${{getActiveList().length}} lecturas`;
            document.getElementById('healthLabel').textContent = isHistoricalView ? 'Salud promedio del día' : 'Índice de Salud de Máquina';
            syncDayPlaybackButton();
        }}

        function changeDay(delta, keepPlaying = false) {{
            const nextDay = delta > 0
                ? availableDays.find(date => date > selectedDay)
                : availableDays.slice().reverse().find(date => date < selectedDay);
            if (!nextDay) return false;
            selectDay(nextDay, keepPlaying);
            return true;
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

            }} else if (esAdvertencia && !esPicoAislado) {{
                estado = "ADVERTENCIA";

            }} else {{
                estado = "NORMAL";

            }}

            const sensorDevs = [
                {{ name: "Vibración RMS del Troquel", z: Math.max(0, (vibM60 - 1.35) / 0.25), comp: "Troquel" }},
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
            if (!list.length) {{
                currentTomaIdx = 0;
                for (const id of ['valTemp', 'valPres', 'valVib', 'valCorr', 'rulValue', 'healthValue']) {{
                    document.getElementById(id).innerText = '—';
                }}
                for (const id of ['badgeTemp', 'badgePres', 'badgeVib', 'badgeCorr']) setBadge(id, 'normal', 'SIN DATOS');
                document.getElementById('healthBarFill').style.width = '0%';
                document.getElementById('healthDesc').innerText = 'Sin datos';
                document.getElementById('sliderTimeText').innerText = 'Sin tomas para esta fecha';
                document.getElementById('statusTitle').innerText = 'SIN LECTURAS REGISTRADAS';
                updateChart();
                return;
            }}
            if (idx >= list.length) idx = list.length - 1;
            if (idx < 0) idx = 0;
            if (isHistoricalView) idx = list.length - 1;
            currentTomaIdx = idx;
            const sample = isHistoricalView ? dailyAverageSample(list) : list[idx];
            const diag = calculateRULAndState(sample);

            document.getElementById('sliderTimeText').innerText = isHistoricalView ? `${{list.length}} lecturas · Día completo` : `Toma #${{idx + 1}} de ${{list.length}} · ${{sample.t}}`;
            document.getElementById('timeSlider').value = idx;

            document.getElementById('valTemp').innerText = sample.temp.toFixed(1);

            document.getElementById('valPres').innerText = sample.pres.toFixed(1);

            document.getElementById('valVib').innerText = sample.vib.toFixed(2);

            document.getElementById('valCorr').innerText = sample.corr.toFixed(1);

            setBadge('badgeTemp', sample.temp_m < 58 ? 'normal' : (sample.temp_m < 68 ? 'warning' : 'critical'), sample.temp_m < 58 ? '' : (sample.temp_m < 68 ? 'ELEVADA' : 'CRÍTICA'));
            setBadge('badgePres', sample.pres_m > 204 ? 'normal' : (sample.pres_m > 185 ? 'warning' : 'critical'), sample.pres_m > 204 ? '' : (sample.pres_m > 185 ? 'BAJA' : 'CRÍTICA'));
            setBadge('badgeVib', sample.vib_m < 2.2 ? 'normal' : (sample.vib_m < 3.8 ? 'warning' : 'critical'), sample.vib_m < 2.2 ? '' : (sample.vib_m < 3.8 ? 'ALERTA' : 'CRÍTICA'));
            setBadge('badgeCorr', sample.corr_m < 23.5 ? 'normal' : (sample.corr_m < 28.5 ? 'warning' : 'critical'), sample.corr_m < 23.5 ? '' : (sample.corr_m < 28.5 ? 'SOBRECARGA' : 'CRÍTICA'));

            const banner = document.getElementById('statusBanner');
            const title = document.getElementById('statusTitle');
            const rulVal = document.getElementById('rulValue');

            const health = isHistoricalView ? list.reduce((sum, row) => sum + Math.max(5, Math.min(100, 100 - row.estres * 8.5)), 0) / list.length : Math.max(5, Math.min(100, 100 - (diag.estres * 8.5)));
            document.getElementById('healthValue').innerText = health.toFixed(1) + '%';
            document.getElementById('healthBarFill').style.width = health + '%';

            if (diag.estado === 'NORMAL') {{
                banner.className = 'status-banner normal';
                title.innerText = '🟢 ESTADO: OPERACIÓN NORMAL';
                rulVal.className = 'rul-val normal';
                rulVal.innerText = `RUL DEL MODELO: ${{sample.rul.toFixed(2)}} DÍAS`;

                document.getElementById('healthValue').style.color = 'var(--green-main)';
                document.getElementById('healthBarFill').style.background = 'var(--green-main)';
                document.getElementById('healthDesc').innerHTML = 'Condición: <b>Excelente</b>';

            }} else if (diag.estado === 'ADVERTENCIA') {{
                banner.className = 'status-banner warning';
                title.innerText = '🟡 DEGRADACIÓN DETECTADA (ALERTA PREVENTIVA)';
                rulVal.className = 'rul-val warning';
                rulVal.innerText = `FALTAN ${{diag.rulDias.toFixed(1)}} DÍAS PARA LA FALLA`;

                document.getElementById('healthValue').style.color = 'var(--yellow-main)';
                document.getElementById('healthBarFill').style.background = 'var(--yellow-main)';
                document.getElementById('healthDesc').innerHTML = 'Condición: <b>Degradación Sostenida</b>';

            }} else {{
                banner.className = 'status-banner critical';
                title.innerText = '🔴 PARADA CRÍTICA INMINENTE';
                rulVal.className = 'rul-val critical';
                rulVal.innerText = `🚨 FALTAN ${{diag.rulDias.toFixed(2)}} DÍAS (${{(diag.rulDias * 10).toFixed(1)}} HORAS)`;

                document.getElementById('healthValue').style.color = 'var(--red-main)';
                document.getElementById('healthBarFill').style.background = 'var(--red-main)';
                document.getElementById('healthDesc').innerHTML = 'Condición: <b>Peligro Crítico</b>';

            }}

            if (isHistoricalView) {{
                title.innerText = '📊 RESUMEN DEL DÍA';
                rulVal.innerText = `RUL PROMEDIO: ${{sample.rul.toFixed(2)}} DÍAS`;
            }}

            updateChart();
        }}

        function setBadge(id, cls, text) {{
            const el = document.getElementById(id);
            el.className = 'badge ' + cls;
            el.innerText = text;
            el.hidden = text === '';
        }}

        function selectScenarioSensor(sensor) {{
            selectedScenarioSensor = sensor;
            updateScenarioButtons();
            if (activeScenarioKey !== 'completa') {{
                const kind = activeScenarioKey.split('_')[1];
                const key = `${{sensor}}_${{kind}}`;
                selectScenario(SCENARIO_METADATA[key].available ? kind : 'completa');
            }}
        }}

        function updateScenarioButtons() {{
            for (const [kind, id, label] of [['normal','btnPreset1','Normal'], ['pico','btnPreset2','Pico transitorio'], ['desgaste','btnPreset3','Desgaste sostenido'], ['critico','btnPreset4','Crítico']]) {{
                const info = SCENARIO_METADATA[`${{selectedScenarioSensor}}_${{kind}}`];
                const button = document.getElementById(id);
                button.disabled = !info.available;
                button.innerText = info.available ? `${{label}} (80 tomas)` : `${{label}} · Sin ejemplo disponible`;
                button.title = info.available ? info.criterion : info.reason;
                button.style.opacity = info.available ? '1' : '0.5';
            }}
        }}

        function selectScenario(scenarioKey, keepPlaying = false) {{
            if (scenarioKey !== 'completa' && !scenarioKey.includes('_')) scenarioKey = `${{selectedScenarioSensor}}_${{scenarioKey}}`;
            if (scenarioKey !== 'completa' && !SCENARIO_METADATA[scenarioKey].available) return;
            updateScenarioButtons();
            if (!keepPlaying && isPlaying) togglePlay();
            activeScenarioKey = scenarioKey;
            if (scenarioKey !== 'completa') isHistoricalView = false;
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
            const buttonKind = scenarioKey === 'completa' ? 'completa' : scenarioKey.split('_')[1];
            if (buttons[buttonKind]) {{
                const el = document.getElementById(buttons[buttonKind]);
                if (el) el.classList.add('active');
            }}

            const list = getActiveList();
            const slider = document.getElementById('timeSlider');
            if (scenarioKey !== 'completa') selectedDay = list[0].date;
            slider.max = Math.max(0, list.length - 1);
            slider.disabled = !list.length || isHistoricalView;
            document.getElementById('playBtn').disabled = !list.length || isHistoricalView;
            chartCaches.clear();
            document.getElementById('dayCalendar').value = scenarioKey === 'completa' ? selectedDay : list[0].date;
            document.getElementById('daySummary').innerText = list.length ? `${{list.length}} tomas · ${{list[0].hm}} a ${{list[list.length - 1].hm}}` : 'No hay registros para esta fecha';
            syncHeaderCalendar();
            configureDayView();

            if (scenarioKey === 'completa') {{
                document.getElementById('chartTitle').innerText = `📈 Tomas del día · ${{selectedDay}}`;
                slider.value = 0;
                updateUI(0);
            }} else {{
                const info = SCENARIO_METADATA[scenarioKey];
                document.getElementById('chartTitle').innerText = `📈 ${{info.sensor}} · ${{info.kind}} · ${{info.start.slice(0,10)}}`;
                
                let startIdx = info.kind === 'pico' ? info.anchor_index : 0;
                slider.value = startIdx;
                updateUI(startIdx);
            }}
        }}

        function stepToma(delta) {{
            if (isHistoricalView) return;
            const list = getActiveList();
            let next = currentTomaIdx + delta;
            if (activeScenarioKey === 'completa' && list.length) {{
                if (next >= list.length && changeDay(1, isPlaying)) return;
                if (next < 0 && changeDay(-1, isPlaying)) {{ updateUI(getActiveList().length - 1); return; }}
            }}
            if (next < 0) next = 0;
            if (next >= list.length) next = list.length - 1;
            updateUI(next);
        }}

        function onSliderChange(val) {{
            if (isHistoricalView) return;
            updateUI(parseInt(val, 10));
        }}

        function togglePlay() {{
            if (isHistoricalView) return;
            const btn = document.getElementById('playBtn');
            const list = getActiveList();
            if (!list.length) return;
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                btn.innerText = '▶ Iniciar Reproducción (Toma a Toma)';
            }} else {{
                isPlaying = true;
                btn.innerText = '⏸ Pausar Reproducción';
                playInterval = setInterval(() => {{
                    let next = currentTomaIdx + 1;
                    if (next >= getActiveList().length) {{
                        if (activeScenarioKey === 'completa' && changeDay(1, true)) return;
                        togglePlay(); return;
                    }}
                    updateUI(next);
                }}, 1000);
            }}
            syncDayPlaybackButton();
        }}

        function updateChart() {{
            if (document.getElementById('dashboard').hidden) return;
            if (!document.getElementById('dayCharts').open) return;
            for (const [sensor, canvasId] of [['vib', 'trendVib'], ['temp', 'trendTemp'], ['pres', 'trendPres'], ['corr', 'trendCorr']]) {{
                drawSensorChart(sensor, canvasId);
            }}
        }}

        function drawSensorChart(sensorKey, canvasId) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const theme = getComputedStyle(document.documentElement);
            const themeColor = name => theme.getPropertyValue(name).trim();
            const drawMode = document.getElementById('drawModeSelect').value; // 'full' o 'progressive'

            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const w = rect.width;
            const h = rect.height;
            const padLeft = 64;
            const padRight = 30;
            const padTop = 34;
            const padBottom = 64;

            ctx.clearRect(0, 0, w, h);

            const fullList = getActiveList();
            if (!fullList || fullList.length < 2) return;

            const totalT = fullList.length;
            const rawKey = sensorKey;
            const meanKey = sensorKey + '_m';

            // Escala Y
            const cacheKey = `${{activeScenarioKey}}:${{selectedDay}}:${{sensorKey}}:${{w}}:${{h}}:${{window.devicePixelRatio}}`;
            const chartCache = chartCaches.get(sensorKey);
            let minVal = Infinity;
            let maxVal = -Infinity;
            if (chartCache && chartCache.key === cacheKey) {{
                minVal = chartCache.minVal;
                maxVal = chartCache.maxVal;
            }} else {{
                for (const sample of fullList) {{
                    const raw = sample[rawKey];
                    const mean = sample[meanKey] ?? raw;
                    minVal = Math.min(minVal, raw, mean);
                    maxVal = Math.max(maxVal, raw, mean);
                }}
                minVal *= 0.95;
                maxVal *= 1.05;
            }}
            if (maxVal === minVal) {{ maxVal += 1; minVal -= 1; }}

            const getX = (i) => padLeft + (i / (totalT - 1)) * (w - padLeft - padRight);
            const getY = (val) => h - padBottom - ((val - minVal) / (maxVal - minVal)) * (h - padTop - padBottom);

            // Reutilizar el trazo del sensor mientras avanza el cursor.
            if (drawMode === 'full' && chartCache && chartCache.key === cacheKey) {{
                ctx.drawImage(chartCache.canvas, 0, 0, w, h);
            }} else {{

            // 1. Cuadrícula horizontal y valores Y
            ctx.strokeStyle = themeColor('--chart-grid');
            ctx.lineWidth = 1;
            for (let step = 0; step <= 4; step++) {{
                const y = padTop + (step / 4) * (h - padTop - padBottom);
                ctx.beginPath();
                ctx.moveTo(padLeft, y);
                ctx.lineTo(w - padRight, y);
                ctx.stroke();

                const v = maxVal - (step / 4) * (maxVal - minVal);
                ctx.fillStyle = themeColor('--text-muted');
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(v.toFixed(1), padLeft - 6, y + 4);
            }}

            // 2. Eje X: Secuencia de Tomas
            
            // Micro-ticks a lo largo del eje
            const microStep = Math.max(1, Math.ceil(totalT / Math.max(1, Math.floor(w / 5))));
            ctx.beginPath();
            ctx.strokeStyle = themeColor('--chart-grid');
            ctx.lineWidth = 1;
            for (let i = 0; i < totalT; i += microStep) {{
                const x = getX(i);
                ctx.moveTo(x, h - padBottom);
                ctx.lineTo(x, h - padBottom + 4);
            }}
            ctx.stroke();

            // Intervalo de marcas principales según longitud de la secuencia
            const stepTomas = Math.max(1, Math.ceil((totalT - 1) / Math.max(1, Math.floor((w - padLeft - padRight) / 130))));
            for (let i = 0; i < totalT; i += stepTomas) {{
                const x = getX(i);
                
                // Línea vertical punteada
                ctx.beginPath();
                ctx.strokeStyle = themeColor('--chart-grid');
                ctx.setLineDash([3, 3]);
                ctx.moveTo(x, padTop);
                ctx.lineTo(x, h - padBottom);
                ctx.stroke();
                ctx.setLineDash([]);

                // Tick principal
                ctx.beginPath();
                ctx.strokeStyle = themeColor('--chart-raw');
                ctx.lineWidth = 1.2;
                ctx.moveTo(x, h - padBottom);
                ctx.lineTo(x, h - padBottom + 6);
                ctx.stroke();

                // Etiqueta de número de toma
                ctx.fillStyle = themeColor('--blue-accent');
                ctx.font = 'bold 14px sans-serif';
                ctx.textAlign = i === 0 ? 'left' : (i === totalT - 1 ? 'right' : 'center');
                ctx.fillText(`Toma #${{i + 1}}`, x, h - padBottom + 22);

                // Fecha/hora de la toma
                ctx.fillStyle = themeColor('--text-muted');
                ctx.font = '14px sans-serif';
                if (totalT > 100) {{
                    ctx.fillText(fullList[i].hm, x, h - padBottom + 41);
                }} else {{
                    ctx.fillText(fullList[i].hm, x, h - padBottom + 41);
                }}
            }}

            // Asegurar etiqueta en la última toma
            const lastIdx = totalT - 1;
            if (lastIdx % stepTomas !== 0) {{
                const lx = getX(lastIdx);
                ctx.beginPath();
                ctx.strokeStyle = themeColor('--chart-raw');
                ctx.moveTo(lx, h - padBottom);
                ctx.lineTo(lx, h - padBottom + 6);
                ctx.stroke();

                ctx.fillStyle = themeColor('--blue-accent');
                ctx.font = 'bold 14px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(`Toma #${{lastIdx + 1}}`, lx, h - padBottom + 22);

                ctx.fillStyle = themeColor('--text-muted');
                ctx.font = '14px sans-serif';
                if (totalT > 100) {{
                    ctx.fillText(fullList[lastIdx].hm, lx, h - padBottom + 41);
                }} else {{
                    ctx.fillText(fullList[lastIdx].hm, lx, h - padBottom + 41);
                }}
            }}

            // Pie descriptivo del eje
            ctx.fillStyle = themeColor('--text-muted');
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'left';
            if (activeScenarioKey === 'completa') {{
                ctx.fillText(selectedDay, padLeft, h - 4);
            }} else {{
                ctx.fillText(fullList[0].date, padLeft, h - 4);
            }}

            ctx.textAlign = 'right';
            ctx.fillText(`${{totalT}} tomas`, w - padRight, h - 4);

            // Cantidad de puntos a dibujar
            const drawCount = drawMode === 'progressive' ? (currentTomaIdx + 1) : totalT;

            // 3. Trazo Instantáneo
            if (drawCount > 1) {{
                ctx.beginPath();
                ctx.strokeStyle = themeColor('--chart-raw');
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
                ctx.strokeStyle = themeColor('--chart-mean');
                ctx.lineWidth = totalT > 100 ? 2.4 : 2.8;
                for (let i = 0; i < drawCount; i++) {{
                    const x = getX(i);
                    const mVal = fullList[i][meanKey] || fullList[i][rawKey];
                    if (i === 0) ctx.moveTo(x, getY(mVal));
                    else ctx.lineTo(x, getY(mVal));
                }}
                ctx.stroke();
            }}

            if (drawMode === 'full') {{
                const background = document.createElement('canvas');
                background.width = canvas.width;
                background.height = canvas.height;
                background.getContext('2d').drawImage(canvas, 0, 0);
                chartCaches.set(sensorKey, {{ key: cacheKey, canvas: background, minVal, maxVal }});
            }}
            }}

            // 5. Cursor y marcador de la toma activa
            if (currentTomaIdx >= 0 && currentTomaIdx < totalT) {{
                const curX = getX(currentTomaIdx);
                const curY = getY(fullList[currentTomaIdx][rawKey]);
                const curSample = fullList[currentTomaIdx];
                const curDiag = calculateRULAndState(curSample);

                const markerColor = curDiag.estado === 'CRITICO' ? '#ef4444' : (curDiag.estado === 'ADVERTENCIA' ? themeColor('--chart-mean') : themeColor('--blue-accent'));

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
                const label = `Toma #${{currentTomaIdx + 1}} (${{curSample.hm}}): ${{curSample[rawKey].toFixed(2)}}`;
                ctx.font = 'bold 16px sans-serif';
                const textWidth = ctx.measureText(label).width;
                const boxX = Math.max(padLeft, Math.min(w - padRight - textWidth - 12, curX - textWidth / 2 - 6));
                
                ctx.fillStyle = themeColor('--chart-label-bg');
                ctx.fillRect(boxX, padTop + 2, textWidth + 12, 28);
                ctx.strokeStyle = markerColor;
                ctx.lineWidth = 1;
                ctx.strokeRect(boxX, padTop + 2, textWidth + 12, 28);

                ctx.fillStyle = themeColor('--chart-label-text');
                ctx.textAlign = 'left';
                ctx.fillText(label, boxX + 6, padTop + 22);
            }}
        }}

        document.getElementById('loginForm').addEventListener('submit', event => {{
            event.preventDefault();
            const username = document.getElementById('loginUsername');
            const password = document.getElementById('loginPassword');
            const error = document.getElementById('loginError');
            if (username.value !== 'usuario' || password.value !== 'artefacto') {{
                error.hidden = false;
                password.focus();
                return;
            }}
            error.hidden = true;
            password.value = '';
            document.getElementById('loginScreen').hidden = true;
            document.getElementById('dashboard').hidden = false;
            renderTable();
            selectScenario('completa');
            togglePlay();
            document.querySelector('#headerCalendar > summary').focus({{ preventScroll: true }});
            window.scrollTo(0, 0);
        }});

        window.addEventListener('load', () => {{
            updateThemeButton();
        }});

        document.getElementById('dayCharts').addEventListener('toggle', updateChart);
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
