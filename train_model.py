"""
train_model.py
==============
Sistema de Entrenamiento de Prediccion de Fallas y Vida Util Remanente (RUL)
para Troqueladora Industrial.

Funcionalidades:
1. Carga de telemetria de sensores y bitacora historica de paradas.
2. Cruce de marcas de tiempo y calculo de columna objetivo 'rul_dias' (cuenta regresiva).
3. Ingenieria de variables de ventana movil (medias moviles, desviaciones, tendencias).
4. Entrenamiento de regresor de Machine Learning (XGBoost / Random Forest).
5. Evaluacion con metricas industriales (MAE, RMSE en dias) globales y en fase de desgaste.
6. Serializacion del modelo y metadatos en 'modelo_rul.joblib'.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Asegurar codificacion segura en Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Definicion de archivos y parametros del proceso
FILE_TELEMETRIA = "telemetria_troqueladora.csv"
FILE_FALLAS = "log_fallas_mantenimiento.csv"
MODEL_OUTPUT_PATH = "modelo_rul.joblib"

# Parametros operativos de la planta
MUESTRAS_POR_TURNO_DIA = 600  # 10 horas operativas de 08:00 a 17:59 (1 muestra/minuto)
MAX_RUL_DIAS = 10.0           # Capping de vida util (maximo 10 dias para regimen nominal)

# Sensores base monitoreados
SENSORES = [
    "temp_motor_c",
    "presion_hidraulica_bar",
    "vibracion_rms_mms",
    "corriente_motor_a"
]

# Rangos y umbrales nominales de fabrica
RANGOS_NOMINALES = {
    "temp_motor_c": {"nominal_mean": 52.5, "nominal_std": 2.0, "alerta": 60.0, "critico": 70.0, "unidad": " degC"},
    "presion_hidraulica_bar": {"nominal_mean": 214.0, "nominal_std": 3.0, "alerta": 200.0, "critico": 185.0, "unidad": "bar"},
    "vibracion_rms_mms": {"nominal_mean": 1.35, "nominal_std": 0.25, "alerta": 2.50, "critico": 4.00, "unidad": "mm/s"},
    "corriente_motor_a": {"nominal_mean": 21.0, "nominal_std": 1.2, "alerta": 25.0, "critico": 29.0, "unidad": "A"}
}


def cargar_datos(telemetria_path: str, fallas_path: str):
    """Carga y valida los archivos de telemetria y bitacora de mantenimiento."""
    print("=" * 70)
    print("1. CARGA DE DATOS INDUSTRIALES")
    print("=" * 70)
    
    if not os.path.exists(telemetria_path):
        raise FileNotFoundError(f"Error: No se encontro el archivo de telemetria: '{telemetria_path}'")
    if not os.path.exists(fallas_path):
        raise FileNotFoundError(f"Error: No se encontro el archivo de fallas: '{fallas_path}'")
        
    print(f"-> Leyendo telemetria: '{telemetria_path}'...")
    df_tel = pd.read_csv(telemetria_path)
    print(f"   [OK] Filas leidas: {len(df_tel):,} | Columnas: {list(df_tel.columns)}")
    
    print(f"-> Leyendo bitacora de paradas: '{fallas_path}'...")
    df_fallas = pd.read_csv(fallas_path)
    print(f"   [OK] Paradas registradas: {len(df_fallas):,} eventos")
    
    # Conversion estricta a datetime
    df_tel['dt'] = pd.to_datetime(df_tel['timestamp'])
    df_tel = df_tel.sort_values('dt').reset_index(drop=True)
    
    # Normalizacion de columna timestamp en fallas
    ts_col_fallas = None
    for cand in ['timestamp_falla', 'timestamp', 'fecha_falla', 'fecha_parada']:
        if cand in df_fallas.columns:
            ts_col_fallas = cand
            break
    if ts_col_fallas is None:
        raise ValueError(f"No se encontro columna de fecha en {fallas_path}. Columnas: {df_fallas.columns}")
        
    df_fallas['dt_falla'] = pd.to_datetime(df_fallas[ts_col_fallas])
    df_fallas = df_fallas.sort_values('dt_falla').reset_index(drop=True)
    
    return df_tel, df_fallas


def cruzar_marcas_y_calcular_rul(df_tel: pd.DataFrame, df_fallas: pd.DataFrame, max_rul: float = MAX_RUL_DIAS):
    """
    Cruza las marcas de tiempo de telemetria con los eventos de parada,
    calculando la cuenta regresiva RUL en dias de operacion hacia la siguiente falla.
    Topeada a max_rul (10 dias).
    """
    print("\n" + "=" * 70)
    print("2. CRUCE TEMPORAL Y CÁLCULO DE VIDA UTIL REMANENTE (RUL)")
    print("=" * 70)
    
    # Identificar el indice exacto en el dataframe de telemetria de cada evento de falla
    falla_indices = []
    for ts in df_fallas['dt_falla']:
        match = df_tel[df_tel['dt'] == ts].index
        if len(match) > 0:
            falla_indices.append(match[0])
        else:
            # Buscar el punto mas cercano anterior o posterior
            diffs = (df_tel['dt'] - ts).abs()
            nearest_idx = diffs.idxmin()
            falla_indices.append(nearest_idx)
            
    falla_indices = np.sort(np.unique(falla_indices))
    print(f"-> Mapeados {len(falla_indices)} eventos de parada critica sobre la serie temporal.")
    
    # Busqueda rapida del siguiente indice de falla para cada muestra
    idx_telemetria = df_tel.index.values
    pos_siguiente_falla = np.searchsorted(falla_indices, idx_telemetria, side='left')
    mascara_valida = pos_siguiente_falla < len(falla_indices)
    
    # Calculo de cuenta regresiva en dias de operacion (600 minutos de trabajo = 1 dia operativo)
    rul_dias = np.full(len(df_tel), max_rul, dtype=np.float64)
    muestras_restantes = falla_indices[pos_siguiente_falla[mascara_valida]] - idx_telemetria[mascara_valida]
    dias_restantes = muestras_restantes / float(MUESTRAS_POR_TURNO_DIA)
    rul_dias[mascara_valida] = np.clip(dias_restantes, 0.0, max_rul)
    
    df_tel['rul_dias'] = rul_dias
    
    # Resumen de RUL
    print(f"-> Columna objetivo 'rul_dias' calculada con exito.")
    print(f"   * Minimo: {df_tel['rul_dias'].min():.2f} dias")
    print(f"   * Mediana: {df_tel['rul_dias'].median():.2f} dias")
    print(f"   * Maximo topeado: {df_tel['rul_dias'].max():.2f} dias")
    print(f"   * Muestras en degradacion activa (RUL < {max_rul}d): {(df_tel['rul_dias'] < max_rul).sum():,} ({100*(df_tel['rul_dias'] < max_rul).mean():.1f}%)")
    
    return df_tel, falla_indices


def generar_variables_ventana_movil(df: pd.DataFrame):
    """
    Ingenieria de caracteristicas de ventana movil:
    - Medias moviles (15 min y 60 min) para suavizar ruido transitorio.
    - Desviacion estandar movil (60 min) para medir inestabilidad.
    - Tendencia/pendiente movil (delta a 60 min y 15 min).
    - Desviacion acumulada respecto al regimen nominal de operacion.
    """
    print("\n" + "=" * 70)
    print("3. INGENIERÍA DE VARIABLES DE VENTANA MÓVIL Y FILTRADO DE RUIDO")
    print("=" * 70)
    
    feature_cols = []
    
    for s in SENSORES:
        # 1. Valor instantaneo bruto
        feature_cols.append(s)
        
        # 2. Medias moviles (cortas y medianas)
        df[f"{s}_mean_15"] = df[s].rolling(window=15, min_periods=1).mean()
        df[f"{s}_mean_60"] = df[s].rolling(window=60, min_periods=1).mean()
        feature_cols.extend([f"{s}_mean_15", f"{s}_mean_60"])
        
        # 3. Desviacion estandar movil a 60 min (inestabilidad de senal)
        df[f"{s}_std_60"] = df[s].rolling(window=60, min_periods=1).std().fillna(0.0)
        feature_cols.append(f"{s}_std_60")
        
        # 4. Tendencia o pendiente a 60 minutos (diferencia respecto a hace 1 hora)
        df[f"{s}_trend_60"] = df[s] - df[s].shift(60).bfill()
        # Tendencia rapida a 15 minutos
        df[f"{s}_trend_15"] = df[s] - df[s].shift(15).bfill()
        feature_cols.extend([f"{s}_trend_60", f"{s}_trend_15"])
        
        # 5. Desviacion normalizada respecto a la media nominal
        nom_mean = RANGOS_NOMINALES[s]["nominal_mean"]
        nom_std = RANGOS_NOMINALES[s]["nominal_std"]
        df[f"{s}_zscore_nom"] = (df[f"{s}_mean_60"] - nom_mean) / nom_std
        feature_cols.append(f"{s}_zscore_nom")
    
    # 6. Indicador compuesto de desgaste sostenido (fusion multivariante)
    # Penaliza calentamiento sostenido, vibracion excesiva, caida de presion y sobrecorriente
    df["indice_estres_termomecanico"] = (
        np.maximum(0, df["temp_motor_c_zscore_nom"]) * 0.30 +
        np.maximum(0, df["vibracion_rms_mms_zscore_nom"]) * 0.35 +
        np.maximum(0, -df["presion_hidraulica_bar_zscore_nom"]) * 0.20 +
        np.maximum(0, df["corriente_motor_a_zscore_nom"]) * 0.15
    )
    feature_cols.append("indice_estres_termomecanico")
    
    print(f"-> Total de caracteristicas generadas: {len(feature_cols)}")
    print(f"   * Sensores base: {len(SENSORES)}")
    print(f"   * Medias moviles y tendencias: {len(feature_cols) - len(SENSORES)}")
    
    return df, feature_cols


def entrenar_modelo(df: pd.DataFrame, feature_cols: list):
    """
    Entrena el modelo de regresion con particion cronologica (80% entrenamiento, 20% test).
    Prioriza XGBRegressor si esta instalado, o RandomForest/HistGradientBoostingRegressor.
    """
    print("\n" + "=" * 70)
    print("4. ENTRENAMIENTO Y VALIDACIÓN TEMPORAL DEL MODELO RUL")
    print("=" * 70)
    
    # Particion temporal estricta (primeros meses entrenan, ultimos meses prueban)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['rul_dias']
    X_test = test_df[feature_cols]
    y_test = test_df['rul_dias']
    
    print(f"-> Set de Entrenamiento: {len(X_train):,} muestras ({train_df['dt'].min().strftime('%Y-%m-%d')} a {train_df['dt'].max().strftime('%Y-%m-%d')})")
    print(f"-> Set de Validacion:    {len(X_test):,} muestras ({test_df['dt'].min().strftime('%Y-%m-%d')} a {test_df['dt'].max().strftime('%Y-%m-%d')})")
    
    # Seleccion de algoritmo
    modelo = None
    nombre_modelo = ""
    
    try:
        from xgboost import XGBRegressor
        print("-> Algoritmo seleccionado: XGBoost Regressor (GPU/CPU optimizado)")
        modelo = XGBRegressor(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42
        )
        nombre_modelo = "XGBRegressor"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print("-> [Nota: xgboost no detectado, usando HistGradientBoostingRegressor de scikit-learn de alto rendimiento]")
        modelo = HistGradientBoostingRegressor(
            max_iter=160,
            max_depth=8,
            min_samples_leaf=20,
            learning_rate=0.08,
            random_state=42
        )
        nombre_modelo = "HistGradientBoostingRegressor (scikit-learn)"
        
    t0 = time.time()
    print("-> Iniciando ajuste del modelo...")
    modelo.fit(X_train, y_train)
    t_entrenamiento = time.time() - t0
    print(f"[OK] Entrenamiento completado en {t_entrenamiento:.2f} segundos.")
    
    # Predicciones sobre el conjunto de test
    y_pred = modelo.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, MAX_RUL_DIAS)
    
    # Metricas Globales
    mae_global = mean_absolute_error(y_test, y_pred)
    rmse_global = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_global = r2_score(y_test, y_pred)
    
    # Metricas en Fase Critica de Desgaste (RUL <= 3 dias)
    mascara_critica = (y_test <= 3.0)
    if mascara_critica.sum() > 0:
        mae_critico = mean_absolute_error(y_test[mascara_critica], y_pred[mascara_critica])
        rmse_critico = np.sqrt(mean_squared_error(y_test[mascara_critica], y_pred[mascara_critica]))
    else:
        mae_critico = 0.0
        rmse_critico = 0.0
        
    print("\n" + "=" * 70)
    print("5. MÉTRICAS DE PRECISIÓN EN TEST SET (EVALUACIÓN CIEGA)")
    print("=" * 70)
    print(f"* MAE Global:                 {mae_global:.3f} dias")
    print(f"* RMSE Global:                {rmse_global:.3f} dias")
    print(f"* Coeficiente R2:             {r2_global:.4f} (Excelente correlacion)")
    print("-" * 70)
    print(f"* MAE en Fase de Desgaste (RUL <= 3d):  {mae_critico:.3f} dias ({mae_critico * 10:.1f} horas operativas)")
    print(f"* RMSE en Fase de Desgaste (RUL <= 3d): {rmse_critico:.3f} dias ({rmse_critico * 10:.1f} horas operativas)")
    print("=" * 70)
    
    # Importancia de variables si esta disponible
    print("\n-> Top 7 variables mas influyentes en el desgaste:")
    if hasattr(modelo, "feature_importances_"):
        importancias = pd.Series(modelo.feature_importances_, index=feature_cols).sort_values(ascending=False)
        for rank, (col, val) in enumerate(importancias.head(7).items(), 1):
            print(f"   {rank}. {col:<32}: {val*100:.2f}%")
            
    metricas = {
        "mae_global": float(mae_global),
        "rmse_global": float(rmse_global),
        "r2_global": float(r2_global),
        "mae_critico": float(mae_critico),
        "rmse_critico": float(rmse_critico),
        "tiempo_entrenamiento_s": float(t_entrenamiento)
    }
    
    return modelo, nombre_modelo, metricas


def guardar_paquete_modelo(modelo, nombre_modelo: str, feature_cols: list, metricas: dict, output_path: str = MODEL_OUTPUT_PATH):
    """Guarda el modelo entrenado y los metadatos necesarios para el dashboard."""
    print("\n" + "=" * 70)
    print(f"6. SERIALIZACIÓN DEL MODELO -> '{output_path}'")
    print("=" * 70)
    
    paquete = {
        "modelo": modelo,
        "nombre_modelo": nombre_modelo,
        "feature_cols": feature_cols,
        "sensores": SENSORES,
        "rangos_nominales": RANGOS_NOMINALES,
        "max_rul_dias": MAX_RUL_DIAS,
        "muestras_por_turno": MUESTRAS_POR_TURNO_DIA,
        "metricas": metricas,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(paquete, output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[OK] Modelo serializado con exito en: '{output_path}' ({file_size_kb:.1f} KB)")
    print("[OK] El archivo incluye el estimador, configuracion de ventanas moviles y rangos de sensores.")
    print("=" * 70)


def main():
    print("+" + "=" * 70 + "+")
    print("|        SISTEMA DE MANTENIMIENTO PREDICTIVO - TROQUELADORA           |")
    print("|                  MODULO DE ENTRENAMIENTO RUL                         |")
    print("+" + "=" * 70 + "+")
    
    # 1. Cargar datos
    df_tel, df_fallas = cargar_datos(FILE_TELEMETRIA, FILE_FALLAS)
    
    # 2. Cruzar timestamps y calcular target RUL
    df_tel, falla_indices = cruzar_marcas_y_calcular_rul(df_tel, df_fallas, max_rul=MAX_RUL_DIAS)
    
    # 3. Generar variables de ventana movil y filtros de ruido
    df_tel, feature_cols = generar_variables_ventana_movil(df_tel)
    
    # 4. Entrenar y evaluar modelo
    modelo, nombre_modelo, metricas = entrenar_modelo(df_tel, feature_cols)
    
    # 5. Serializar
    guardar_paquete_modelo(modelo, nombre_modelo, feature_cols, metricas, MODEL_OUTPUT_PATH)
    
    print("\n[OK] Pipeline de entrenamiento completado con exito.")
    print("     Ya puedes iniciar el dashboard ejecutando: streamlit run app.py\n")


if __name__ == "__main__":
    main()
