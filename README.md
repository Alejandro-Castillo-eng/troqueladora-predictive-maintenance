# ⚙️ Sistema de Mantenimiento Predictivo & Tablero Andon 4.0 · Prensa Troqueladora Industrial

Sistema integral de analítica predictiva de fallas y monitoreo en tiempo real para prensas troqueladoras industriales de alta cadencia. Integra modelos de Machine Learning (XGBoost) para estimación de **Vida Útil Remanente (RUL - Remaining Useful Life)** y un **Tablero Andon 4.0** interactivo para la detección temprana de anomalías electromecánicas e hidráulicas.

---

## 📌 Características Principales

- **🧠 Algoritmo de Estimación RUL (XGBoost Regressor)**:
  - Predice con precisión el número de días restantes antes de una parada no programada ($RUL \le 10$ días).
  - Basado en ventanas móviles (15 y 60 min), tendencias de pendiente, desviaciones respecto a tolerancias nominales e **Índice de Estrés Termomecánico**.
- **🛡️ Filtro Inteligente Anti-Falsas Alarmas**:
  - Distingue picos transitorios y ruido de sensor aislado (por rebotes o variaciones eléctricas puntuales) de patrones de **desgaste acumulado sostenido**.
- **🌐 Tablero Andon Web Autónomo (`index.html`)**:
  - Archivo HTML estandarizado con JavaScript y CSS nativo (sin dependencias externas).
  - Incluye un calendario para consultar la telemetría día por día y 4 escenarios de validación operativa.
  - Cada día conserva todas sus filas. La reproducción recorre una lectura por paso y, después de la última toma del día, limpia el gráfico y continúa con el siguiente día disponible. Se detiene al terminar el dataset.
  - Por defecto, el gráfico muestra la adquisición progresiva del día. También permite ver el trazo completo del día elegido. Los horarios corresponden a las lecturas reales, sin rellenar pausas con datos inventados.
  - Vibración, temperatura, presión y corriente se muestran en cuatro gráficos separados y sincronizados con la misma fecha y toma.
  - El botón de la cabecera alterna entre modo claro y oscuro, adapta también los gráficos y guarda la preferencia en el navegador.
  - El resumen lateral se genera únicamente al llegar a la última toma registrada del día. Incluye promedios, mínimos, máximos, estados del panel y RUL estimado; conserva el último resumen al avanzar al siguiente día.
  - El selector de sensor ofrece ejemplos de normalidad, pico transitorio, desgaste sostenido y estado crítico extraídos exclusivamente del CSV de test de 2026. Cada ventana contiene 80 filas consecutivas y muestra sus fechas, filas y criterio de selección.
  - Si no existe un ejemplo con los umbrales establecidos, el botón aparece deshabilitado. Actualmente no hay ejemplos de pico transitorio en temperatura ni presión.
  - Las lecturas conservan los valores originales del CSV y el RUL muestra la predicción del modelo guardado. Los escenarios no reentrenan el modelo ni constituyen una confirmación de falla real.
  - Listo para ser visualizado en local o alojado directamente en **GitHub Pages**.
- **📊 Aplicación Interactiva en Streamlit (`app.py`)**:
  - Dashboard completo con gráficos interactivos Plotly WebGL, análisis de causa raíz, matriz de correlaciones y contraste con la bitácora de paradas (*Ground Truth*).

---

## 🎛️ Variables de Telemetría Monitoreadas

| Sensor | Rango Nominal | Unidad | Subsistema Asociado |
|---|---|---|---|
| 🌡️ **Temperatura Motor** | 50.0 - 55.0 | °C | Devanados, ventilación y refrigeración de planta |
| 💧 **Presión Hidráulica** | 210.0 - 220.0 | bar | Bomba principal, acumuladores y válvula de alivio |
| 📳 **Vibración RMS** | 1.00 - 1.60 | mm/s | Troquel |
| ⚡ **Corriente Motor** | 20.0 - 22.5 | A | Consumo eléctrico del estator y resistencia mecánica |

---

## 📁 Estructura del Repositorio

```text
├── index.html                           # Dashboard interactivo autónomo (Web / GitHub Pages)
├── daily_summary.js                     # Resumen al completar las tomas del día
├── daily_summary.css                    # Diseño del resumen lateral
├── build_html.py                        # Generador del HTML con datos embebidos
├── app.py                               # Dashboard avanzado en Streamlit
├── train_model.py                       # Pipeline de entrenamiento y validación de Machine Learning
├── entrenamiento_modelo_rul.ipynb       # Notebook con el entrenamiento explicado paso a paso
├── probar_modelo_metropt3.py            # Prueba del modelo con el dataset público MetroPT-3
├── probar_modelo_industrial_syn.py      # Prueba del modelo con Industrial Machine PdM (syn)
├── generar_dataset_v2.py                # Telemetría 2025 v2 con comportamiento calibrado (Industrial syn)
├── entrenar_modelo_v2.py                # Entrena modelo_rul_v2.joblib con el mismo método
├── comparar_modelos_v1_v2.py            # Compara modelo original y v2 (incluye MetroPT-3)
├── probar_modelos_otros_datasets.py     # Prueba ambos modelos con SKAB y Hydraulic Systems
├── telemetria_troqueladora_v2.csv       # Telemetría 2025 v2 (mismas fechas, rangos y fallas)
├── modelo_rul_v2.joblib                 # Modelo entrenado con la telemetría v2
├── modelo_rul.joblib                    # Modelo entrenado serializado (XGBoost + metadatos)
├── telemetria_troqueladora.csv          # Telemetría histórica de entrenamiento (2025)
├── telemetria_troqueladora_test_2026.csv # Telemetría simulada de prueba (2026)
├── log_fallas_mantenimiento.csv         # Bitácora histórica de eventos de falla (Ground Truth)
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Documentación técnica
```

---

## 🚀 Puesta en Marcha

### 1. Visualización Rápida en Navegador
Para abrir el Tablero Andon interactivo sin necesidad de instalar Python:
1. Clona el repositorio o descarga y extrae su ZIP completo. Mantén `index.html`, `daily_summary.js` y `daily_summary.css` en la misma carpeta.
2. Haz doble clic en [`index.html`](index.html) para abrirlo en Google Chrome, Microsoft Edge, Firefox, etc.

### 2. Ejecución de la Aplicación Streamlit

```bash
# Clonar el repositorio
git clone https://github.com/Alejandro-Castillo-eng/troqueladora-predictive-maintenance.git
cd troqueladora-predictive-maintenance

# Instalar dependencias
pip install -r requirements.txt

# Iniciar la aplicación
streamlit run app.py
```

### 3. Re-entrenar el Modelo Predictivo

```bash
python train_model.py
```

### 4. Ver el Entrenamiento Paso a Paso (Jupyter Notebook)

`entrenamiento_modelo_rul.ipynb` recorre los mismos pasos que `train_model.py` (carga, exploración, cálculo del RUL, variables, partición, entrenamiento, evaluación e importancia de variables) con explicaciones y gráficos. Se puede leer directamente en GitHub o ejecutar en local:

```bash
pip install jupyter
jupyter notebook entrenamiento_modelo_rul.ipynb
```

El notebook entrena una copia en memoria y **no sobrescribe** `modelo_rul.joblib`.

### 5. Probar el Modelo con un Dataset Público (MetroPT-3)

```bash
python probar_modelo_metropt3.py
```

Descarga [MetroPT-3](https://doi.org/10.24432/C5VW3R) (compresor real del metro de Oporto) en `datos_externos/`, traslada sus sensores a la escala de la troqueladora y mide el error del modelo guardado sin reentrenarlo. Resultados e informe en `output/prueba_metropt3/`. En esta prueba el modelo **no generaliza**: MAE de 29–49 horas y poca capacidad de distinguir los días previos a una falla (AUC 0,51–0,66), porque las fallas reales son mucho más sutiles que las de los datos de entrenamiento.

### 6. Probar el Modelo con un Dataset con las Mismas 4 Variables (Industrial Machine PdM)

```bash
python probar_modelo_industrial_syn.py
```

Usa [Industrial Machine Predictive Maintenance (syn)](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance) (20 máquinas simuladas con temperatura, vibración, corriente, presión y RUL). Resultados e informe en `output/prueba_industrial_syn/`. Aunque tiene las mismas variables, el modelo **no estima bien el RUL** (MAE de 22–25 h, peor que predecir siempre el promedio; AUC 0,63–0,64). Sus alertas críticas sí se concentran cerca de la falla, pero solo aparecen antes de 20 de las 63 fallas.

### 7. Dataset v2 Calibrado y Modelo v2

```bash
python generar_dataset_v2.py
python entrenar_modelo_v2.py
python comparar_modelos_v1_v2.py
```

`telemetria_troqueladora_v2.csv` conserva las columnas, fechas, rangos normales y las 37 fallas de la bitácora, pero toma de Industrial syn las fluctuaciones y la forma del desgaste (fallas de 0–2,4 desviaciones en lugar de 15–20). El modelo v2 detecta esas fallas sutiles en su validación (AUC 1,00, 4 de 4 alertas), pero en datos reales (MetroPT-3) no mejora al original (AUC 0,61–0,65). Informe en `output/comparacion_v1_v2/informe.md`. Los archivos originales no se modifican.

### 8. Otros Datasets Reales: SKAB y Hydraulic Systems

```bash
python probar_modelos_otros_datasets.py
```

Prueba ambos modelos con [SKAB](https://github.com/waico/SKAB) (bomba de agua con anomalías provocadas, ~4 MB) y [Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) (banco hidráulico con componentes degradados). Sin RUL en ninguno, se mide si el RUL predicho baja cuando el equipo está peor. En SKAB ningún modelo detecta las anomalías (AUC 0,37–0,53). En Hydraulic Systems el modelo v2 baja el RUL con el sistema degradado (AUC 0,91), pero solo por la falla del enfriador, que sube la temperatura. Informe en `output/prueba_otros_datasets/informe.md`.

---

## 🧠 Arquitectura del Modelo Predictivo

El modelo se entrena a partir del cruce de telemetría continua y marcas de tiempo de las paradas no programadas documentadas por el equipo de mantenimiento.
1. **Etiquetado RUL**: Cuenta regresiva lineal en días hacia la falla más próxima, acotada a un umbral operativo de 10 días.
2. **Ingeniería de Características**:
   - Medias móviles a corto y mediano plazo (`mean_15`, `mean_60`).
   - Dispersión y volatilidad (`std_60`).
   - Pendientes instantáneas (`trend_15`, `trend_60`).
   - Z-score respecto a valores nominales de fábrica.
   - Índice sintético de severidad termomecánica.
3. **Métricas de Desempeño**:
   - Modelo reentrenado con cuatro sensores y 29 características de entrada.
   - **MAE global**: 1.130 días; **RMSE global**: 1.432 días; **R²**: 0.8174.
   - **MAE en fase crítica (RUL ≤ 3 días)**: 0.048 días operativos (aproximadamente 29 minutos).
   - Estas métricas corresponden al último 20 % de 2025; el primer 80 % se utiliza para el ajuste.
   - El dataset de 2026 se utiliza para generar predicciones independientes. Para calcular su error se necesita una bitácora de fallas de 2026.

---

## 👨‍💻 Autor
Desarrollado por **Alejandro Castillo**.
