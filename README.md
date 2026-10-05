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
