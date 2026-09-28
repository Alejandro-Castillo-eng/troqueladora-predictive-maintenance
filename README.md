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
  - Soporta visualización de la secuencia completa anual (enero a diciembre de 2026) y 4 escenarios de validación operativa.
  - Listo para ser visualizado en local o alojado directamente en **GitHub Pages**.
- **📊 Aplicación Interactiva en Streamlit (`app.py`)**:
  - Dashboard completo con gráficos interactivos Plotly WebGL, análisis de causa raíz, matriz de correlaciones y contraste con la bitácora de paradas (*Ground Truth*).

---

## 🎛️ Variables de Telemetría Monitoreadas

| Sensor | Rango Nominal | Unidad | Subsistema Asociado |
|---|---|---|---|
| 🌡️ **Temperatura Motor** | 50.0 - 55.0 | °C | Devanados, ventilación y refrigeración de planta |
| 💧 **Presión Hidráulica** | 210.0 - 220.0 | bar | Bomba principal, acumuladores y válvula de alivio |
| 📳 **Vibración RMS** | 1.00 - 1.60 | mm/s | Rodamientos de apoyo, bancada y guías de deslizamiento |
| ⚡ **Corriente Motor** | 20.0 - 22.5 | A | Consumo eléctrico del estator y resistencia mecánica |

---

## 📁 Estructura del Repositorio

```text
├── index.html                           # Dashboard interactivo autónomo (Web / GitHub Pages)
├── build_html.py                        # Generador del HTML con datos embebidos
├── app.py                               # Dashboard avanzado en Streamlit
├── train_model.py                       # Pipeline de entrenamiento y validación de Machine Learning
├── modelo_rul.joblib                    # Modelo entrenado serializado (XGBoost + metadatos)
├── telemetria_troqueladora.csv          # Telemetría histórica de entrenamiento (2025)
├── telemetria_troqueladora_test_2026.csv # Telemetría de prueba y validación (2026)
├── log_fallas_mantenimiento.csv         # Bitácora histórica de eventos de falla (Ground Truth)
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Documentación técnica
```

---

## 🚀 Puesta en Marcha

### 1. Visualización Rápida en Navegador
Para abrir el Tablero Andon interactivo sin necesidad de instalar Python:
1. Clona el repositorio o descarga el archivo `index.html`.
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
   - **MAE**: < 0.85 días en conjunto de prueba independiente.
   - **Alerta temprana promedio**: Detección con 3.5 a 5.0 días de antelación antes de una parada crítica.

---

## 👨‍💻 Autor
Desarrollado por **Alejandro Castillo**.
