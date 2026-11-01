# Prueba de los modelos con otros dos datasets reales: SKAB y Hydraulic Systems

**Conclusión:**
- **SKAB:** ninguno de los dos modelos detecta las anomalías. El AUC queda entre 0,37 y 0,53, es decir, azar o incluso al revés.
- **Hydraulic Systems:** el modelo v2 baja claramente el RUL cuando el sistema está muy degradado (AUC 0,91), pero solo reacciona a **un** componente: el enfriador, cuya falla sube la temperatura del aceite unos 19 °C. No reacciona a la válvula, a la fuga de la bomba ni al acumulador. El modelo original reacciona menos (AUC 0,78) y pasa el 74 % del tiempo normal en advertencia.

En ambos datasets se confirma que los modelos dependen casi solo de la temperatura.

Ninguno de los dos datasets trae RUL. Por eso **no se mide el error en horas**, sino si el RUL predicho **baja cuando el equipo está peor**. Se probaron `modelo_rul.joblib` (original) y `modelo_rul_v2.joblib` (v2), sin reentrenar.

## 1. SKAB (Skoltech Anomaly Benchmark)

**Qué es:** un banco de pruebas real con una bomba de agua (Skoltech, Rusia). Tiene 34 experimentos de unos 20 minutos, con una lectura por segundo. En cada uno se provoca una anomalía de unos 7 minutos a partir del minuto 10: cierre de válvulas, desbalance del rotor, cavitación, fugas, agua caliente...

| Variable del modelo | Sensor de SKAB |
|---|---|
| Temperatura | `Temperature`: temperatura del cuerpo del motor (°C) |
| Presión | `Pressure`: presión después de la bomba (bar) |
| Vibración | `Accelerometer1RMS`: vibración en aceleración (g) |
| Corriente | `Current`: corriente del motor (A) |

**Traslado:** se toma como media normal la parte previa a la anomalía de cada experimento, y como desviación normal la del experimento sin anomalías.

**Dos escalas de tiempo**, porque los experimentos duran menos que la ventana de 60 tomas del modelo:
- **Real:** lecturas promediadas por minuto. Quedan solo unas 20 filas por experimento.
- **Comprimida:** cada segundo cuenta como una fila. Es una prueba favorable para el modelo, porque las ventanas se llenan.

| Modelo · escala | RUL medio normal | RUL medio en anomalía | AUC anomalía vs normal | Experimentos donde baja el RUL |
|---|---:|---:|---:|---:|
| Original · minuto | 6,85 | 7,09 | 0,45 | 16 de 34 |
| v2 · minuto | 6,52 | 7,28 | **0,37** | 7 de 34 |
| Original · segundo | 6,77 | 6,62 | 0,53 | 21 de 34 |
| v2 · segundo | 5,99 | 5,72 | 0,53 | 20 de 34 |

El AUC es la probabilidad de que una toma anómala reciba menos RUL que una normal: 0,5 equivale al azar.

**Resultado:** ninguno de los dos modelos distingue las anomalías. Con la escala real, el v2 incluso da **más** RUL durante la anomalía. Ninguna anomalía alcanzó el nivel crítico de forma relevante: como mucho el 0,3 % del tiempo. Gráfico: `skab_ejemplos.png`.

**Límite:** las anomalías de SKAB son cambios bruscos de pocos minutos, no un desgaste progresivo. Es una prueba exigente para un modelo de RUL.

## 2. Condition Monitoring of Hydraulic Systems (UCI)

**Qué es:** un banco hidráulico real de laboratorio (ZeMA, Alemania). Tiene 2 205 ciclos de 60 s, y en cada ciclo cuatro componentes están en distintos niveles de degradación:
- Enfriador.
- Válvula.
- Fuga interna de la bomba.
- Acumulador.

Cada ciclo se resumió en una fila, así que 1 ciclo = 1 minuto.

| Variable del modelo | Sensor | Rango por ciclo |
|---|---|---|
| Temperatura | `TS1` (°C) | 35,3–57,9 |
| Presión | `PS1` (bar) | 155,4–180,9 |
| Vibración | `VS1` (**mm/s**, como en la troqueladora) | 0,52–0,84 |
| Corriente | `EPS1`: **potencia** del motor (W), a falta de corriente | 2 362–2 741 |

**Severidad:** cada componente va de 0 (óptimo) a 1 (cerca de la falla total), y la severidad total va de 0 a 4. La base normal son los 50 ciclos estables con severidad ≤ 0,5. Los "muy degradados" son los 760 ciclos estables con severidad ≥ 2.

| Modelo | RUL medio en la base | RUL medio muy degradado | AUC degradado vs base | Correlación RUL–severidad | Base en advertencia |
|---|---:|---:|---:|---:|---:|
| Original | 3,81 | 3,35 | 0,78 | −0,35 | **74 %** |
| v2 | **6,34** | **2,46** | **0,91** | −0,48 | 20 % |

### Por componente: RUL medio del modelo v2

| Componente | Óptimo → cerca de la falla total |
|---|---|
| **Enfriador** (100 → 20 → 3 %) | **6,37 → 1,69 → 1,68** |
| Válvula (100 → 90 → 80 → 73 %) | 3,29 → 3,22 → 3,31 → 3,25 |
| Fuga de la bomba (0 → 1 → 2) | 3,66 → 2,80 → 3,33 |
| Acumulador (130 → 115 → 100 → 90 bar) | 3,82 → 3,04 → 2,94 → 3,26 |

Gráficos: `hidraulico_por_componente.png` y `hidraulico_ciclos.png`.

**Resultado:** el buen AUC del v2 se debe casi por completo al **enfriador**. Cuando el enfriador falla, la temperatura del aceite sube de 36 °C (100 %) a 45 °C (20 %) y a 55 °C (3 %), y el modelo, que depende sobre todo de la temperatura, reacciona. Con la válvula, la bomba y el acumulador el RUL apenas cambia ni sigue un orden.

Además, en este banco los tres niveles del enfriador se registraron en **tres bloques de tiempo seguidos**. Parte del efecto podría deberse a cambios a lo largo del ensayo y no solo al enfriador.

El modelo original pasa el 74 % del tiempo normal en advertencia: con los rangos de este banco, todo le parece degradado.

## Qué se puede afirmar

- Con datos reales de otras máquinas, los modelos **solo detectan fallas que suben claramente la temperatura**. Las fallas que afectan a la presión, el caudal o la vibración no se detectan.
- El modelo v2 se comporta algo mejor que el original en el banco hidráulico: menos falsas advertencias y más diferencia entre normal y degradado. En SKAB no.
- Junto con MetroPT-3, son tres datasets reales en los que ningún modelo estima el RUL de forma útil.

## Límites

- Ninguno de los dos datasets tiene RUL ni fallas que lleguen a ocurrir. Solo se mide coherencia, no error.
- SKAB: experimentos muy cortos y anomalías bruscas. La escala comprimida es artificial.
- Hydraulic Systems: degradación provocada en laboratorio, potencia en lugar de corriente y enfriador confundido con el orden temporal.
- Las equivalencias de sensores son aproximadas. Por ejemplo, vibración en g frente a mm/s en SKAB.

## Reproducción

```bash
python probar_modelos_otros_datasets.py
```

Descarga SKAB (~4 MB, GitHub `waico/SKAB`) y Hydraulic Systems (~76 MB comprimido, UCI, CC BY 4.0) en `datos_externos/` si no existen. Esa carpeta no se sube a git. No modifica modelos ni datos.
