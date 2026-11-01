# Validación externa del modelo con máquinas CNC (Industrial Machine PdM syn)

**Conclusión:** con las 5 máquinas CNC del dataset público, el modelo original **detecta parcialmente la cercanía de una falla**:
- En las últimas 10 h antes de una falla, asigna menos vida restante que lejos de ella en el **69 %** de los casos. El intervalo de confianza del 95 % va del 58 % al 81 %.
- Emitió una **alerta crítica antes de 9 de las 15 fallas (60 %)**.
- Sus alertas críticas son **45 veces más frecuentes cerca de la falla que lejos**.

**No estima con precisión las horas que faltan:** su error medio (21,9 h) no mejora al de predecir siempre el promedio (20,2 h).

## Qué se probó

| | |
|---|---|
| Modelo | `modelo_rul.joblib` (original), **sin reentrenar** |
| Dataset | Industrial Machine Predictive Maintenance (syn), Kaggle, CC0. Es **simulado** |
| Máquinas | Solo las **5 CNC** (máquinas 1 a 5) |
| Periodo | 1 al 14 de enero de 2024. 65 826 minutos evaluables (~1 100 h) |
| Fallas | **15**: 5 de rodamiento, 2 hidráulicas, 2 de sobrecalentamiento, 1 eléctrica y 5 sin tipo indicado |
| Preparación | Igual que `probar_modelo_industrial_syn.py`. Una toma por minuto. Temperatura y vibración con sus **valores originales** (están en los rangos de la troqueladora). Corriente y presión trasladadas a la escala de la troqueladora usando el estado normal de las CNC (RUL > 48 h). Se excluyen los minutos con RUL de relleno (0,5 h) |
| RUL | `rul_hours` ÷ 10. En el modelo, 1 unidad = 600 tomas de 1 min = 10 h |

**Por qué CNC:** es la máquina de mecanizado industrial más parecida a una prensa. Hay que dejarlo claro: **la elección se hizo después de ver los resultados de los cuatro tipos**. Con las 20 máquinas juntas, el resultado es AUC 0,64 y alerta en 16 de 63 fallas. Las CNC son el tipo con mejor resultado.

## Resultados

| Métrica | Resultado | IC 95 % | En palabras |
|---|---:|---:|---|
| **AUC: últimas 10 h frente a ≥ 48 h de la falla** | **0,69** | 0,58–0,81 | En el 69 % de los casos, una toma cercana a la falla recibe menos RUL que una lejana (50 % = azar) |
| AUC: últimas 24 h frente a ≥ 48 h | 0,66 | 0,59–0,75 | |
| **Fallas con alerta crítica previa** (RUL ≤ 1 durante 30 min) | **9 de 15 (60 %)** | 36 %–80 % | |
| Antelación de esas alertas | mediana 26 h | (de 2 a 60 h) | Avisa antes de lo que indica "RUL ≤ 1" (10 h) |
| Alertas críticas por cada 100 h: cerca (≤ 10 h) / intermedio / lejos (≥ 48 h) | 11,8 / 3,5 / **0,26** | | **45 veces más** frecuentes cerca que lejos |
| Tiempo lejos de la falla en crítico | 0,8 % | | Casi sin falsas alertas críticas |
| Tiempo lejos de la falla en advertencia (RUL ≤ 4) | **67,8 %** | | El umbral de advertencia **no sirve** aquí: casi siempre está activo |
| RUL predicho medio: últimas 10 h / lejos | 2,50 / 3,64 | | Baja, pero poco |
| Correlación de rangos entre RUL real y predicho | 0,24 | | Débil |
| Error medio absoluto | 21,9 h | | La referencia "siempre el promedio" da 20,2 h; R² −0,31 |

El IC 95 % del AUC se calculó con bootstrap remuestreando los ciclos de falla (2 000 repeticiones). El de las alertas usa el intervalo de Wilson.

### Por máquina

| Máquina | Fallas | Con alerta crítica | AUC (10 h) |
|---:|---:|---:|---:|
| 1 | 3 | 2 | 0,74 |
| 2 | 3 | 2 | 0,74 |
| 3 | 3 | 2 | 0,78 |
| 4 | 4 | 2 | 0,67 |
| 5 | 2 | 1 | 0,53 |

### Por tipo de falla

| Tipo de falla | Fallas | Con alerta crítica | RUL predicho medio en las últimas 10 h |
|---|---:|---:|---:|
| **Rodamiento** | 5 | **5** | 1,25 |
| Hidráulica | 2 | 2 | 1,77 |
| Sobrecalentamiento | 2 | 1 | 2,72 |
| Eléctrica | 1 | 0 | 3,26 |
| Sin tipo indicado | 5 | 1 | 3,76 |

El modelo detecta mejor las fallas de **rodamiento**: la vibración de las CNC sube hasta ~3 mm/s antes de la falla, y la vibración es la segunda variable que más usa el modelo. Las fallas eléctricas y las que no tienen tipo indicado casi no cambian la temperatura ni la vibración, y el modelo no las detecta.

Gráficos:
- `maquinas_cnc.png`: las 5 máquinas, con ✓ o ✗ en cada falla.
- `rul_predicho_vs_real_cnc.png`: RUL predicho frente a real.
- `sensores_cnc_antes_de_falla.png`: sensores de las CNC antes de la falla.

## Cómo se comportan los sensores de las CNC antes de la falla

- **Temperatura:** sube de ~48–53 °C (más de 60 h antes) a ~62–64 °C (últimas 10 h).
- **Vibración:** sube de ~1,5–2,0 a ~3,0 mm/s.
- **Corriente:** sube un poco, de ~8–10 a ~11–12 A.
- **Presión:** no cambia.
- La subida es gradual en las últimas ~40 h. Pero queda mezclada con la variación normal entre los modos reposo, normal y pico, por eso la detección es parcial.

## Cómo presentarlo

> "Como validación externa se usó un dataset público independiente con las mismas cuatro variables. Se analizaron las 5 máquinas CNC: 15 fallas, 14 días. Sin reentrenar, el modelo distingue los momentos cercanos a una falla de los lejanos en el 69 % de los casos (IC 95 %: 58–81 %) y emitió alerta crítica antes de 9 de las 15 fallas, con muy pocas falsas alertas críticas. No estima con precisión las horas restantes. Con todos los tipos de máquina el resultado es menor (64 %), y con datos reales de otras máquinas (MetroPT-3, SKAB, Hydraulic Systems) no se observó capacidad predictiva útil."

## Límites

- El dataset es **simulado**. Valida el modelo frente a otra simulación, no frente a una prensa real.
- Solo hay **15 fallas** en 5 máquinas, por eso los intervalos de confianza son amplios.
- **CNC se eligió después de ver los resultados por tipo.** Debe mencionarse junto con el resultado global (64 %).
- Las lecturas originales, cada ~16 min, se interpolaron a 1 min.
- El umbral de advertencia (RUL ≤ 4) está activo casi siempre. Solo el umbral crítico aporta información.

## Reproducción

```bash
python validar_modelo_cnc.py
```

Usa el dataset de `datos_externos/industrial_syn/` (lo descarga si falta, ~2 MB) y no modifica modelos ni datos.
