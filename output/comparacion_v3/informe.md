# Dataset v3 (fallas reales de 2024) y comparación de los tres modelos

**Conclusión:** se construyó un dataset de 2024 sobre las **27 fallas reales** de la prensa (`log_fallas_2024.csv`). Cada causa tiene su propia firma en los sensores, según el cuadro causa → variables, y las magnitudes son realistas. Con esos datos, el modelo v3 resuelve un problema mucho más difícil que los anteriores: en su propia validación distingue el último día antes de la falla con un **AUC de 0,67**, frente a 1,00 de los modelos anteriores. Es un valor más creíble para datos con fallas sutiles, repentinas y de causas distintas.

En las pruebas externas, el v3 **no mejora en general**:
- Mejora algo en SKAB (0,60 frente a 0,53).
- Queda entre los otros dos en Hydraulic Systems (0,87).
- Empeora en MetroPT-3 (0,46–0,59) y en Industrial syn (0,59).
- Sus alertas críticas aparecen antes de más fallas, pero con **muchas más falsas alarmas**, así que no indican una mejor detección.

## 1. Cómo se construyó el dataset v3

Script: `generar_dataset_v3.py`. Salida: `telemetria_troqueladora_2024_v3.csv`.

| Elemento | Origen |
|---|---|
| **Fallas: fecha, causa, inicio y fin** | **Reales**: `log_fallas_2024.csv`, 27 eventos de 2024 |
| Calendario | Real: lunes a sábado, 08:00–17:59. Sin lecturas durante cada parada (inicio → fin) |
| Comportamiento de los sensores | **Simulado**, con parámetros de referencias técnicas generales |

### Firma de cada causa (cuadro del usuario)

| Causa | Fallas | Qué cambia antes de la falla | Magnitud | Duración del desgaste |
|---|---:|---|---|---|
| Desajuste de presión de corte | 13 | **Presión ↓** y fluctúa más | −8 a −18 bar | 0,5–4 días de turno |
| Desalineación del troquel | 8 | **Vibración ↑** y fluctúa más | hasta 2,8–4,5 mm/s (zonas C–D de la ISO 10816-3) | 1–6 días |
| Sobrecarga del motor principal | 6 | **Corriente ↑** y **temperatura ↑** con retraso térmico | 110–125 % de la corriente nominal; +5 a +15 °C | 1–5 días |

### Otros elementos

- **Variabilidad entre fallas.** La forma del desgaste puede ser lineal, exponencial o escalonada, y la intensidad varía. **6 de las 27 fallas son casi repentinas**: 30–180 minutos de aviso y la mitad de la magnitud.
- **La reparación reinicia el desgaste de esa causa.** Por ejemplo, EV-14 ocurrió durante la reparación de EV-13 y casi no tiene desgaste propio.
- **Operación normal:**
  - Cambios de carga por trabajo o troquel, cada 1–4 h, del 92 % al 108 %, que afectan a la corriente, la presión, la temperatura y la vibración.
  - Calentamiento del motor al inicio de cada turno.
  - Variación diaria.
  - Picos transitorios aislados de vibración y presión.
- **Valores normales** iguales a los del proyecto: 52 °C, 214 bar, 1,3 mm/s y 21 A.
- **No se usó ningún dataset externo para calibrar.** Así, MetroPT-3, Industrial syn, SKAB y Hydraulic Systems siguen siendo pruebas independientes para el v3.

Parámetros de cada falla en `output/dataset_v3/parametros_por_falla.csv`. Gráficos en `output/dataset_v3/`:
- `ejemplos_por_causa.png`
- `anio_2024.png`

## 2. Modelo v3

Script: `entrenar_modelo_v3.py`. Salida: `modelo_rul_v3.joblib`. Mismo método de `train_model.py`. La falla se sitúa en la hora de inicio. Los dos pares de fallas solapadas se convierten en un solo punto de falla cada uno, así que quedan 25 puntos.

- Entrenamiento: 1 de enero a 19 de octubre de 2024.
- Validación: 19 de octubre a 31 de diciembre, con 6 fallas, **3 de ellas repentinas**.

Variables más importantes:

| | Original | v2 | **v3** |
|---|---|---|---|
| 1 | temperatura media 15 min: 73,7 % | z-score de temperatura: 28,8 % | **índice de estrés: 37,8 %** |
| 2 | vibración: 11,4 % | temperatura media 60 min: 13,5 % | z-score de presión: 5,4 % |
| 3 | temperatura media 60 min: 1,9 % | índice de estrés: 11,6 % | presión media 60 min: 5,3 % |

El v3 es el primero que **usa las cuatro variables**: presión, vibración y corriente además de la temperatura. El índice de estrés las combina todas.

## 3. Comparación (AUC: 0,5 = azar, 1 = distingue siempre)

| Prueba | Original | v2 | v3 |
|---|---:|---:|---:|
| Validación propia (último 20 %) | 1,00 | 1,00 | **0,67** |
| MetroPT-3 (real), base fija | 0,66 | 0,65 | 0,59 |
| MetroPT-3 (real), base móvil | 0,62 | 0,61 | 0,46 |
| Industrial syn, 20 máquinas | 0,64 | 0,69* | 0,59 |
| Industrial syn, 5 CNC | 0,69 | 0,71* | 0,59 |
| SKAB (real), minuto | 0,45 | 0,37 | 0,50 |
| SKAB (real), segundo | 0,53 | 0,53 | **0,60** |
| Hydraulic Systems (real) | 0,78 | **0,91** | 0,87 |

\* No es independiente: el dataset v2 se construyó con Industrial syn.

### Alertas críticas y falsas alarmas

| Prueba | Modelo | Fallas con alerta previa | Tiempo normal o lejano en crítico |
|---|---|---:|---:|
| Validación propia | original / v2 / v3 | 4 de 4 / 3 de 4 / **2 de 6** | 0 % / 0 % / 0 % |
| MetroPT-3, base fija | original / v2 / v3 | 0 de 4 / 2 de 4 / 3 de 4 | 0 % / 6,8 % / **4,4 %** |
| Industrial syn, 20 máquinas | original / v2 / v3 | 16 / 38 / 44 de 63 | 0,3 % / 2,3 % / **9,2 %** |
| Industrial syn, CNC | original / v2 / v3 | 9 / 13 / 11 de 15 | 0,8 % / 3,5 % / **11,8 %** |

Con el v3 aparecen alertas críticas antes de más fallas, pero también pasa entre el 4 % y el 12 % del tiempo **sin falla cercana** en estado crítico. Esas alertas no muestran mejor detección.

### Hydraulic Systems por componente (RUL predicho medio, de óptimo a casi falla total)

| Componente | Original | v2 | v3 |
|---|---|---|---|
| Enfriador (sube la temperatura) | 3,9 → 3,5 → 3,0 | 6,4 → 1,7 → 1,7 | **5,8 → 2,8 → 0,7** |
| Válvula | sin tendencia | sin tendencia | 3,2 → 3,2 → 3,2 → 2,9 |
| Fuga de la bomba | sin tendencia | sin tendencia | sin tendencia |
| Acumulador | sin tendencia | sin tendencia | sin tendencia |

Los tres modelos responden sobre todo a la falla que sube la temperatura. El v3 responde de forma más ordenada (baja en cada nivel) y apenas reacciona a la válvula en su peor nivel.

## Por qué el v3 no mejora con datos externos

1. **El problema que aprende es más difícil, y eso es más realista.** Con fallas de magnitud moderada, un 22 % de fallas repentinas y un desgaste de solo 0,5–6 días, el RUL a 10 días no se puede deducir de los sensores durante gran parte del ciclo. En su validación, 3 de las 6 fallas son repentinas.
2. **Hay pocos ejemplos por causa:** 6 sobrecargas, 8 desalineaciones y 13 desajustes de presión. Cada firma se aprende con muy pocos casos.
3. **Las firmas son específicas de una prensa.** Una fuga de aire en un compresor (MetroPT-3) o una válvula en un banco hidráulico no se parecen a un desajuste de presión de corte ni a una desalineación de troquel.

## Qué se puede afirmar

- El **dataset v3 es el más defendible** de los tres: sus fallas (fechas, horas, causas y duración de las paradas) son **reales**, y cada causa afecta solo a sus variables, con magnitudes justificadas por referencias técnicas.
- Su validación propia (AUC 0,67) es una **estimación más honesta** de lo que se podría esperar con fallas de este tipo que el 1,00 de los modelos anteriores.
- Ningún modelo entrenado con datos simulados, ni siquiera calibrados, **generaliza de forma útil a máquinas reales distintas**. Para validar de verdad hace falta telemetría real de la prensa junto con esta bitácora real.

## Límites

- El comportamiento de los sensores sigue siendo **simulado**. Solo las fallas son reales.
- La validación propia del v3 tiene solo 6 fallas, 3 de ellas repentinas.
- Las magnitudes y duraciones son supuestos razonables, no mediciones de la prensa.

## Reproducción

```bash
python generar_dataset_v3.py
```

```bash
python entrenar_modelo_v3.py
```

```bash
python comparar_modelos_v3.py
```

Ningún script modifica los datos ni los modelos anteriores.
