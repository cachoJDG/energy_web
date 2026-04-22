# Como correr y usar el proyecto

## Requisitos

- Tener `python3` instalado.
- Estar dentro de la carpeta del proyecto (`energy_web`).

## Opcion rapida (recomendada)

Usa el script de arranque:

```bash
./run_project.sh
```

Que hace este script:

1. Crea `.venv` si no existe.
2. Instala dependencias desde `requirements.txt`.
3. Levanta la app con Streamlit.

Cuando arranca, Streamlit muestra una URL local (normalmente `http://localhost:8501`).

## Primer uso en Linux/macOS

Si al ejecutarlo te dice permiso denegado:

```bash
chmod +x run_project.sh
./run_project.sh
```

## Uso dentro de la app

1. En el sidebar, en **Source**, elige:
   - `Demo dataset` para probar rapido, o
   - `Upload file` para subir tus datos.
2. Sube un archivo:
   - CSV (`.csv`) o Excel (`.xlsx`, `.xls`).
   - Puedes usar el archivo del repo: `household_data.xlsx`.
3. Elige el horizonte de prediccion:
   - `Next 24 hours` o
   - `Next 7 days`.
4. Revisa las pestanas:
   - **Overview**: metricas y consumo historico,
   - **Forecast**: rendimiento del modelo y pronostico,
   - **Patterns**: patrones horarios/dias,
   - **Recommendations**: sugerencias automaticas.

## Que cambio para mejorar uploads (importante)

- La app ahora compara dos enfoques en validacion:
  - `RandomForest`
  - `Naive seasonal (24h)`
- Para el forecast futuro usa automaticamente el que mejor MAE tenga.
- En **Forecast** veras el modelo activo en `Active forecast for future horizon`.

## Data quality checks

En **Overview** hay un bloque llamado **Data quality checks** con:

- `rows`
- `time_range`
- `coverage_before_fill_pct`
- `missing_hours_before_fill`
- `mean_kWh`
- `max_kWh`
- `converted_from_cumulative`

Si `converted_from_cumulative = 1`, la app detecto que la serie parecia acumulada y la convirtio a consumo horario por diferencia.

Esto mejora mucho los graficos y evita curvas sin sentido al entrenar.

## Prints en consola al subir archivo

Al hacer upload, veras logs como:

```text
[UPLOAD] file=... rows=... range=... coverage_before_fill=... cumulative_converted=...
[UPLOAD] Loaded OPSD-style data (series: ...)
```

Sirve para debug rapido del dataset seleccionado.

## Formatos de datos soportados

- Formato simple:
  - `datetime` + `consumption_kWh`, o
  - `fecha/date` + `hora/hour` + columna de consumo.
- Formato Open Power System Data (multi-header), como `household_data.xlsx`.

## Detener la app

En la terminal donde corre Streamlit, presiona `Ctrl + C`.
