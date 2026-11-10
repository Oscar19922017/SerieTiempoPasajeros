# ✈️ ¿Cuántos pasajeros tendremos el próximo año?

Aplicación de **Streamlit** que pronostica los pasajeros mensuales de una
aerolínea usando la serie histórica "airline passengers" (1949–1960).

Responde una pregunta de negocio sencilla: **¿cuántos pasajeros habrá en 1961?**,
y de paso enseña *cómo se compara* el desempeño de varios modelos de pronóstico.

## ¿Qué se ve en la app?

Cuatro pestañas, cada una con su gráfico y una explicación en palabras simples:

1. **La historia** — pasajeros mes a mes durante 12 años.
2. **La temporada** — qué meses del año son temporada alta y baja.
3. **El examen** — se esconden los últimos 24 meses y se comparan **5 modelos**
   para ver cuál se equivoca menos (con una tabla de errores y el ganador).
4. **El pronóstico** — los 12 meses de 1961, eligiendo el modelo con un menú.

### Modelos comparados

| Modelo | Idea |
| --- | --- |
| **Copiar el año pasado** | cada mes igual al mismo mes del último año |
| **Suavizamiento exponencial (Holt-Winters)** | tendencia + temporada |
| **SARIMA** | modelo clásico (configuración "airline" 0,1,1)(0,1,1,12) |
| **Regresión lineal** | una recta que crece + el efecto de cada mes |
| **Prophet** | modelo moderno que aprende tendencia y temporada solo |

En la prueba sobre los últimos 24 meses gana el **Suavizamiento exponencial
(Holt-Winters)** con ~6.4 % de error promedio.

## Archivos del proyecto

```
ProyectoSerieTiempo/
├── app.py                 # La aplicación (Streamlit)
├── datos.py               # Consigue los datos (internet o archivo local)
├── requirements.txt       # Dependencias para Streamlit Community Cloud
├── iniciar_app.bat        # Doble clic para abrir la app en Windows
├── .streamlit/
│   └── config.toml        # Tema y configuración
├── Datos/
│   ├── airline-passengers.csv     # Copia local de los datos (respaldo)
│   ├── apple_diario.csv
│   └── ventas_diarias_olist.csv
├── _prueba_datos.py       # Prueba de datos.py
└── _prueba_app.py         # Prueba de la lógica de app.py (sin abrir la web)
```

> Los datos se bajan desde GitHub (jbrownlee/Datasets). Si no hay internet, la
> app usa automáticamente `Datos/airline-passengers.csv`.

## Correr la app en tu computador

Con [uv](https://docs.astral.sh/uv/) (recomendado):

```bash
uv run python -m streamlit run app.py
```

Alternativa con pip:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Luego abre <http://localhost:8501>.

## Desplegar en Streamlit Community Cloud

1. **Sube el proyecto a GitHub** (este repositorio).
2. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
3. Clic en **"New app"** → **"Deploy a public app from GitHub"**.
4. Elige el repositorio, la rama (`main`) y el **archivo principal: `app.py`**.
5. Antes de publicar, abre **"Advanced settings"** y elige
   **Python 3.12** (es la versión con la que se probó todo).
6. Clic en **"Deploy"**. La nube instalará `requirements.txt` y arrancará la app.

### Notas importantes del despliegue

- **Versión de Python:** elígela en *Advanced settings* / ajustes de la app.
  Streamlit Cloud ya **no** usa `runtime.txt`. Si dejas la versión por defecto
  (3.14) algunos paquetes podrían no tener instaladores y el despliegue fallaría.
- **`requirements.txt` y no `uv.lock`:** Streamlit Cloud le da prioridad a
  `uv.lock` por encima de `requirements.txt`. Por eso `uv.lock` está en el
  `.gitignore`: así la nube usa `requirements.txt`, que es lo que queremos.
- **Prophet:** la versión 1.5.0 ya incluye su modelo compilado y su motor
  (`cmdstan`), por lo que **no** necesita `packages.txt` ni compilador.

## Notas técnicas

- Toda la lógica de cálculo vive en `app.py` (funciones puras) y la pantalla se
  dibuja al final, dentro de `main()`. Así los cálculos se pueden probar sin
  abrir la web.
- Si un modelo llegara a fallar en el servidor, la app **lo omite y avisa**
  en lugar de caerse, para que el resto siga funcionando.
- Los modelos pesados usan `st.cache_data` para no recalcular en cada clic.
