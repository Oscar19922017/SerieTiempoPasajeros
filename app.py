"""
app.py
------
Aplicación de Streamlit que pronostica los pasajeros de avión.

Está dividida en 4 pestañas:
1. La historia   -> cómo han cambiado los pasajeros mes a mes.
2. La temporada  -> qué meses del año son los más fuertes y los más flojos.
3. El examen     -> probamos VARIOS modelos y vemos cuál gana.
4. El pronóstico -> la predicción de los 12 meses de 1961 con el modelo que elijas.

Modelos que comparamos:
- Copiar el año pasado
- Suavizamiento exponencial (Holt-Winters)
- SARIMA
- Regresión lineal
- Prophet

Para ejecutarla:  uv run python -m streamlit run app.py
"""

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from datos import leer_pasajeros

# --- Silencio los mensajes técnicos de Prophet y su motor de cálculo ---
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Nombres de los meses en español, para mostrar fechas bonitas.
MESES_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

# --- Nombres de los modelos y un color para cada uno ---
NOMBRE_COPIAR = "Copiar el año pasado"
NOMBRE_SUAVIZADO = "Suavizamiento exponencial (Holt-Winters)"
NOMBRE_SARIMA = "SARIMA"
NOMBRE_REGRESION = "Regresión lineal"
NOMBRE_PROPHET = "Prophet"

ORDEN_MODELOS = [
    NOMBRE_COPIAR,
    NOMBRE_SUAVIZADO,
    NOMBRE_SARIMA,
    NOMBRE_REGRESION,
    NOMBRE_PROPHET,
]

COLORES_MODELOS = {
    NOMBRE_COPIAR: "#d9822b",
    NOMBRE_SUAVIZADO: "#2f7ed8",
    NOMBRE_SARIMA: "#8e44ad",
    NOMBRE_REGRESION: "#2e9e5b",
    NOMBRE_PROPHET: "#c0392b",
}


# ---------------------------------------------------------------------------
# Funciones de ayuda (cálculos sencillos).
# ---------------------------------------------------------------------------

def nombre_mes(fecha) -> str:
    """Convierte una fecha en un texto del tipo 'enero 1949'."""
    return f"{MESES_ES[fecha.month - 1]} {fecha.year}"


def total_anio(tabla: pd.DataFrame, anio: int) -> int:
    """Suma de pasajeros de todo un año."""
    return int(tabla.loc[tabla["Month"].dt.year == anio, "Passengers"].sum())


def promedio_por_mes(tabla: pd.DataFrame) -> pd.Series:
    """Promedio de pasajeros para cada mes del año (enero a diciembre)."""
    copia = tabla.copy()
    copia["MesNum"] = copia["Month"].dt.month
    promedio = copia.groupby("MesNum")["Passengers"].mean()
    return promedio.reindex(range(1, 13))


def error_porcentual(real, pronostico) -> float:
    """Error promedio en porcentaje (MAPE).

    Compara lo que pasó de verdad con lo pronosticado.
    Un número pequeño significa un mejor pronóstico.
    """
    real = np.asarray(real, dtype=float)
    pronostico = np.asarray(pronostico, dtype=float)
    return float(np.mean(np.abs(real - pronostico) / real) * 100)


def _fechas_futuras(ultima_fecha, n: int) -> pd.DatetimeIndex:
    """Los siguientes n meses, empezando justo después de ultima_fecha."""
    inicio = pd.Timestamp(ultima_fecha) + pd.offsets.MonthBegin(1)
    return pd.date_range(start=inicio, periods=n, freq="MS")


# ---------------------------------------------------------------------------
# Los modelos. Cada uno recibe la serie de entrenamiento (mensual) y cuántos
# meses pronosticar, y devuelve una lista de valores pronosticados.
# ---------------------------------------------------------------------------

def pronostico_copiar(serie: pd.Series, n: int) -> np.ndarray:
    """Cada mes igual al mismo mes del último año conocido."""
    ultimos_12 = serie.to_numpy(dtype=float)[-12:]
    return np.array([ultimos_12[i % 12] for i in range(n)], dtype=float)


def pronostico_suavizamiento(serie: pd.Series, n: int) -> np.ndarray:
    """Suavizamiento exponencial de Holt-Winters (tendencia y temporada)."""
    modelo = ExponentialSmoothing(
        serie, trend="add", seasonal="mul", seasonal_periods=12
    ).fit()
    return modelo.forecast(n).to_numpy(dtype=float)


def pronostico_sarima(serie: pd.Series, n: int) -> np.ndarray:
    """SARIMA: un modelo clásico para series con tendencia y temporada.

    Uso la configuración "airline" de Box-Jenkins (0,1,1)(0,1,1,12), que es la
    estándar para esta serie de pasajeros.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        # Silencio avisos de convergencia para no ensuciar la pantalla.
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            serie,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return modelo.get_forecast(n).predicted_mean.to_numpy(dtype=float)


def _diseno_regresion(fechas: pd.DatetimeIndex, fecha_inicial) -> np.ndarray:
    """Arma las variables que usa la regresión lineal.

    - una columna con el paso del tiempo (los meses transcurridos),
    - y una columna por cada mes del año (para capturar la temporada).
    """
    meses_transcurridos = (fechas.year - fecha_inicial.year) * 12 + (
        fechas.month - fecha_inicial.month
    )
    columnas = [np.asarray(meses_transcurridos, dtype=float)]
    for mes in range(2, 13):  # enero queda como referencia
        columnas.append((np.asarray(fechas.month) == mes).astype(float))
    return np.column_stack(columnas)


def pronostico_regresion(serie: pd.Series, n: int) -> np.ndarray:
    """Regresión lineal: una recta que crece, más el efecto de cada mes."""
    fecha_inicial = serie.index[0]
    matriz = _diseno_regresion(serie.index, fecha_inicial)
    valores = serie.to_numpy(dtype=float)
    # Resuelvo la regresión (mínimos cuadrados).
    coeficientes, *_ = np.linalg.lstsq(matriz, valores, rcond=None)
    fechas_futuras = _fechas_futuras(serie.index[-1], n)
    matriz_futura = _diseno_regresion(fechas_futuras, fecha_inicial)
    return matriz_futura @ coeficientes


def pronostico_prophet(serie: pd.Series, n: int) -> np.ndarray:
    """Prophet: modelo moderno que aprende tendencia y temporada solo."""
    from prophet import Prophet

    datos = pd.DataFrame({"ds": serie.index, "y": serie.to_numpy(dtype=float)})
    modelo = Prophet()
    modelo.fit(datos)
    futuro = modelo.make_future_dataframe(periods=n, freq="MS")
    pronostico = modelo.predict(futuro)
    return pronostico["yhat"].to_numpy(dtype=float)[-n:]


# Tabla que une cada nombre con su función.
MODELOS = {
    NOMBRE_COPIAR: pronostico_copiar,
    NOMBRE_SUAVIZADO: pronostico_suavizamiento,
    NOMBRE_SARIMA: pronostico_sarima,
    NOMBRE_REGRESION: pronostico_regresion,
    NOMBRE_PROPHET: pronostico_prophet,
}


# ---------------------------------------------------------------------------
# El examen: escondemos los últimos meses y probamos todos los modelos.
# ---------------------------------------------------------------------------

def preparar_examen(tabla: pd.DataFrame, meses_ocultos: int = 24) -> dict:
    """Esconde los últimos meses y mide el error de cada modelo.

    Devuelve, para cada modelo, sus valores pronosticados y su error.
    """
    visible = tabla.iloc[:-meses_ocultos].reset_index(drop=True)
    real = tabla.iloc[-meses_ocultos:].reset_index(drop=True)

    serie_visible = (
        visible.set_index("Month")["Passengers"].astype(float).asfreq("MS")
    )
    real_vals = real["Passengers"].astype(float).to_numpy(dtype=float)

    pronosticos = {}
    errores = {}
    fallos = {}
    for nombre in ORDEN_MODELOS:
        try:
            pred = MODELOS[nombre](serie_visible, meses_ocultos)
            pronosticos[nombre] = np.asarray(pred, dtype=float)
            errores[nombre] = error_porcentual(real_vals, pred)
        except Exception as error:
            # Si un modelo falla (por ejemplo, por falta de un paquete),
            # lo apartamos en vez de tumbar toda la aplicación.
            fallos[nombre] = f"{type(error).__name__}: {error}"

    disponibles = [n for n in ORDEN_MODELOS if n in pronosticos]
    ganador = min(errores, key=errores.get) if errores else None

    return {
        "visible": visible,
        "real": real,
        "pronosticos": pronosticos,
        "errores": errores,
        "disponibles": disponibles,
        "fallos": fallos,
        "ganador": ganador,
    }


def pronostico_con_modelo(tabla: pd.DataFrame, nombre_modelo: str, n: int = 12) -> pd.Series:
    """Entrena el modelo elegido con TODOS los datos y pronostica n meses."""
    serie = tabla.set_index("Month")["Passengers"].astype(float).asfreq("MS")
    valores = MODELOS[nombre_modelo](serie, n)
    fechas = _fechas_futuras(serie.index[-1], n)
    return pd.Series(np.asarray(valores, dtype=float), index=fechas)


# --- Envoltorios con caché: evitan recalcular todo en cada clic. ---
# (Solo se usan dentro de la app; las funciones de arriba quedan puras.)
@st.cache_data(show_spinner=False)
def examen_con_cache(tabla: pd.DataFrame) -> dict:
    return preparar_examen(tabla, meses_ocultos=24)


@st.cache_data(show_spinner=False)
def pronostico_con_cache(tabla: pd.DataFrame, nombre_modelo: str) -> pd.Series:
    return pronostico_con_modelo(tabla, nombre_modelo, n=12)


# ---------------------------------------------------------------------------
# La pantalla (todo lo que se ve en la web).
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Pasajeros de avión",
        page_icon="✈️",
        layout="wide",
    )

    st.title("¿Cuántos pasajeros tendremos el próximo año?")

    # Leo los datos con la función de datos.py (internet o archivo local).
    tabla = leer_pasajeros()
    st.caption(
        f"Datos desde {nombre_mes(tabla['Month'].min())} "
        f"hasta {nombre_mes(tabla['Month'].max())} "
        f"({len(tabla)} meses en total)."
    )

    # --- Los 3 indicadores de arriba ---
    pasajeros_1949 = total_anio(tabla, 1949)
    pasajeros_1960 = total_anio(tabla, 1960)
    crecimiento = pasajeros_1960 / pasajeros_1949

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Pasajeros en 1949",
        f"{pasajeros_1949:,}",
        help="Total de pasajeros que volaron durante todo el año 1949.",
    )
    col2.metric(
        "Pasajeros en 1960",
        f"{pasajeros_1960:,}",
        help="Total de pasajeros que volaron durante todo el año 1960.",
    )
    col3.metric(
        "Cuántas veces creció",
        f"{crecimiento:.1f}×",
        help="Cuántas veces es 1960 comparado con 1949.",
    )

    # --- Calculo el examen una sola vez (lo usan 2 pestañas) ---
    with st.spinner("Entrenando los modelos... la primera vez tarda unos segundos."):
        examen = examen_con_cache(tabla)

    # --- Las 4 pestañas ---
    pestana1, pestana2, pestana3, pestana4 = st.tabs(
        ["La historia", "La temporada", "El examen", "El pronóstico"]
    )

    # ===================== PESTAÑA 1: LA HISTORIA =====================
    with pestana1:
        st.subheader("Pasajeros mes a mes")

        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=tabla["Month"],
                y=tabla["Passengers"],
                name="Pasajeros",
                line=dict(color="#2f7ed8", width=2),
                fill="tozeroy",
                fillcolor="rgba(47, 126, 216, 0.15)",
            )
        )
        figura.update_layout(
            xaxis_title="Año",
            yaxis_title="Pasajeros por mes",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(figura)

        st.info(
            "Cada punto es la cantidad de pasajeros que volaron en un mes. "
            "La línea sube de forma constante: la aviación no dejó de crecer en estos 12 años. "
            "Además hay un patrón que se repite cada año (sube y baja dentro del mismo año), "
            "y eso es justo lo que usaremos para pronosticar."
        )

    # ===================== PESTAÑA 2: LA TEMPORADA =====================
    with pestana2:
        st.subheader("¿Qué meses del año son los más fuertes?")

        promedio = promedio_por_mes(tabla)
        mes_normal = float(tabla["Passengers"].mean())

        figura = go.Figure()
        figura.add_trace(
            go.Bar(
                x=[MESES_ES[m - 1].capitalize() for m in range(1, 13)],
                y=promedio.to_numpy(),
                marker_color="#2f7ed8",
                name="Promedio del mes",
            )
        )
        figura.add_hline(
            y=mes_normal,
            line_dash="dash",
            line_color="#d9822b",
            annotation_text="Mes normal",
            annotation_position="top left",
        )
        figura.update_layout(
            xaxis_title="Mes del año",
            yaxis_title="Pasajeros promedio",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(figura)

        st.info(
            "Esta gráfica promedia los 12 años para mostrar un año 'típico'. "
            "Las barras por encima de la línea naranja (marzo a agosto) son temporada alta: "
            "los meses donde más se vuela. Las barras por debajo (enero, febrero, noviembre) "
            "son temporada baja. Para el negocio, esto significa que conviene tener más aviones "
            "y personal en la primera mitad del año."
        )

        mes_mas_fuerte = int(promedio.idxmax())
        mes_mas_flojo = int(promedio.idxmin())
        st.caption(
            f"Mes más fuerte: {MESES_ES[mes_mas_fuerte - 1]} "
            f"({promedio.max():.0f} pasajeros promedio). "
            f"Mes más flojo: {MESES_ES[mes_mas_flojo - 1]} "
            f"({promedio.min():.0f} pasajeros promedio)."
        )

    # ===================== PESTAÑA 3: EL EXAMEN =====================
    with pestana3:
        st.subheader("Ponemos a prueba varios modelos")
        st.write(
            "Escondemos los últimos 24 meses (1959 y 1960) y pedimos a **"
            f"{len(examen['disponibles'])} modelos** que los adivinen, sin haberlos visto. "
            "Al final comparamos con la realidad para ver cuál se equivocó menos."
        )

        visible = examen["visible"]
        real = examen["real"]

        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=visible["Month"],
                y=visible["Passengers"],
                name="Real (lo que el modelo pudo ver)",
                line=dict(color="#9aa0a6", width=2),
            )
        )
        figura.add_trace(
            go.Scatter(
                x=real["Month"],
                y=real["Passengers"],
                name="Real (lo que pasó de verdad)",
                line=dict(color="#1f2d3d", width=3),
            )
        )
        for nombre in examen["disponibles"]:
            figura.add_trace(
                go.Scatter(
                    x=real["Month"],
                    y=examen["pronosticos"][nombre],
                    name=nombre,
                    line=dict(
                        color=COLORES_MODELOS[nombre],
                        width=2,
                        dash="dash" if nombre == NOMBRE_COPIAR else "solid",
                    ),
                )
            )
        figura.add_vline(
            x=real["Month"].iloc[0],
            line_dash="dot",
            line_color="#9aa0a6",
        )
        figura.update_layout(
            xaxis_title="Año",
            yaxis_title="Pasajeros por mes",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(figura)

        st.info(
            "La línea gris es lo que los modelos conocían; a partir de la línea punteada empieza "
            "lo que tenían que adivinar. La línea oscura es la realidad y las de colores son los "
            "pronósticos de cada modelo. Cuanto más pegada esté una línea de color a la línea "
            "oscura, mejor es ese modelo. 'Copiar el año pasado' (naranja punteada) se queda corto "
            "porque ignora el crecimiento."
        )

        # Tabla con el error de cada modelo, ordenada de mejor a peor.
        tabla_error = pd.DataFrame(
            {
                "Modelo": examen["disponibles"],
                "Error promedio (%)": [
                    round(examen["errores"][nombre], 1)
                    for nombre in examen["disponibles"]
                ],
            }
        ).sort_values("Error promedio (%)").reset_index(drop=True)
        tabla_error.insert(0, "Puesto", range(1, len(tabla_error) + 1))

        st.write("**¿Cuánto se equivocó cada modelo? (de mejor a peor)**")
        st.dataframe(tabla_error, hide_index=True)

        st.caption(
            "Ojo con SARIMA: da un error parecido a 'Copiar el año pasado'. Eso pasa porque "
            "SARIMA se apoya en los cambios de un mes a otro y no logra seguir un crecimiento "
            "tan rápido como el de 1959-1960. No está mal hecho: es una limitación real del método."
        )

        # Si algún modelo no se pudo usar, lo avisamos (pero la app sigue funcionando).
        if examen["fallos"]:
            nombres_fallidos = ", ".join(examen["fallos"].keys())
            st.warning(
                f"Estos modelos no se pudieron usar en este servidor y se omitieron: "
                f"**{nombres_fallidos}**."
            )

        ganador = examen["ganador"]
        if ganador is not None:
            st.success(
                f"Ganó **{ganador}**, con un error promedio de "
                f"{examen['errores'][ganador]:.1f}%. Cuanto más pequeño el número, mejor: "
                "es el porcentaje en que se equivocó, en promedio, cada mes. "
                "En la pestaña **El pronóstico** puedes usar este modelo para 1961."
            )
        else:
            st.error("Ningún modelo pudo calcularse en este servidor.")

    # ===================== PESTAÑA 4: EL PRONÓSTICO =====================
    with pestana4:
        st.subheader("Los 12 meses de 1961")

        modelo_elegido = st.selectbox(
            "Elige el modelo para el pronóstico (por defecto, el que ganó el examen):",
            options=examen["disponibles"],
            index=examen["disponibles"].index(examen["ganador"]),
        )

        with st.spinner(f"Calculando el pronóstico con {modelo_elegido}..."):
            pronostico = pronostico_con_cache(tabla, modelo_elegido)

        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=tabla["Month"],
                y=tabla["Passengers"],
                name="Real (histórico)",
                line=dict(color="#2f7ed8", width=2),
            )
        )
        figura.add_trace(
            go.Scatter(
                x=pronostico.index,
                y=pronostico.to_numpy(),
                name=f"Pronóstico 1961 ({modelo_elegido})",
                line=dict(color=COLORES_MODELOS[modelo_elegido], width=3, dash="dash"),
            )
        )
        figura.update_layout(
            xaxis_title="Año",
            yaxis_title="Pasajeros por mes",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(figura)

        st.info(
            f"La línea azul es la historia real y la punteada es el pronóstico para 1961, "
            f"hecho con **{modelo_elegido}** usando todos los datos disponibles. "
            "El modelo espera que 1961 siga creciendo y que repita el patrón de temporada "
            "alta y baja de cada año. Puedes cambiar de modelo arriba para comparar."
        )

        tabla_pronostico = pd.DataFrame(
            {
                "Mes": [nombre_mes(fecha) for fecha in pronostico.index],
                "Pasajeros esperados": [int(round(v)) for v in pronostico.to_numpy()],
            }
        )
        st.write(f"**Pasajeros esperados mes a mes ({modelo_elegido}):**")
        st.dataframe(tabla_pronostico, hide_index=True)

        total_1961 = int(round(pronostico.sum()))
        st.caption(
            f"En total, para 1961 se esperan alrededor de {total_1961:,} pasajeros "
            f"según {modelo_elegido}."
        )


# Streamlit ejecuta este archivo como programa principal, así que main() corre.
# Si en cambio importamos el archivo (por ejemplo, para probarlo), main() NO corre.
if __name__ == "__main__":
    main()
