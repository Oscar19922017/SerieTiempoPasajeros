"""
datos.py
--------
Se encarga de conseguir la tabla de pasajeros de avión.

Funciona así:
1. Primero intenta bajar los datos desde internet.
2. Si no hay internet, usa la copia que está guardada en la carpeta local.

La tabla que devuelve tiene dos columnas:
- Month: el mes (en formato de fecha real, por ejemplo 1949-01-01)
- Passengers: cuántos pasajeros viajaron ese mes
"""

from pathlib import Path

import pandas as pd

# Dirección de internet donde están los datos originales.
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

# Ruta del archivo guardado en el computador (el "plan B" sin internet).
# Busco en "Datos" y también en "datos" por si el nombre cambia de mayúsculas.
CARPETA = Path(__file__).parent
RUTAS_LOCALES = [
    CARPETA / "Datos" / "airline-passengers.csv",
    CARPETA / "datos" / "airline-passengers.csv",
]


def leer_pasajeros() -> pd.DataFrame:
    """Devuelve la tabla de pasajeros por mes (columnas Month y Passengers).

    Intenta leer desde internet y, si no puede, desde el archivo local.
    Además imprime de dónde sacó los datos.
    """
    origen = "internet"

    try:
        # Intento 1: bajar los datos desde internet.
        tabla = pd.read_csv(URL)
    except Exception:
        # Intento 2: si no hay internet, busco el archivo local.
        tabla = None
        for ruta in RUTAS_LOCALES:
            if ruta.exists():
                tabla = pd.read_csv(ruta)
                origen = f"archivo local ({ruta.name})"
                break
        if tabla is None:
            # Si tampoco existe el archivo local, aviso con un error claro.
            raise FileNotFoundError(
                "No hubo internet y no encontré el archivo local "
                "'airline-passengers.csv' en la carpeta Datos."
            )

    # La columna Month viene como texto (por ejemplo "1949-01").
    # La convierto a fecha real para que sea fácil graficarla después.
    tabla["Month"] = pd.to_datetime(tabla["Month"], format="%Y-%m")

    print(f"Datos leídos desde: {origen}")
    return tabla


if __name__ == "__main__":
    # Si se ejecuta este archivo directamente, muestro las primeras filas.
    print(leer_pasajeros().head())
