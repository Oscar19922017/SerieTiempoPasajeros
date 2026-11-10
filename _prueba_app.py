"""
_prueba_app.py
--------------
Revisa que la lógica de app.py funcione, SIN abrir la app.

Importa las funciones de app.py (por eso no se ejecuta la pantalla) y las
prueba una por una, imprimiendo resultados.
"""

import time

import app
from datos import leer_pasajeros


def main() -> None:
    tabla = leer_pasajeros()
    print(f"Filas totales: {len(tabla)}")

    # Indicadores
    p1949 = app.total_anio(tabla, 1949)
    p1960 = app.total_anio(tabla, 1960)
    print(f"Pasajeros 1949: {p1949} | 1960: {p1960} | Crecimiento: {p1960 / p1949:.2f}x")

    # Promedio por mes
    promedio = app.promedio_por_mes(tabla)
    print(f"Promedios por mes (ene..dic): {[round(v) for v in promedio.tolist()]}")

    # Examen: probar cada modelo por separado (con su tiempo).
    print("\n--- El examen: error de cada modelo (escondiendo 24 meses) ---")
    examen = app.preparar_examen(tabla, meses_ocultos=24)
    for nombre in app.ORDEN_MODELOS:
        valor = examen["pronosticos"][nombre]
        error = examen["errores"][nombre]
        print(f"  {nombre:45s} error = {error:5.1f}%  ({len(valor)} meses pronosticados)")
    print(f"Ganador: {examen['ganador']}")

    # El pronóstico de 1961 con cada modelo.
    print("\n--- Pronóstico de 1961 con cada modelo ---")
    for nombre in app.ORDEN_MODELOS:
        t0 = time.perf_counter()
        pron = app.pronostico_con_modelo(tabla, nombre, n=12)
        dt = time.perf_counter() - t0
        print(
            f"  {nombre:45s} total 1961 = {round(pron.sum()):6d} "
            f"(en {dt:4.1f}s)  primer mes = {app.nombre_mes(pron.index[0])}"
        )

    print("\nOK: todos los modelos corrieron sin errores.")


if __name__ == "__main__":
    main()
