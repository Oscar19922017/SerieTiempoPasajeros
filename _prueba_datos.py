"""
_prueba_datos.py
----------------
Script de prueba: revisa que datos.py funcione y resume la serie.

Muestra:
- cuántos meses hay,
- desde y hasta cuándo,
- el mes con menos pasajeros,
- el mes con más pasajeros.
"""

from datos import leer_pasajeros


def main() -> None:
    # Leo la tabla de pasajeros.
    tabla = leer_pasajeros()

    # Cuento cuántas filas (meses) hay.
    total_meses = len(tabla)

    # Primer y último mes de la serie.
    primer_mes = tabla["Month"].min()
    ultimo_mes = tabla["Month"].max()

    # Mes con menos pasajeros y mes con más pasajeros.
    fila_min = tabla.loc[tabla["Passengers"].idxmin()]
    fila_max = tabla.loc[tabla["Passengers"].idxmax()]

    # Formato de fecha sencillo: "mes de año" (por ejemplo, "enero 1949").
    def bonito(fecha) -> str:
        meses = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ]
        return f"{meses[fecha.month - 1]} {fecha.year}"

    print()
    print("=== Resumen de la serie de pasajeros ===")
    print(f"Cantidad de meses:          {total_meses}")
    print(f"Desde:                      {bonito(primer_mes)} ({primer_mes.date()})")
    print(f"Hasta:                      {bonito(ultimo_mes)} ({ultimo_mes.date()})")
    print(
        f"Mes con MENOS pasajeros:    {bonito(fila_min['Month'])} "
        f"-> {int(fila_min['Passengers'])} pasajeros"
    )
    print(
        f"Mes con MÁS pasajeros:      {bonito(fila_max['Month'])} "
        f"-> {int(fila_max['Passengers'])} pasajeros"
    )


if __name__ == "__main__":
    main()
