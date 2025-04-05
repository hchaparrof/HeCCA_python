import os
import pandas as pd
from hecca_final import ejecutar_bloque
from ingreso_datos import generar_algoritmo
import multiprocessing as mp
import copy
RUTAESTACIONES = 'C://Users//ASUS//Desktop//datos//unal//semillero//CAUDAL//resultados//2'
diccionario_espejo: dict = {
  "archivos": {
    "archivo_base": "C:/Users/ASUS/Desktop/datos/unal/semillero/repo_hecca/est_25027400.csv",
    "archivo_maximos": -1,
    "archivo_minimos": -1,
    "archivo_apoyo": -1,
    "archivo_enso": "oni_3.csv"
  },
  "existencia_enso": False,
  "areas": -1,
  "umbrales": -1,
  "estacion_hidrologica": "Nombre_de_la_estacion",
  "organismo": "ideam",
  "revision_iha": -1,
  "anio_hidrologico": -1
}
estaciones = []
resultados = []
# 1. leer cuantos archivos hay en la carpeta ruta_estaciones y poner las rutas en la variable estaciones
# 2. crear un código multiproceso que para cada ruta en estaciones cambie diccionario espejo para que 'archivo_base'
# sea esa ruta para pasarlo a generar_algoritmo, que devuelve una lista de objetos, que ejecuten su metodo principal_funcion()
# 3. guardar por ahí todos los resultados en la lista resultados

def procesar_estacion(ruta_archivo):
    diccionario_local = copy.deepcopy(diccionario_espejo)
    diccionario_local["archivos"]["archivo_base"] = ruta_archivo

    try:
        objetos:list = generar_algoritmo(diccionario_local)
        print(objetos)
        resultados_locales = ejecutar_bloque(objetos)
        return resultados_locales
    except Exception as e:
        print(f"Error procesando {ruta_archivo}: {e}")
        return []

if __name__ == "__main__":
    procesar_estacion("C://c//Users//ASUS//Desktop//datos//unal//semillero//repo_hecca//est_12027050")
if __name__ == "__main__2":
    estaciones = [
        os.path.join(RUTAESTACIONES, archivo)
        for archivo in os.listdir(RUTAESTACIONES)
        if archivo.endswith(".csv")
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados_multiproceso = pool.map(procesar_estacion, estaciones)

    # Flatten the list of lists
    resultados = [res for sublist in resultados_multiproceso for res in sublist]

    print(f"Se procesaron {len(resultados)} resultados.")