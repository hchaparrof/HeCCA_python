import os
from typing import List, Optional
import pandas as pd
import estado_algoritmo
from hecca_final import ejecutar_bloque
from ingreso_datos import generar_algoritmo
import multiprocessing as mp
import copy
import csv
RUTAESTACIONES = 'C://Users//ASUS//Desktop//datos//unal//semillero//CAUDAL//resultados//2'
RUTARESULTADOS = 'C://Users//ASUS//Desktop//datos//unal//semillero//repo_hecca//resultados//'
diccionario_espejo: dict = {
  "archivos": {
    "archivo_base": "C://Users//ASUS//Desktop//datos//unal//semillero//repo_hecca//est_12027050.csv",
    "archivo_maximos": -1,
    "archivo_minimos": -1,
    "archivo_apoyo": -1,
    "archivo_enso": "oni_3.csv"
  },
  "existencia_enso": True,
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
        objetos:Optional[list[estado_algoritmo.EstadoAlgoritmo]] = generar_algoritmo(diccionario_espejo)
        print(objetos)
        resultados_locales = ejecutar_bloque(objetos)
        return resultados_locales
    except Exception as e:
        print(f"Error procesando {ruta_archivo}: {e}")
        return []
def main_2():
    return procesar_estacion("C://Users//ASUS//Desktop//datos//unal//semillero//repo_hecca//est_12027050.csv")
def guardar_datos(objeto_resultado: estado_algoritmo.EstadoAlgoritmo):
    ENCABEZADO: List[str] = ["formato", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    estacion: int = objeto_resultado.codigo_est
    ruta_est = os.path.join(RUTARESULTADOS, str(estacion)) + os.path.sep
    if not os.path.exists(ruta_est):
        os.makedirs(ruta_est)
    ENSO: str = objeto_resultado.str_apoyo
    tipo: str = "ideam" if isinstance(objeto_resultado, estado_algoritmo.EstadoIdeam) else "anla"
    formato: str = "_" + tipo + "_" + ENSO + "_"
    ruta_general: str = os.path.join(ruta_est, f"_{estacion}{formato}_amb.csv")
    objeto_resultado.data_alter.to_csv(ruta_general)
    resultados_aprov: str = os.path.join(ruta_est, f"{estacion}_resultados_.csv")
    archivo_nuevo = not os.path.exists(resultados_aprov)
    with open(resultados_aprov, mode='a', newline='', encoding='utf-8') as archivo:
        writer = csv.writer(archivo)
        if archivo_nuevo:
            writer.writerow(ENCABEZADO)  # encabezados
        fila: List = [formato + "_porc"] + (objeto_resultado.df2['%_aprov'].to_list())
        writer.writerow(fila)
        fila: List = [formato + "_caud"] + (objeto_resultado.df2['Q_aprov'].to_list())
        writer.writerow(fila)


    pass
if __name__ == "__main___3":
    main_2()
if __name__ == "__main__":
    estaciones = [
        os.path.join(RUTAESTACIONES, archivo)
        for archivo in os.listdir(RUTAESTACIONES)
        if archivo.endswith(".csv")
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados_multiproceso = pool.map(procesar_estacion, estaciones)

    # Flatten the list of lists
    resultados = [res for sublist in resultados_multiproceso for res in sublist]
    for resultado in resultados:
        guardar_datos(resultado)
    print(f"Se procesaron {len(resultados)} resultados.")
def procesar_cuenca(ruta_principal: str, ruta_secundaria: str,
                    area_principal: float, area_secundaria: float):

    dicc_local = copy.deepcopy(diccionario_espejo)
    dicc_local["archivos"]["archivo_base"] = ruta_principal
    dicc_local["archivos"]["archivo_apoyo"] = ruta_secundaria
    dicc_local["areas"] = [area_principal, area_secundaria]
    

    try:
        objetos: Optional[tuple[list[estado_algoritmo.EstadoAnla], list[estado_algoritmo.EstadoIdeam]]] = generar_algoritmo(dicc_local)
        print("objetos", objetos, type(objetos))
        if objetos is None or not objetos[1]:
            print(f"No se generaron resultados válidos para {ruta_principal}")
            return "sin_resultado"

        ideam_objs = objetos[1]
        cuenca_nombre = os.path.basename(ruta_principal).replace(".csv", "")

        for ideam in ideam_objs:
            ideam.principal_funcion()
            str_apoyo = ideam.str_apoyo or "sin_nombre"
            carpeta_resultados = f"C://Users//ASUS//Desktop//datos//unal//trabajo de grado//resultados//{str_apoyo}"
            os.makedirs(carpeta_resultados, exist_ok=True)

            # Guardar los datos
            ideam.df2.to_csv(os.path.join(carpeta_resultados, f"df2_{cuenca_nombre}.csv"), index=False)
            ideam.data_alter.to_csv(os.path.join(carpeta_resultados, f"data_alter_{cuenca_nombre}.csv"), index=False)

            # Umbrales simples
            df_umbrales_csv = pd.DataFrame(list(ideam.umbrales.items()), columns=["umbral", "valor"])
            df_umbrales_csv.to_csv(os.path.join(carpeta_resultados, f"umbrales_{cuenca_nombre}.csv"), index=False)

            # Umbrales DataFrame
            for nombre, df in ideam.df_umbrales.items():
                if not df.empty:
                    nombre_archivo = f"{nombre}_{cuenca_nombre}.csv"
                    df.to_csv(os.path.join(carpeta_resultados, nombre_archivo), index=False)

        return "ok"

    except Exception as e:
        print(f"Error procesando cuenca {ruta_principal}: {e}")
        return f"error: {e}"
def hacer_trabajo_actual():
    ruta_trabajo = "C://Users//ASUS//Desktop//datos//unal//trabajo de grado//trabajo_actual.csv"
    ruta_cuencas = "C://Users//ASUS//Desktop//datos//unal//trabajo de grado//cuencas_a_estudiar_2.csv"
    carpeta_cuencas = "C://Users//ASUS//Desktop//datos//unal//trabajo de grado//cuencas//"

    # Crear archivo trabajo_actual.csv si no existe
    if not os.path.exists(ruta_trabajo):
        df_vacio = pd.DataFrame(columns=["Serie_principal", "Serie_secundaria", "resultado"])
        df_vacio.to_csv(ruta_trabajo, index=False)

    # Leer archivos
    df_trabajo = pd.read_csv(ruta_trabajo)
    df_cuencas = pd.read_csv(ruta_cuencas)

    for _, fila in df_cuencas.iterrows():
        Serie_principal = int(fila["Serie_principal"])
        Serie_secundaria = int(fila["Serie_secundaria"])

        # Saltar si ya fue procesada
        print(df_trabajo.head())
        ya_hecho = ((df_trabajo["Serie_principal"] == Serie_principal) & 
                    (df_trabajo["Serie_secundaria"] == Serie_secundaria)).any()
        if ya_hecho:
            print(f"Saltando cuenca {Serie_principal}-{Serie_secundaria}, ya procesada.")
            continue

        area_principal = float(fila["area_principal"])
        area_secundaria = float(fila["area_secundaria"])

        ruta_principal = os.path.join(carpeta_cuencas, f"{Serie_principal}.csv")
        ruta_secundaria = os.path.join(carpeta_cuencas, f"{Serie_secundaria}.csv")

        resultados = procesar_cuenca(ruta_principal, ruta_secundaria, area_principal, area_secundaria)

        nueva_fila = {
            "Serie_principal": Serie_principal,
            "Serie_secundaria": Serie_secundaria,
            "resultado": str(resultados)[:1000]
        }
        df_trabajo = pd.concat([df_trabajo, pd.DataFrame([nueva_fila])], ignore_index=True)
        df_trabajo.to_csv(ruta_trabajo, index=False)
