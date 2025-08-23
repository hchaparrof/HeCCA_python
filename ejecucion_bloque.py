import json
import os
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import IhaEstado
import estado_algoritmo
import funciones_anla
from hecca_final import ejecutar_bloque
from ingreso_datos import generar_algoritmo
import multiprocessing as mp
import copy
import csv
import exportar_anla_creado
RUTAESTACIONES = 'C://Users//ASUS//Desktop//datos//unal//semillero//CAUDAL//resultados//2'
RUTARESULTADOS = 'C://Users//ASUS//Desktop//datos//unal//semillero//repo_hecca//resultados//'
with open("setup.json", "r") as archivo_json:
    diccionario_espejo = json.load(archivo_json)
estaciones = []
resultados = []
# 1. leer cuantos archivos hay en la carpeta ruta_estaciones y poner las rutas en la variable estaciones
# 2. crear un código multiproceso que para cada ruta en estaciones cambie diccionario espejo para que 'archivo_base'
# sea esa ruta para pasarlo a generar_algoritmo, que devuelve una lista de objetos, que ejecuten su metodo principal_funcion()
# 3. guardar por ahí todos los resultados en la lista resultados


def procesar_estacion(ruta_archivo):
    diccionario_local = copy.deepcopy(diccionario_espejo)
    diccionario_local["archivos"]["archivo_base"] = ruta_archivo
    objetos: Optional[list[estado_algoritmo.EstadoAlgoritmo]] = generar_algoritmo(diccionario_espejo)
    print(objetos, "algo")
    resultados_locales = ejecutar_bloque(objetos)
    return resultados_locales
    # except Exception as e:
    #    print(f"Error procesando {ruta_archivo}: {e}")
    #    return []


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
if __name__ == "__main___4":
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
    objetos: Optional[tuple[list[estado_algoritmo.EstadoAnla], list[estado_algoritmo.EstadoIdeam]]] = generar_algoritmo(dicc_local)
    print("objetos", objetos, type(objetos))
    if objetos is None:
        print(f"No se generaron resultados válidos para {ruta_principal}")
        return "sin_resultado"

    ideam_objs = objetos[1]
    cuenca_nombre = os.path.basename(ruta_principal).replace(".csv", "")

    for ideam in ideam_objs:
        ideam.principal_funcion()
        str_apoyo = ideam.str_apoyo or "sin_nombre"
        carpeta_resultados = f"C://Users//ASUS//Desktop//datos//unal//trabajo de grado//resultados_8_prueba//ideam//{str_apoyo}"
        os.makedirs(carpeta_resultados, exist_ok=True)
        print(ideam.anio_hidrologico, "anio_hidrologico", cuenca_nombre)
        print(ideam.ajuste, "ajuste", cuenca_nombre)
        # Guardar los datos
        ideam.df2.to_csv(os.path.join(carpeta_resultados, f"df2_{cuenca_nombre}.csv"), index=True)
        ideam.data_alter.to_csv(os.path.join(carpeta_resultados, f"data_alter_{cuenca_nombre}.csv"), index=True)

        # Umbrales simples
        df_umbrales_csv = pd.DataFrame(list(ideam.umbrales.items()), columns=["umbral", "valor"])
        df_umbrales_csv.to_csv(os.path.join(carpeta_resultados, f"umbrales_{cuenca_nombre}.csv"), index=True)

        # Umbrales DataFrame
        for nombre, df in ideam.df_umbrales.items():
            if not df.empty:
                nombre_archivo = f"{nombre}_{cuenca_nombre}.csv"
                df.to_csv(os.path.join(carpeta_resultados, nombre_archivo), index=True)
    # return ideam_objs
    anla_objs = objetos[0]
    cuenca_nombre = os.path.basename(ruta_principal).replace(".csv", "")
    # todo separar esto en exportar MADS y exportar ANLA

    for anla in anla_objs:
        anla.principal_funcion()
        str_apoyo = anla.str_apoyo or "sin_nombre"
        carpeta_resultados = f"C://Users//ASUS//Desktop//datos//unal//trabajo de grado//resultados//anla//otros//{str_apoyo}"
        os.makedirs(carpeta_resultados, exist_ok=True)
        anla.df2.to_csv(os.path.join(carpeta_resultados, f"df2_{cuenca_nombre}.csv"), index=True)
        anla.data_alter.to_csv(os.path.join(carpeta_resultados, f"data_alter_{cuenca_nombre}.csv"), index=True)
        # Exportar DataFrames principales
        anla.data_ref.to_csv(os.path.join(carpeta_resultados, f"data_ref_{cuenca_nombre}.csv"), index=True)
        anla.df_cdc_normal.to_csv(os.path.join(carpeta_resultados, f"df_cdc_normal_{cuenca_nombre}.csv"), index=True)
        anla.df_cdc_alterada.to_csv(os.path.join(carpeta_resultados, f"df_cdc_alterada_{cuenca_nombre}.csv"), index=True)
        anla.data_alter2.to_csv(os.path.join(carpeta_resultados, f"data_alter2_{cuenca_nombre}.csv"), index=True)

        # Exportar listas como CSV de una columna
        listas_exportables = {
            "propuesta_inicial_ref": anla.propuesta_inicial_ref,
            "caud_final": anla.caud_final,
            "q95": anla.q95,
            "q7_10": anla.q7_10,
            "cdc_normales": anla.cdc_normales,
            "cdc_alterados": anla.cdc_alterados,
            "caud_return_normal": anla.caud_return_normal,
            "caud_return_alterado": anla.caud_return_alterado
        }

        for nombre, lista in listas_exportables.items():
            df = pd.DataFrame(lista, columns=[nombre])
            df.to_csv(os.path.join(carpeta_resultados, f"{nombre}_{cuenca_nombre}.csv"), index=True)

        # Exportar resultados si tienen DataFrames
        for etiqueta, resultado in {
            "ori": anla.resultados_ori,
            "alterada": anla.resultados_alterada,
            "ref": anla.resultados_ref
        }.items():
            if resultado is not None:
                # Exportar el DataFrame principal (cdc)
                resultado.cdc.to_csv(
                    os.path.join(carpeta_resultados, f"resultado_{etiqueta}_cdc_{cuenca_nombre}.csv"),
                    index=True
                )

                # Exportar caudales de retorno
                pd.DataFrame(resultado.caud_return, columns=["caud_return"]).to_csv(
                    os.path.join(carpeta_resultados, f"resultado_{etiqueta}_caud_return_{cuenca_nombre}.csv"),
                    index=True
                )

                # Exportar los años del cdc
                pd.DataFrame(resultado.cdc_anios, columns=["anio"]).to_csv(
                    os.path.join(carpeta_resultados, f"resultado_{etiqueta}_anios_{cuenca_nombre}.csv"),
                    index=True
                )

    return "ok"
    # except Exception as e:
    #     print(f"Error procesando cuenca {ruta_principal}: {e}")
    #     return f"error: {e}"


def export_instancia_ideam(resultado: estado_algoritmo.EstadoIdeam, carpeta_resultados: str) -> None:
    str_apoyo = resultado.str_apoyo or "sin_nombre"
    cuenca_nombre = resultado.codigo_est
    carpeta_resultados = os.path.join(carpeta_resultados, str_apoyo)
    os.makedirs(carpeta_resultados, exist_ok=True)
    # Guardar los datos
    resultado.df2.to_csv(os.path.join(carpeta_resultados, f"df2_{cuenca_nombre}.csv"), index=True)
    resultado.data_alter.to_csv(os.path.join(carpeta_resultados, f"data_alter_{cuenca_nombre}.csv"), index=True)
    graficar_caud_amb(resultado.data, resultado.data_alter, carpeta_resultados, f"Caudal_ambiental_{resultado.str_apoyo}", resultado.umbrales)
    # Umbrales simples
    df_umbrales_csv = pd.DataFrame(list(resultado.umbrales.items()), columns=["umbral", "valor"])
    df_umbrales_csv.to_csv(os.path.join(carpeta_resultados, f"umbrales_{cuenca_nombre}.csv"), index=True)

    # Umbrales DataFrame
    for nombre, df in resultado.df_umbrales.items():
        if not df.empty:
            nombre_archivo = f"{nombre}_{cuenca_nombre}.csv"
            df.to_csv(os.path.join(carpeta_resultados, nombre_archivo), index=True)
    return None

def export_instancia_anla(resultado_general: estado_algoritmo.EstadoAnla, ruta: str):
    str_apoyo = resultado_general.str_apoyo or "sin_nombre"
    cuenca_nombre = resultado_general.codigo_est
    ruta = os.path.join(ruta, str_apoyo)
    os.makedirs(ruta, exist_ok=True)
    resultado_general.df2.to_csv(os.path.join(ruta, f"df2_{cuenca_nombre}.csv"), index=True)
    resultado_general.data_alter.to_csv(os.path.join(ruta, f"data_alter_{cuenca_nombre}.csv"), index=True)
    graficar_caud_amb(resultado_general.data, resultado_general.data_alter, ruta, f"Caudal_ambiental_{resultado_general.str_apoyo}", None)
    # Exportar DataFrames principales
    resultado_general.data_ref.to_csv(os.path.join(ruta, f"data_ref_{cuenca_nombre}.csv"), index=True)
    resultado_general.df_cdc_normal.to_csv(os.path.join(ruta, f"df_cdc_normal_{cuenca_nombre}.csv"), index=True)
    resultado_general.df_cdc_alterada.to_csv(os.path.join(ruta, f"df_cdc_alterada_{cuenca_nombre}.csv"), index=True)
    resultado_general.data_alter2.to_csv(os.path.join(ruta, f"data_alter2_{cuenca_nombre}.csv"), index=True)

    # Exportar listas como CSV de una columna
    listas_exportables = {
        "propuesta_inicial_ref": resultado_general.propuesta_inicial_ref,
        "caud_final": resultado_general.caud_final,
        "q95": resultado_general.q95,
        "q7_10": resultado_general.q7_10,
        "cdc_normales": resultado_general.cdc_normales,
        "cdc_alterados": resultado_general.cdc_alterados,
        "caud_return_normal": resultado_general.caud_return_normal,
        "caud_return_alterado": resultado_general.caud_return_alterado
    }

    for nombre, lista in listas_exportables.items():
        df = pd.DataFrame(lista, columns=[nombre])
        df.to_csv(os.path.join(ruta, f"{nombre}_{cuenca_nombre}.csv"), index=True)

    # Exportar resultados si tienen DataFrames
    for etiqueta, resultado in {
        "ori": resultado_general.resultados_ori,
        "alterada": resultado_general.resultados_alterada,
        "ref": resultado_general.resultados_ref
    }.items():
        if resultado is not None:
            # Exportar el DataFrame principal (cdc)
            resultado.cdc.to_csv(
                os.path.join(ruta, f"resultado_{etiqueta}_cdc_{cuenca_nombre}.csv"),
                index=True
            )

            # Exportar caudales de retorno
            pd.DataFrame(resultado.caud_return, columns=["caud_return"]).to_csv(
                os.path.join(ruta, f"resultado_{etiqueta}_caud_return_{cuenca_nombre}.csv"),
                index=True
            )

            # Exportar los años del cdc
            pd.DataFrame(resultado.cdc_anios, columns=["anio"]).to_csv(
                os.path.join(ruta, f"resultado_{etiqueta}_anios_{cuenca_nombre}.csv"),
                index=True
            )
    if resultado_general.resultados_ori is not None and resultado_general.resultados_alterada is not None:
        graficar_cdc(
            resultado_general.resultados_ori.cdc,
            resultado_general.resultados_alterada.cdc,
            ruta,
            titulo=f"CDC {cuenca_nombre} (Natural vs Alterada)",
        )
    ruta_nat = os.path.join(ruta, 'iha','natural')
    ruta_alt = os.path.join(ruta, 'iha', 'alterado')
    exportar_anla_creado.exportar_iha_real(resultado_general.resultados_ori.iah_result, ruta_nat)
    exportar_anla_creado.exportar_iha_real(resultado_general.resultados_alterada.iah_result, ruta_alt)

def export_resultados_ideam(resultados_ideam: Optional[List[estado_algoritmo.EstadoIdeam]], ruta: str) -> int:
    if resultados_ideam is None:
        return 0
    ruta = os.path.join(ruta, "MADS")
    for resultado in resultados_ideam:
        export_instancia_ideam(resultado, ruta)
    export_compilado(resultados_ideam, ruta)
    return 0


def graficar_cdc(natural, alterado, ruta, titulo):
    """
    Grafica la curva de duración de caudales (CDC) para caudales naturales y alterados.

    Parámetros:
        natural  (pd.DataFrame): Índice datetime, columnas 'Valor' y 'cumsum'.
        alterado (pd.DataFrame): Índice datetime, columnas 'Valor' y 'cumsum'.
        ruta     (str): Ruta donde guardar la imagen.
        titulo   (str): Título de la gráfica.
    """
    # Umbrales de frecuencia
    # ruta = os.path.join(ruta, "cdc.png")
    os.makedirs(ruta, exist_ok=True)
    ruta = os.path.join(ruta, "cdc.png")
    cdc_umbrales = [0.70, 0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995]

    # Crear figura con proporción 2.2:1
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    # Graficar curvas CDC
    ax.plot(natural["cumsum"], natural["Valor"], label="Caudal natural", color="blue", linewidth=1)
    ax.plot(alterado["cumsum"], alterado["Valor"], label="Caudal alterado", color="red", linewidth=1)

    # Líneas verticales de umbral
    for freq in cdc_umbrales:
        ax.axvline(x=freq, color="gray", linestyle="--", linewidth=0.7)

    # Configuración de ejes
    ax.set_xlabel("Frecuencia relativa")
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.7)

    # Guardar y cerrar
    plt.tight_layout()
    os.makedirs(ruta, exist_ok=True)
    plt.savefig(ruta, dpi=300)
    plt.close()


def graficar_caud_amb(natural, alterado, ruta, titulo, umbrales: Optional[List] = None):
    """
    Genera una gráfica comparando caudales naturales y alterados,
    con umbrales como líneas horizontales.

    Parámetros:
        natural  (pd.DataFrame): Índice datetime, columna 'Valor'.
        alterado (pd.DataFrame): Índice datetime, columna 'Valor'.
        umbrales (list): Lista con 4 valores numéricos [u1, u2, u3, u4].
        ruta     (str): Ruta donde guardar la imagen.
        titulo   (str): Título de la gráfica.
    """
    # Crear figura con proporción 2.2:1
    os.makedirs(ruta, exist_ok=True)
    ruta = os.path.join(ruta, "caud_amb.png")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)  # 11/5 ≈ 2.2

    # Graficar las series
    ax.plot(natural.index, natural["Valor"], label="Caudal natural", color="blue", linewidth=1)
    ax.plot(alterado.index, alterado["Valor"], label="Caudal alterado", color="red", linewidth=1)

    # Dibujar umbrales
    colores_umbral = ["green", "orange", "purple", "brown"]
    nombres_umbral = ["Q15", "QB", "QTQ", "Q10"]  # puedes cambiarlos
    print(umbrales)
    if umbrales is not None:
        for i, (nombre, valor) in enumerate(umbrales.items()):
            ax.axhline(
                y=valor,
                color=colores_umbral[i % len(colores_umbral)],
                linestyle="--",
                linewidth=0.8,
                label=f"{nombre} = {valor:.2f}"
            )
    # Leyenda y etiquetas
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_xlabel("Fecha")
    ax.legend()
    ax.set_title(titulo)

    # Grid
    ax.grid(True, linestyle=":", linewidth=0.7)

    # Ajustar formato de fecha (opcional, si matplotlib detecta mal)
    fig.autofmt_xdate()

    # Guardar imagen
    plt.tight_layout()
    plt.savefig(ruta, dpi=300)
    plt.close()


def export_compilado(resultados: Optional[List[estado_algoritmo.EstadoAlgoritmo]], ruta: str, umbrales: Optional[list] = None) -> None:
    ruta = os.path.join(ruta, "compilado")
    os.makedirs(ruta, exist_ok=True)

    # Unir DataFrames de cada resultado
    df_naturales = [objeto.data for objeto in resultados]
    df_alterados = [objeto.data_alter2 for objeto in resultados]
    archivo_natural = unir_df(df_naturales)
    archivo_alterado = unir_df(df_alterados)
    archivo_natural.to_csv(os.path.join(ruta, "natural.csv"))
    archivo_alterado.to_csv(os.path.join(ruta, "alterado.csv"))
    # Graficar comparación de caudales
    graficar_caud_amb(
        archivo_natural,
        archivo_alterado,
        ruta,
        titulo="Caudal_ambiental",
        umbrales=umbrales
    )

    # ===== NATURAL =====
    ruta_nat = os.path.join(ruta, "natural")
    os.makedirs(ruta_nat, exist_ok=True)

    cdc_mads_natural: pd.DataFrame = funciones_anla.generar_cdc(archivo_natural)
    iha_mads_natural: IhaEstado.IhaEstado = IhaEstado.IhaEstado(archivo_natural)
    iha_mads_natural.calcular_iha()

    ruta_mads_cdc_nat: str = os.path.join(ruta_nat, 'cdc_natural.csv')
    cdc_mads_natural.to_csv(ruta_mads_cdc_nat, index=True)
    exportar_anla_creado.exportar_iha_real(iha_mads_natural, ruta_nat)

    # ===== ALTERADO =====
    ruta_alt = os.path.join(ruta, "alterado")
    os.makedirs(ruta_alt, exist_ok=True)

    cdc_mads_alterado: pd.DataFrame = funciones_anla.generar_cdc(archivo_alterado)
    iha_mads_alterado: IhaEstado.IhaEstado = IhaEstado.IhaEstado(archivo_alterado)
    iha_mads_alterado.calcular_iha()

    ruta_mads_cdc_alt: str = os.path.join(ruta_alt, 'cdc_alterado.csv')
    cdc_mads_alterado.to_csv(ruta_mads_cdc_alt, index=True)
    exportar_anla_creado.exportar_iha_real(iha_mads_alterado, ruta_alt)


def unir_df(caudales_array: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if caudales_array:
        df_total: pd.DataFrame = pd.concat(caudales_array).sort_index()
        caudales_resultado: pd.DataFrame = df_total.groupby(df_total.index).max()
        return caudales_resultado
    else:
        print(f"No hay datos para unir")
        return None


def export_resultados_anla(resultados_anla: Optional[List[estado_algoritmo.EstadoAnla]], ruta: str) -> int:
    if resultados_anla is None:
        return 0
    ruta = os.path.join(ruta, "ANLA")
    for resultado in resultados_anla:
        export_instancia_anla(resultado, ruta)
    export_compilado(resultados_anla, ruta)
    return 0


def ejecutar_cuenca_completa(diccionario: dict) -> None:
    resultados = generar_algoritmo(diccionario)
    for resultado in resultados:
        for instancia in resultado:
            instancia.principal_funcion()
    resultados_ideam = resultados[1]
    resultados_anla = resultados[0]
    ruta_exportar = diccionario["ruta_carpeta_salida"]
    nombre_estacion = resultados_ideam[0].codigo_est
    ruta_exportar = os.path.join(ruta_exportar, str(nombre_estacion))
    export_resultados_ideam(resultados_ideam, ruta_exportar)
    export_resultados_anla(resultados_anla, ruta_exportar)



def hacer_trabajo_actual():
    ruta_trabajo: str = "C://Users//ASUS//Desktop//datos//unal//trabajo de grado//trabajo_actual_ideam_4.csv"
    # ruta_cuencas: str = "C://Users//ASUS//Desktop//datos//unal//semillero//CAUDAL//resultados"
    ruta_cuencas: str = "C://Users//ASUS//Desktop//datos//unal//trabajo de grado//cuencas//"
    ruta_anios: str = "C://Users//ASUS//Desktop//datos//unal//semillero//consistencia//Consistencia"
    lista_numeros: list[tuple[int, str]] = []
    for carpeta_raiz, carpetas, archivos in os.walk(ruta_cuencas):
        for archivo in archivos:
            if archivo.endswith(".csv"):
                nombre_sin_ext = os.path.splitext(archivo)[0]
                if nombre_sin_ext.isdigit():
                    ruta_completa = os.path.join(carpeta_raiz, archivo)
                    lista_numeros.append((int(nombre_sin_ext), ruta_completa))
    # Crear archivo trabajo_actual.csv si no existe
    if not os.path.exists(ruta_trabajo):
        df_vacio = pd.DataFrame(columns=["Serie_principal", "resultado"])
        df_vacio.to_csv(ruta_trabajo, index=False)

    # Leer archivos
    df_trabajo = pd.read_csv(ruta_trabajo)
    # df_cuencas = pd.read_csv(ruta_cuencas)
    for nombre, ruta in lista_numeros:
        ruta_anios_utiles = os.path.join(ruta_anios, f"Resumen_{nombre}.xls")
        datos = list(pd.read_excel(ruta_anios_utiles, sheet_name="RESULTADOS_GRAFICADOS")['Año'])
        if len(datos) > 0:
            diccionario_espejo['anios_utiles'] = datos
        diccionario_espejo['archivos']['archivo_base'] = ruta
        ejecutar_cuenca_completa(diccionario_espejo)
        nueva_fila = {
            "Serie_principal": nombre,
            "resultado": 100
        }
        df_trabajo = pd.concat([df_trabajo, pd.DataFrame([nueva_fila])], ignore_index=True)
        df_trabajo.to_csv(ruta_trabajo, index=False)
    # hasta aquí


def exportardf_completo(caudales_array: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if caudales_array:
        df_total: pd.DataFrame = pd.concat(caudales_array).sort_index()
        caudales_resultado: pd.DataFrame = df_total.groupby(df_total.index).max()
        return caudales_resultado
        # print(f"Exportado: {str_export}")
    else:
        print(f"No hay datos para exportar")
        return None


def exportar_anla(objetos, ruta_principal):
    anla_objs = objetos[0]
    cuenca_nombre = os.path.basename(ruta_principal).replace(".csv", "")

    for anla in anla_objs:
        anla.principal_funcion()
        str_apoyo = anla.str_apoyo or "sin_nombre"
        carpeta_resultados = f"C://Users//ASUS//Desktop//datos//unal//trabajo de grado//resultados//anla//{str_apoyo}"
        os.makedirs(carpeta_resultados, exist_ok=True)

        # Exportar DataFrames principales
        anla.data_ref.to_csv(os.path.join(carpeta_resultados, f"data_ref_{cuenca_nombre}.csv"), index=False)
        anla.df_cdc_normal.to_csv(os.path.join(carpeta_resultados, f"df_cdc_normal_{cuenca_nombre}.csv"), index=False)
        anla.df_cdc_alterada.to_csv(os.path.join(carpeta_resultados, f"df_cdc_alterada_{cuenca_nombre}.csv"), index=False)
        anla.data_alter2.to_csv(os.path.join(carpeta_resultados, f"data_alter2_{cuenca_nombre}.csv"), index=False)

        # Exportar listas como CSV de una columna
        listas_exportables = {
            "propuesta_inicial_ref": anla.propuesta_inicial_ref,
            "caud_final": anla.caud_final,
            "q95": anla.q95,
            "q7_10": anla.q7_10,
            "cdc_normales": anla.cdc_normales,
            "cdc_alterados": anla.cdc_alterados,
            "caud_return_normal": anla.caud_return_normal,
            "caud_return_alterado": anla.caud_return_alterado
        }

        for nombre, lista in listas_exportables.items():
            df = pd.DataFrame(lista, columns=[nombre])
            df.to_csv(os.path.join(carpeta_resultados, f"{nombre}_{cuenca_nombre}.csv"), index=False)

        # Exportar resultados si tienen DataFrames
        for etiqueta, resultado in {
            "ori": anla.resultados_ori,
            "alterada": anla.resultados_alterada,
            "ref": anla.resultados_ref
        }.items():
            if resultado and hasattr(resultado, 'df'):
                resultado.df.to_csv(os.path.join(carpeta_resultados, f"resultado_{etiqueta}_{cuenca_nombre}.csv"), index=False)

    return "ok"
if __name__ == "__main__":
    hacer_trabajo_actual()