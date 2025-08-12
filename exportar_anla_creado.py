import json
import os
from typing import List

import pandas as pd

import IhaEstado
import estado_algoritmo
import funciones_anla
import ingreso_datos

# C:\Users\ASUS\Desktop\datos\unal\trabajo de grado\latex\resultados_anla_2\anla\otros Carpeta anla
cuencas = [
    54077010,
    44017060,
    32067030,
    29067150,
    22027020
]

RUTA_BASE = r'C:\Users\ASUS\Desktop\datos\unal\trabajo de grado\latex\codigo\hecca'

RUTA_RESULTADOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..', 'resultados_7'))
ruta_cuencas = os.path.abspath(os.path.join(RUTA_BASE, '..', '..', '..', 'cuencas'))
ruta_anla = os.path.abspath(os.path.join(RUTA_BASE, '..', '..', 'resultados_anla_2', 'anla', 'otros'))

estados: List[str] = ['ninio', 'ninia', 'normal']
# RUTA_BASE: str = 'C:\\Users\\ASUS\\Desktop\\datos\\unal\\trabajo de grado\\latex\\codigo\\hecca'
# RUTA_RESULTADOS: str = os.path.join(RUTA_BASE,'..','..','resultados_7')
# ruta_cuencas: str = os.path.join(RUTA_BASE,'..','..','..','cuencas')
# ruta_anla: str = os.path.join(RUTA_BASE,'..','..','resultados_anla_2','anla','otros')
# instancias_anla: Optional[tuple[list[estado_algoritmo.EstadoAlgoritmo]]] = None


def exportar_iha(objeto_2: estado_algoritmo.ResultadosAnla, ruta: str):
    objeto_iha: IhaEstado.IhaEstado = objeto_2.iah_result
    for i in range(1, 6):
        df_temp: pd.DataFrame = getattr(objeto_iha, f'grupo_{i}')
        ruta_a_exp: str = os.path.join(ruta, f"grupo_{i}.csv")
        os.makedirs(os.path.dirname(ruta_a_exp), exist_ok=True)
        df_temp.to_csv(ruta_a_exp)


def exportar_iha_real(objeto_iha: IhaEstado.IhaEstado, ruta: str):
    for i in range(1, 6):
        df_temp: pd.DataFrame = getattr(objeto_iha, f'grupo_{i}')
        print(df_temp)
        ruta_a_exp: str = os.path.join(ruta, f"grupo_{i}.csv")
        os.makedirs(os.path.dirname(ruta_a_exp), exist_ok=True)
        df_temp.to_csv(ruta_a_exp)


if __name__ == '__main__':
    with open('setup.json', "r") as archivo_json:
        datos_dict: dict = json.load(archivo_json)
    if datos_dict:
        for cuenca in cuencas:
            path_cuenca: str = os.path.join(ruta_cuencas, f"{cuenca}.csv")
            datos_dict['archivos']['archivo_base'] = path_cuenca
            instancia_temp = ingreso_datos.generar_algoritmo(datos_dict)
            lista_temp: List[estado_algoritmo.EstadoAnla] = instancia_temp[0]
            for objeto in lista_temp:
                estado: str = objeto.str_apoyo
                ruta_alter: str = os.path.join(ruta_anla, estado, f"data_alter_{cuenca}.csv")
                df_alter: pd.DataFrame = pd.read_csv(ruta_alter, index_col=0, parse_dates=True)
                objeto.data_alter2 = df_alter
                objeto.ajuste = 5
                objeto.resultados_ori = funciones_anla.calc_resultados(objeto.data, objeto.ajuste)
                ruta_ori: str = os.path.join(RUTA_RESULTADOS, str(cuenca), estado, 'natural')
                objeto.resultados_alterada = funciones_anla.calc_resultados(objeto.data_alter2, objeto.ajuste)
                ruta_alt: str = os.path.join(RUTA_RESULTADOS, str(cuenca), estado, 'alterada')
                exportar_iha(objeto.resultados_ori, ruta_ori)
                exportar_iha(objeto.resultados_alterada, ruta_alt)
