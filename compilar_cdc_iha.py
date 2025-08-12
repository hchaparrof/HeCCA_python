import os
from typing import Optional, List

import pandas as pd
import exportar_anla_creado
import funciones_anla
import ejecucion_bloque
from IhaEstado import IhaEstado


CARPETA_TRABAJO: str = 'C:\\Users\\ASUS\\Desktop\\datos\\unal\\trabajo de grado\\latex'
CARPETA_RESULTADOS: str = 'C:\\Users\\ASUS\\Desktop\\datos\\unal\\trabajo de grado\\latex\\resultados_compilados'
# C:\Users\ASUS\Desktop\datos\unal\trabajo de grado\latex\ambientales compilados
CUENCAS: List[int] = [
	54077010,
	44017060,
	32067030,
	29067150,
	22027020
]
resultados_anla: str = os.path.join(
	CARPETA_TRABAJO,
	'ambientales compilados',
	'anla')
resultados_mads: str = os.path.join(
	CARPETA_TRABAJO,
	'ambientales compilados',
	'mads')

if __name__ == '__main__':
	for cuenca in CUENCAS:
		ruta_cuenca_mads: str = os.path.join(resultados_mads, f'data_alter_{cuenca}.csv')
		ruta_cuenca_anla: str = os.path.join(resultados_anla, f'data_alter_{cuenca}.csv')
		# Para lo de CDC
		archivo_mads: pd.DataFrame = pd.read_csv(ruta_cuenca_mads, index_col=0, parse_dates=True)
		archivo_anla: pd.DataFrame = pd.read_csv(ruta_cuenca_anla, index_col=0, parse_dates=True)
		cdc_mads: pd.DataFrame = funciones_anla.generar_cdc(archivo_mads)
		cdc_anla: pd.DataFrame = funciones_anla.generar_cdc(archivo_anla)
		# para lo de IHA
		iha_mads: IhaEstado = IhaEstado(archivo_mads)
		iha_anla: IhaEstado = IhaEstado(archivo_anla)
		iha_mads.calcular_iha()
		iha_anla.calcular_iha()
		ruta_anla_resultados: str = os.path.join(resultados_anla, str(cuenca))
		ruta_mads_resultados: str = os.path.join(resultados_mads, str(cuenca))
		print()
		ruta_mads_cdc: str = os.path.join(ruta_mads_resultados, 'cdc.csv')
		os.makedirs(os.path.dirname(ruta_mads_cdc), exist_ok=True)
		ruta_anla_cdc: str = os.path.join(ruta_anla_resultados, 'cdc.csv')
		os.makedirs(os.path.dirname(ruta_anla_cdc), exist_ok=True)
		cdc_mads.to_csv(ruta_mads_cdc, index=True)
		cdc_anla.to_csv(ruta_anla_cdc, index=True)

		exportar_anla_creado.exportar_iha_real(iha_mads, ruta_mads_resultados)
		exportar_anla_creado.exportar_iha_real(iha_anla, ruta_anla_resultados)



