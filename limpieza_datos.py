import datetime
from typing import Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime as today_date_time
from sklearn.linear_model import LinearRegression
import re


class ErrorFecha(Exception):
	pass


def find_months(text: str) -> bool:
	english_months_regex = (r'\b(?:January|February|March|April|May|June|July|August|September|'
													r'October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b')
	spanish_months_regex = (r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|'
													r'noviembre|diciembre|ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\b')
	english_matches = re.findall(english_months_regex, text, flags=re.IGNORECASE)
	spanish_matches = re.findall(spanish_months_regex, text, flags=re.IGNORECASE)
	all_matches = english_matches + spanish_matches
	return bool(all_matches)


def formato_fecha(datos: pd.DataFrame) -> tuple[int, Optional[str]]:
	"""
  @param datos: pd.DataFrame con alguna fecha formatada
  @return: tuple (int: codigo_validez: (0: formato valido, retorna str; 1: formato valido no ambiguo no retorna str;
   2: formato no valido, no retorna str).
   str: formato de fechas, en caso de no valido se retorna "")
  """
	primero: str = ""
	segundo: str = ""
	# tercero: str = ""
	a = datos.sample()
	c = a.index[0]
	if not (c + 1 in datos.index):
		c -= 2
	b = datos.loc[c + 1]
	if len(a['Fecha'].iloc[0]) < 10:
		return 2, None
	if find_months(a['Fecha'].iloc[0]):
		return 1, None
	excedente = None
	num_dos_punt = a['Fecha'].iloc[0].count(":")
	if num_dos_punt == 0:
		excedente = ""
	elif num_dos_punt == 1:
		excedente = " %H:%M"
	elif num_dos_punt == 2:
		excedente = " %H:%M:%S"
	else:
		raise ErrorFecha("no se entiende el formato de fecha, por favor"
						" cambielo a un formato sin ambiguedades, recomendamos yyyy/mm/dd")
	str_year = "%Y"
	str_month = "%m"
	str_day = "%d"
	# separador = ()
	if a['Fecha'].iloc[0][4].isdigit():  # empieza con dia o mes
		tercero = str_year
		separador = (a['Fecha'].iloc[0][2], a['Fecha'].iloc[0][2])
		a = pd.to_datetime(a['Fecha'], dayfirst=True)
		b = pd.to_datetime(b['Fecha'], dayfirst=True)
		dif_fechas = b - a
		if isinstance(dif_fechas, pd.Series):
			dif_fechas = dif_fechas.iloc[0]  # Obtener el primer valor de la serie
		try:
			if abs(dif_fechas.days) == 1:
				primero = str_day
				segundo = str_month
			else:
				primero = str_month
				segundo = str_day
		except AttributeError:
			pass
	else:  # empieza con año
		separador = (a['Fecha'].iloc[0][4], a['Fecha'].iloc[0][7])
		primero = str_year
		segundo = str_month
		tercero = str_day
	return 0, primero + separador[0] + segundo + separador[1] + tercero + excedente


def datos_anomalos(df: pd.DataFrame, retirar_anomalos: bool = True) -> pd.DataFrame:
	""" Recibe el dataframe de trabajo ordenado, con huecos y datos anomalos y devuelve el dataframe de trabajo """
	if not retirar_anomalos:
		return df
	for mes in range(1, 13):
		Q1 = df.loc[df.index.month == mes, 'Valor'].quantile(0.25)
		Q3 = df.loc[df.index.month == mes, 'Valor'].quantile(0.75)
		IQR = Q3 - Q1
		limite_inferior = Q1 - 1.5 * IQR
		limite_superior = Q3 + 1.5 * IQR
		anomalos_mes = (
						((df['Valor'] < limite_inferior) |
						 (df['Valor'] > limite_superior)) &
						(df.index.month == mes)
		)

		df.loc[anomalos_mes, 'Valor'] = np.nan
	return df


def fecha_cruda(df1: pd.DataFrame) -> tuple[bool, float, float, float, float]:
	"""
  Toma un pd.DataFrame y devuelve las fechas utilizables, porcentaje de datos faltantes, años totales y si es funcional
  Parametros:
    df1 (pd.DataFrame): Dataframe crudo igual que como entrega el dhime
  returns:
    bool: True si es funcional, es funcional si tiene las columnas 'Fecha' y 'Valor'.
    float: el porcentaje de datos faltante
    int: año inicial
    int: año final
    int: años utilizables
  """
	funcional = verificar_columnas(df1)
	if funcional:
		df2 = organize_df(df1)
		a = summarize_missing_values(df2)
		b = prim_ult(df2)
		return True, a, int(b[0].year), int(b[1].year), b[2]
	else:
		return False, 0, 0, 0, 0


def verificar_columnas(dataframe: pd.DataFrame) -> bool:
	"""
  Determina si el dataframe recibido cuenta con las columnas 'Fecha' y 'Valor'
  parámetros:
    dataframe: Dataframe a verificar
  return:
    bool: True si sí las tiene
  """

	return {'Valor', 'Fecha'}.issubset(set(dataframe.columns))


def prim_ult(df1: pd.DataFrame) -> tuple[int, int, int]:
	"""
  Recibe un dataframe y devuelve primer año completo, ultimo año completo y la cantidad de años completos entre ellas
  parámetros:
    df1: pd.DataFrame a verificar
  return:
     int: primer año completo
     int: ultimo año completo
     int: años disponibles
  """
	# Se necesitan años completos entonces
	min_date = datetime.date(df1.index[0].year, df1.index[0].month, df1.index[0].day)
	if min_date.month != 1 or min_date.day != 1:
		min_date = datetime.date(df1.index[0].year + 1, 1, 1)
	max_date = datetime.date(df1.index[-1].year, df1.index[-1].month, df1.index[-1].day)
	if max_date.month != 12 or max_date.day != 31:
		max_date = datetime.date(df1.index[-1].year - 1, 12, 31)
	delta_anios = max_date.year - min_date.year + 1
	# min = int(min_date.year)
	# max = int(max_date.year)
	return min_date, max_date, delta_anios


"""###funciones de llenado"""


# Función para resumir la información de los datos faltantes


def summarize_missing_values(complete_data: pd.DataFrame) -> float:
	"""
  resume información de los datos faltantes, recibe un dataframe y retorna el porcentaje de datos faltante
  parámetros:
    complete_data: pd.DataFrame dataframe a verificar
  return:
    float: porcentaje de datos faltantes
  """
	mask = complete_data.isna()
	list_missing_values = complete_data[mask['Valor']].index
	length_missing_values = len(list_missing_values)
	length_total_values = len(complete_data)
	ratio_missing_values = length_missing_values / length_total_values
	return ratio_missing_values


"""Llenado completo"""

@DeprecationWarning
def funcion_miguel(df_base: pd.DataFrame, df_apoyo: pd.DataFrame = None) -> bool:
	return True
	# if df_apoyo:
	#   pass
	#   # hacer algo
	# else:
	#   df_base.min()
	#   pass
	#   # normal
	# todo esto
	# todo esto


def process_df(df_base: pd.DataFrame, df_apoyo: Optional[pd.DataFrame] = None, areas: Optional[tuple] = None) -> Optional[pd.DataFrame]:
	"""
    Procesa un DataFrame con o sin un DataFrame de apoyo para llenar los valores NaN en el primer DataFrame.

    Parameters
    ----------
    df_base: pandas.DataFrame
        El DataFrame principal a procesar.
    df_apoyo: pandas.DataFrame, opcional
        El DataFrame de apoyo que se utilizará para llenar los valores NaN en df_base. Por defecto es None.
    areas: tuple, opcional
        Las áreas que se usarán para calcular las proporciones. Por defecto es None.

    Returns
    -------
    pandas.DataFrame
        El DataFrame df_base procesado con los valores NaN reemplazados, el formato de salida es index: Fecha y 'cuenca-base' para los valores de caudal
    """
	df_base, df_apoyo = organize_df(df_base, df_apoyo)
	if areas is not None:
		df_base = df_base / areas[0]
		df_apoyo = df_apoyo / areas[1]
	if df_apoyo is not None:
		df_base_2, df_apoyo_2 = fill_na_values(df_base, df_apoyo)
		X = df_apoyo_2.values.reshape(-1, 1)
		Y = df_base_2.values
		model: LinearRegression = LinearRegression().fit(X, Y)
		slope = model.coef_[0]
		intercept = model.intercept_
		new_series = slope * df_apoyo + intercept
		df_base = df_base.fillna(new_series)
	df_base = datos_anomalos(df_base, True)
	df_base = fill_data_na(df_base)
	df_return = sacar_anios(df_base)
	# todo aquí se pone lo de miguel
	if areas is not None:
		df_return = df_return * areas[0]
	# df_return.rename(columns={'Valor': 'cuenca-base'}, inplace=True)
	return df_return

def parse_dict_fe(json_entrada: dict) -> pd.DataFrame:
  start_date = json_entrada['data'][0]['data'][0]['startDate']
  final_date = json_entrada['data'][0]['data'][0]['endDate']
  Valor = json_entrada['data'][0]['data'][0]['series']

  date_range = pd.date_range(start=start_date, end=final_date)
  df_new = pd.DataFrame({'Valor': Valor}, index=date_range)

  # Reset the index to a numerical index starting from 0
  df_new = df_new.reset_index(drop=False)

  # Rename the old index column to 'Fecha'
  df_new = df_new.rename(columns={'index': 'Fecha'})

  # Convert 'Fecha' column to string and format
  df_new['Fecha'] = df_new['Fecha'].astype(str).str.slice(0, 10)
  df_new['Valor'] = pd.to_numeric(df_new['Valor'], errors='coerce')
  return df_new

def organize_df(df_base: pd.DataFrame, df_apoyo: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
	"""
    Esta función recibe dos DataFrames, uno obligatorio `df_base` y otro opcional `df_apoyo`.
    La función convierte la columna 'Fecha' en tipo datetime y
    reindexa los DataFrames con un rango de fechas completo desde la primera hasta la última fecha presente en `df_base`
    Si `df_apoyo` es diferente de None, también se procesa `df_apoyo` y se retornan ambos DataFrames reindexados.
    En caso contrario, solo se retorna `df_base` reindexado.

    Argumentos:
    df_base -- un DataFrame con dos columnas 'Fecha' y 'Valor'
    df_apoyo -- un DataFrame con dos columnas 'Fecha' y 'Valor' (opcional)

    Retorna:
    (base, apoyo) -- tupla con los DataFrames `df_base` y `df_apoyo` reindexados (si `df_apoyo` no es None)
    base -- el DataFrame `df_base` reindexado (si `df_apoyo` es None)
    """
	base = df_base[['Fecha', 'Valor']].copy()
	formato = formato_fecha(base)
	if formato[0] == 0:  # formato encontrado utilizable
		base['Fecha'] = pd.to_datetime(base['Fecha'], format=formato[1])
	elif formato[0] == 1:  # formato encontrado pero pandas lo entiende mejor
		base['Fecha'] = pd.to_datetime(base['Fecha'])
	else:  # formato no encontrado
		raise ErrorFecha("no se entiende el formato de fecha, por favor"
										 " cambielo a un formato sin ambigüedades, recomendamos yyyy/mm/dd")
	# base['Fecha'] = pd.to_datetime(base['Fecha'], yearfirst=True)
	size = base['Fecha'].size
	# days = pd.date_range(base.at[0,'Fecha'], base.at[size-1,'Fecha'])
	days = pd.date_range(base.iloc[0]['Fecha'], base.iloc[-1]['Fecha'])
	days_ind = pd.DatetimeIndex(days)
	base = base.set_index('Fecha')
	base = base.reindex(days)
	base, anios = anios_vacios(base)
	if df_apoyo is not None:
		apoyo = df_apoyo[['Fecha', 'Valor']].copy()
		apoyo['Fecha'] = pd.to_datetime(apoyo['Fecha'])
		apoyo = apoyo.set_index('Fecha')
		apoyo = apoyo.reindex(days)
		apoyo = quitar_anio(apoyo, anios)
		return base, apoyo
	else:
		return (base, None)


def organize_df(df_base: pd.DataFrame, df_apoyo: Optional[pd.DataFrame] = None) ->tuple[pd.DataFrame, Optional[pd.DataFrame]]:
	"""
    Recibe dos DataFrames, `df_base` y Optional[df_apoyo].
    convierte la columna 'Fecha' en tipo datetime y
    reindexa los DataFrames con un rango de fechas completo desde la primera hasta la última fecha presente en `df_base`
    Si `df_apoyo` es diferente de None, también se procesa `df_apoyo` y se retornan ambos DataFrames reindexados.
    En caso contrario, solo se retorna `df_base` reindexado.

    Argumentos:
    df_base -- un DataFrame con dos columnas 'Fecha' y 'Valor'
    df_apoyo -- un DataFrame con dos columnas 'Fecha' y 'Valor' (opcional)

    Retorna:
    tuple[base, apoyo] -- DataFrames `df_base` y `df_apoyo` reindexados.
    base -- el DataFrame `df_base` reindexado (si `df_apoyo` es None)
    """
	base = df_base[['Fecha', 'Valor']].copy()
	formato = formato_fecha(base)
	if formato[0] == 0:  # formato encontrado utilizable
		base['Fecha'] = pd.to_datetime(base['Fecha'], format=formato[1])
	elif formato[0] == 1:  # formato encontrado pero pandas lo entiende mejor
		base['Fecha'] = pd.to_datetime(base['Fecha'])
	else:  # formato no encontrado
		raise ErrorFecha("no se entiende el formato de fecha, por favor"
										 " cambielo a un formato sin ambigüedades, recomendamos yyyy/mm/dd")
	# base['Fecha'] = pd.to_datetime(base['Fecha'], yearfirst=True)
	size = base['Fecha'].size
	# days = pd.date_range(base.at[0,'Fecha'], base.at[size-1,'Fecha'])
	days = pd.date_range(base.iloc[0]['Fecha'], base.iloc[-1]['Fecha'])
	days_ind = pd.DatetimeIndex(days)
	base = base.set_index('Fecha')
	# print(base.head())
	base = base.reindex(days)
	base, anios = anios_vacios(base)
	if df_apoyo is not None:
		apoyo = df_apoyo[['Fecha', 'Valor']].copy()
		apoyo['Fecha'] = pd.to_datetime(apoyo['Fecha'])
		apoyo = apoyo.set_index('Fecha')
		apoyo = apoyo.reindex(days)
		apoyo = quitar_anio(apoyo, anios)
		return base, apoyo
	else:
		return base, None


def anios_vacios(df_propio: pd.DataFrame) -> tuple[pd.DataFrame, set]:
	prueba_grupe_2 = df_propio.groupby([df_propio.index.year])
	df = pd.DataFrame()
	df = df.assign(Fecha=None, Nas=None)
	for group, data_filter in prueba_grupe_2:
		year = group[0]
		row = [year, data_filter.isna().sum().iloc[0] / 365]
		df.loc[len(df)] = row
	anios_completos = set(df[df['Nas'] < 0.9]['Fecha'])
	df_propio = quitar_anio(df_propio, anios_completos)
	return df_propio, anios_completos


def quitar_anio(df: pd.DataFrame, anios: set) -> pd.DataFrame:
	return df[df.index.year.isin(anios)].copy()


def fill_na_values(df_base: pd.DataFrame, df_apoyo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
    Elimina las filas en dos DataFrames donde al menos un valor es NaN.

    Parameters
    ----------
    df_base: pandas.DataFrame
        El DataFrame principal a procesar.
    df_apoyo: pandas.DataFrame
        El DataFrame de apoyo que se utilizará para llenar los valores NaN en df_base.

    Returns
    -------
    tuple (pandas.DataFrame, pandas.DataFrame)
        Una tupla con los dos DataFrames sin filas con NaN.
    """
	mask = df_base.isna() | df_apoyo.isna()
	df_base = df_base[~mask].copy()
	df_apoyo = df_apoyo[~mask].copy()
	df_base.dropna(inplace=True)
	df_apoyo.dropna(inplace=True)
	return df_base, df_apoyo


def fill_data_na(data: pd.DataFrame) -> pd.DataFrame:
	"""
    Completa los valores faltantes en el DataFrame `data` utilizando el promedio del día y mes correspondiente.

    Argumentos:
    data: pandas.DataFrame
        DataFrame con los datos a completar. Debe tener una columna 'cuenca-base' que es la que se completará.

    Retorna:
    pandas.DataFrame
        DataFrame con los valores faltantes completados.
    """
	complete_data_day_month = data.groupby(by=[data.index.day, data.index.month])
	complete_data_day_month_description = complete_data_day_month.describe()

	def fill_caudal_day(row: pd.tseries) -> float:
		"""
    Función auxiliar que completa un valor faltante en la columna 'Valor' en el DataFrame `data`.

    Argumentos:
    row: pandas.Series
        Fila en el DataFrame con los datos a completar.

    Retorna:
    float
        El valor completado para la columna 'Valor'.
    """
		if np.isnan(row['Valor']):
			summary_day = complete_data_day_month_description.loc[row.name.day, row.name.month]
			return complete_data_day_month_description.loc[row.name.day, row.name.month].loc[('Valor', 'mean')]
		return row['Valor']

	data['Valor'] = data.apply(fill_caudal_day, axis=1)
	return data
# test [

def sacar_anios(df1: pd.DataFrame) -> pd.DataFrame:
	"""
    Función que recorta un DataFrame `df1` indexado con fechas,
    eliminando el año correspondiente a la primera y última fechas.

    Argumentos:
    df1: pandas.DataFrame
        DataFrame indexado con fechas que se quiere recortar.

    Retorna:
    pandas.DataFrame
        DataFrame con el mismo contenido que `df1` pero sin el año correspondiente a la primera y última fechas.
    """
	min_date, max_date, delta_anios = prim_ult(df1)
	Today_time = today_date_time.min.time()
	min_date_corr = today_date_time.combine(min_date, Today_time)
	max_date_corr = today_date_time.combine(max_date, Today_time)
	df1 = df1[df1.index >= min_date_corr][df1[df1.index >= min_date_corr].index <= max_date_corr].copy()
	return df1
