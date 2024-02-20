import datetime
import numpy as np
import pandas as pd
from datetime import datetime as todaysDateTime
from sklearn.linear_model import LinearRegression
def datos_anomalos(df, retirar_anomalos=True):
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


def fecha_cruda(df1):
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


def verificar_columnas(dataframe):
  """
  Determina si el dataframe recibido cuenta con las columnas 'Fecha' y 'Valor'
  parámetros:
    dataframe: Dataframe a verificar
  return:
    bool: True si sí las tiene
  """

  return {'Valor', 'Fecha'}.issubset(set(dataframe.columns))


def prim_ult(df1):
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
    min_date = datetime.date(df1.index[0].year + 1,  1, 1)
  max_date = datetime.date(df1.index[-1].year, df1.index[-1].month, df1.index[-1].day)
  if max_date.month != 12 or max_date.day != 31:
    max_date = datetime.date(df1.index[-1].year - 1, 12, 31)
  delta_anios = max_date.year-min_date.year+1
  min = int(min_date.year)
  max = int(max_date.year)
  return min_date, max_date, delta_anios


"""###funciones de llenado"""

# Función para resumir la información de los datos faltantes


def summarize_missing_values(complete_data):
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
  ratio_missing_values = length_missing_values/length_total_values
  return ratio_missing_values


"""Llenado completo"""


def process_df(df_base, df_apoyo=None, areas: tuple = None):
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
        El DataFrame df_base procesado con los valores NaN reemplazados, si es posible.
    """
  area1, area2 = areas
  if df_apoyo is not None:
    df_base, df_apoyo = organize_df(df_base, df_apoyo)
  else:
    df_base = organize_df(df_base)
  if areas is not None:
    df_base = df_base / area1
    df_apoyo = df_apoyo / area2
  if df_apoyo is not None:
    df_base_2, df_apoyo_2 = fill_na_values(df_base, df_apoyo)
    X = df_apoyo_2.values.reshape(-1, 1)
    Y = df_base_2.values
    model = LinearRegression().fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    new_series = slope * df_apoyo + intercept
    df_base = df_base.fillna(new_series)
  df_base = datos_anomalos(df_base, True)
  df_base = fill_data_na(df_base)
  df_return = sacar_anios(df_base)
  if areas is not None:
    df_return = df_return*area1
  return df_return


def organize_df(df_base, df_apoyo=None):
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
  base['Fecha'] = pd.to_datetime(base['Fecha'], dayfirst=True)
  size = base['Fecha'].size
  # days = pd.date_range(base.at[0,'Fecha'], base.at[size-1,'Fecha'])
  days = pd.date_range(base.iloc[0]['Fecha'], base.iloc[-1]['Fecha'])
  days_ind = pd.DatetimeIndex(days)
  base = base.set_index('Fecha')
  base = base.reindex(days)
  if df_apoyo is not None:
    apoyo = df_apoyo[['Fecha', 'Valor']].copy()
    apoyo['Fecha'] = pd.to_datetime(apoyo['Fecha'])
    apoyo = apoyo.set_index('Fecha')
    apoyo = apoyo.reindex(days)
    return base, apoyo
  else:
    return base


def fill_na_values(df_base, df_apoyo):
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


def fill_data_na(data):
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

  def fill_caudal_day(row):
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


def sacar_anios(df1):
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
  Today_time = todaysDateTime.min.time()
  min_date_corr = todaysDateTime.combine(min_date, Today_time)
  max_date_corr = todaysDateTime.combine(max_date, Today_time)
  df1 = df1[df1.index >= min_date_corr][df1[df1.index >= min_date_corr].index <= max_date_corr].copy()
  return df1
