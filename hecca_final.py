# -*- coding: utf-8 -*-
"""##librerias"""
import pandas as pd
import numpy as np
from funciones_ideam import buscar_umbrales
from funciones_ideam import df_eventos
from funciones_ideam import nombrar_evento
from funciones_ideam import contar_eventos
from funciones_ideam import org_df2_1
from funciones_ideam import org_df2_2
from funciones_ideam import formar_alter
from funciones_ideam import org_alt
from funciones_ideam import crear_evento
import datetime
# import matplotlib.pyplot as plt
# from datetime import date as todaysDate
from datetime import datetime as todaysDateTime
# from numpy.ma.core import empty
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import seaborn as sns
from estado_algoritmo import EstadoAlgoritmo
from scipy.stats import f
from scipy.stats import ttest_ind
from limpieza_datos import process_df
from comprobacion_ideam import calibrar_mes

"""##clases"""


# class Evento_completo:
  # """
  # Reune todas las caracteristicas del evento como
  # magnitud, intensidad, etc. también reune el evento en sí en un df.
  # """
  # mes = -1  # es el mes en que se encuentra el evento, no existen eventos multimensuales
  # magnitud = -1.0  # la suma de los caudales excedentes al umbral del evento, m3/s
  # intensidad = -1.0  # magnitud del evento sobre la duracion m3/(s*día)
  # duracion = -1  # duracion en días del evento (día)
  #
  # def __init__(self, df1, umbral, umbralstr):
  #   self.df1 = df1.copy()
  #   self.umbral = umbral
  #   self.umbralstr = umbralstr
  #   self.organizar_df()
  #   self.set_intensidad()
  #   self.organizar_mes()
  #
  # def organizar_df(self):
  #   """
  #   Toma el DataFrame con los datos del evento y le
  #   añade la columba unbralstr en la que va a estar el excedente del caudal sobre el umbral
  #   """
  #   self.df1 = self.df1[['cuenca-base', self.umbralstr]]
  #   self.df1.loc[:, self.umbralstr] = abs(self.df1['cuenca-base'] - self.umbral)
  #
  # def set_intensidad(self):
  #   """Determina los parametros hidrologicos del evento, magnitud duración e intensidad"""
  #   self.magnitud = self.df1[self.umbralstr].sum()
  #   self.duracion = self.df1[self.umbralstr].size
  #   self.intensidad = self.magnitud/self.duracion
  #
  # def organizar_mes(self):
  #   """
  #   Establece el mes en el que se encuentra el evento, como no
  #   existen eventos multimensuales se toma cualquier dato de mes
  #   """
  #   self.mes = self.df1.index.month.min()


"""
## funciones

###funciones utilitarias: llenado de datos y esas cosas, luego ampliare esto de acuerdo a lo que halla
"""

'''
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

'''
"""###funciones estadisticas"""
'''

def mad_2(datos_4):
    """
    Calcula la desviación absoluta media (MAD) de una serie de datos.
    """
    mean = datos_4.mean()
    deviations = abs(datos_4 - mean)
    return deviations.mean()


def var_test(x, y, ratio=1, alternative="two-sided", conf_level=0.95):
    # Eliminar valores faltantes
    x = np.array(x)[np.isfinite(x)]
    y = np.array(y)[np.isfinite(y)]

    # Calcular grados de libertad y varianzas
    df_x = len(x) - 1
    df_y = len(y) - 1
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    # Calcular estadístico F y p-value
    estimate = var_x / var_y
    statistic = estimate / ratio
    pval = f.sf(statistic, df_x, df_y)
    if alternative == "two-sided":
        pval = 2 * min(pval, 1 - pval)
        beta = (1 - conf_level) / 2
        cint = [estimate / f.ppf(1 - beta, df_x, df_y), estimate / f.ppf(beta, df_x, df_y)]
    elif alternative == "greater":
        pval = 1 - pval
        cint = [estimate / f.ppf(conf_level, df_x, df_y), np.inf]
    else:
        cint = [0, estimate / f.ppf(1 - conf_level, df_x, df_y)]

    # Crear objeto htest y devolver resultado
    htest = {
        "statistic": statistic,
        "parameter": {"num df": df_x, "denom df": df_y},
        "p.value": pval,
        "conf.int": cint,
        "estimate": estimate,
        "null.value": ratio,
        "alternative": alternative,
        "method": "F test to compare two variances",
        "data.name": f"{x} and {y}"
    }

    return htest['p.value']


def ttest_pvalue_scipy(d1, d2, equal_vari):
    """Recibe dos series de datos numpy y un booleano de si las varianzas son iguales.
     Devuelve el resultado del t-test de scipy.
     """
    a = ttest_ind(d1, d2, equal_var=equal_vari)
    return a[1]


def inv_t_test_b(alfa, free_deg):
  """recibe un valor flotante (alfa) y un valor entero (free_deg) y devuelve el t test inverso de forma b"""
  return stats.t.ppf((1-(1-alfa)/2), free_deg)


def inv_t_test_a(alfa, free_deg):
  """recibe un valor flotante (alfa) y un valor entero (free_deg) y devuelve el t test inverso de forma a"""
  return stats.t.ppf((1-alfa/2), free_deg)


def promedio_serie(s1, prueba):
  if prueba:
    return s1.mean()
  return -1


def varianza_serie(s1, prueba, magnitud, alt):
  if prueba:
    if alt:
      return s1.var()
    if magnitud:
      return s1.var()
    if mad_2(s1) > 1:
      return s1.var()
  if alt:
    return -2
  return -1


def prueba_f_serie(s1, s2, var1, var2):  # -1 ="no calcular" -2="revisar"
  if var1 == -1:
    return -1
  if (var2 == 0) or (var2 == -2):
    return -2

  return var_test(s1, s2)


def lib_grad(s1, s2, var1, var2, pruebaf):  # -1 = "no calcular " -2 = "revisar"
  if pruebaf == -2:
    return -2
  if (var1 == -1) or (var2 == -2):
    return -1
  if s1.size+s2.size-2 < 0:
    return -1
  return s1.size+s2.size-2


def prueba_t_serie(s1, s2, pruebaf, grados_lib):  # -2000000 = "revisar" -1000000 = "no calcular"
  if grados_lib == -2:
    return -2000000
  if grados_lib == -1:
    return -1000000
  if pruebaf < 0.05:
    equal_var = False
  else:
    equal_var = True
  # s1=
  return ttest_pvalue_scipy(s1, s2, equal_var)


def anti_t_student(t_st, glib, dug):
  if t_st == -2000000:
    return t_st
  if glib == -1:
    return -1000000
  if dug:
    return inv_t_test_a(t_st, glib)
  return inv_t_test_b(t_st, glib)

'''
'''
def prueba_si_cumple(sb, sa, mes, duracion, magintud):
  prub = Prueba_porc(sb, mes, True, duracion)
  prua = Prueba_porc(sa, mes, False, duracion)
  mean_base = promedio_serie(sb, prub)
  mean_alt = promedio_serie(sa, prua)
  var_base = varianza_serie(sb, prub, magintud, False)
  var_alt = varianza_serie(sa, prua, magintud, True)
  F1 = prueba_f_serie(sb, sa, var_base, var_alt)
  lib1 = lib_grad(sb, sa, var_base, var_alt, F1)
  s1 = sb
  s2 = sa
  Tst = prueba_t_serie(s1, s2, F1, lib1)
  anti_tst = anti_t_student(Tst, lib1, True)
  valor_confianza = anti_t_student(0.95, lib1, False)
  if anti_tst == -1000000:
    return True
  if anti_tst == -2000000:
    return False
  return abs(anti_tst) < abs(valor_confianza)


def cumple(dfb, dfa, mes):  # dfb =ref dfa = alterada
  sb = dfb['Magnitud'][dfb['mes'] == mes]
  sa = dfa['Magnitud'][dfa['mes'] == mes]
  boola = prueba_si_cumple(sb, sa, mes, False, True)
  sb = dfb['Duracion'][dfb['mes'] == mes]
  sa = dfa['Duracion'][dfa['mes'] == mes]
  boolb = prueba_si_cumple(sb, sa, mes, True, False)
  sb = dfb['Intensidad'][dfb['mes'] == mes]
  sa = dfa['Intensidad'][dfa['mes'] == mes]
  boolc = prueba_si_cumple(sb, sa, mes, False, False)
  return (boola and boolb) & boolc
# s1 serie de eventos, mes, mes, es_ref <- ref o no, es_duracion <- duracion o no


def Prueba_porc(s1, mes, es_ref, es_duracion):
  num_anos = 0
  maxs = data.index.year.max()
  mins = data.index.year.min()
  num_anos = maxs-mins+1
  if es_ref:  # si es ref
    porc_aprov = df2.iat[mes-1, 3]
    tam_ref = s1.size
    prueba_est = tam_ref/num_anos
    if prueba_est > 0.05:  # si la prueba est es mayor al 5%
      if tam_ref > 1:  # si tiene mas de un dato
        return True
      return False
    return False
  if not es_duracion:
    if s1.size > 1:
      return True
    return False
  if (s1.size > 1) and (not (s1.mean() == 1)):
    return True
  return False

'''
"""###funciones del algoritmo

####minimos
"""


def listas(dataframe):
    # Obtener las columnas '%_aprov' y 'Q_aprov' como listas
    lista_aprov = dataframe['%_aprov'].tolist()
    lista_q_aprov = dataframe['Q_aprov'].tolist()
    return lista_aprov, lista_q_aprov


def listas_ambiental(cuenca_base, cuenca_comparacion=None, areas=None):
  a = prin_func(cuenca_base, cuenca_comparacion, areas)
  b = listas(a)
  return b


def caudal_ambiental(cuenca_base, cuenca_comparacion=None, areas=None):
  a = prin_func(cuenca_base, cuenca_comparacion, areas)
  return a

'''
def minimos(umbrales: object = None) -> object:
  """
    calcula los umbrales QTR 15, QTQ, QB y Q10, además del minimo revisado por año.

    Parámetros
    ----------
    data: pandas.DataFrame
        El DataFrame principal a procesar.

    Returns
    -------
    DataFrame ( pandas.DataFrame)
      el dataframe con los minimos y promedios revisados
      los umbrales se guardan en las variables globales
    """
  global QTR_15
  global QTQ
  global QB
  global Q10
  global data
  if umbrales is None:
    u_qb = 2.33
    u_qtq = 2
  else:
    u_qb = umbrales[0]
    u_qtq = umbrales[1]
  # df es un dataframe temporar que se sobreescribe eventualmente
  df = pd.DataFrame()
  # set columns
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None)
  # se crea un dataframe en donde se calculan los minimos máximos y promedio por mes y año
  for i in range(data.index.min().year, data.index.max().year+1):
      for j in range(1, 13):
          # filtro los datos por año y mes para hacer el analisis
          data_filter = data[data.index.year == i][data[data.index.year == i].index.month == j]
          row = [str(i)+str(-j), data_filter['cuenca-base'].min(), data_filter['cuenca-base'].max(), data_filter['cuenca-base'].mean()]
          df.loc[len(df)] = row
  # data view
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  # se crea listas con n valores, siendo n el número de años en el dataset, guardando el máximo y minimo por año
  mins = []
  maxs = []
  for i in range(df.index.min().year, df.index.max().year+1):
    mins.append(df[df.index.year == i]['Min'].min())
    maxs.append(df[df.index.year == i]['Max'].max())
  mins = np.array(mins)
  maxs = np.array(maxs)
  mean_min = (mins.mean())
  std_min = (np.std(mins, ddof=1))  # desviacion estandar
  coef_variacion_min = std_min/mean_min
  mean_max = (maxs.mean())
  std_max = (np.std(maxs, ddof=1))
  alpha_min = 1.0079*(coef_variacion_min**-1.084)
  a_alpha_min = (-0.0607*(coef_variacion_min**3))+(0.5502*(coef_variacion_min**2))-(0.4937*coef_variacion_min)+1.003
  beta = (a_alpha_min/mean_min)**(-1)
  # umbrales QTQ y QB
  #
  umbral_QTQ = beta*((-np.log(1-(1/u_qtq)))**(1/alpha_min))
  umbral_Q10 = beta*((-np.log(1-(1/10)))**(1/alpha_min))
  #
  #
  df = pd.DataFrame()  # crea un dataframe donde poner los minimos y maximos por mes de toda la serie
  # set columns
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None, Min_rev=None)
  # calculate mim, max, mean, min_rev and mean_rev
  for i in range(data.index.min().year, data.index.max().year+1):
      for j in range(1, 13):
          # filtro los datos por año y mes para hacer el analisis
          data_filter = data[(data.index.year == i) & (data.index.month == j)]
          row = [str(i)+str(-j),
                 data_filter['cuenca-base'].min(),
                 data_filter['cuenca-base'].max(),
                 data_filter['cuenca-base'].mean(),
                 data_filter['cuenca-base'][data['cuenca-base'] > umbral_Q10].min()]
          df.loc[len(df)] = row
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  alpha_max = (np.sqrt(6)*std_max)/np.pi
  u_max = mean_max-(0.5772*alpha_max)
  yt_QB = -np.log(np.log(u_qb/(u_qb-1)))
  yt_Q15 = -np.log(np.log(15/(15-1)))
  # umbrales QB y QTR15
  #
  umbral_QB = u_max+(yt_QB*alpha_max)
  umbral_Q15 = u_max+(yt_Q15*alpha_max)
  #
  ##
  df = pd.DataFrame()
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None, Min_rev=None, Mean_rev=None)
  for i in range(data.index.min().year, data.index.max().year+1):
      for j in range(1, 13):
          # filtro los datos por año y mes para hacer el analisis
          data_filter = data[(data.index.year == i) & (data.index.month == j)]
          row = [str(i)+str(-j),
                 data_filter['cuenca-base'].min(),
                 data_filter['cuenca-base'].max(),
                 data_filter['cuenca-base'].mean(),
                 data_filter['cuenca-base'][data['cuenca-base'] > umbral_Q10].min(),
                 data_filter['cuenca-base'][(data['cuenca-base'] > umbral_Q10) & (data['cuenca-base'] < umbral_Q15)].mean()]
          df.loc[len(df)] = row
  #########
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  QTR_15 = umbral_Q15
  QB = umbral_QB
  QTQ = umbral_QTQ
  Q10 = umbral_Q10
  return df
'''

"""####funcion principal"""


# def prin_func(crud_base, crud_apoyo=None, areas=None, umbrales=None):
#   preparacion_inicial()
#   global QTR_15
#   global data_alter2
#   global QB
#   global QTQ
#   global Q10
#   global df_qtq_ref
#   global df_qtq_alt
#   global df_qb_ref
#   global df_qb_alt
#   global data
#   global df2
#   global primer_dia
#   global dif
#   global final_dia
#   df_funcional = process_df(crud_base, crud_apoyo, areas)
#   df_funcional = df_funcional.rename(columns={'Valor': 'cuenca-base'})
#   data = df_funcional
#   df_rev = umbrales
#   org_df2_1(estado, df_rev)
#   if (umbrales == None):
#     df_rev = buscar_umbrales(umbrales)
#   for i in range(1, 13):
#     org_df2_2(estado,1, i)
#   data = df_funcional.copy()
#   primer_dia = data.index.min()
#   final_dia = data.index.max()
#   segundo_dia = data[data.index > data.index.min()].index.min()
#   dif = segundo_dia-primer_dia
#   nombrar_evento(data, 'cuenca-base')
#   contar_eventos(data, 'event_QTR15', eventos_qtr15, 'event_QTR15', QTR_15)
#   contar_eventos(data, 'event_QB', eventos_qb, 'event_QTR15', QB)
#   contar_eventos(data, 'event_QTQ', eventos_qtq, 'event_Q10', QTQ)
#   contar_eventos(data, 'event_Q10', eventos_q10, 'event_Q10', Q10)
#   df_qtq_ref = df_eventos(df_qtq_ref, eventos_qtq)
#   df_qb_ref = df_eventos(df_qb_ref, eventos_qb)
#   formar_alter()
#   org_alt()
#   '''for j in range (0,1):
#     for i in range(12,0,-1):
#       print(i)
#       calibrar_mes(i)'''
#   for i in range(1, 13):
#       print(i)
#       calibrar_mes(i)
#   return df2, data_alter2


"""####otra cosa"""


# def org_alt():
#   global df_qtq_alt
#   global df_qb_alt
#   global eventos_rev_qtr15
#   global eventos_rev_qb
#   global eventos_rev_qtq
#   global eventos_rev_q10
#   global QTR_15
#   global Q10
#   global QB
#   global QTQ
#   global data_alter2
#   df_qtq_alt = pd.DataFrame()
#   df_qb_alt = pd.DataFrame()
#   df_qtq_alt.assign(mes=None, Magnitud=None, Duracion=None, Intensidad=None)
#   df_qb_alt.assign(mes=None, Magnitud=None, Duracion=None, Intensidad=None)
#   eventos_rev_qtr15.clear()
#   eventos_rev_qb.clear()
#   eventos_rev_qtq.clear()
#   eventos_rev_q10.clear()
#   nombrar_evento(data_alter2, 'cuenca-base')
#   contar_eventos(data_alter2, 'event_QTR15', eventos_rev_qtr15, 'event_QTR15', QTR_15)
#   contar_eventos(data_alter2, 'event_QB', eventos_rev_qb, 'event_QTR15', QB)
#   contar_eventos(data_alter2, 'event_QTQ', eventos_rev_qtq, 'event_Q10', QTQ)
#   contar_eventos(data_alter2, 'event_Q10', eventos_rev_q10, 'event_Q10', Q10)
#   df_qtq_alt = df_eventos(df_qtq_alt, eventos_rev_qtq)
#   df_qb_alt = df_eventos(df_qb_alt, eventos_rev_qb)
#
#
# def formar_alter():
#   global data_alter
#   global data_alter2
#   global df2
#   global data
#   data_alter = data[['cuenca-base']].copy()
#   data_alter['Aprov_teo'] = np.NaN
#   data_alter['check_3'] = np.NaN
#   data_alter['check_2'] = np.NaN
#   data_alter['Q_ajustado'] = np.NaN
#   data_alter['Q_ambiental'] = np.NaN
#   data_alter['Q_aprov_real'] = np.NaN
#   for i in range(1, 13):
#     a_bool = data_alter.index.month == i  # serie booleana si pertenece al mes correcto
#     data_alter['Aprov_teo'].loc[a_bool] = df2.iat[i-1, 4]  # aprovechamiento teorico = aprovechamiento teorico de ese mes ########
#     data_alter['check_3'].loc[a_bool] = df2.iat[i-1, 1]  # check 3 igual al minimo revisado ########
#     b_bool = data_alter['cuenca-base'] > data_alter['Aprov_teo']  # s.b. si el Q obs es > que el aprov teorico
#     c_bool = a_bool & b_bool  # s.b. si pertenece al mes correcto y es Q obs es > que el aprov teorico
#     d_bool = a_bool & (~b_bool)  # s.b. si pertenece al mes correcto y es Q obs es < que el aprov teorico
#     e_bool = data_alter['cuenca-base'] < data_alter['check_3']  # s.b. si el Q obs es < al minimo revisado
#     f_bool = d_bool & e_bool  # Si mes correcto & Q obs < aprov teo & Q obs < check 3
#     g_bool = d_bool & (~e_bool)  # s.b. si pertenece al mes correcto y el Q obs es > al minimo revisado & Q obs es < aprov teorico
#     data_alter['Q_ajustado'].loc[c_bool] = df2.iat[i-1, 4]  # Q obs > aprov teorico = aprov teorico #########
#     data_alter['Q_ajustado'].loc[f_bool] = 0  # Q obs < aprov teorico = 0 #########
#     data_alter['Q_ajustado'].loc[g_bool] = data_alter['cuenca-base'][g_bool]-data_alter['check_3'][g_bool]  # #######
#     data_alter['check_2'] = data_alter['cuenca-base']-data_alter['Q_ajustado']  # check 2 = Q obs - Q ajustado
#   a_bool = data_alter['cuenca-base'] < data_alter['check_3']  # los caudales observados que sean menores al minimo revisado
#   b_bool = data_alter['check_2'] < data_alter['check_3']  # Q ajustado es menor al minimo revisado
#   data_alter['Q_ambiental'].loc[a_bool] = data_alter['cuenca-base']  # los queQobs sean < al minrev no se le hace nada
#   data_alter['Q_ambiental'].loc[(~a_bool) & b_bool] = data_alter['check_3']  # si el Qobs - Qaprov =min rev
#   data_alter['Q_ambiental'].loc[(~a_bool) & (~b_bool)] = data_alter['check_2']  # si Qobs> min rev y Qobs-Qaprov >min rev, Qobs-aQaprov
#   data_alter['Q_aprov_real'] = data_alter['cuenca-base']-data_alter['Q_ambiental']  # qobs -qambs
#   data_alter2 = data_alter[['Q_ambiental']].copy()
#   data_alter2.rename(columns={'Q_ambiental': 'cuenca-base'}, inplace=True)
#
# '''
# def calibrar_mes(mess):
#   inferior = 0
#   superior = 1
#   org_df2_2(1, mess)
#   formar_alter()
#   org_alt()
#   i = mess
#   if cumple(df_qtq_ref, df_qtq_alt, i) and cumple(df_qb_ref, df_qb_alt, i):
#     return
#   conteo = 0
#   mayor = 0
#   while True:
#     conteo += 1
#     lim_prov = (inferior+superior)/2
#     org_df2_2(lim_prov, mess)
#     formar_alter()
#     org_alt()
#     a = cumple(df_qtq_ref, df_qtq_alt, i) and cumple(df_qb_ref, df_qb_alt, i)
#     # respaldo_print(a,"a",lim_prov)
#     if a:
#       if lim_prov > mayor:
#         mayor = lim_prov
#       inferior = lim_prov
#       if abs(inferior-superior) < 0.01:
#         # respaldo_print(cumple(df_qtq_ref,df_qtq_alt,i), cumple(df_qb_ref,df_qb_alt,i),lim_prov,i)
#         return
#     else:
#       superior = lim_prov
#     if conteo > 2000:
#       org_df2_2(mayor, mess)
#       formar_alter()
#       org_alt()
#       return
#
# '''
# def org_df2_2(aprov, mes):  # porcentaje de aprovechamiento, mes, cambia el porcentaje de aprovechamiento de un mes escogido por el %escogido
#   """recibe un valor porcentual (aprov) y un valor entero (mes) y cambia el valor de aprovechamiento de ese mes ademas de cambiar el caudal aprovechado para ese mes"""
#   global df2
#   df2.loc[df2.index.month == mes, '%_aprov'] = aprov
#   df2['Q_aprov'] = df2['Mean_rev']*df2['%_aprov']
#
#
# def org_df2_1(df):
#   """coge los valores de data(dataframe global) y los usa para crear el df2(df global) con 12 filas y 3 columnas (promedio, min_rev y mean_rev) por mes."""
#   global df2
#   global data
#   df2 = pd.DataFrame()
#   # set columns
#   df2 = df2.assign(Fecha=None, Mean=None, Min_rev=None, Mean_rev=None)
#   # calculate mim, max, mean, min_rev and mean_rev
#   i = data.index.min().year
#   for j in range(1, 13):
#       # filtro los datos por año y mes para hacer el analisis
#       data_filter = df[df.index.month == j]
#       row = [str(i)+str(-j), data_filter['Mean'].mean(), data_filter['Min_rev'].min(), data_filter['Mean_rev'].mean()]
#       df2.loc[len(df2)] = row
#   # data view
#   df2['Fecha'] = pd.to_datetime(df2['Fecha'])
#   df2 = df2.set_index('Fecha')

'''
def DF_eventos(df_objetivo, lista_eventos):
  """función toma un dataframe donde guardar las cosas y una lista con todos los eventos de la categoria que se le den y lo que hace es crear un dataframe con los datos de mes magnitud, etc. para luego poder acceder a ellos más fácil"""
  # set columns
  df_objetivo = df_objetivo.assign(mes=None, Magnitud=None, Intensidad=None, Duracion=None)
  # calculate mim, max, mean, min_rev and mean_rev
  for j in range(len(lista_eventos)):
    # filtro los datos por año y mes para hacer el analisis
    # data_filter=df[df.index.month==j]
    row = [lista_eventos[j].mes, lista_eventos[j].magnitud, lista_eventos[j].intensidad, lista_eventos[j].duracion]
    df_objetivo.loc[len(df_objetivo)] = row
  # data view
  return df_objetivo
'''
'''
def nombrar_evento(dfh, str4_h='cuenca-base'):
  """funcion que coge un dataframe con datos de caudal y determina que días hubo eventos poniendo 1 o 0 donde sea necesario
  Parameters
  -----------
  dfh: dataframe al que se le van a encontrar los eventos
  str4_h: string con el valor 'cuenca-base' se necesita, aunque no se porque
  Return
  ----------
  nada
  """
  global QTR_15
  global QB
  global QTQ
  global Q10
  umbral_Q15 = QTR_15
  umbral_QB = QB
  umbral_Q10 = Q10
  umbral_QTQ = QTQ
  dfh['event_QTR15'] = np.nan
  dfh['event_QB'] = np.nan
  dfh['event_QTQ'] = np.nan
  dfh['event_Q10'] = np.nan
  dfh.loc[dfh[str4_h] > umbral_Q15, 'event_QTR15'] = 1
  dfh.loc[dfh[str4_h] <= umbral_Q15, 'event_QTR15'] = 0
  # #
  dfh.loc[(dfh[str4_h] <= umbral_Q15) & (dfh[str4_h] > umbral_QB), 'event_QB'] = 1
  dfh.loc[~((dfh[str4_h] <= umbral_Q15) & (dfh[str4_h] > umbral_QB)), 'event_QB'] = 0

  dfh.loc[(dfh['cuenca-base'] > umbral_Q10) & (dfh['cuenca-base'] <= umbral_QTQ), 'event_QTQ'] = 1
  dfh.loc[~((dfh[str4_h] > umbral_Q10) & (dfh[str4_h] <= umbral_QTQ)), 'event_QTQ'] = 0
  # #

  dfh.loc[dfh['cuenca-base'] <= umbral_Q10, 'event_Q10'] = 1  # #aqui
  dfh.loc[dfh[str4_h] > umbral_Q10, 'event_Q10'] = 0
  # #data.fillna(0,inplace=True)
'''

# def crear_evento(df21):
#   global primer_dia
#   global dif
#   global final_dia
#   global lista_prueba
#   lista_po = []
#   # contar_eventos(data,'event_QTQ',eventos_qtq,'event_Q10',QTQ)
#   meses_1 = df21.index.month.min()
#   meses_2 = df21.index.month.max()
#   inicio = primer_dia
#   if meses_1 == meses_2:
#     lista_po.append(df21)
#   else:
#     for i in range(meses_1, meses_2+1):
#       df_provisional = df21[df21.index.month == i]
#       num_filas = df_provisional.shape[0]
#       if num_filas == 0:
#         continue
#       lista_po.append(df_provisional)
#   lista_prueba = lista_po
#   return lista_po


# def contar_eventos(df21, eventosh, eventos_umbral, eventosi, umbral_caudal):
#   """recibe un dataframe (df21) con los 1's ya puestos, un string (eventosh) con el tipo de evento, una lista (eventos_umbral), otros string (eventosi) que es el evento umbral mayor, en caso de ser un evento medio y un flotante (umbral_caudal)con el valor del umbral, rellena las listas de eventos con bloques de eventos donde hayan 1's juntos, para formar eventos completos, en el caso de evento QB y QTQ también verifica que no hallan eventos QTR15 o Q10 inmediatamente anterior o siguiente al bloque"""
#   global df2
#   global primer_dia
#   global dif
#   global final_dia
#   inicio = primer_dia
#   # inicio=data[data['event_QTR15']==1].index.min()
#   final = data[(df21.index > inicio) & (df21[eventosh] == 0)].index.min()
#   g = 0
#   while g < 700:
#     # forma los bloques encontrando el primer 1 del df y el primer 0 después de ese 1 y haciendo un bloque de datos entre los dos y luego volviendolo a hacer
#     # pero empeando desde el 0 anterior, hasta que acabe con el df o con el g<700 puesto por si hay algun problema no se quede pensando infinitamente
#     inicio = df21[(df21[eventosh] == 1) & (df21.index >= inicio)].index.min()
#     final = df21[(df21[eventosh] == 0) & (df21.index >= inicio)].index.min()
#     if df21[((df21.index >= inicio) & (df21.index <= df21.index.max())) & (df21[eventosh] == 0)].size == 0:
#       final = df21.index.max()
#     if (not ((eventosh == 'event_QB') | (eventosh == 'event_QTQ'))) | (not (any(df21[eventosi][(df21.index >= inicio-dif) & (df21.index < final+dif)]))):
#       if df21[(df21.index >= inicio) & (df21.index < final)].size == 0:
#         pass
#       else:
#         df_envio = df21[(df21.index >= inicio) & (df21.index < final)].copy()
#         a = crear_evento(df_envio)
#         for i in range(len(a)):
#           eventos_umbral.append(Evento_completo(a[i][(a[i].index >= inicio) & (a[i].index < final)].copy(), umbral_caudal, eventosh))
#     inicio = final
#     if not (any(df21[eventosh][df21.index >= final])):
#       break
#     # if (df21[(df21.index>=inicio)&(df21.index<final)].empty):
#     #  break
#     g = g+1
#   inicio = primer_dia
#   final = final_dia
#   # eventos_rev_qtq[-1].df1.size


"""##variables globales

###umbrales
"""
'''
QTR_15 = -1
QTQ = -1
QB = -1
Q10 = -1
'''
print(QTR_15)
print(QTQ)
print(QB)
print(Q10)

"""###Provisionales"""

primer_dia = 1.0
dif = 1.0
final_dia = 1.0

# DF_minimos_mensual = pd.DataFrame()

porcentajes = np.empty(12)

"""###listas de eventos"""

# natural
eventos_qtr15 = []
eventos_qb = []
eventos_qtq = []
eventos_q10 = []

# alterada
eventos_rev_qtr15 = []
eventos_rev_qb = []
eventos_rev_qtq = []
eventos_rev_q10 = []

"""###dataframes utiles"""

# data es el dataframe donde va a estar
data = pd.DataFrame()
data_alter = pd.DataFrame()
data_alter2 = pd.DataFrame()
# df2 dataframe con el resultado
df2 = pd.DataFrame()
df_prueba = 2

# dataframes con los eventos qtq y qb
df_qtq_ref = pd.DataFrame()
df_qtq_alt = pd.DataFrame()
df_qb_ref = pd.DataFrame()
df_qb_alt = pd.DataFrame()

# esto es provisional
RH = True
RQ = True
# eleccion metodología
metod = "anla"

def preparacion_inicial():
    # global QTR_15, QTQ, QB, Q10
    global primer_dia, dif, final_dia
    global porcentajes  # ,DF_minimos_mensual
    global eventos_qtr15, eventos_qb, eventos_qtq, eventos_q10
    global eventos_rev_qtr15, eventos_rev_qb, eventos_rev_qtq, eventos_rev_q10
    global data, data_alter, data_alter2, df2, data, df_prueba
    global df_qtq_ref, df_qtq_alt, df_qb_ref, df_qb_alt
    global RH, RQ
    global metod
    if metod=="anla":
      pass
    else:
      from estadistica_ideam import *
    # Inicialización de variables globales
    '''QTR_15 = -1
    QTQ = -1
    QB = -1
    Q10 = -1
    '''
    primer_dia = 1.0
    dif = 1.0
    final_dia = 1.0

    # DF_minimos_mensual = pd.DataFrame()

    porcentajes = np.empty(12)

    # Inicialización de listas de eventos
    eventos_qtr15 = []
    eventos_qb = []
    eventos_qtq = []
    eventos_q10 = []

    eventos_rev_qtr15 = []
    eventos_rev_qb = []
    eventos_rev_qtq = []
    eventos_rev_q10 = []

    # Inicialización de dataframes útiles
    data = pd.DataFrame()
    data_alter = pd.DataFrame()
    data_alter2 = pd.DataFrame()
    df2 = pd.DataFrame()
    data = pd.DataFrame()
    df_prueba = 2

    df_qtq_ref = pd.DataFrame()
    df_qtq_alt = pd.DataFrame()
    df_qb_ref = pd.DataFrame()
    df_qb_alt = pd.DataFrame()

    RH = True
    RQ = True