import pandas as pd
import estado_algoritmo
import numpy as np
from scipy.stats import norm, lognorm, gumbel_r, pearson3,  weibull_min


def calcular_7q10(df_completo: pd.DataFrame) -> list:
  '''
  Calcula los 7q10
  @param df_completo:
  @return: lista 12 7q10 por mes
  '''
  def unico_7q10(df: pd.DataFrame) -> float:
    vesn: list = [0] * len(df['cuenca-base'])
    vesln: list = [0] * len(df['cuenca-base'])
    vesp: list = [0] * len(df['cuenca-base'])
    vesg: list = [0] * len(df['cuenca-base'])
    vesw: list = [0] * len(df['cuenca-base'])
    # iniciacion valores
    mun, stdn = norm.fit(df['cuenca-base'])
    pln = lognorm.fit(df['cuenca-base'])
    mug, beta = gumbel_r.fit(df['cuenca-base'])
    a1, m, s = pearson3.fit(df['cuenca-base'])
    shape, loc, scale = weibull_min.fit(df['cuenca-base'])
    for i, x in enumerate(df['cuenca-base']):
      n = normal_distr(x, mun, stdn)
      vesn[i] = n
      n = lognormal_distr(x, pln[0], pln[2])
      vesln[i] = n
      n = gumball_distr(x, mug, beta)
      vesg[i] = n
      n = weibull_min.pdf(x, shape, loc, scale)
      vesw[i] = n
      n = pearson3.pdf(x, a1, m, s)
      vesp[i] = n
    gumball_corr = np.corrcoef(df['cuenca-base'], vesg)[0, 1]
    normal_corr = np.corrcoef(df['cuenca-base'], vesn)[0, 1]
    pearson_corr = np.corrcoef(df['cuenca-base'], vesp)[0, 1]
    lognorm_corr = np.corrcoef(df['cuenca-base'], vesln)[0, 1]
    weibull_corr = np.corrcoef(df['cuenca-base'], vesw)[0, 1]
    # Sacar el mayor de las correlaciones y sacar en 10 años cuanto es el 7Q10
    tiempo_retorno = 10
    prob_ret = 1 - (1 / tiempo_retorno)
    ajuste_seleccionado = mejor_ajuste(np.abs(normal_corr), np.abs(lognorm_corr),
                                       np.abs(gumball_corr), np.abs(pearson_corr),
                                       np.abs(weibull_corr))

    if ajuste_seleccionado == 1:
      return norm.ppf(prob_ret, loc=mun, scale=stdn)  # 84.08149197181189
    elif ajuste_seleccionado == 2:
      return lognorm.ppf(prob_ret, pln[2], pln[2], pln[0])  # 84.08149197181189
    elif ajuste_seleccionado == 3:
      return gumbel_r.ppf(prob_ret, loc=mug, scale=beta)
    elif ajuste_seleccionado == 4:
      return pearson3.ppf(prob_ret, a1, loc=m, scale=s)
    elif ajuste_seleccionado == 5:
      return weibull_min.ppf(prob_ret, shape, loc, scale)
    else:
      return -1.0
    pass
  def mejor_ajuste(a: float, b: float, c: float, d: float, e: float) -> int:
    maximo: float = max(a, b, c, d, e)  # Encuentra el máximo entre los tres valores
    if a == maximo:  # Comprueba si a es el máximo
      return 1
    elif b == maximo:  # Comprueba si b es el máximo
      return 2
    elif c == maximo:  # Si no es a ni b, entonces c es el máximo
      return 3
    elif d == maximo:  # Si no es a ni b, entonces c es el máximo
      return 4
    else:
      return 5

  def gumball_distr(x_gum, mu_gum, beta_gum) -> float:
    return (1 / beta_gum) * np.exp(-(x_gum - mu_gum) / beta_gum - np.exp(-(x_gum - mu_gum) / beta_gum))

  def lognormal_distr(x_ln, mu_ln, sigma_ln) -> float:
    return (1 / (x_ln * sigma_ln * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x_ln) - mu_ln) / sigma_ln) ** 2)

  def normal_distr(x_normal, mun_normal, stdn_normal) -> float:
    return (1 / (stdn_normal * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_normal - mun_normal) / stdn_normal) ** 2)
  promedio_7_dias: pd.DataFrame = df_completo.rolling(7).mean()
  df: pd.DataFrame = promedio_7_dias.groupby(promedio_7_dias.index.year)['cuenca-base'].min()
  meses = [pd.DataFrame()] * 12
  for i in range(1, 13):
    meses[i - 1] = promedio_7_dias[promedio_7_dias.index.month == i]
  q_710s: list = [0]*12
  for i, x in enumerate(meses):
    q_710s[i] = unico_7q10(x)
  return q_710s



def calcular_q95(estado: estado_algoritmo.EstadoAnla):
  # todo funcion provisional
  df = pd.DataFrame()
  # set columns
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None)
  data = estado.data
  # se crea un dataframe en donde se calculan los minimos máximos y promedio por mes y año
  for i in range(data.index.min().year, data.index.max().year + 1):
    for j in range(1, 13):
      # filtro los datos por año y mes para hacer el analisis
      data_filter = data[data.index.year == i][data[data.index.year == i].index.month == j]
      row = [str(i) + str(-j), data_filter['cuenca-base'].min(), data_filter['cuenca-base'].max(),
             data_filter['cuenca-base'].mean()]
      df.loc[len(df)] = row
  # data view
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  '''
  se crea listas con n valores, siendo n el número de años en el dataset, guardando el máximo y minimo por año
  '''
  mins = []
  maxs = []
  for i in range(df.index.min().year, df.index.max().year + 1):
    mins.append(df[df.index.year == i]['Min'].min())
    maxs.append(df[df.index.year == i]['Max'].max())
  mins = np.array(mins)
  # maxs = np.array(maxs)
  mean_min = (mins.mean())
  std_min = (np.std(mins, ddof=1))  # desviacion estandar
  coef_variacion_min = std_min / mean_min
  # mean_max = (maxs.mean())
  # std_max = (np.std(maxs, ddof=1))
  alpha_min = 1.0079 * (coef_variacion_min ** -1.084)
  a_alpha_min = (-0.0607 * (coef_variacion_min ** 3)) + (0.5502 * (coef_variacion_min ** 2)) - (
          0.4937 * coef_variacion_min) + 1.003
  beta = (a_alpha_min / mean_min) ** (-1)
  # umbrales QTQ y QB
  #
  # umbral_QTQ = beta * ((-np.log(1 - (1 / u_qtq))) ** (1 / alpha_min))
  estado.q95 = beta * ((-np.log(1 - (1 / 95))) ** (1 / alpha_min))


def prin_func(estado: estado_algoritmo.EstadoAnla) -> pd.DataFrame:
  estado.q7_10 = calcular_7q10(estado.data)
  calcular_q95(estado)
  return pd.DataFrame()
