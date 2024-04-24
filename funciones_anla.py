import pandas as pd
import estado_algoritmo
import numpy as np
from scipy.stats import norm, lognorm, gumbel_r, pearson3,  weibull_min


def calc_caud_retor(df: pd.DataFrame, tiempo_ret: int) -> float:
  tamanio_array: int = len(df['cuenca-base'])
  vesn: np.ndarray = np.empty(tamanio_array, dtype=np.float32)  # [0] * len(df['cuenca-base'])
  vesln: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesp: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesg: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesw: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
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
  tiempo_retorno = tiempo_ret
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


def calcular_7q10(df_completo: pd.DataFrame) -> list:
  '''
  Calcula los 7q10
  @param df_completo:
  @return: lista 12 7q10 por mes
  '''

  promedio_7_dias: pd.DataFrame = df_completo.rolling(7).mean()
  df: pd.DataFrame = promedio_7_dias.groupby(promedio_7_dias.index.year)['cuenca-base'].min()
  meses = [pd.DataFrame()] * 12
  for i in range(1, 13):
    meses[i - 1] = promedio_7_dias[promedio_7_dias.index.month == i]
  q_710s: list = [0]*12
  for i, x in enumerate(meses):
    q_710s[i] = calc_caud_retor(x, 10)
  return q_710s


def calcular_q95(estado: estado_algoritmo.EstadoAnla):
  meses: list = [pd.DataFrame()] * 12
  q95s: list = [0]*12
  for i in range(1, 13):
    meses[i-1] = estado.data[estado.data.index.month == i]
  for i, x in enumerate(meses):
    q95s[i] = calc_caud_retor(x, 95)
  return q95s


def generar_cdc(datos: pd.DataFrame) -> pd.DataFrame:
  ordenados_2 = datos.sort_values(by='cuenca-base', ascending=False)
  ordenados_2['cumsum'] = ordenados_2['cuenca-base'].cumsum() / sum(ordenados_2['cuenca-base'])
  return ordenados_2


def calc_normal(estado: estado_algoritmo.EstadoAnla) -> None:
  estado.df_cdc_normal = generar_cdc(estado.data)
  estado.cdc_normales = np.interp([0.70, 0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995], estado.df_cdc_normal['cumsum'],
                                  estado.df_cdc_alterada['cuenca-base'])
  estado.caud_return_normal = caud_retorn_anla(estado.data, estado.anios_retorn)


def calc_alterado(estado: estado_algoritmo.EstadoAnla) -> None:
  estado.data_alterado = estado.data.copy()
  for index, row in estado.data_alterado.iterrows():
      month: int = row.name.month - 1
      estado.data_alterado.at[index, 'cuenca-base'] = min(row['cuenca-base'], estado.propuesta_caudal[month])
  estado.df_cdc_alterada = generar_cdc(estado.data_alterado)
  estado.cdc_alterados = np.interp([0.70, 0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995], estado.df_cdc_alterada['cumsum'],
                                   estado.df_cdc_alterada['cuenca-base'])
  estado.caud_return_alterado = caud_retorn_anla(estado.data_alter, estado.anios_retorn)

# def cdc_valor(cdc: pd.DataFrame, valor: float) -> float:
#   p1 = cdc.loc[cdc[cdc['cumsum']>valor].index[0]]
#   p2 = cdc.loc[cdc[cdc['cumsum']<valor].index[-1]]
#   return interpolacion(p1['cuenca-base'],p2['cuenca-base'],p1['cumsum'],p2['cumsum'],valor)


def caud_retorn_anla(df: pd.DataFrame, anios: list) -> list:
  resultado: list = [0]*len(anios)
  for i in anios:
    resultado[i] = calc_caud_retor(df, anios[i])
  return resultado


def prin_func(estado: estado_algoritmo.EstadoAnla) -> pd.DataFrame:
  estado.q7_10 = calcular_7q10(estado.data)
  estado.q95 = calcular_q95(estado)
  estado.propuesta_caudal = np.minimum(estado.q7_10, estado.q95)
  calc_normal(estado)
  calc_alterado(estado)
  # todo lo de R
  # todo iteracion
  return pd.DataFrame()
