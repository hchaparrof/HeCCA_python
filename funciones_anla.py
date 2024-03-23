import pandas as pd
import estado_algoritmo
import numpy as np
from scipy.stats import norm, lognorm, gumbel_r, pearson3,  weibull_min


def calcular_q15(df_completo: pd.DataFrame) -> float:
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
  df_minimos_por_ano: pd.DataFrame = promedio_7_dias.groupby(promedio_7_dias.index.year)['cuenca-base'].min()
  mun, stdn = norm.fit(df_minimos_por_ano['cuenca-base'])
  vesn: list = []
  for x in df_minimos_por_ano['cuenca-base']:
    n = normal_distr(x, mun, stdn)
    vesn.append(n)
  corn = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesn)[0, 1]
  pln = lognorm.fit(df_minimos_por_ano['cuenca-base'])
  vesln = []
  for x in df_minimos_por_ano['cuenca-base']:
    n = lognormal_distr(x, pln[0], pln[2])
    vesln.append(n)
  corln = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesln)[0, 1]
  mug, beta = gumbel_r.fit(df_minimos_por_ano['cuenca-base'])
  vesg = []
  for x in df_minimos_por_ano['cuenca-base']:
    n = gumball_distr(x, mug, beta)
    vesg.append(n)
  corg = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesg)[0, 1]
  a1, m, s = pearson3.fit(df_minimos_por_ano['cuenca-base'])
  vesp = []
  for x in df_minimos_por_ano['cuenca-base']:
    n = pearson3.pdf(x, a1, m, s)
    vesp.append(n)
  corp = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesp)[0, 1]
  # Weibull
  shape, loc, scale = weibull_min.fit(df_minimos_por_ano['cuenca-base'])
  vesw = []
  for x in df_minimos_por_ano['cuenca-base']:
    n = weibull_min.pdf(x, shape, loc, scale)
    vesw.append(n)
  corw = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesw)[0, 1]
  # Sacar el mayor de las correlaciones y sacar en 10 años cuanto es el 7Q10
  tiempo_retorno = 10
  pe = 1 - (1 / tiempo_retorno)
  ajuste_seleccionado = mejor_ajuste(np.abs(corn), np.abs(corln), np.abs(corg), np.abs(corp), np.abs(corw))

  if ajuste_seleccionado == 1:
    return norm.ppf(pe, loc=mun, scale=stdn)  # 84.08149197181189
  elif ajuste_seleccionado == 2:
    return lognorm.ppf(pe, pln[2], pln[2], pln[0])  # 84.08149197181189
  elif ajuste_seleccionado == 3:
    return gumbel_r.ppf(pe, loc=mug, scale=beta)
  elif ajuste_seleccionado == 4:
    return pearson3.ppf(pe, a1, loc=m, scale=s)
  elif ajuste_seleccionado == 5:
    return weibull_min.ppf(pe, shape, loc, scale)
  else:
    return -1.0


def prin_func(estado: estado_algoritmo.EstadoAnla) -> pd.DataFrame:
  estado.q7_10 = calcular_q15(estado.data)
  return pd.DataFrame()
