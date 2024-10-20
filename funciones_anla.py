from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, gumbel_r, pearson3, weibull_min

import estado_algoritmo
from IhaEstado import IhaEstado


def calc_caud_retor(df: pd.DataFrame, tiempo_ret: int) -> float:
  tamanio_array: int = len(df['Valor'])
  vesn: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesln: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesp: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesg: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  vesw: np.ndarray = np.empty(tamanio_array, dtype=np.float32)
  # iniciacion valores
  mun, stdn = norm.fit(df['Valor'])
  pln = lognorm.fit(df['Valor'])
  mug, beta = gumbel_r.fit(df['Valor'])
  a1, m, s = pearson3.fit(df['Valor'])
  shape, loc, scale = weibull_min.fit(df['Valor'])
  for i, x in enumerate(df['Valor']):
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
  gumball_corr = np.corrcoef(df['Valor'], vesg)[0, 1]
  normal_corr = np.corrcoef(df['Valor'], vesn)[0, 1]
  pearson_corr = np.corrcoef(df['Valor'], vesp)[0, 1]
  lognorm_corr = np.corrcoef(df['Valor'], vesln)[0, 1]
  weibull_corr = np.corrcoef(df['Valor'], vesw)[0, 1]
  # Sacar el mayor de las correlaciones y sacar en 10 años cuanto es el 7Q10
  tiempo_retorno = tiempo_ret
  prob_ret = 1 - (1 / tiempo_retorno)
  ajuste_seleccionado = mejor_ajuste(np.abs(normal_corr), np.abs(lognorm_corr),
                                     np.abs(gumball_corr), np.abs(pearson_corr),
                                     np.abs(weibull_corr))
  if ajuste_seleccionado == 1:
    return norm.ppf(prob_ret, loc=mun, scale=stdn)
  elif ajuste_seleccionado == 2:
    return lognorm.ppf(prob_ret, pln[2], pln[2], pln[0])
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
  """
  Calcula los 7q10
  @param df_completo:
  @return: lista 12 7q10 por mes
  """

  promedio_7_dias: pd.DataFrame = df_completo.rolling(7).mean()
  df: pd.DataFrame = promedio_7_dias.groupby(promedio_7_dias.index.year)['Valor'].min()
  meses = [pd.DataFrame()] * 12
  for i in range(1, 13):
    meses[i - 1] = promedio_7_dias[promedio_7_dias.index.month == i]
  q_710s: list = [0] * 12
  for i, x in enumerate(meses):
    q_710s[i] = calc_caud_retor(x, 10)
  return q_710s


def calcular_q95(estado: estado_algoritmo.EstadoAnla):
  meses: list = [pd.DataFrame()] * 12
  q95s: list = [0] * 12
  for i in range(1, 13):
    meses[i - 1] = estado.data[estado.data.index.month == i]
  for i, x in enumerate(meses):
    q95s[i] = calc_caud_retor(x, 95)
  return q95s


def generar_cdc(datos: pd.DataFrame) -> pd.DataFrame:
  ordenados_2 = datos.sort_values(by='Valor', ascending=False)
  ordenados_2['cumsum'] = ordenados_2['Valor'].cumsum() / sum(ordenados_2['Valor'])
  return ordenados_2


# def calc_normal(estado: estado_algoritmo.EstadoAnla) -> None:
#   estado.df_cdc_normal = generar_cdc(estado.data)
#   estado.cdc_normales = np.interp([0.70, 0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995], estado.df_cdc_normal['cumsum'],
#                                   estado.df_cdc_alterada['Valor'])
#   estado.caud_return_normal = caud_retorn_anla(estado.data, estado.anios_retorn)


def calc_resultados(datos: pd.DataFrame) -> estado_algoritmo.ResultadosAnla:
  cdc: pd.DataFrame = generar_cdc(datos)
  cdc_anios: np.ndarray = np.interp(estado_algoritmo.EstadoAnla.cdc_umbrales, cdc['cumsum'],
                                    cdc['Valor'])
  anios_retorno: list[float] = caud_retorn_anla(datos, estado_algoritmo.EstadoAnla.anios_retorn)
  resultados_iha: IhaEstado = IhaEstado(datos)
  resultados_iha.calcular_iha()
  resultados: estado_algoritmo.ResultadosAnla = estado_algoritmo.ResultadosAnla(cdc=cdc, cdc_anios=cdc_anios,
                                                                                caud_return=anios_retorno,
                                                                                iah_result=resultados_iha)
  return resultados


def recortar_caudal(serie: pd.DataFrame, caudales: Sequence[float]) -> pd.DataFrame:
  data_alterado = serie.copy()
  for index, row in data_alterado.iterrows():
    month: int = row.name.month - 1
    data_alterado.at[index, 'Valor'] = min(row['Valor'], caudales[month])
  return data_alterado


def calc_alterado(data: pd.DataFrame, caudales) -> (pd.DataFrame, estado_algoritmo.ResultadosAnla):
  data_alterado = recortar_caudal(data, caudales)
  calc_resultados(data_alterado)
  return data_alterado, estado_algoritmo.ResultadosAnla


def caud_retorn_anla(df: pd.DataFrame, anios: list) -> list[float]:
  resultado: list = [0] * len(anios)
  for i in anios:
    resultado[i] = calc_caud_retor(df, anios[i])
  return resultado


def general_month_mean(data: pd.DataFrame) -> pd.DataFrame:
  df = pd.DataFrame()
  df = df.assign(Fecha=None, Mean=None)
  grupo = df.groupby([data.index.year, data.index.month])
  for group, data_filter in grupo:
    year, month = group
    mean_value = data_filter['Valor'].mean()
    row = [f"{year}-{month}", mean_value]
    df.loc[len(df)] = row
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  return df


# @dataclass
# class ResultadosAnla:
#   cdc: pd.DataFrame
#   cdc_anios: np.ndarray
#   caud_return: list[float]
#   iah_result: IhaEstado.IhaEstado

def dividir_resultados(resultados_natural: estado_algoritmo.ResultadosAnla,
                       resultados_otro: estado_algoritmo.ResultadosAnla) -> list:
  resultados_cdc: list[float] = []
  resultados_retorno: list[float] = []
  for a, b in zip(resultados_natural.cdc_anios, resultados_otro.cdc_anios):
    resultados_cdc.append(b / a)
    pass
  for a, b in zip(resultados_natural.caud_return, resultados_otro.caud_return):
    resultados_retorno.append(b / a)
  resultados_iha: list[float] = resultados_natural.iah_result - resultados_otro.iah_result
  return list[resultados_cdc, resultados_retorno, resultados_iha]
  # < 0.5 cdc con y sin proyecto
  # > 0.6 periodos de retorno
  # umbrales es media mas o menos desviación estandar caudal natural
  pass


def prin_func(estado: estado_algoritmo.EstadoAnla) -> pd.DataFrame:
  # calculo elementos iniciales
  estado.df_month_mean = general_month_mean(estado.data)
  estado.q7_10 = calcular_7q10(estado.data)
  estado.q95 = calcular_q95(estado)
  estado.propuesta_inicial_ref = np.minimum(estado.q7_10, estado.q95)
  # calculo estado normal
  estado.resultados_ori = calc_resultados(estado.data)
  # calculo estado referencia
  estado.data_ref, estado.resultados_ref = calc_alterado(estado.data, estado.propuesta_inicial_ref)
  # calibracion estado objetivo
  for i in range(2000):
    # cambiar los caudales, puede ser algoritmos geneticos o minimize o algo así
    # estado.caud_final = alguna cosa aquí
    estado.data_alter, estado.resultados_alterada = calc_alterado(estado.data, estado.caud_final)
    resultados_prov = comparar_resultados(estado.resultados_ori, estado.resultados_alterada)
    # comparar los resultados
    # eso es complicado
  return pd.DataFrame()
def comparar_resultados(a, b):
  # todo
  return a