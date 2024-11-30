from collections.abc import Sequence, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, gumbel_r, pearson3, weibull_min, lognorm

import estado_algoritmo
from IhaEstado import IhaEstado


def calc_caud_retor(df: pd.DataFrame, tiempo_ret: int) -> float:
  tamanio_array: int = len(df['Valor'])
  array_valor: np.ndarray = df['Valor'].to_numpy()
  # iniciacion valores
  # df.to_csv('prueba_func.csv')
  mun, stdn = norm.fit(df['Valor'])
  # pln = lognorm.fit(df['Valor'])
  mug, beta = gumbel_r.fit(df['Valor'])
  a1, m, s = pearson3.fit(df['Valor'])
  shape, loc, scale = weibull_min.fit(df['Valor'])
  vesn = normal_distr(array_valor, mun, stdn)
  # vesln = lognormal_distr(array_valor, pln[0], pln[2])
  vesg = gumball_distr(array_valor, mug, beta)
  vesw = weibull_min.pdf(array_valor, shape, loc, scale)
  vesp = pearson3.pdf(array_valor, a1, m, s)
  gumball_corr = np.corrcoef(df['Valor'], vesg)[0, 1]
  normal_corr = np.corrcoef(df['Valor'], vesn)[0, 1]
  pearson_corr = np.corrcoef(df['Valor'], vesp)[0, 1]
  # lognorm_corr = np.corrcoef(df['Valor'], vesln)[0, 1]
  weibull_corr = np.corrcoef(df['Valor'], vesw)[0, 1]

  # Sacar el mayor de las correlaciones y sacar en 10 años cuanto es el 7Q10
  tiempo_retorno = tiempo_ret
  prob_ret = 1 - (1 / tiempo_retorno)
  ajuste_seleccionado = mejor_ajuste(np.abs(normal_corr), 0, # np.abs(lognorm_corr),
                                     np.abs(gumball_corr), np.abs(pearson_corr),
                                     np.abs(weibull_corr))
  if ajuste_seleccionado == 1:
    return norm.ppf(prob_ret, loc=mun, scale=stdn)
  elif ajuste_seleccionado == 2:
    # return lognorm.ppf(prob_ret, pln[2], pln[2], pln[0])
    pass
  elif ajuste_seleccionado == 3:
    return gumbel_r.ppf(prob_ret, loc=mug, scale=beta)
  elif ajuste_seleccionado == 4:
    return pearson3.ppf(prob_ret, a1, loc=m, scale=s)
  elif ajuste_seleccionado == 5:
    return weibull_min.ppf(prob_ret, shape, loc, scale)
  else:
    return -1.0
  pass


def mejor_ajuste(*args) -> int:
  a, b, c, d, e = args
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
    meses[i - 1] = promedio_7_dias[promedio_7_dias.index.month == i].dropna()
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


def calc_resultados(datos: pd.DataFrame, debug_flag: bool = False) -> estado_algoritmo.ResultadosAnla:
  if debug_flag:
    # print(datos, "datos_alterado")
    pass
  cdc: pd.DataFrame = generar_cdc(datos)
  cdc_anios: np.ndarray = np.interp(estado_algoritmo.EstadoAnla.cdc_umbrales, cdc['cumsum'],
                                    cdc['Valor'])
  anios_retorno: list[float] = caud_retorn_anla(datos, estado_algoritmo.EstadoAnla.anios_retorn)
  resultados_iha: IhaEstado = IhaEstado(datos)
  resultados_iha.calcular_iha()
  resultados: estado_algoritmo.ResultadosAnla = estado_algoritmo.ResultadosAnla(cdc=cdc, cdc_anios=cdc_anios,
                                                                                caud_return=anios_retorno,
                                                                                iah_result=resultados_iha)
  if debug_flag:
    pass  # print(resultados, "resultados_iha")
  return resultados


def recortar_caudal(original: pd.DataFrame, caudales: Sequence[float]) -> pd.DataFrame:
  data_alterado = original.copy()
  meses = data_alterado.index.month - 1
  data_alterado['Valor'] = np.minimum(data_alterado['Valor'].values, np.array(caudales)[meses])
  return data_alterado


def calc_alterado(data: pd.DataFrame, caudales) -> (pd.DataFrame, estado_algoritmo.ResultadosAnla):
  data_alterado = recortar_caudal(data, caudales)
  resultados_alterado: estado_algoritmo.ResultadosAnla = calc_resultados(data_alterado, True)
  return data_alterado, resultados_alterado


def caud_retorn_anla(df: pd.DataFrame, anios: list) -> list[float]:
  resultado: list = [0] * len(anios)
  for i, anio in enumerate(anios):
    resultado[i] = calc_caud_retor(df, anio)
  return resultado


def general_month_mean(data: pd.DataFrame) -> pd.DataFrame:
  df = pd.DataFrame()
  df = df.assign(Fecha=None, Mean=None)
  grupo = data.groupby([data.index.year, data.index.month])
  for group, data_filter in grupo:
    year, month = group
    mean_value = data_filter['Valor'].mean()
    row = [f"{year}-{month}", mean_value]
    df.loc[len(df)] = row
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  return df


def mensual_mean(data: pd.DataFrame) -> (np.ndarray, np.ndarray):
  datos_agrupados = data.groupby(data.index.month)
  promedios = np.array((datos_agrupados.mean())['Valor'])
  maximos = np.array((datos_agrupados.max())['Valor'])
  return promedios, maximos


# @dataclass
# class ResultadosAnla:
#   cdc: pd.DataFrame
#   cdc_anios: np.ndarray
#   caud_return: list[float]
#   iah_result: IhaEstado.IhaEstado

def dividir_resultados(resultados_natural: estado_algoritmo.ResultadosAnla,
                       resultados_otro: estado_algoritmo.ResultadosAnla) -> list[list[float]]:
  resultados_cdc: list[float] = []
  resultados_retorno: list[float] = []
  for a, b in zip(resultados_natural.cdc_anios, resultados_otro.cdc_anios):
    resultados_cdc.append(a * 0.5 / b)
    pass
  for a, b in zip(resultados_natural.caud_return, resultados_otro.caud_return):
    resultados_retorno.append(a * 0.6 / b)
  resultados_iha: list[float] = resultados_natural.iah_result - resultados_otro.iah_result
  return [resultados_cdc, resultados_retorno, resultados_iha]
  # > 0.5 cdc con y sin proyecto
  # > 0.6 periodos de retorno
  # umbrales es media más o menos desviación estandar caudal natural
  # pass


def funcion_costo(resultados_01: list, caudales: list):
  costo: float = funcion_castigo(resultados_01) + funcion_suma(caudales)
  print(costo, "funcion_costo", caudales)
  return costo


def funcion_castigo(resultados: list) -> float:
  C1, C2, C3 = 1, 4, 1
  sumatoria = 0

  def castigar(a: Iterable | float):
    nonlocal sumatoria
    if isinstance(a, Iterable):
      for i in a:
        castigar(i)
    else:
      if a < 1:
        pass
      else:
        sumatoria += C1 * np.exp((a - C3) * C2)

  castigar(resultados)
  return sumatoria


def funcion_suma(caudales: list, euclidiana: bool = False) -> float:
  epsilon = 2E-3
  if euclidiana:
    suma = np.sqrt(np.sum(np.array(caudales) ** 2))
  else:
    suma = np.array(caudales).sum()
  if suma <= epsilon:
    return 20000
  return suma


def prin_func(estado: estado_algoritmo.EstadoAnla) -> pd.DataFrame:
  # calculo elementos iniciales
  estado.df_month_mean = general_month_mean(estado.data)
  estado.q7_10 = calcular_7q10(estado.data)
  estado.q95 = calcular_q95(estado)
  estado.data.to_csv("estado_data_prueba_1.csv")
  estado.propuesta_inicial_ref = np.minimum(estado.q7_10, estado.q95)
  # calculo estado normal
  estado.resultados_ori = calc_resultados(estado.data)
  # calculo estado referencia
  estado.data_ref, estado.resultados_ref = calc_alterado(estado.data, estado.propuesta_inicial_ref)

  def funcion_objetivo(caudal_propuesto) -> float:
    """
    Función objetivo que calcula el costo para un conjunto dado de caudales.
    """
    print(caudal_propuesto, "caudal_propuesto")
    estado.data_alter, estado.resultados_alterada = calc_alterado(estado.data, caudal_propuesto)
    # Utiliza la función comparar_resultados con el caudal propuesto

    return comparar_resultados(estado.resultados_ori, estado.resultados_alterada, caudal_propuesto)

  max_mean = mensual_mean(estado.data)
  propuesta_inicial = max_mean[0]
  resultado = minimize(
    funcion_objetivo,
    x0=propuesta_inicial,  # Valor inicial (propuesta inicial de caudales)
    method='Nelder-Mead',
    # method='L-BFGS-B',  # Método de optimización
    bounds=[(0, limite) for limite in max_mean[1]]  # Limita los caudales a ser positivos
  )
  print(resultado.x, "resultado")
  estado.data_alter2 = estado.data_alter

  return pd.DataFrame()


def comparar_resultados(natural: estado_algoritmo.ResultadosAnla, alterado: estado_algoritmo.ResultadosAnla,
                        caudales: list) -> float:
  resultados_a_comparar = dividir_resultados(natural, alterado)
  return funcion_costo(resultados_a_comparar, caudales)
