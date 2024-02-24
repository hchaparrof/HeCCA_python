import pandas as pd

from estadistica_ideam import *
from estado_algoritmo import EstadoIdeam
from funciones_ideam import *


def prueba_si_cumple(estado: EstadoIdeam, sb: pd.Series, sa: pd.Series, mes: int, duracion, magintud):
  prub = Prueba_porc(estado, sb, mes, True, duracion)
  prua = Prueba_porc(estado, sa, mes, False, duracion)
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


def cumple(estado: EstadoIdeam, dfb, dfa, mes):  # dfb =ref dfa = alterada
  sb = dfb['Magnitud'][dfb['mes'] == mes]
  sa = dfa['Magnitud'][dfa['mes'] == mes]
  boola = prueba_si_cumple(estado, sb, sa, mes, False, True)
  sb = dfb['Duracion'][dfb['mes'] == mes]
  sa = dfa['Duracion'][dfa['mes'] == mes]
  boolb = prueba_si_cumple(estado, sb, sa, mes, True, False)
  sb = dfb['Intensidad'][dfb['mes'] == mes]
  sa = dfa['Intensidad'][dfa['mes'] == mes]
  boolc = prueba_si_cumple(estado, sb, sa, mes, False, False)
  return (boola and boolb) & boolc
# s1 serie de eventos, mes, mes, es_ref <- ref o no, es_duracion <- duracion o no


def Prueba_porc(estado: EstadoIdeam, s1, mes, es_ref, es_duracion):
  data = estado.data
  df2 = estado.df2
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


def calibrar_mes(estado: EstadoIdeam, mess):
  df_qtq_ref = estado.df_umbrales['df_qtq_ref']
  df_qtq_alt = estado.df_umbrales['df_qtq_alt']
  df_qb_ref = estado.df_umbrales['df_qb_ref']
  df_qb_alt = estado.df_umbrales['df_qb_alt']
  inferior = 0
  superior = 1
  org_df2_2(estado, 1, mess)
  formar_alter(estado)
  org_alt(estado)
  i = mess
  if cumple(estado, df_qtq_ref, df_qtq_alt, i) and cumple(estado, df_qb_ref, df_qb_alt, i):
    return
  conteo = 0
  mayor = 0
  while True:
    conteo += 1
    lim_prov = (inferior+superior)/2
    org_df2_2(estado, lim_prov, mess)
    formar_alter(estado)
    org_alt(estado)
    a = cumple(estado, df_qtq_ref, df_qtq_alt, i) and cumple(estado, df_qb_ref, df_qb_alt, i)
    if a:
      if lim_prov > mayor:
        mayor = lim_prov
      inferior = lim_prov
      if abs(inferior-superior) < 0.01:
        return
    else:
      superior = lim_prov
    if conteo > 2000:
      org_df2_2(estado, mayor, mess)
      formar_alter(estado)
      org_alt(estado)
      return
