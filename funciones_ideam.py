import pandas as pd
from evento_completo import EventoCompleto
from estado_algoritmo import EstadoIdeam
from estadistica_ideam import *


def buscar_umbrales(estado: EstadoIdeam) -> pd.DataFrame:
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
  data = estado.data
  # print(data)
  if estado.h_umbrales['QB'] is None:
    u_qb = 2.33
    u_qtq = 2
  else:
    u_qb = estado.h_umbrales[0]
    u_qtq = estado.h_umbrales[1]
  # df es un dataframe temporar que se sobreescribe eventualmente
  df = pd.DataFrame()
  # set columns
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None)
  # se crea un dataframe en donde se calculan los minimos máximos y promedio por mes y año
  grouped = data.groupby([data.index.year, data.index.month])
  for group, data_filter in grouped:
    year, month = group
    row = [f"{year}-{month}", data_filter['cuenca-base'].min(), data_filter['cuenca-base'].max(),
           data_filter['cuenca-base'].mean()]
    df.loc[len(df)] = row
  # data view
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  '''
  se crea listas con n valores, siendo n el número de años en el dataset, guardando el máximo y minimo por año
  '''
  mins = df.groupby(df.index.year)['Min'].min()
  maxs = df.groupby(df.index.year)['Max'].max()
  mins = np.array(mins)
  maxs = np.array(maxs)
  # print(mins, maxs, "hola_mins")
  mean_min = (mins.mean())
  std_min = (np.std(mins, ddof=1))  # desviacion estandar
  coef_variacion_min = std_min / mean_min
  mean_max = (maxs.mean())
  std_max = (np.std(maxs, ddof=1))
  alpha_min = 1.0079 * (coef_variacion_min ** -1.084)
  a_alpha_min = (-0.0607 * (coef_variacion_min ** 3)) + (0.5502 * (coef_variacion_min ** 2)) - (
            0.4937 * coef_variacion_min) + 1.003
  beta = (a_alpha_min / mean_min) ** (-1)
  # umbrales QTQ y QB
  #
  umbral_QTQ = beta * ((-np.log(1 - (1 / u_qtq))) ** (1 / alpha_min))
  umbral_Q10 = beta * ((-np.log(1 - (1 / 10))) ** (1 / alpha_min))
  #
  #
  df = pd.DataFrame()  # crea un dataframe donde poner los minimos y maximos por mes de toda la serie
  # set columns
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None, Min_rev=None)
  # calculate mim, max, mean, min_rev and mean_rev
  grouped = data.groupby([data.index.year, data.index.month])
  for group, data_filter in grouped:
    year, month = group
    min_value = data_filter['cuenca-base'].min()
    max_value = data_filter['cuenca-base'].max()
    mean_value = data_filter['cuenca-base'].mean()
    min_rev_value = data_filter.loc[data_filter['cuenca-base'] > umbral_Q10, 'cuenca-base'].min()
    row = [f"{year}-{month}", min_value, max_value, mean_value, min_rev_value]
    df.loc[len(df)] = row
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  # df = df.set_index('Fecha')
  alpha_max = (np.sqrt(6) * std_max) / np.pi
  u_max = mean_max - (0.5772 * alpha_max)
  yt_QB = -np.log(np.log(u_qb / (u_qb - 1)))
  yt_Q15 = -np.log(np.log(15 / (15 - 1)))
  # umbrales QB y QTR15
  #
  umbral_QB = u_max + (yt_QB * alpha_max)
  umbral_Q15 = u_max + (yt_Q15 * alpha_max)
  # print(umbral_Q15, "umbral_q15")
  #
  ##
  df = pd.DataFrame()
  df = df.assign(Fecha=None, Min=None, Max=None, Mean=None, Min_rev=None, Mean_rev=None)
  grouped = data.groupby([data.index.year, data.index.month])
  for group, data_filter in grouped:
    year, month = group
    min_value = data_filter['cuenca-base'].min()
    max_value = data_filter['cuenca-base'].max()
    mean_value = data_filter['cuenca-base'].mean()
    min_rev_value = data_filter.loc[data_filter['cuenca-base'] > umbral_Q10, 'cuenca-base'].min()
    mean_rev_value = data_filter.loc[
      (data_filter['cuenca-base'] > umbral_Q10) & (data_filter['cuenca-base'] < umbral_Q15), 'cuenca-base'].mean()
    row = [f"{year}-{month}", min_value, max_value, mean_value, min_rev_value, mean_rev_value]
    df.loc[len(df)] = row
  #########
  df['Fecha'] = pd.to_datetime(df['Fecha'])
  df = df.set_index('Fecha')
  # print(df)
  estado.umbrales['QTR15'] = umbral_Q15
  estado.umbralesQB = umbral_QB
  estado.QTR10 = umbral_Q10
  estado.QTQ = umbral_QTQ
  if estado.h_umbrales is not None:
    estado.setear_umbrales([umbral_Q15, umbral_QB, umbral_QTQ, umbral_Q10])
  return df


def df_eventos(df_objetivo, lista_eventos):
  """función toma un dataframe donde guardar las cosas y una lista con todos
  los eventos de la categoria que se le den y lo que hace es crear un dataframe
  con los datos de mes magnitud, etc. para luego poder acceder a ellos más fácil
  """
  # set columns
  df_objetivo = df_objetivo.assign(mes=None, Magnitud=None, Intensidad=None, Duracion=None)
  # calculate mim, max, mean, min_rev and mean_rev
  for j in range(len(lista_eventos)):
    # filtro los datos por año y mes para hacer el analisis
    row = [lista_eventos[j].mes, lista_eventos[j].magnitud, lista_eventos[j].intensidad, lista_eventos[j].duracion]
    df_objetivo.loc[len(df_objetivo)] = row
  return df_objetivo


def nombrar_evento(estado: EstadoIdeam, dfh, str4_h='cuenca-base'):
  """funcion que coge un dataframe con datos de caudal y determina
  que días hubo eventos poniendo 1 o 0 donde sea necesario
  Parameters
  -----------
  estado: instancia del algoritmo en el que se esta trabajando
  dfh: dataframe al que se le van a encontrar los eventos
  str4_h: string con el valor 'cuenca-base' se necesita, aunque no se porque
  Return
  ----------
  nada
  """
  # global QTR_15
  # global QB
  # global QTQ
  # global Q10
  umbral_Q15 = estado.umbrales['QTR15']
  umbral_QB = estado.umbrales['QB']
  umbral_Q10 = estado.umbrales['Q10']
  umbral_QTQ = estado.umbrales['QTQ']
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

  dfh.loc[dfh['cuenca-base'] <= umbral_Q10, 'event_Q10'] = 1
  dfh.loc[dfh[str4_h] > umbral_Q10, 'event_Q10'] = 0


def crear_evento(estado: EstadoIdeam, df21):
  primer_dia = estado.primer_dia
  dif = estado.dif
  final_dia = estado.final_dia
  # global lista_prueba
  lista_po = []
  # contar_eventos(data,'event_QTQ',eventos_qtq,'event_Q10',QTQ)
  meses_1 = df21.index.month.min()
  meses_2 = df21.index.month.max()
  inicio = primer_dia
  if meses_1 == meses_2:
    lista_po.append(df21)
  else:
    for i in range(meses_1, meses_2 + 1):
      df_provisional = df21[df21.index.month == i]
      num_filas = df_provisional.shape[0]
      if num_filas == 0:
        continue
      lista_po.append(df_provisional)
  return lista_po


def contar_eventos(estado: EstadoIdeam, df21: pd.DataFrame, eventosh: str,
                   eventos_umbral: list, eventosi: str, umbral_caudal: float):
  """recibe un dataframe (df21) con los 1's ya puestos, un string (eventosh) con el tipo de evento, una lista
  (eventos_umbral), otros string (eventosi) que es el evento umbral mayor, en caso de ser un evento medio y
   un flotante (umbral_caudal)con el valor del umbral, rellena las listas de eventos con bloques de eventos
   donde hayan 1's juntos, para formar eventos completos, en el caso de evento QB y QTQ también verifica que no
    hallan eventos QTR15 o Q10 inmediatamente anterior o siguiente al bloque"""
  data = estado.data
  df2 = estado.df2
  primer_dia = estado.primer_dia
  dif = estado.dif
  final_dia = estado.final_dia
  inicio = primer_dia
  # print(inicio, type(inicio))
  # inicio=data[data['event_QTR15']==1].index.min()
  final = data[(df21.index > inicio) & (df21[eventosh] == 0)].index.min()
  g = 0
  while g < 700:
    # forma los bloques encontrando el primer 1 del df y el primer 0 después de ese 1 y haciendo un bloque de datos entre los dos y luego volviendolo a hacer
    # pero empeando desde el 0 anterior, hasta que acabe con el df o con el g<700 puesto por si hay algun problema no se quede pensando infinitamente
    inicio = df21[(df21[eventosh] == 1) & (df21.index >= inicio)].index.min()
    final = df21[(df21[eventosh] == 0) & (df21.index >= inicio)].index.min()
    if df21[((df21.index >= inicio) & (df21.index <= df21.index.max())) & (df21[eventosh] == 0)].size == 0:
      final = df21.index.max()
    if (not ((eventosh == 'event_QB') | (eventosh == 'event_QTQ'))) | (
    not (any(df21[eventosi][(df21.index >= inicio - dif) & (df21.index < final + dif)]))):
      if df21[(df21.index >= inicio) & (df21.index < final)].size == 0:
        pass
      else:
        df_envio = df21[(df21.index >= inicio) & (df21.index < final)].copy()
        a = crear_evento(estado, df_envio)
        for i in range(len(a)):
          eventos_umbral.append(
            EventoCompleto(a[i][(a[i].index >= inicio) & (a[i].index < final)].copy(), umbral_caudal, eventosh))
    inicio = final
    if not (any(df21[eventosh][df21.index >= final])):
      break
    # if (df21[(df21.index>=inicio)&(df21.index<final)].empty):
    #  break
    g = g + 1
  inicio = primer_dia
  final = final_dia


def org_alt(estado: EstadoIdeam):
  estado.reiniciar_alter()
  df_qtq_alt = estado.df_umbrales['df_qtq_alt']
  df_qb_alt = estado.df_umbrales['df_qb_alt']
  eventos_rev_qtr15 = estado.listas_eventos['eventos_rev_qtr15']
  eventos_rev_qb = estado.listas_eventos['eventos_rev_qb']
  eventos_rev_qtq = estado.listas_eventos['eventos_rev_qtq']
  eventos_rev_q10 = estado.listas_eventos['eventos_rev_q10']
  QTR_15 = estado.umbrales['QTR15']
  Q10 = estado.umbrales['Q10']
  QB = estado.umbrales['QB']
  QTQ = estado.umbrales['QTQ']
  data_alter2 = estado.data_alter2
  # ###
  df_qtq_alt = pd.DataFrame()
  df_qb_alt = pd.DataFrame()
  df_qtq_alt.assign(mes=None, Magnitud=None, Duracion=None, Intensidad=None)
  df_qb_alt.assign(mes=None, Magnitud=None, Duracion=None, Intensidad=None)
  eventos_rev_qtr15.clear()
  eventos_rev_qb.clear()
  eventos_rev_qtq.clear()
  eventos_rev_q10.clear()
  nombrar_evento(estado, estado.data_alter2, 'cuenca-base')
  contar_eventos(estado, estado.data_alter2, 'event_QTR15', eventos_rev_qtr15, 'event_QTR15', QTR_15)
  estado.listas_eventos['eventos_rev_qtr15'] = eventos_rev_qtr15
  contar_eventos(estado, estado.data_alter2, 'event_QB', eventos_rev_qb, 'event_QTR15', QB)
  estado.listas_eventos['eventos_rev_qb'] = eventos_rev_qb
  contar_eventos(estado, estado.data_alter2, 'event_QTQ', eventos_rev_qtq, 'event_Q10', QTQ)
  estado.listas_eventos['eventos_rev_qtq'] = eventos_rev_qtq
  contar_eventos(estado, estado.data_alter2, 'event_Q10', eventos_rev_q10, 'event_Q10', Q10)
  estado.listas_eventos['eventos_rev_q10'] = eventos_rev_q10
  df_qtq_alt = df_eventos(df_qtq_alt, eventos_rev_qtq)
  df_qb_alt = df_eventos(df_qb_alt, eventos_rev_qb)
  estado.df_umbrales['df_qtq_alt'] = df_qtq_alt
  estado.df_umbrales['df_qb_alt'] = df_qb_alt


def formar_alter(estado: EstadoIdeam):
  data_alter = estado.data_alter
  data_alter2 = estado.data_alter2
  df2 = estado.df2
  data = estado.data
  data_alter = data[['cuenca-base']].copy()
  data_alter['Aprov_teo'] = np.NaN
  data_alter['check_3'] = np.NaN
  data_alter['check_2'] = np.NaN
  data_alter['Q_ajustado'] = np.NaN
  data_alter['Q_ambiental'] = np.NaN
  data_alter['Q_aprov_real'] = np.NaN
  #print(data_alter.index.month, "iiiiiiiiiiiiiiiiii")
  for i in range(1, 13):
    # print(estado.data.index.dtype, " iiiiiii ", data_alter.index.dtype)
    # print(i, "hahahahahah")
    # print(data_alter.index.month == 1, " iiiiiiiiiiiiiii")
    a_bool = data_alter.index.month == i  # serie booleana si pertenece al mes correcto
    data_alter.loc[a_bool, 'Aprov_teo'] = df2.iat[i - 1, 4]  # aprovechamiento teorico = aprovechamiento teorico de ese mes ########
    data_alter.loc[a_bool, 'check_3'] = df2.iat[i - 1, 1]  # check 3 igual al minimo revisado ########
    b_bool = data_alter['cuenca-base'] > data_alter['Aprov_teo']  # s.b. si el Q obs es > que el aprov teorico
    c_bool = a_bool & b_bool  # s.b. si pertenece al mes correcto y es Q obs es > que el aprov teorico
    d_bool = a_bool & (~b_bool)  # s.b. si pertenece al mes correcto y es Q obs es < que el aprov teorico
    e_bool = data_alter['cuenca-base'] < data_alter['check_3']  # s.b. si el Q obs es < al minimo revisado
    f_bool = d_bool & e_bool  # Si mes correcto & Q obs < aprov teo & Q obs < check 3
    g_bool = d_bool & (~e_bool)  # s.b. si pertenece al mes correcto y el Q obs es > al minimo revisado & Q obs es < aprov teorico
    data_alter.loc[c_bool,'Q_ajustado'] = df2.iat[i - 1, 4]  # Q obs > aprov teorico = aprov teorico #########
    data_alter.loc[f_bool, 'Q_ajustado'] = 0  # Q obs < aprov teorico = 0 #########
    data_alter.loc[g_bool, 'Q_ajustado'] = data_alter['cuenca-base'][g_bool] - data_alter['check_3'][g_bool]  # #######
    data_alter['check_2'] = data_alter['cuenca-base'] - data_alter['Q_ajustado']  # check 2 = Q obs - Q ajustado
  a_bool = data_alter['cuenca-base'] < data_alter['check_3']  # los caudales observados que sean menores al minimo revisado
  b_bool = data_alter['check_2'] < data_alter['check_3']  # Q ajustado es menor al minimo revisado
  data_alter.loc[a_bool, 'Q_ambiental'] = data_alter['cuenca-base']  # los queQobs sean < al minrev no se le hace nada
  data_alter.loc[(~a_bool) & b_bool, 'Q_ambiental'] = data_alter['check_3']  # si el Qobs - Qaprov =min rev
  data_alter.loc[(~a_bool) & (~b_bool), 'Q_ambiental'] = data_alter['check_2']  # si Qobs> min rev y Qobs-Qaprov >min rev, Qobs-aQaprov
  data_alter['Q_aprov_real'] = data_alter['cuenca-base'] - data_alter['Q_ambiental']  # qobs -qambs
  data_alter2 = data_alter[['Q_ambiental']].copy()
  data_alter2.rename(columns={'Q_ambiental': 'cuenca-base'}, inplace=True)
  estado.data_alter = data_alter
  estado.data_alter2 = data_alter2


def org_df2_2(estado: EstadoIdeam, aprov, mes):  # porcentaje de aprovechamiento, mes, cambia el porcentaje de aprovechamiento de un mes escogido por el %escogido
  """recibe un valor porcentual (aprov) y un valor entero (mes) y cambia el valor de aprovechamiento de ese mes ademas de cambiar el caudal aprovechado para ese mes"""
  df2 = estado.df2
  df2.loc[df2.index.month == mes, '%_aprov'] = aprov
  df2['Q_aprov'] = df2['Mean_rev'] * df2['%_aprov']
  estado.df2 = df2


def org_df2_1(estado: EstadoIdeam, df):
  """coge los valores de data(dataframe global) y los usa para crear el df2(df global) con 12 filas y 3 columnas (promedio, min_rev y mean_rev) por mes."""
  df2 = estado.df2
  data = estado.data
  # global df2
  # global data
  df2 = pd.DataFrame()
  # set columns
  df2 = df2.assign(Fecha=None, Mean=None, Min_rev=None, Mean_rev=None)
  # calculate mim, max, mean, min_rev and mean_rev
  i = data.index.min().year
  for j in range(1, 13):
    # filtro los datos por año y mes para hacer el analisis
    data_filter = df[df.index.month == j]
    row = [str(i) + str(-j), data_filter['Mean'].mean(), data_filter['Min_rev'].min(), data_filter['Mean_rev'].mean()] # cambiado
    df2.loc[len(df2)] = row
  # data view
  df2['Fecha'] = pd.to_datetime(df2['Fecha'])
  df2 = df2.set_index('Fecha')
  # print(df2)
  estado.df2 = df2

def numeros_naturales():
  n = 1
  while True:
    yield n
    n += 1


def prin_func(estado: EstadoIdeam) -> (pd.DataFrame, pd.DataFrame):
  data = estado.data
  df_rev = buscar_umbrales(estado)
  org_df2_1(estado, df_rev)
  if (estado.h_umbrales == None):
    df_rev = buscar_umbrales(estado)
  for i in range(1, 13):
    org_df2_2(estado, 1, i)
  primer_dia = data.index.min()
  estado.primer_dia = primer_dia
  estado.final_dia = estado.data.index.max()
  final_dia = data.index.max()
  segundo_dia = data[data.index > data.index.min()].index.min()
  dif = segundo_dia - primer_dia
  estado.dif = dif
  nombrar_evento(estado, data, 'cuenca-base')
  contar_eventos(estado, data, 'event_QTR15', estado.listas_eventos['eventos_qtr15'], 'event_QTR15', estado.umbrales['QTR15'])
  contar_eventos(estado, data, 'event_QB', estado.listas_eventos['eventos_qb'], 'event_QTR15', estado.umbrales['QB'])
  contar_eventos(estado, data, 'event_QTQ', estado.listas_eventos['eventos_qtq'], 'event_Q10', estado.umbrales['QTQ'])
  contar_eventos(estado, data, 'event_Q10', estado.listas_eventos['eventos_q10'], 'event_Q10', estado.umbrales['Q10'])
  estado.df_umbrales['df_qtq_ref'] = df_eventos(estado.df_umbrales['df_qtq_ref'], estado.listas_eventos['eventos_qtq'])
  estado.df_umbrales['df_qb_ref'] = df_eventos(estado.df_umbrales['df_qb_ref'], estado.listas_eventos['eventos_qb'])
  formar_alter(estado)
  org_alt(estado)
  #print(cumple(estado, estado.df_umbrales['df_qtq_ref'], estado.df_umbrales['df_qtq_alt'], i) and cumple(estado, estado.df_umbrales['df_qb_ref'], estado.df_umbrales['df_qb_alt'], i))
  for i in range(1, 13):
    print(f"empezando el {i} mes")
    calibrar_mes(estado, i)
  return estado.df2, estado.data_alter2
# cumple(estado, estado.df_umbrales['df_qtq_ref'], estado.df_umbrales['df_qtq_alt'], 1) and cumple(estado, estado.df_umbrales['df_qb_ref'], estado.df_umbrales['df_qb_alt'], 1)
# org_df2_2(estado, 1, 1)


def prueba_si_cumple(estado: EstadoIdeam, sb: pd.Series, sa: pd.Series, mes: int, duracion, magintud):
  prub = prueba_porc(estado, sb, mes, True, duracion)
  prua = prueba_porc(estado, sa, mes, False, duracion)
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


def cumple(estado: EstadoIdeam, dfb:pd.DataFrame, dfa:pd.DataFrame, mes:int) -> bool:  # dfb =ref dfa = alterada
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


def prueba_porc(estado: EstadoIdeam, s1: pd.Series, mes: int, es_ref: bool, es_duracion: bool) -> bool:
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
  generador_2 = numeros_naturales()
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
    a = cumple(estado, estado.df_umbrales['df_qtq_ref'], estado.df_umbrales['df_qtq_alt'], i) and cumple(estado, estado.df_umbrales['df_qb_ref'], estado.df_umbrales['df_qb_alt'], i)
    # print(next(generador_2), "aaa", "calibrar_mes")
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
