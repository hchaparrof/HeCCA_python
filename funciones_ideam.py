import pandas as pd
import numpy as np
from evento_completo import EventoCompleto
from estado_algoritmo import EstadoIdeam


def buscar_umbrales(estado: EstadoIdeam, umbrales: list = None) -> object:
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
  '''
  se crea listas con n valores, siendo n el número de años en el dataset, guardando el máximo y minimo por año
  '''
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
  #df = df.set_index('Fecha')
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
  estado.QTR15 = umbral_Q15
  estado.QB = umbral_QB
  estado.QTR10 = umbral_Q10
  estado.QTQ = umbral_QTQ
  return df

def df_eventos(df_objetivo, lista_eventos):
  """función toma un dataframe donde guardar las cosas y una lista con todos los eventos de la categoria que se le den y lo que hace es crear un dataframe con los datos de mes magnitud, etc. para luego poder acceder a ellos más fácil"""
  # set columns
  df_objetivo = df_objetivo.assign(mes=None, Magnitud=None, Intensidad=None, Duracion=None)
  # calculate mim, max, mean, min_rev and mean_rev
  for j in range(len(lista_eventos)):
    # filtro los datos por año y mes para hacer el analisis
    row = [lista_eventos[j].mes, lista_eventos[j].magnitud, lista_eventos[j].intensidad, lista_eventos[j].duracion]
    df_objetivo.loc[len(df_objetivo)] = row
  return df_objetivo

def nombrar_evento(estado: EstadoIdeam,dfh, str4_h='cuenca-base'):
  """funcion que coge un dataframe con datos de caudal y determina que días hubo eventos poniendo 1 o 0 donde sea necesario
  Parameters
  -----------
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
  #global lista_prueba
  lista_po = []
  # contar_eventos(data,'event_QTQ',eventos_qtq,'event_Q10',QTQ)
  meses_1 = df21.index.month.min()
  meses_2 = df21.index.month.max()
  inicio = primer_dia
  if meses_1 == meses_2:
    lista_po.append(df21)
  else:
    for i in range(meses_1, meses_2+1):
      df_provisional = df21[df21.index.month == i]
      num_filas = df_provisional.shape[0]
      if num_filas == 0:
        continue
      lista_po.append(df_provisional)
  return lista_po

def contar_eventos(estado: EstadoIdeam, df21, eventosh, eventos_umbral, eventosi, umbral_caudal):
  """recibe un dataframe (df21) con los 1's ya puestos, un string (eventosh) con el tipo de evento, una lista (eventos_umbral), otros string (eventosi) que es el evento umbral mayor, en caso de ser un evento medio y un flotante (umbral_caudal)con el valor del umbral, rellena las listas de eventos con bloques de eventos donde hayan 1's juntos, para formar eventos completos, en el caso de evento QB y QTQ también verifica que no hallan eventos QTR15 o Q10 inmediatamente anterior o siguiente al bloque"""
  data = estado.data
  df2 = estado.df2
  primer_dia = estado.primer_dia
  dif = estado.dif
  final_dia = estado.final_dia
  inicio = primer_dia
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
    if (not ((eventosh == 'event_QB') | (eventosh == 'event_QTQ'))) | (not (any(df21[eventosi][(df21.index >= inicio-dif) & (df21.index < final+dif)]))):
      if df21[(df21.index >= inicio) & (df21.index < final)].size == 0:
        pass
      else:
        df_envio = df21[(df21.index >= inicio) & (df21.index < final)].copy()
        a = crear_evento(estado, df_envio)
        for i in range(len(a)):
          eventos_umbral.append(EventoCompleto(a[i][(a[i].index >= inicio) & (a[i].index < final)].copy(), umbral_caudal, eventosh))
    inicio = final
    if not (any(df21[eventosh][df21.index >= final])):
      break
    # if (df21[(df21.index>=inicio)&(df21.index<final)].empty):
    #  break
    g = g+1
  inicio = primer_dia
  final = final_dia
def org_alt(estado: EstadoIdeam):
  df_qtq_alt = estado.df_umbrales['df_qtq_alt']
  df_qb_alt = estado.df_umbrales['df_qb_alt']
  eventos_rev_qtr15 = estado.listas_eventos['eventos_rev_qtr15']
  eventos_rev_qb = estado.listas_eventos['eventos_rev_qb']
  eventos_rev_qtq = estado.listas_eventos['eventos_rev_qtq']
  eventos_rev_q10 = estado.listas_eventos['eventos_rev_q10']
  QTR_15 = estado.umbrales['QTR_15']
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
  nombrar_evento(data_alter2, 'cuenca-base')
  contar_eventos(estado, data_alter2, 'event_QTR15', eventos_rev_qtr15, 'event_QTR15', QTR_15)
  contar_eventos(estado, data_alter2, 'event_QB', eventos_rev_qb, 'event_QTR15', QB)
  contar_eventos(estado, data_alter2, 'event_QTQ', eventos_rev_qtq, 'event_Q10', QTQ)
  contar_eventos(estado, data_alter2, 'event_Q10', eventos_rev_q10, 'event_Q10', Q10)
  df_qtq_alt = df_eventos(df_qtq_alt, eventos_rev_qtq)
  df_qb_alt = df_eventos(df_qb_alt, eventos_rev_qb)
  estado.df_umbrales['df_qtq_alt'] = df_qtq_alt
  estado.df_umbrales['df_qb_alt'] = df_qb_alt

def formar_alter(estado: EstadoIdeam):
  data_alter = estado.data_alter
  data_alter2 = estado.data_alter2
  df2 = estado.df2
  data = estado.data
  # global data_alter
  # global data_alter2
  # global df2
  # global data
  data_alter = data[['cuenca-base']].copy()
  data_alter['Aprov_teo'] = np.NaN
  data_alter['check_3'] = np.NaN
  data_alter['check_2'] = np.NaN
  data_alter['Q_ajustado'] = np.NaN
  data_alter['Q_ambiental'] = np.NaN
  data_alter['Q_aprov_real'] = np.NaN
  for i in range(1, 13):
    a_bool = data_alter.index.month == i  # serie booleana si pertenece al mes correcto
    data_alter['Aprov_teo'].loc[a_bool] = df2.iat[
      i - 1, 4]  # aprovechamiento teorico = aprovechamiento teorico de ese mes ########
    data_alter['check_3'].loc[a_bool] = df2.iat[i - 1, 1]  # check 3 igual al minimo revisado ########
    b_bool = data_alter['cuenca-base'] > data_alter['Aprov_teo']  # s.b. si el Q obs es > que el aprov teorico
    c_bool = a_bool & b_bool  # s.b. si pertenece al mes correcto y es Q obs es > que el aprov teorico
    d_bool = a_bool & (~b_bool)  # s.b. si pertenece al mes correcto y es Q obs es < que el aprov teorico
    e_bool = data_alter['cuenca-base'] < data_alter['check_3']  # s.b. si el Q obs es < al minimo revisado
    f_bool = d_bool & e_bool  # Si mes correcto & Q obs < aprov teo & Q obs < check 3
    g_bool = d_bool & (~e_bool)  # s.b. si pertenece al mes correcto y el Q obs es > al minimo revisado & Q obs es < aprov teorico
    data_alter['Q_ajustado'].loc[c_bool] = df2.iat[i - 1, 4]  # Q obs > aprov teorico = aprov teorico #########
    data_alter['Q_ajustado'].loc[f_bool] = 0  # Q obs < aprov teorico = 0 #########
    data_alter['Q_ajustado'].loc[g_bool] = data_alter['cuenca-base'][g_bool] - data_alter['check_3'][g_bool]  # #######
    data_alter['check_2'] = data_alter['cuenca-base'] - data_alter['Q_ajustado']  # check 2 = Q obs - Q ajustado
  a_bool = data_alter['cuenca-base'] < data_alter['check_3']  # los caudales observados que sean menores al minimo revisado
  b_bool = data_alter['check_2'] < data_alter['check_3']  # Q ajustado es menor al minimo revisado
  data_alter['Q_ambiental'].loc[a_bool] = data_alter['cuenca-base']  # los queQobs sean < al minrev no se le hace nada
  data_alter['Q_ambiental'].loc[(~a_bool) & b_bool] = data_alter['check_3']  # si el Qobs - Qaprov =min rev
  data_alter['Q_ambiental'].loc[(~a_bool) & (~b_bool)] = data_alter['check_2']  # si Qobs> min rev y Qobs-Qaprov >min rev, Qobs-aQaprov
  data_alter['Q_aprov_real'] = data_alter['cuenca-base'] - data_alter['Q_ambiental']  # qobs -qambs
  data_alter2 = data_alter[['Q_ambiental']].copy()
  data_alter2.rename(columns={'Q_ambiental': 'cuenca-base'}, inplace=True)
  estado.data_alter = data_alter
  estado.data_alter2 = data_alter2


def org_df2_2(estado: EstadoIdeam,aprov, mes):  # porcentaje de aprovechamiento, mes, cambia el porcentaje de aprovechamiento de un mes escogido por el %escogido
  """recibe un valor porcentual (aprov) y un valor entero (mes) y cambia el valor de aprovechamiento de ese mes ademas de cambiar el caudal aprovechado para ese mes"""
  df2 = estado.df2
  df2.loc[df2.index.month == mes, '%_aprov'] = aprov
  df2['Q_aprov'] = df2['Mean_rev'] * df2['%_aprov']
  estado.df2 = df2

def org_df2_1(estado: EstadoIdeam,df):
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
    row = [str(i) + str(-j), data_filter['Mean'].mean(), data_filter['Min_rev'].min(), data_filter['Mean_rev'].mean()]
    df2.loc[len(df2)] = row
  # data view
  df2['Fecha'] = pd.to_datetime(df2['Fecha'])
  df2 = df2.set_index('Fecha')
  estado.df2 = df2
