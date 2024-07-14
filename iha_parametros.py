# -*- coding: utf-8 -*-
"""IHA Parametros.ipynb
Automatically generated by Colab.
Original file is located at
    https://colab.research.google.com/drive/1MZyHjMwFuCaI1ASBdriY7DFMpNL8MTXF
"""
# Parametros IHA - Adaptación HeCCA_2.0
import pandas as pd
import numpy as np
"""Serie de caudales diarios medios de ejemplo csv"""


def Setdata(file_path):
  data = pd.read_csv(file_path, usecols=[0, 1], header=None, names=['Date', 'Valor'])
  data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
  data['Valor'] = pd.to_numeric(data['Valor'], errors='coerce')
  """Quita la primera fila del dataframe"""
  data = data.iloc[1:]
  data['Year'] = data['Date'].dt.year
  data['Month'] = data['Date'].dt.month
  start_year = data['Year'].min()
  end_year = data['Year'].max()
  return data, start_year, end_year


def Iha_parameter1(data, start_year, end_year):
  """Parametros Grupo_1 calculo de la media a partir de los caudales diarios para los 12 meses de cada año"""
  from datetime import datetime, timedelta
  """Se crea el dataframe donde se guardaran los parametros del grupo 1"""
  Group1_IHA = pd.DataFrame(index=range(1, 13), columns=range(start_year, end_year+1))
  """Se crea el rango de fechas para el año"""
  for year in range(start_year, end_year+1):
    for month in range(1, 13):
      monthly_data = data[(data['Year'] == year) & (data['Month'] == month)]
      if not monthly_data.empty:
        mean_value = np.nanmean(monthly_data['Valor'])
        Group1_IHA.loc[month, year] = mean_value
      else:
        Group1_IHA.loc[month, year] = np.nan
  return Group1_IHA


def Iha_parameter2(data, start_year, end_year):
  """Parametros Grupo_2 calcula la magnitud y duración de caudales anuales extremos,
  determina el caudal anual minimo y maximo para diferentes duraciones (1 dia, 3 dias, 7 dias, 30 dias y 90 dias)
  Se inicializa un dataframe para guardar los parametros"""
  Group2_IHA = pd.DataFrame()
  for year in range(start_year, end_year+1):
    """Ordenar los datos en orden ascendente"""
    caudales_diarios = data[data['Year'] == year]['Valor'].values
    if len(caudales_diarios) == 0:
      continue
    Sorted_Yearly = np.sort(caudales_diarios.flatten())
    """Calcular los minimos"""
    Group2_IHA.loc[0, year] = Sorted_Yearly[0]              # Caudal anual minimo 1 dia
    Group2_IHA.loc[1, year] = np.mean(Sorted_Yearly[0:3])   # Caudal anual minimo 3 dia
    Group2_IHA.loc[2, year] = np.mean(Sorted_Yearly[0:7])   # Caudal anual minimo 7 dia
    Group2_IHA.loc[3, year] = np.mean(Sorted_Yearly[0:30])  # Caudal anual minimo 30 dia
    Group2_IHA.loc[4, year] = np.mean(Sorted_Yearly[0:90])  # Caudal anual minimo 90 dia
    """Calcular los máximos"""
    Group2_IHA.loc[5, year] = Sorted_Yearly[-1]             # Caudal anual máximo 1 dia
    Group2_IHA.loc[6, year] = np.mean(Sorted_Yearly[-3:])   # Caudal anual máximo 3 dia
    Group2_IHA.loc[7, year] = np.mean(Sorted_Yearly[-7:])   # Caudal anual máximo 7 dia
    Group2_IHA.loc[8, year] = np.mean(Sorted_Yearly[-30:])  # Caudal anual máximo 30 dia
    Group2_IHA.loc[9, year] = np.mean(Sorted_Yearly[-90:])  # Caudal anual máximo 90 dia
  return Group2_IHA


def Iha_parameter3(data, start_year, end_year):
  """Fecha de ocurrencia de los caudales extremos Grupo 3 parametros IHA """
  columns = [f'Year_{year}' for year in range(start_year, end_year+1)]
  """Se inicializa el dataframe para guardar los parametros"""
  Group3_IHA = pd.DataFrame()
  for year in range(start_year, end_year+1):
    caudales_diarios = data[data['Year'] == year]['Valor'].values
    if len(caudales_diarios) == 0:
      continue
    """Encontrar las fechas de los caudales máximos y mínimos"""
    INDXmax = np.where(caudales_diarios == caudales_diarios.max())[0]+1  # +1 para convertir de índice a fecha calendario Juliano
    INDXmin = np.where(caudales_diarios == caudales_diarios.min())[0]+1
    """Ajustar las fechas en casos de multiples ocurrencias"""
    Group3_IHA.loc['Julian_date_max', f'Year_{year}'] = np.ceil(INDXmax.mean())
    Group3_IHA.loc['Julian_date_min', f'Year_{year}'] = np.ceil(INDXmin.mean())
  return Group3_IHA


"""Frecuencia y duración de pulsos altos y bajos de caudal
Bajos pulsos: Periodos durante los cuales el caudal medio esta por debajo del 25th percentil de todos los caudales previos
Altos pulsos: Periodos durante los cuales el caudal excede el 75th percentil.
"""


def get_pulse_duration(indices):
  """Función para obtener la duración de los pulsos"""
  transitions = np.diff(np.concatenate(([0], indices, [0])))
  pulse_starts = np.where(transitions == 1)[0]
  pulse_ends = np.where(transitions == -1)[0]
  return pulse_ends - pulse_starts


def Iha_parameter4(data, start_year, end_year):
  """"Calcular los percentiles (25th,50th,75th)"""
  percentiles = np.percentile(data['Valor'].dropna(), [25, 50, 75])
  """Se inicializa el dataframe para guardar los parametros"""
  Group4_IHA = pd.DataFrame()
  """Identificar los indices de los pulsos altos y bajos"""
  for year in range(start_year, end_year+1):
    daily_flow = data[data['Year'] == year]['Valor'].values
    if len(daily_flow) == 0:
      continue
    """Identificar los indices de los pulsos altos y bajos"""
    high_pulse_index = daily_flow >= percentiles[2]
    low_pulse_index = daily_flow <= percentiles[0]
    """Contar el número de pulsos altos y bajos"""
    high_pulse_count = np.sum(high_pulse_index)
    low_pulse_count = np.sum(low_pulse_index)
    """Duración de los pulsos altos y bajos"""
    high_pulse_duration = np.sum(get_pulse_duration(high_pulse_index))
    low_pulse_duraiton = np.sum(get_pulse_duration(low_pulse_index))
    """Almacenar los resulatdos en el DataFrame"""
    Group4_IHA.loc['High_pulse_count', f'Year_{year}'] = high_pulse_count
    Group4_IHA.loc['Low_pulse_count', f'Year_{year}'] = low_pulse_count
    Group4_IHA.loc['High_pulse_duration', f'Year_{year}'] = high_pulse_duration
    Group4_IHA.loc['Low_pulse_duration', f'Year_{year}'] = low_pulse_duraiton

  return Group4_IHA


def Ihaparameter5(data):
  """Tasa y frecuencia de cambios en las condiciones hidrologicas
  Se inicializa el dataframe para guardar los parametros"""
  Group5_IHA = pd.DataFrame()
  """Diferencia de días consecutivos"""
  data['Diff'] = data.groupby('Year')['Valor'].diff()
  """Indice,valor y media de diferencias positivas y negativas"""
  for year in data['Year'].unique():
    yearly_data = data[data['Year'] == year]
    if yearly_data.empty:
      continue
    """Para diferencias positivas"""
    PositiveDiff_INDX = yearly_data['Diff'] > 0
    Positiv_diff_val = yearly_data.loc[PositiveDiff_INDX, 'Diff'].dropna().values
    Group5_IHA.at['Num_Rises', year] = len(Positiv_diff_val)  # Número de aumentos
    Group5_IHA.at['Mean_Pos_Diff', year] = np.mean(Positiv_diff_val) if len(Positiv_diff_val) > 0 else np.nan  # Media de diferencias positivas

    """Para diferencias negativas"""
    NegativeDiff_INDX = yearly_data['Diff'] < 0
    Negativ_diff_val = yearly_data.loc[NegativeDiff_INDX, 'Diff'].dropna().values
    Group5_IHA.at['Num_Falls', year] = len(Negativ_diff_val)  # Número de caidas
    Group5_IHA.at['Mean_Neg_Diff', year] = np.mean(Negativ_diff_val) if len(Negativ_diff_val) > 0 else np.nan  # Media de diferencias negativas

  return Group5_IHA