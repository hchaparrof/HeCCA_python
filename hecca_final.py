# -*- coding: utf-8 -*-
"""##librerias"""
import pandas as pd
import ingreso_datos
import estado_algoritmo
import concurrent.futures


def main():
  instancia_algoritmo: list[estado_algoritmo.EstadoAlgoritmo] | None = ingreso_datos.generar_algoritmo_json()
  if instancia_algoritmo is None:
    print("Error en el proceso")
    return
  with concurrent.futures.ThreadPoolExecutor() as executor:
    resultados = executor.map(lambda estado: estado.principal_funcion(), instancia_algoritmo)
  # for estado in instancia_algoritmo:
  #   estado.principal_funcion()
  caudales_ambientales: list[pd.DataFrame] = [pd.DataFrame()]*len(instancia_algoritmo)
  for i, instancia in enumerate(instancia_algoritmo):
    caudales_ambientales[i] = instancia.data_alter.rename(columns={'Q_ambiental': 'Valor'})[['Valor']]
  df_total = pd.concat(caudales_ambientales).sort_index().copy()
  df_total.to_csv('caudal_ambiental.csv')
  print("labor finalizada ")


if __name__ == '__main__':
  main()
