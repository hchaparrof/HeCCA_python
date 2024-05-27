# -*- coding: utf-8 -*-
"""##librerias"""
import pandas as pd
import ingreso_datos
import estado_algoritmo
import concurrent.futures


def export_resultados(array_onj: list, str_export: str) -> None:
  caudales_array: list[pd.DataFrame] = [pd.DataFrame()] * len(array_onj)
  for i, instancia in enumerate(array_onj):
    caudales_array[i] = instancia.data_alter[['Valor']]
  if caudales_array:
    df_total = pd.concat(caudales_array).sort_index().copy()
    df_total.sort_index(inplace=True)
    caudales_resultado = df_total.groupby(df_total.index).max()
    caudales_resultado.to_csv(str_export)


def main():
  instancia_algoritmo: list[estado_algoritmo.EstadoAlgoritmo] | None = ingreso_datos.generar_algoritmo_json()
  if instancia_algoritmo is None:
    print("Error en el proceso")
    return
  # for instance in instancia_algoritmo:
  #   instance.principal_funcion()
  with concurrent.futures.ThreadPoolExecutor() as executor:
    resultados = executor.map(lambda estado: estado.principal_funcion(), instancia_algoritmo)
  print("hola")
  array_ideam: list[estado_algoritmo.EstadoIdeam] = []
  array_anla: list[estado_algoritmo.EstadoAnla] = []
  for obj in instancia_algoritmo:
    if isinstance(obj, estado_algoritmo.EstadoIdeam):
      array_ideam.append(obj)
    elif isinstance(obj, estado_algoritmo.EstadoAnla):
      array_anla.append(obj)
  export_resultados(array_ideam, 'caudal_ambiental_ideam.csv')
  export_resultados(array_anla, 'caudal_ambiental_anla.csv')
  print("labor finalizada ")


if __name__ == '__main__':
  main()
