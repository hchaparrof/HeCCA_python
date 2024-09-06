# -*- coding: utf-8 -*-
"""##librerias"""

import pandas as pd
import ingreso_datos
import estado_algoritmo
from typing import Optional
# import concurrent.futures


def export_resultados(array_onj: list, str_export: str) -> None:
  caudales_array: list[pd.DataFrame] = [pd.DataFrame()] * len(array_onj)
  for i, instancia in enumerate(array_onj):
    print(f"{i}: {instancia.data_alter2}", "hola_23")
    caudales_array[i] = instancia.data_alter[['Valor']]
  if caudales_array:
    df_total = pd.concat(caudales_array).sort_index().copy()
    df_total.sort_index(inplace=True)
    caudales_resultado = df_total.groupby(df_total.index).max()
    caudales_resultado.to_csv(str_export)


def ejecutar_funcion(objeto: estado_algoritmo.EstadoAlgoritmo):
  return objeto.principal_funcion()


def main():

  instancia_algoritmo: Optional[list[estado_algoritmo.EstadoAlgoritmo]] = ingreso_datos.ejecutar_algoritmo_ruta("setup.json")  # ingreso_datos.generar_algoritmo_json()
  if instancia_algoritmo is None:
    print("Error en el proceso")
    return
  for instance in instancia_algoritmo:
    instance.principal_funcion()
  print(len(instancia_algoritmo))
  # with concurrent.futures.ProcessPoolExecutor() as executor:
  #   # print("Executando...")
  #   try:
  #     resultados = executor.map(ejecutar_funcion, instancia_algoritmo)
  #   except Exception as e:
  #     print(f"Exception occurred: {e}")
  print("hola")
  array_ideam: list[estado_algoritmo.EstadoIdeam] = []
  array_anla: list[estado_algoritmo.EstadoAnla] = []
  hola = 0
  for obj in instancia_algoritmo:
    if isinstance(obj, estado_algoritmo.EstadoIdeam):
      array_ideam.append(obj)
      obj.data_alter.to_csv('datos_' + str(hola) + '.csv')
      hola += 1
    elif isinstance(obj, estado_algoritmo.EstadoAnla):
      array_anla.append(obj)
  print("terminada_parte_multi")
  export_resultados(array_ideam, 'caudal_ambiental_ideam.csv')
  export_resultados(array_anla, 'caudal_ambiental_anla.csv')
  print("labor finalizada ")


if __name__ == '__main__':
  main()
