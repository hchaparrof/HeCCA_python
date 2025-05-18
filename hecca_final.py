# -*- coding: utf-8 -*-

import pandas as pd
import ingreso_datos
import funciones_anla
import estado_algoritmo
from typing import Optional


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


def ejecutar_funcion(objeto: estado_algoritmo.EstadoAlgoritmo) -> estado_algoritmo.EstadoAlgoritmo:
  objeto.principal_funcion()
  print("hola")
  return objeto

def ejecutar_bloque(instancia_algoritmo: Optional[list[estado_algoritmo.EstadoAlgoritmo]]) -> Optional[tuple[list[estado_algoritmo.EstadoAnla], list[estado_algoritmo.EstadoIdeam]]]:
  if instancia_algoritmo is None:
    return None
  print("hola_1")
  print(type(instancia_algoritmo))
  for instance in instancia_algoritmo:
    instance.principal_funcion()
  array_ideam: list[estado_algoritmo.EstadoIdeam] = []
  array_anla: list[estado_algoritmo.EstadoAnla] = []
  hola = 0
  print("hola_2")
  for obj in instancia_algoritmo:
    if isinstance(obj, estado_algoritmo.EstadoIdeam):
      array_ideam.append(obj)
      # obj.data_alter.to_csv('datos_' + str(hola) + '.csv')
      hola += 1
    elif isinstance(obj, estado_algoritmo.EstadoAnla):
      array_anla.append(obj)
  return array_anla, array_ideam

def main():
  instancia_algoritmo: Optional[list[estado_algoritmo.EstadoAlgoritmo]] = ingreso_datos.ejecutar_algoritmo_ruta("setup.json")  
  resultados = ejecutar_bloque(instancia_algoritmo)
  if resultados == None:
    print("Error en el proceso")
    return 
  else:
    array_anla, array_ideam = ejecutar_bloque(instancia_algoritmo)
  export_resultados(array_ideam, 'caudal_ambiental_ideam.csv')
  export_resultados(array_anla, 'caudal_ambiental_anla.csv')
  print("labor finalizada ")


if __name__ == '__main__':
  main()
