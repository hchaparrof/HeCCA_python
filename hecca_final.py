# -*- coding: utf-8 -*-
"""##librerias"""
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


if __name__ == '__main__':
  main()
