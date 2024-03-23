# -*- coding: utf-8 -*-
"""##librerias"""
import ingreso_datos
import estado_algoritmo
import concurrent.futures


def main():
  instancia_algoritmo: list[estado_algoritmo.EstadoAlgoritmo] = ingreso_datos.generar_algoritmo_json()
  with concurrent.futures.ThreadPoolExecutor() as executor:
    resultados = executor.map(lambda estado: estado.principal_funcion(), instancia_algoritmo)
  # for estado in instancia_algoritmo:
  #   estado.principal_funcion()


if __name__ == '__main__':
  main()
