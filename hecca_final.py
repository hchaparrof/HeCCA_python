# -*- coding: utf-8 -*-
"""##librerias"""
import ingreso_datos
import estado_algoritmo


def main():
  instancia_algoritmo: list[estado_algoritmo.EstadoAlgoritmo] = ingreso_datos.generar_algoritmo_json()
  for estado in instancia_algoritmo:
    estado.principal_funcion()


if __name__ == '__main__':
  main()
