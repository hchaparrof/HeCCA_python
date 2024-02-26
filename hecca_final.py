# -*- coding: utf-8 -*-
"""##librerias"""
import ingreso_datos
from estado_algoritmo import *


def main():
    instancia_algoritmo: EstadoAlgoritmo = ingreso_datos.iniciar_algoritmo()
    instancia_algoritmo.principal_funcion()

if __name__ == '__main__':
    main()
