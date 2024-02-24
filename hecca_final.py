# -*- coding: utf-8 -*-
"""##librerias"""
import ingreso_datos
from estado_algoritmo import *


if __name__ == '__main__':
    instancia_algoritmo: EstadoAlgoritmo = ingreso_datos.iniciar_algoritmo()
    instancia_algoritmo.principal_funcion()
