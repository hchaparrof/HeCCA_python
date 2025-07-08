from dataclasses import dataclass
from typing import ClassVar, List, Optional

import numpy as np
import pandas as pd
import IhaEstado


class EstadoAlgoritmo:
  def __init__(self, data_inicial: pd.DataFrame, ruta_m: str, anio_hidrologico: int, codigo_est: int):
    self.ruta: str = ruta_m
    self.data: pd.DataFrame = data_inicial
    self.data_alter: pd.DataFrame = pd.DataFrame()
    self.df2: pd.DataFrame = pd.DataFrame()
    self.primer_dia = 1
    self.dif = 1
    self.final_dia = 1
    self.str_apoyo: str = "normal"
    self.df_month_mean: pd.DataFrame = pd.DataFrame()
    self.df_month_mean_rev: pd.DataFrame = pd.DataFrame()
    self.ajuste: int = -1
    self.anio_hidrologico: int = anio_hidrologico
    self.codigo_est: int = codigo_est

  def preparacion_comun(self, *args):
    from funciones_anla import determinar_ajuste
    determinar_ajuste(self)

  def principal_funcion(self):
    pass

  def to_csv(self):
    def modificar_nombre_archivo(nombre_archivo: str) -> str:
      # Dividir el nombre del archivo y su extensión
      nombre, extension = nombre_archivo.rsplit('.', 1)
      # Añadir "_arreglado" al nombre del archivo
      nuevo_nombre = nombre + "_arreglado_" + self.str_apoyo + "." + extension
      return nuevo_nombre

    # print("hola")
    self.df2.to_csv(modificar_nombre_archivo(self.ruta))
    # self.data_alter.to_csv(self.str_apoyo)


class EstadoIdeam(EstadoAlgoritmo):
  def __init__(self, data_inicial: pd.DataFrame, data_dict: dict,
               extremos: list[Optional[pd.DataFrame]] = None, codigo_est: int = -1):
    # print(data_dict)
    super().__init__(data_inicial, data_dict['archivos']['archivo_base'], data_dict['anio_hidrologico'], codigo_est)
    # print(extremos)
    if extremos:
      pass
    else:
      extremos = [pd.DataFrame(), pd.DataFrame()]
    self.data_min: pd.DataFrame = extremos[0]
    self.data_max: pd.DataFrame = extremos[1]
    if data_dict['umbrales'] == -1:
      h_umbrales = (None, None)
    else:
      h_umbrales = data_dict['umbrales']
    self.data_alter = pd.DataFrame()
    self.data_alter2 = pd.DataFrame()
    self.umbrales = {
      'QTR15': -1,
      'Q10': -1,
      'QB': -1,
      'QTQ': -1
    }
    self.df_umbrales = {
      'df_qtq_alt': pd.DataFrame(),
      'df_qtq_ref': pd.DataFrame(),
      'df_qb_alt': pd.DataFrame(),
      'df_qb_ref': pd.DataFrame(),
      'df_q15_ref': pd.DataFrame(),
      'df_q15_alt': pd.DataFrame(),
      'df_q10_ref': pd.DataFrame(),
      'df_q10_alt': pd.DataFrame()
    }
    self.listas_eventos = {
      'eventos_rev_qtr15': [],
      'eventos_rev_qb': [],
      'eventos_rev_qtq': [],
      'eventos_rev_q10': [],
      'eventos_qtr15': [],
      'eventos_qb': [],
      'eventos_qtq': [],
      'eventos_q10': []
    }
    self.h_umbrales = {
      'QB': h_umbrales[0],
      'QTQ': h_umbrales[1]
    }
    self.porcentajes = np.empty(12)
    #todo revisar porcentajes
    self.preparacion_comun()

  def preparacion_comun(self):
    import funciones_ideam
    super().preparacion_comun(self)
    funciones_ideam.umbrales_serie(self, True)
    funciones_ideam.calcular_df_resumen(self)

  def principal_funcion(self):
    # print("principal funcion_ideam")
    import funciones_ideam
    funciones_ideam.prin_func(self)
    self.to_csv()

  def setear_umbrales(self, f_umbrales: list):
    self.umbrales['QTR15'] = f_umbrales[0]
    self.umbrales['QB'] = f_umbrales[1]
    self.umbrales['QTQ'] = f_umbrales[2]
    self.umbrales['Q10'] = f_umbrales[3]

  def reiniciar_alter(self):
    self.df_umbrales['df_qtq_alt'] = pd.DataFrame()
    self.df_umbrales['df_qb_alt'] = pd.DataFrame()
    self.df_umbrales['df_qtq_alt'].assign(mes=None, Magnitud=None, Duracion=None, Intensidad=None)
    self.df_umbrales['df_qb_alt'].assign(mes=None, Magnitud=None, Duracion=None, Intensidad=None)
    self.listas_eventos['eventos_rev_qtr15'].clear()
    self.listas_eventos['eventos_rev_qb'].clear()
    self.listas_eventos['eventos_rev_qtq'].clear()
    self.listas_eventos['eventos_rev_q10'].clear()


class EstadoAnla(EstadoAlgoritmo):
  cdc_umbrales: list = [0.70, 0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995]
  anios_retorn: list = [2, 5, 10, 25]
  def __init__(self, data_inicial: pd.DataFrame, ruta_m: str, anio_hidrologico: int, codigo_est: int, grupos_iha: List[int] | int):
    super().__init__(data_inicial, ruta_m, anio_hidrologico, codigo_est)
    if grupos_iha == -1:
      self.grupos_a_analizar = [3,4,5]
    else: 
      self.grupos_a_analizar = grupos_iha
    self.data_ref: pd.DataFrame = pd.DataFrame()
    self.propuesta_inicial_ref: list[float] = [0] * 12
    self.caud_final: list[float] = [0] * 12
    self.q95: list = [0]*12
    self.q7_10: list = [0]*12
    self.df_cdc_normal: pd.DataFrame = pd.DataFrame()
    self.df_cdc_alterada: pd.DataFrame = pd.DataFrame()
    self.cdc_normales: list = [0]*8
    self.cdc_alterados: list = [0]*8
    self.caud_return_normal: list = [0]*4
    self.caud_return_alterado: list = [0] * 4
    self.resultados_ori: ResultadosAnla
    self.resultados_alterada: ResultadosAnla
    self.resultados_ref: ResultadosAnla
    self.data_alter2: pd.DataFrame

  def principal_funcion(self):
    # print("princial_funcion_anla")
    from funciones_anla import prin_func
    prin_func(self)
    self.to_csv()

  def actualizar_caudal(self):
    self.data_alter = self.data.copy()
    for i in range(self.data_alter.size):
      self.data_alter.iloc[i] = min(self.data_alter.iloc[i]['Valor'],
                                    self.propuesta_inicial_ref[self.data_alter.iloc[i].name.month - 1])


# ////
@dataclass
class ResultadosAnla:
  cdc: pd.DataFrame
  cdc_anios: np.ndarray
  caud_return: list[float]
  iah_result: IhaEstado.IhaEstado
  grupos_iha: ClassVar[List[int] | int]
