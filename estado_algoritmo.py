import numpy as np
import pandas as pd


class EstadoAlgoritmo:
  def __init__(self, data_inicial):
    self.data = data_inicial
    self.data_alter = pd.DataFrame()
    self.df2 = pd.DataFrame()
    self.primer_dia = 1
    self.dif = 1
    self.final_dia = 1

  def principal_funcion(self):
    pass


class EstadoIdeam(EstadoAlgoritmo):
  def __init__(self, data_inicial, h_umbrales=None):
    super().__init__(data_inicial)
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
      'df_qb_ref': pd.DataFrame()
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

  def principal_funcion(self):
    from funciones_ideam import prin_func
    prin_func(self)

  def setear_umbrales(self, f_umbrales: list):
    self.umbrales['QTR15'] = f_umbrales[0]
    self.umbrales['QB'] = f_umbrales[1]
    self.umbrales['QTQ'] = f_umbrales[2]
    self.umbrales['Q10'] = f_umbrales[3]


class EstadoAnla(EstadoAlgoritmo):
  def __init__(self, data_inicial):
    super().__init__(data_inicial)
    self.primera_iteracion = None
    self.q95 = None
    self.q7_10 = None

  def principal_funcion(self):
    from funciones_anla import prin_func
    prin_func(self)