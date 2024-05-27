import numpy as np
import pandas as pd


class EstadoAlgoritmo:
  def __init__(self, data_inicial, ruta_m: str):
    self.ruta = ruta_m
    self.data = data_inicial
    self.data_alter = pd.DataFrame()
    self.df2: pd.DataFrame = pd.DataFrame()
    self.primer_dia = 1
    self.dif = 1
    self.final_dia = 1
    self.str_apoyo = ""
    self.df_month_mean = pd.DataFrame()
    self.df_month_mean_rev = pd.DataFrame()
    #self.preparacion_comun()

  def preparacion_comun(self):
    pass

  def principal_funcion(self):
    pass

  def to_csv(self):
    def modificar_nombre_archivo(nombre_archivo: str) -> str:
      # Dividir el nombre del archivo y su extensión
      nombre, extension = nombre_archivo.rsplit('.', 1)
      # Añadir "_arreglado" al nombre del archivo
      nuevo_nombre = nombre + "_arreglado_" + self.str_apoyo + "." + extension
      return nuevo_nombre

    print("hola")
    self.df2.to_csv(modificar_nombre_archivo(self.ruta))
    # self.data_alter.to_csv(self.str_apoyo)


class EstadoIdeam(EstadoAlgoritmo):
  def __init__(self, data_inicial, ruta_m: str, h_umbrales=(None, None)):
    super().__init__(data_inicial, ruta_m)
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
    self.preparacion_comun()
  def preparacion_comun(self):
    from funciones_ideam import buscar_umbrales
    self.df_month_mean = buscar_umbrales(self)

  def principal_funcion(self):
    from funciones_ideam import prin_func
    prin_func(self)
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
  def __init__(self, data_inicial, ruta_m: str):
    super().__init__(data_inicial, ruta_m)
    self.propuesta_caudal: list = [0] * 12
    self.q95: list = [0]*12
    self.q7_10: list = [0]*12
    self.df_cdc_normal: pd.DataFrame = pd.DataFrame()
    self.df_cdc_alterada: pd.DataFrame = pd.DataFrame()
    self.cdc_umbrales: list = [0.70, 0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995]
    self.anios_retorn: list = [2, 5, 10, 25]
    self.cdc_normales: list = [0]*8
    self.cdc_alterados: list = [0]*8
    self.caud_return_normal: list = [0]*4
    self.caud_return_alterado: list = [0] * 4

  def principal_funcion(self):
    from funciones_anla import prin_func
    prin_func(self)
    self.to_csv()

  def actualizar_caudal(self):
    self.data_alter = self.data.copy()
    for i in range(self.data_alter.size):
      self.data_alter.iloc[i] = min(self.data_alter.iloc[i]['Valor'],
                                    self.propuesta_caudal[self.data_alter.iloc[i].name.month - 1])
