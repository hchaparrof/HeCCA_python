import iha_parametros as ih
import pandas as pd


class IhaEstado:
  def __init__(self, data_real: pd.DataFrame | None):
    self.grupo_1 = 1
    self.grupo_2 = 1
    self.grupo_3 = 1
    self.grupo_4 = 1
    self.grupo_5 = 1
    self.data = data_real

    self.start_year = None
    self.end_year = None

  def __eq__(self, other):
    pass

  def __ne__(self, other):
    return not self.__eq__(other)

  def __truediv__(self, otro):
    if isinstance(otro, IhaEstado):
      base = IhaEstado(None)
      base.grupo_1 = self.grupo_1/otro.grupo_1
      base.grupo_2 = self.grupo_2 / otro.grupo_2
      base.grupo_3 = self.grupo_3 / otro.grupo_3
      base.grupo_4 = self.grupo_4 / otro.grupo_4
      base.grupo_5 = self.grupo_5 / otro.grupo_5
      return base
    elif isinstance(otro, (int, float)):
      base = IhaEstado(None)
      base.grupo_1 = self.grupo_1 / otro
      base.grupo_2 = self.grupo_2 / otro
      base.grupo_3 = self.grupo_3 / otro
      base.grupo_4 = self.grupo_4 / otro
      base.grupo_5 = self.grupo_5 / otro
      return base
      # return MiClase(self.valor / otro)
    else:
      return NotImplemented

  def calcular_iha(self):

    self.grupo_1 = ih.Iha_parameter1(self.data, self.start_year, self.end_year)
    self.grupo_2 = ih.Iha_parameter2(self.data, self.start_year, self.end_year)
    self.grupo_3 = ih.Iha_parameter3(self.data, self.start_year, self.end_year)
    self.grupo_4 = ih.Iha_parameter4(self.data, self.start_year, self.end_year)
    self.grupo_5 = ih.Ihaparameter5(self.data)
