from typing import Optional

import iha_parametros as ih
import pandas as pd


class IhaEstado:
  def __init__(self, data_real: Optional[pd.DataFrame]):
    self.grupo_1 = 1
    self.grupo_2 = 1
    self.grupo_3 = 1
    self.grupo_4 = 1
    self.grupo_5 = 1
    self.data_cruda = data_real
    self.data: Optional[pd.DataFrame] = None
    self.start_year = None
    self.end_year = None

  def __eq__(self, other):
    pass

  def __ne__(self, other):
    return not self.__eq__(other)

  def unir_grupos(elemento):
    df_unido = pd.concat(
      [elemento.grupo_1, elemento.grupo_2, elemento.grupo_3, elemento.grupo_4,
       elemento.grupo_5], axis=0, ignore_index=True)
    return df_unido

  def __sub__(self, other):
    if isinstance(other, IhaEstado):
      base = IhaEstado.unir_grupos(self)
      otro = IhaEstado.unir_grupos(other)
      serie_resultado = ((base['Valor'] - otro['Valor']).abs()) / base['std']
      return list[serie_resultado]
    return NotImplemented

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
    self.data, self.start_year, self.end_year = ih.set_data(self.data_cruda)
    print("prueba_cambio")
    # self.start_year = self.data.index.year.min()
    # self.end_year = self.data.index.year.max()
    print(self.start_year, self.end_year)
    self.grupo_1 = ih.Iha_parameter1(self.data, self.start_year, self.end_year)
    self.grupo_2 = ih.Iha_parameter2(self.data, self.start_year, self.end_year)
    self.grupo_3 = ih.Iha_parameter3(self.data, self.start_year, self.end_year)
    self.grupo_4 = ih.Iha_parameter4(self.data, self.start_year, self.end_year)
    self.grupo_5 = ih.Iha_parameter5(self.data)
