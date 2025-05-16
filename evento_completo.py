import pandas as pd
class EventoCompleto:
  """
    Reune todas las caracteristicas del evento como
    magnitud, intensidad, etc. también reune el evento en sí en un df.
    """
  mes = -1  # es el mes en que se encuentra el evento, no existen eventos multimensuales
  magnitud = -1.0  # la suma de los caudales excedentes al umbral del evento, m3/s
  intensidad = -1.0  # magnitud del evento sobre la duracion m3/(s*día)
  duracion = -1  # duracion en días del evento (día)

  def __init__(self, df1: pd.DataFrame, umbral: float, umbralstr: str):
    self.df1: pd.DataFrame = df1.copy()
    self.umbral: float = umbral
    self.umbralstr: str = umbralstr
    self.organizar_df()
    self.set_intensidad()
    self.organizar_mes()

  def organizar_df(self):
    """
    Toma el DataFrame con los datos del evento y le
    añade la columba unbralstr en la que va a estar el excedente del caudal sobre el umbral
    """
    self.df1 = self.df1[['Valor', self.umbralstr]]
    self.df1.loc[:, self.umbralstr] = abs(self.df1['Valor'] - self.umbral)

  def set_intensidad(self):
    """Determina los parametros hidrologicos del evento, magnitud duración e intensidad"""
    self.magnitud = self.df1[self.umbralstr].sum()
    self.duracion = self.df1[self.umbralstr].size
    self.intensidad = self.magnitud / self.duracion

  def organizar_mes(self):
    """
    Establece el mes en el que se encuentra el evento, como no
    existen eventos multimensuales se toma cualquier dato de mes
    """
    self.mes = self.df1.index.month.min()
