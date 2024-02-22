class EstadoAlgoritmo:
  def __init__(self):
    self.data = None
    self.data_alterada = None
    self.df2 = None
    self.primer_dia = None
    self.dif = None
    self.final_dia = None


class EstadoIdeam(EstadoAlgoritmo):
  def __init__(self):
    super().__init__()
    self.data_alter = None
    self.data_alter2 = None
    self.umbrales = {
      'QTR15': None,
      'Q10': None,
      'QB': None,
      'QTQ': None
    }
    self.df_umbrales = {
      'df_qtq_alt': None,
      'df_qtq_ref': None,
      'df_qb_alt': None,
      'df_qb_ref': None
    }
    self.listas_eventos = {
      'eventos_rev_qtr15': None,
      'eventos_rev_qb': None,
      'eventos_rev_qtq': None,
      'eventos_rev_q10': None
    }


class EstadoAnla(EstadoAlgoritmo):
  def __init__(self):
    super().__init__()
    self.primera_iteracion = None
    self.q95 = None
    self.q7_10 = None
