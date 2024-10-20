import pandas as pd
from limpieza_datos import process_df  # , ErrorFecha
# import ingreso_datos
import estado_algoritmo
# from typing import Optional
from ingreso_datos import crear_objeto_estado

MY_DICT = {
  "archivos": {
    "archivo_base": -1,
    "archivo_maximos": -1,
    "archivo_minimos": -1,
    "archivo_apoyo": -1,
    "archivo_enso": -1
  },
  "existencia_enso": False,
  "areas": -1,
  "umbrales": -1,
  "estacion_hidrologica": "Nombre_de_la_estacion",
  "organismo": "ideam",
  "existencia_umbrales": False
}
# RUTA_ENSO = "RUTA_ENSO.csv"


def funcion_prueba_miguel(df_crudo: pd.DataFrame) -> pd.DataFrame:
  # MY_DICT["archivos"]["archivo_enso"] = RUTA_ENSO
  datos_usables = process_df(df_crudo)
  instancia_usable: list[estado_algoritmo.EstadoIdeam] = crear_objeto_estado(datos_usables, MY_DICT)
  return instancia_usable[0].umbrales

if __name__ == '__main__':
  df_crudo = pd.read_csv('C:\\Users\\ASUS\\Desktop\\datos\\unal\\semillero\\repo_hecca\\est_25027400.csv')
  print(funcion_prueba_miguel(df_crudo))