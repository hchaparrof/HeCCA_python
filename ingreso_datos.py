import pandas as pd
import json
import estado_algoritmo
from limpieza_datos import process_df


def iniciar_algoritmo() -> estado_algoritmo.EstadoAlgoritmo:
  with open("setup.json", "r") as archivo_json:
    datos = json.load(archivo_json)
    base = pd.read_csv(datos['archivos']['archivo_base'])
    apoyo = None
    areas = None
    if datos['existencia_archivo_apoyo']:
      if datos['existencia_areas']:
        areas = datos['areas']
      else:
        areas = None
      apoyo = pd.read_csv(datos['archivo_apoyo'])
    df_limpio = process_df(base, apoyo, areas)
    if datos['organismo'] == 'anla':
      return estado_algoritmo.EstadoAlgoritmo(df_limpio, datos['archivos']['archivo_base'])
    else:
      if datos['existencia_umbrales']:
        return estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'], datos['umbrales'])
      return estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'])


'''def crear_lista(estado: estado_algoritmo.EstadoAlgoritmo, enso: pd.Dataframe) -> list:
  df_completo = estado.data
'''
