import json
import pandas as pd
import estado_algoritmo
from limpieza_datos import process_df


def generar_algoritmo_json() -> list[estado_algoritmo.EstadoAlgoritmo]:
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
    objeto_base: estado_algoritmo.EstadoAlgoritmo
    if datos['organismo'] == 'anla':
      objeto_base = estado_algoritmo.EstadoAnla(df_limpio, datos['archivos']['archivo_base'])
    else:
      if datos['existencia_umbrales']:
        objeto_base = estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'], datos['umbrales'])
      else:
        objeto_base = estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'])
    if datos["existencia_enso"]:
      return crear_lista(objeto_base, datos["archivo_enso"])
    return [objeto_base]


def generar_algoritmo_fn(datos: (str, str), areas: tuple = None, umbrales: tuple = (None, None),
                         organismo: str = "ideam", enso: tuple = None) -> estado_algoritmo.EstadoAlgoritmo | None:
  apoyo = None
  try:
    base = pd.read_csv(datos[0])
  except:
    print("No hay ningun archivo base")
    return None
  if enso is not None:
    # todo lo que es de Enso
    pass
  if len(datos) > 1:
    # si hay apoyo
    try:
      apoyo = pd.read_csv(datos[1])
    except:
      print("No hay ningun archivo de apoyo")
      apoyo = None
  base = process_df(base, apoyo, areas)
  if organismo == "ideam":
    return estado_algoritmo.EstadoIdeam(base, datos[0], umbrales)
  elif organismo == "anla":
    return estado_algoritmo.EstadoAnla(base, datos[0])


def crear_lista(estado: estado_algoritmo.EstadoAlgoritmo, enso: str) -> list[estado_algoritmo.EstadoAlgoritmo]:
  datos_csv = estado.data
  with open(enso, 'r') as file:
    # Leer el contenido del archivo CSV
    datos = file.read()
  nino_set = set(datos['niño'])
  nina_set = set(datos['niña'])
  datos_nino = datos_csv[datos_csv.index.year.isin(nino_set)].copy()
  datos_nina = datos_csv[datos_csv.index.year.isin(nina_set)].copy()
  if datos['normal'] == -1:
    datos_normal = datos_csv[~(datos_csv.index.year.isin(nino_set) & datos_csv.index.year.isin(nina_set))].copy()
  else:
    normal_set = nina_set = set(datos['normal'])
    datos_normal = datos_csv[datos_csv.index.year.isin(normal_set)].copy()
  return [datos_nino, datos_nina, datos_normal]
