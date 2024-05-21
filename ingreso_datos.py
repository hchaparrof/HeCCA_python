import json
import pandas as pd
import estado_algoritmo
from limpieza_datos import process_df, ErrorFecha
import copy


def generar_algoritmo_json() -> list[estado_algoritmo.EstadoAlgoritmo] | None:
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
    try:
      df_limpio: pd.DataFrame | None = process_df(base, apoyo, areas)
    except ErrorFecha as e:
      print(e)
      return None
    objeto_base: estado_algoritmo.EstadoAlgoritmo
    if datos['organismo'] == 'anla':
      objeto_base = estado_algoritmo.EstadoAnla(df_limpio, datos['archivos']['archivo_base'])
    else:
      if datos['existencia_umbrales']:
        objeto_base = estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'], datos['umbrales'])
      else:
        objeto_base = estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'])
    if datos["existencia_enso"]:
      return crear_lista(objeto_base, datos['archivos']['archivo_enso'])  # datos["archivo_enso"])
    return [objeto_base]

# todo esta funcion no esta actualizada como la otra


def generar_algoritmo_fn(datos: (str, str), areas: tuple = None, umbrales: tuple = (None, None),
                         organismo: str = "ideam", enso: tuple = None) -> estado_algoritmo.EstadoAlgoritmo | None:
  apoyo = None
  try:
    base = pd.read_csv(datos[0])
  except FileNotFoundError:
    print("No hay ningun archivo base")
    return None
  if enso is not None:
    # todo lo que es de Enso
    pass
  if len(datos) > 1:
    # si hay apoyo
    try:
      apoyo = pd.read_csv(datos[1])
    except FileNotFoundError:
      print("No hay ningun archivo de apoyo")
      apoyo = None
  base = process_df(base, apoyo, areas)
  if organismo == "ideam":
    return estado_algoritmo.EstadoIdeam(base, datos[0], umbrales)
  elif organismo == "anla":
    return estado_algoritmo.EstadoAnla(base, datos[0])


def crear_lista(estado: estado_algoritmo.EstadoAlgoritmo, enso: str) -> list[estado_algoritmo.EstadoAlgoritmo]:
  estado_normal = copy.deepcopy(estado)
  estado_ninio = copy.deepcopy(estado)
  estado_ninia = copy.deepcopy(estado)
  with open(enso, 'r') as file:
    # Leer el contenido del archivo CSV
    datos_enso = json.load(file)
  nino_set = set(datos_enso['ninio'])
  nina_set = set(datos_enso['ninia'])
  datos_nino = estado_ninio.data[estado_ninio.data.index.year.isin(nino_set)].copy()
  datos_nina = estado_ninia.data[estado_ninia.data.index.year.isin(nina_set)].copy()
  if datos_enso['normal'] == -1:
    datos_normal = estado_normal.data[~(estado_normal.data.index.year.isin(nino_set) |
                                        estado_normal.data.index.year.isin(nina_set))].copy()
  else:
    normal_set = set(datos_enso['normal'])
    datos_normal = estado_normal.data[estado_normal.data.index.year.isin(normal_set)].copy()
  estado_normal.data = datos_normal
  estado_normal.str_apoyo = "normal"
  estado_ninio.data = datos_nino
  estado_ninio.str_apoyo = "ninio"
  estado_ninia.data = datos_nina
  estado_ninia.str_apoyo = "ninia"
  return [estado_normal, estado_ninia, estado_ninio]
