import json
from typing import Iterable, List, Optional

import pandas as pd
import IhaEstado
import estado_algoritmo
from limpieza_datos import process_df, ErrorFecha
import copy
from mapa_anio import det_anio_hid


def ejecutar_algoritmo_ruta(ruta_json: str) -> Optional[list[estado_algoritmo.EstadoAlgoritmo]]:
  """
  reciba una ruta valida hacia un archivo json de configuracion valido
  devuelve una lista con instancias de algoritmo.
  @param ruta_json: (string) ruta del archivo json valido
  @return: list[estado_algoritmo.EstadoAlgoritmo] lista con las instancias del algoritmo a ejecutar
  """
  with open(ruta_json, "r") as archivo_json:
    datos = json.load(archivo_json)
  return generar_algoritmo(datos)


def procesar_datos(base: pd.DataFrame, apoyo: Optional[pd.DataFrame] = None,
                   areas: Optional[list] = None, anios_utiles: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
  """
  toma las series de base y apoyo y retorna un df limpio sin datos faltantes, ni anomalos.
  @param base: (pd.DataFrame) serie de datos a análiza
  @param apoyo: serie de datos cuenca de apoyo para rellenar, None si no hay
  @param areas: las areas de las cuencas para rellenar datos, None si no hay
  @return: df limpio con la serie a análizar
  """
  try:
    df_limpio = process_df(base, apoyo, areas, anios_utiles)
  except ErrorFecha as e:
    print(e)
    return None
  return df_limpio


def crear_objeto_estado(df_limpio: pd.DataFrame, datos: dict, codigo_est: int) -> Optional[List[estado_algoritmo.EstadoAlgoritmo]]:
  """

  @param df_limpio: serie de datos sin datos faltantes.
  @param datos: diccionario de configuración valido
  @param codigo_est: entero, el código de la estación en el ideam
  @return: list[estado_algoritmo.EstadoAlgoritmo] una lista de 1 o 2 elementos EstadoAlgoritmo, ya sea EstadoAnla, EstadoIdeam o
  """
  objetos_estado: List[estado_algoritmo.EstadoAlgoritmo] = []
  print(datos['organismo'], 'organismo')
  organismo = datos.get('organismo')

  if organismo not in {'anla', 'ideam', 'ambas'}:
    print("Error: el valor de 'organismo' no es válido. Debe ser 'anla', 'ideam' o 'ambas'.")
    return None
  if datos['organismo'] != 'ideam':
    objetos_estado.append(
      estado_algoritmo.EstadoAnla(
        df_limpio,
        datos['archivos']['archivo_base'],
        datos['anio_hidrologico'],
        codigo_est,
        datos['grupos_iha']
      )
    )

  if datos['organismo'] != 'anla':
    extremos: list[pd.DataFrame] = []
    minimo_prov = None
    maximo_prov = None
    if datos['archivos']['archivo_maximos'] == -1 and datos['archivos']['archivo_minimos'] == -1:
      extremos = [pd.DataFrame(), pd.DataFrame()]
    else:
      if datos['archivos']['archivo_maximos'] != -1:
        try:
          maximo_prov: Optional[pd.DataFrame] = pd.read_csv(datos['archivos']['archivo_maximos'])
          maximo_prov = (procesar_datos(base=maximo_prov))
        except ValueError:
          maximo_prov = None
        except:
          print("otro_error_lectura_maximos")
          maximo_prov = None
        finally:
          pass
      if datos['archivos']['archivo_minimos'] != -1:
        try:
          minimo_prov: Optional[pd.DataFrame] = pd.read_csv(datos['archivos']['archivo_minimos'])
          minimo_prov = (procesar_datos(base=maximo_prov))
        except ValueError:
          minimo_prov = None
        except:
          print("otro_error_lectura_minimos")
        finally:
          pass
      try:
        extremos.append(minimo_prov)
      except AttributeError:
        extremos.append(pd.DataFrame())
        pass
      finally:
        pass
      try:
        extremos.append(maximo_prov)
      except AttributeError:
        extremos.append(pd.DataFrame())
      finally:
        pass
    if not datos['umbrales'] == -1:
      objetos_estado.append(
        estado_algoritmo.EstadoIdeam(df_limpio, datos,
                                     extremos=extremos, codigo_est=codigo_est))  # ['archivos']['archivo_base'], datos['umbrales']))
    else:
      objetos_estado.append(estado_algoritmo.EstadoIdeam(df_limpio, datos, codigo_est=codigo_est))
  return objetos_estado


def generar_algoritmo(datos: dict) -> Optional[tuple[Optional[list[estado_algoritmo.EstadoAnla]], Optional[list[estado_algoritmo.EstadoIdeam]]]]:
  """
  Toma un diccionario valido de configuración y retorna una lista de instancias del algoritmo a ejecutar
  @param datos:
  @return: Optional[list[estado_algoritmo.EstadoAlgoritmo]] lista de las instancias del algoritmo a ejecutar,
    pueden ser desde 1 hasta 6 elementos en la lista
  """
  print(datos)
  if datos['grupos_iha'] == -1:
    IhaEstado.iha_grupos = [3,4,5]
  elif isinstance(datos['grupos_iha'], Iterable):
    IhaEstado.iha_grupos = datos['grupos_iha']
  anios_utiles = None if datos['anios_utiles'] == -1 else datos['anios_no_utilos']
  base: pd.DataFrame = pd.read_csv(datos['archivos']['archivo_base'])
  # print(datos)
  apoyo: Optional[pd.DataFrame] = pd.read_csv(datos['archivos']['archivo_apoyo']) if (datos['archivos']['archivo_apoyo'] != -1) else None
  areas: List[float] = datos['areas'] if (datos['areas'] != -1) else None
  codigo_est: int = base['CodigoEstacion'].min()
  df_limpio: Optional[pd.DataFrame] = procesar_datos(base, apoyo, areas, anios_utiles)
  print(set(df_limpio.index.year))
  if df_limpio is None:
    return None
  if datos['anio_hidrologico'] == -1:
    datos['anio_hidrologico'] = det_anio_hid(base)
  objeto_base: List[estado_algoritmo.EstadoIdeam | estado_algoritmo.EstadoAnla] = crear_objeto_estado(df_limpio, datos, codigo_est)

  if not datos.get("existencia_enso"):
    if isinstance(objeto_base, estado_algoritmo.EstadoIdeam):
      return None, objeto_base
    return objeto_base, None

  lista_anla: List[estado_algoritmo.EstadoAnla] = []
  lista_ideam: List[estado_algoritmo.EstadoIdeam] = []

  for obj in objeto_base:
    if isinstance(obj, estado_algoritmo.EstadoAnla):
      lista_anla = crear_lista(obj, datos['archivos']['archivo_enso'])
    elif isinstance(obj, estado_algoritmo.EstadoIdeam):
      lista_ideam = crear_lista(obj, datos['archivos']['archivo_enso'])

  return lista_anla, lista_ideam


def generar_algoritmo_json() -> Optional[list[estado_algoritmo.EstadoAlgoritmo]]:
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
    elif datos['organismo'] == 'ideam':
      if datos['existencia_umbrales']:
        objeto_base = estado_algoritmo.EstadoIdeam(df_limpio, datos)  # ['archivos']['archivo_base'], datos['umbrales'])
      else:
        objeto_base = estado_algoritmo.EstadoIdeam(df_limpio, datos['archivos']['archivo_base'])
    if datos["existencia_enso"]:
      return crear_lista(objeto_base, datos['archivos']['archivo_enso'])  # datos["archivo_enso"])
    return [objeto_base]


def crear_lista(estado: estado_algoritmo.EstadoAlgoritmo, enso_csv: str) -> list[estado_algoritmo.ResultadosAnla | estado_algoritmo.EstadoIdeam]:
  meses_list = [
    'DJF',
    'JFM',
    'FMA',
    'MAM',
    'AMJ',
    'MJJ',
    'JJA',
    'JAS',
    'ASO',
    'SON',
    'OND',
    'NDJ'
  ]
  # Hacemos una copia profunda de los estados para cada categoría ENSO
  estado_normal = copy.deepcopy(estado)
  estado_ninio = copy.deepcopy(estado)
  estado_ninia = copy.deepcopy(estado)
  # Leer el contenido del archivo CSV
  datos_enso = pd.read_csv(enso_csv, index_col=0)
  # Inicializar DataFrames vacíos para cada categoría
  datos_nino = pd.DataFrame()
  datos_nina = pd.DataFrame()
  datos_normal = pd.DataFrame()
  # Iterar sobre cada año en los datos ENSO
  for year in datos_enso.index:
    for month in range(1, 13):  # De enero (1) a diciembre (12)
      enso_value = datos_enso.loc[year, meses_list[month - 1]]
      if enso_value == -1:
        datos_nino = pd.concat(
          [datos_nino, estado.data[(estado.data.index.year == year) & (estado.data.index.month == month)]])
      elif enso_value == 1:
        datos_nina = pd.concat(
          [datos_nina, estado.data[(estado.data.index.year == year) & (estado.data.index.month == month)]])
      else:
        datos_normal = pd.concat(
          [datos_normal, estado.data[(estado.data.index.year == year) & (estado.data.index.month == month)]])
  # Asignar los datos correspondientes a cada estado
  estado_normal.data = datos_normal
  estado_normal.str_apoyo = "normal"
  estado_ninio.data = datos_nino
  estado_ninio.str_apoyo = "ninio"
  estado_ninia.data = datos_nina
  estado_ninia.str_apoyo = "ninia"
  return [estado_normal, estado_ninia, estado_ninio]
