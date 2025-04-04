import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


SHAPEFILE_PATH = "disuelto_anio_hidrologico.gpkg"  # Ruta al archivo .shp
ATTRIBUTE_NAME = "gridcode"  # Nombre del atributo a extraer

def det_anio_hid(df: pd.DataFrame) -> int:
    """
    Determina el atributo asociado al punto con latitud y longitud mínima del DataFrame.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que debe contener las columnas 'Latitud' y 'Longitud'.

    Retorna:
    --------
    int
        Valor entero del atributo obtenido, o -1 si ocurre un error.
    """
    try:
        latitud: float = df['Latitud'].min()
        longitud: float = df['Longitud'].min()
        resultado = get_polygon_attribute(SHAPEFILE_PATH, latitud, longitud, ATTRIBUTE_NAME)
        return int(resultado)
    except (KeyError, TypeError, ValueError, Exception) as e:
        return -1



def get_polygon_attribute(shapefile: str, x: float, y: float, attribute: str) -> int:
    # Cargar el shapefile
    gdf = gpd.read_file(shapefile)
    
    # Crear un punto con las coordenadas dadas
    point = Point(y,x)
    
    # Buscar el polígono que contiene el punto
    polygon = gdf[gdf.geometry.contains(point)]
    
    # Si el punto está dentro de algún polígono, devolver el atributo
    if not polygon.empty:
        return polygon.iloc[0][attribute]
    else:
        return -1



