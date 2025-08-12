import os
import pandas as pd
from typing import List
from typing import Optional

ruta_base: str = './'

CARPETA_TRABAJO: str = os.path.join(ruta_base, 'resultados_4', 'ideam')
CARPETA_RESULTADOS: str = os.path.join(ruta_base, 'ambientales compilados', 'mads')
# resultados_anla_2/anla/otros/ninia/df_cdc_alterada_44017060.csv
cuencas: List[int] = [
    54077010,
    44017060,
    32067030,
    29067150,
    22027020
]

subcarpetas = ["ninia", "ninio", "normal"]


def construir_ruta_archivo(subcarpeta: str, cuenca: int) -> str:
    nombre_archivo = f"data_alter_{cuenca}.csv"
    ruta = os.path.join(CARPETA_TRABAJO, subcarpeta, nombre_archivo)
    return ruta if os.path.isfile(ruta) else None


def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    primera_col = df.columns[0]
    df[primera_col] = pd.to_datetime(df[primera_col], errors='coerce')
    df = df.set_index(primera_col)
    df.index.name = "Fecha"
    return df


def leer_archivo(ruta: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(ruta)
        return preparar_dataframe(df)
    except Exception as e:
        print(f"Error leyendo {ruta}: {e}")
        return None


def unir_df(caudales_array: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if caudales_array:
        df_total: pd.DataFrame = pd.concat(caudales_array).sort_index()
        caudales_resultado: pd.DataFrame = df_total.groupby(df_total.index).max()
        return caudales_resultado
    else:
        print(f"No hay datos para unir")
        return None


def exportardf_completo(caudales_array: List[pd.DataFrame], nombre_export: str) -> None:
    str_export: str = os.path.join(CARPETA_RESULTADOS, f"{nombre_export}.csv")
    if caudales_array:
        caudales_resultado: pd.DataFrame = unir_df(caudales_array)
        caudales_resultado.to_csv(str_export)
        print(f"Exportado: {str_export}")
    else:
        print(f"No hay datos para exportar: {nombre_export}")


if __name__ == "__main__":
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    for cuenca in cuencas:
        dataframes = []
        for subcarpeta in subcarpetas:
            ruta_archivo = construir_ruta_archivo(subcarpeta, cuenca)
            if ruta_archivo:
                df = leer_archivo(ruta_archivo)
                if df is not None:
                    dataframes.append(df)
            else:
                print(f"No encontrado: {subcarpeta}/data_alter_{cuenca}.csv")

        # Unificamos los tres archivos de la cuenca y exportamos solo uno
        exportardf_completo(dataframes, nombre_export=f"data_alter_{str(cuenca)}")