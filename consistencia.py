import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import json


def filtrar_datos(df, filter_dicts, min_valor, max_valor):
    """
    Filtra los datos del DataFrame según el rango general y excepciones definidas en el diccionario.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos con columnas 'Fecha' y 'slope'.
    filter_dicts (dict): Diccionario que contiene listas de años para 'ninio' y 'ninia'.
    min_valor (float): Valor mínimo para el filtro general.
    max_valor (float): Valor máximo para el filtro general.

    Retorna:
    tuple: Dos pandas Series, la primera contiene fechas fuera del rango filtrado y la segunda contiene fechas dentro del rango.
    """
    # Filtrar por rango general
    dentro_rango_df = df[(df['slope'] >= min_valor) & (df['slope'] <= max_valor)]

    # Aplicar excepciones de ninio (puede ser menor)
    ninio_anios = filter_dicts['ninio']
    excepciones_ninio = df[(df['Fecha'].isin(ninio_anios)) & (df['slope'] < min_valor)]

    # Aplicar excepciones de ninia (puede ser mayor)
    ninia_anios = filter_dicts['ninia']
    excepciones_ninia = df[(df['Fecha'].isin(ninia_anios)) & (df['slope'] > max_valor)]

    # Combinar todos los datos dentro del rango y las excepciones
    dentro_rango_df = pd.concat([dentro_rango_df, excepciones_ninio, excepciones_ninia]).drop_duplicates().reset_index(
        drop=True)

    # Ordenar por fecha
    dentro_rango_df = dentro_rango_df.sort_values(by='Fecha').reset_index(drop=True)
    dentro_rango_df1 = dentro_rango_df['Fecha']

    # Filtrar las fechas fuera del rango utilizando dentro_rango_df1
    fuera_rango_df = df[~df['Fecha'].isin(dentro_rango_df1)]

    dentro_rango_df1 = dentro_rango_df['Fecha']
    fuera_rango_df1 = fuera_rango_df['Fecha']
    return fuera_rango_df1, dentro_rango_df1


def procesar_archivos_csv(csv_folder, json_file):
    """
    Procesa y crea todos los archivos CSV en una carpeta, de caudal acumulado, años inconsistentes
     y consitentes, ademas realiza las graficas con el análisis de regresión lineal,

    Parámetros:
    csv_folder (str): Ruta a la carpeta que contiene los archivos CSV.
    json_file (str): Ruta al archivo JSON que contiene las excepciones para los años 'ninio' y 'ninia'.
    """
    output_dir = 'GraficasPendientes'
    # Verificar y crear el directorio de salida si no existe
    # Crear carpetas si no existen
    for folder in [output_dir, 'Acumulado', 'CSV']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Listar todos los archivos CSV en la carpeta
    for file_name in os.listdir(csv_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_folder, file_name)
            # Leer el csv
            data = pd.read_csv(file_path, parse_dates=['Fecha'], dayfirst=True, low_memory=False)
            # Obtener el nombre base del archivo sin la extensión
            base_name = os.path.splitext(file_name)[0]

            # Asegurarse de que 'Fecha' es datetime
            data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce', dayfirst=True)

            # Verificar si hay valores nulos en 'Fecha' y 'Valor' después de la conversión y eliminarlos
            data = data.dropna(subset=['Fecha', 'Valor'])
            # Asegurarse de que 'Valor' es numérico
            data['Valor'] = pd.to_numeric(data['Valor'], errors='coerce')

            # Crear data frame que va a contener la pendiente
            df = pd.DataFrame()
            df = df.assign(Fecha=None, slope=None, std_err=None, r_values=None)
            # Obtener los años únicos en la columna 'Fecha'
            years = data['Fecha'].dt.year.unique()
            datos = pd.DataFrame()
            array = []

            # Crear el DataFrame dfAcumu
            dfAcumu = pd.DataFrame({
                'Fecha': data['Fecha'],
                'Valor': data['Valor'].cumsum()
            })
            # Extraer el año de la fecha
            dfAcumu['Año'] = dfAcumu['Fecha'].dt.year

            # Calcular el caudal acumulado acumulativo por año en dfAcumu_anual
            dfAcumu_anual = dfAcumu.groupby('Año')['Valor'].max().reset_index()

            # Carpeta donde se guardará el archivo
            carpeta_salida = "Acumulado"
            # Nombre completo del archivo CSV
            archivo_salida = os.path.join(carpeta_salida, f"{base_name}_CauAcum.csv")
            # Guardar el DataFrame en un archivo CSV
            dfAcumu_anual.to_csv(archivo_salida, index=False)
            # Iterar a través de los años y hacer lo que necesites con los datos de cada año
            for year in years:
                # Filtrar los datos para el año actual
                data_year = data[data['Fecha'].dt.year == year]
                # Datos acumulados
                datos['caudal_acumulado'] = data_year['Valor'].cumsum()
                # Regresión lineal
                slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(datos)),
                                                                         datos['caudal_acumulado'])
                array.append((slope, std_err, year, r_value,))
                datos = pd.DataFrame()
            for i in array:
                row = [i[2], i[0], i[1], i[3]]
                df.loc[len(df)] = row
            # Calcular el promedio y la desviación estándar de las pendientes
            mean_slope = df['slope'].mean()
            std_slope = df['slope'].std()
            min_valor = mean_slope - std_slope
            max_valor = mean_slope + std_slope

            # JSON proporcionado
            # Cargar JSON desde el archivo
            with open(json_file, 'r') as file:
                filter_dicts = json.load(file)

            Datos_anomalos, Datos_filtrados = filtrar_datos(df, filter_dicts, min_valor, max_valor)

            # Guardar en csv
            Datos_filtrados.to_csv(f'Csv/{base_name}_SerieFinalRevisada.csv', index=False)
            Datos_anomalos.to_csv(f'Csv/{base_name}_añosatipicos.csv', index=False)

            # Configurar las gráficas para el primer DataFrame
            fig, axs = plt.subplots(1, 1, figsize=(25, 7),
                                    sharex=True)  # Aumentar el tamaño de la figura horizontalmente
            fig.suptitle(f'Gráficas del archivo: {file_name}', fontsize=16)
            # Gráfica de slope
            axs.plot(df['Fecha'], df['slope'], marker='o', linestyle='-')
            axs.axhline(mean_slope, color='r', linestyle='--', label='Promedio')
            axs.axhline(mean_slope + std_slope, color='g', linestyle='--', label='Promedio + Desv. Estándar')
            axs.axhline(mean_slope - std_slope, color='b', linestyle='--', label='Promedio - Desv. Estándar')
            axs.set_title('Slope vs Tiempo')
            axs.set_ylabel('Slope')
            axs.grid(True)
            axs.legend()

            # Asegurarse de que los años se muestren correctamente en el eje x
            plt.xticks(df['Fecha'].astype(int))

            # Rotar las etiquetas del eje x para que no se amontonen
            plt.setp(axs.get_xticklabels(), rotation=45, ha='right')
            # Mostrar las gráficas
            plt.tight_layout()

            # Guardar la primera figura como PNG con el nombre del archivo y sufijo _slope
            fig.savefig(os.path.join(output_dir, f'{base_name}_slope.png'))

            plt.show()

            fig2, ax2 = plt.subplots(figsize=(15, 6), sharex=True)
            ax2.plot(dfAcumu_anual['Año'], dfAcumu_anual['Valor'], marker='o', linestyle='-',
                     linewidth=1)  # Ajustar el grosor de la línea
            plt.title('Caudal Acumulado vs Tiempo')
            plt.xlabel('Fecha')
            plt.ylabel('Caudal Acumulado')
            plt.grid(True)

            # Configurar el eje x para mostrar cada año de manera explícita
            ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ajustar el localizador para enteros
            ax2.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: '{:d}'.format(int(x))))  # Formatear como enteros

            # Ajustar el tamaño de la fuente de los ticks del eje x
            plt.xticks(fontsize=10)
            # Rotar las etiquetas del eje x para que no se amontonen
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            # Mostrar la gráfica
            plt.tight_layout()

            # Guardar la segunda figura como PNG con el nombre del archivo y sufijo _caudal_acumulado
            fig2.savefig(os.path.join(output_dir, f'{base_name}_caudal_acumulado.png'))

            plt.show()

def homogeneidad_helmert(df: pd.DataFrame) -> bool:
  datos_usables_anuales  = df.groupby(df.index.year).mean()
  promedio_anual = datos_usables_anuales['Valor'].mean()
  datos_usables_anuales['mayor_promedio'] = (datos_usables_anuales['Valor'] > promedio_anual).astype(int)
  datos_usables_anuales['helmert'] = np.nan
  for i in range(1, datos_usables_anuales.index.size):
    if datos_usables_anuales.iat[i,1] == datos_usables_anuales.iat[i-1,1]:
      datos_usables_anuales.iat[i, 2] = 1
    else:
      datos_usables_anuales.iat[i, 2] = -1
  suma_de_helmert = datos_usables_anuales['helmert'].sum()
  otro_dato = np.sqrt(datos_usables_anuales.index.size-1)
  return suma_de_helmert < otro_dato

def homogeneidad_kendall(df: pd.DataFrame) -> bool:
  VCRITICA = 1.64
  datos_usables_anuales = df.groupby(df.index.year).mean()
  datos_usables_anuales['SI'] = np.nan
  datos_usables_anuales['TI'] = np.nan
  for i in range( datos_usables_anuales.index.size):
    valor_inicial = datos_usables_anuales.iat[i, 0]
    datos_usables_anuales.iat[i, 1] = datos_usables_anuales[datos_usables_anuales['Valor'] > valor_inicial].index.size
    datos_usables_anuales.iat[i, 2] = datos_usables_anuales[datos_usables_anuales['Valor'] < valor_inicial].index.size
  sum_si = datos_usables_anuales['SI'].sum()
  sum_ti = datos_usables_anuales['TI'].sum()
  s_mayor = sum_si - sum_ti
  return s_mayor < VCRITICA

# Ejemplo de llamada a la función
# csv_folder = 'estaciones_consistencia_csv'
# json_file = 'enso.json'
#
# procesar_archivos_csv(csv_folder, json_file)