Código completo del progreso de la herramienta hecca en python, incluye la implementación de la herramienta para el calculo del caudal ambiental por la metodología del ideam y se preveé sumarle la metodología del anla.

Uso:
En el archivo setup.json cambiar los datos de archivo_base, archivo_apoyo y archivo_enso, por las ubicaciones de los archivos en su computador.
En el archivo setup.json llenar los datos que sean necesarios, en caso de que "existencia_areas" o "existencia_umbrales" sean "false" no importa los valores de "areas" o "umbrales".
"organismo" establece el algoritmo a utilizar, existen 2 metodologías diferentes: IDEAM o ANLA, y se selecciona como "ideam" y "anla" respectivamente.
"areas" Se refiere a las áreas de las cuencas de estudio para las dos estaciones, la base y la de apoyo, en caso de que no se tenga cuenca de apoyo no importa el valor y si no se tiene areas se debe poner false en "existencia_areas", se asume que las áreas estan en m3/s, pero desde que esten en la misma dimensión no debería ser importante.
"umbrales" Se refiere a los umbrales morfometricos de la cuenca de estudio en el lugar de estudio, son los umbrales QB y QTQ o caudal de banca llena y caudal de perdida de conectividad, eso se consigue con un estudio hidraulico para la cuenca, si no se tiene estudios hidraulicos se debe poner false en "existencia_umbrales" en ese caso los umbrales seran tomados como los caudales extremos con un periodo de retorno 2.33 para QB y 2 para QTQ.

Una vez las configuraciones estan completas para correr el código hay que ejecutar el script de la siguiente manera:
Primero hay que instalar los requerimientos, para lo cual primero hay que asegurarse de tener python en el sistema y ejecutar los siguientes comandos en windows:
pip install virtualenv
python -m virtualenv entorno_hecca 
entorno_hecca\Scripts\activate
pip install -r requirements.txt
y luego ahora si correr el código con:
python3 -OO hecca_final.py
 o 
python -OO hecca_final.py
dependiendo de como este configurado su path.
En linux es:
pip install virtualenv
python -m virtualenv entorno_hecca
source entorno_hecca/bin/activate
pip install -r requirements.txt
y luego ahora si correr el código con:
python3 -OO hecca_final.py
 o 
python -OO hecca_final.py
# 🎨 Proyecto de Análisis y Procesamiento de Datos Hidrológicos

⭐ **Repositorio Principal**  
Este repositorio contiene código y herramientas para el análisis y procesamiento de datos relacionados con eventos hidrológicos, así como su comparación y validación con parámetros estadísticos y normativos.

---

📊 **Tabla de Contenidos**
1. [Introducción](#introducción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Uso](#uso)
6. [Contribuciones](#contribuciones)
7. [Autores](#autores)

---

🔬 **Introducción**  
Este proyecto facilita el análisis de eventos hidrológicos, incluyendo:
- Procesamiento y limpieza de datos (archivos CSV).
- Estadísticas sobre caudales y precipitaciones.
- Comparación de eventos con indicadores como ENOS (El Niño-Oscilación del Sur).
- Generación de parámetros IHA (Indicators of Hydrologic Alteration).

---

🌐 **Estructura del Proyecto**  
├── docs/                # Documentación del proyecto
├── IhaEstado.py       # Procesa estados hidrológicos según normas ANLA
├── estadistica_ideam.py # Cálculos estadísticos de datos
├── limpieza_datos.py   # Limpieza de datos crudos
├── ingreso_datos.py    # Manejo de ingreso de información
├── funciones_*         # Funciones auxiliares para diferentes propósitos
├── enso.json           # Datos relacionados con ENOS
├── oni_3.csv          # Archivo con información climática (ONI)
├── README.md          # Este archivo
Detalles por archivo
•  IhaEstado.py: Implementa el cálculo y la clasificación de estados hidrológicos basado en las normas ANLA, incluyendo pruebas preliminares para garantizar resultados consistentes.
•	Clase IhaEstado:
o	Atributos:
	grupo_1 a grupo_5: Almacenan diferentes parámetros calculados según las métricas de IHA.
	data_cruda: Contiene el DataFrame original proporcionado como entrada.
	data: Almacena los datos procesados y filtrados.
	start_year, end_year: Identifican el rango temporal de los datos.
o	Métodos:
	calcular_iha: Ejecuta los cálculos de los parámetros IHA, divididos en cinco grupos, utilizando funciones del módulo iha_parametros.
	unir_grupos: Combina los grupos de datos en un único DataFrame para análisis conjunto.
	Operadores sobrescritos: __sub__ y __truediv__ para comparar y dividir instancias de IhaEstado.
•  enso.json: Archivo JSON que almacena datos relacionados con eventos de El Niño y La Niña. Este archivo contiene:
•	Una lista de años asociados con eventos de El Niño (ninio).
•	Una lista de años asociados con eventos de La Niña (ninia).
•	Una clave normal que indica valores estándar cuando no se presentan eventos ENOS específicos.
•  consistencia.py: Realiza el análisis de consistencia y homogeneidad de datos hidrológicos. Las principales funciones son:
•	filtrar_datos: Filtra datos según un rango general y excepciones para años específicos.
•	procesar_archivos_csv: Procesa múltiples archivos CSV, generando análisis de caudal acumulado, detectando años consistentes y atípicos, y generando gráficas.
•	homogeneidad_helmert: Evalúa la homogeneidad usando el método de Helmert.
•	homogeneidad_kendall: Evalúa la homogeneidad utilizando el test de Kendall.


•	estadistica_ideam.py: Realiza análisis estadísticos de los datos obtenidos del IDEAM, con énfasis en la detección de patrones y tendencias en series temporales.
•  comprobacion_ideam.py: Contiene funciones diseñadas para realizar pruebas estadísticas sobre datos hidrológicos de referencia y alterados, incluyendo:
•	prueba_si_cumple: Evalúa diferencias estadísticas entre series hidrológicas.
•	cumple: Determina si una serie alterada cumple con parámetros normativos en comparación con una de referencia.
•	Prueba_porc: Calcula porcentajes aprobatorios según criterios específicos.
•	calibrar_mes: Ajusta parámetros de un mes para garantizar cumplimiento con condiciones establecidas.
•	
•	limpieza_datos.py: Encargado de limpiar y estructurar los datos crudos provenientes de diferentes fuentes para su posterior análisis.
•	ingreso_datos.py: Automatiza el proceso de ingreso y verificación de datos para garantizar la integridad de la información.
•	funciones_anla.py y funciones_ideam.py: Contienen funciones auxiliares específicas para manejar cálculos, formatos de datos y transformaciones necesarias en las diferentes etapas del proyecto.
•	enso.json: Archivo JSON que almacena datos relacionados con eventos de El Niño y La Niña.
•	oni_3.csv: Archivo CSV que contiene datos climáticos históricos relevantes para el análisis.
________________________________________
🔧 Requisitos
•	Python 3.8 o superior
•	Bibliotecas listadas en requirements.txt
________________________________________
🔄 Instalación
1.	Clona el repositorio:
$ git clone https://github.com/tuusuario/tu-repo.git
2.	Accede al directorio:
$ cd tu-repo
3.	Instala los requisitos:
$ pip install -r requirements.txt
________________________________________
🔍 Uso
Ejecución de scripts principales
1.	Limpieza de datos:
$ python limpieza_datos.py
2.	Procesamiento de estados hidrológicos:
$ python IhaEstado.py
3.	Generación de estadísticas:
$ python estadistica_ideam.py
________________________________________
💡 Contribuciones
✅ Las contribuciones son bienvenidas. Por favor, sigue estos pasos:
1.	Haz un fork del repositorio.
2.	Crea una rama para tus cambios:
$ git checkout -b feature/nueva-funcionalidad
3.	Realiza tus cambios y haz un commit:
$ git commit -m "Agrega nueva funcionalidad"
4.	Envía un pull request.
________________________________________
👤 Autores

