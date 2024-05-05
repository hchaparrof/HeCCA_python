Código completo del progreso de la herramienta hecca en python, incluye la implementación de la herramienta para el calculo del caudal ambiental por la metodología del ideam y se preveé sumarle la metodología del anla.

Uso:
En el archivo setup.json cambiar los datos de archivo_base, archivo_apoyo y archivo_enso, por las ubicaciones de los archivos en su computador.
En el archivo setup.json llenar los datos que sean necesarios, en caso de que "existencia_areas" o "existencia_umbrales" sean "false" no importa los valores de "areas" o "umbrales".
"organismo" establece el algoritmo a utilizar, existen 2 metodologías diferentes: IDEAM o ANLA, y se selecciona como "ideam" y "anla" respectivamente.
"areas" Se refiere a las áreas de las cuencas de estudio para las dos estaciones, la base y la de apoyo, en caso de que no se tenga cuenca de apoyo no importa el valor y si no se tiene areas se debe poner false en "existencia_areas", se asume que las áreas estan en m3/s, pero desde que esten en la misma dimensión no debería ser importante.
"umbrales" Se refiere a los umbrales morfometricos de la cuenca de estudio en el lugar de estudio, son los umbrales QB y QTQ o caudal de banca llena y caudal de perdida de conectividad, eso se consigue con un estudio hidraulico para la cuenca, si no se tiene estudios hidraulicos se debe poner false en "existencia_umbrales" en ese caso los umbrales seran tomados como los caudales extremos con un periodo de retorno 2.33 para QB y 2 para QTQ.

Una vez las configuraciones estan completas para correr el código hay que ejecutar el script de la siguiente manera:
Primero hay que instalar los requerimientos, para lo cual primero hay que asegurarse de tener python en el sistema y ejecutar los siguientes comandos:
pip install virtualenv
python -m virtualenv entorno_hecca 
pip install -r requirements.txt
y luego ahora si correr el código con:
python3 -OO hecca_final.py
 o 
python -OO hecca_final.py

dependiendo de como este configurado su path.
