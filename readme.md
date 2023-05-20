# Avance Tarea 4

El proyecto se estructura de las siguientes carpetas:

- /.: carpeta principal, contiene los script para ejecutar el proyecto y otras carpetas de utilidad
- /figures: carpeta para guardar las figuras exportadas en formato .pdf
- /utils: carpeta para guardar algunas funciones auxiliares a usar en el proyecto

El archivo requirements.txt contiene todas las dependencias necesarias para ejecutar este proyecto. Se resaltan las siguientes librerías:

- pandas==2.0.1
- numpy==1.24.2
- matplotlib==3.7.1
- gym==0.23.1
- torch==2.0.0
- PyYAML==6.0 

Además, los experimentos fueron ejecutados sobre python==3.9.

Ejecutar este proyecto es sencillo, sólo es necesario seguir los siguientes pasos:

1. Instalar las librerías necesarias: `pip install -r requirements.txt`
2. Replicar los experimentos: `python run.py`