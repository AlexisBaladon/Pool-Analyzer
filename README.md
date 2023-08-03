## Integrantes:

- Alexis Baladón
- Ignacio Viscardi
- Facundo Pedreira

## Página web:

La página web fue creada con Angular y se encuentra hosteada en:

https://alexisbaladon.pages.fing.edu.uy/timag-presentation

Todas las rutas utilizadas son relativas, pero en caso de cambiar algun directorio del proyecto
podrían ser de interés los siguientes archivos:

- .gitlab-ci.yml (raíz): Pipeline de CI/CD
- package.json (src/page): Tiene ruta de proyecto (--base-href=/timag-presentation/")
- angular.json (src/page): Tiene ruta de output de build ("outputPath": "../../public")

## Instrucciones de programa:

Para instalar las dependencias, debe ejecutar el siguiente comando:
'pip install -r requirements.txt'

El software del proyecto está dividido en varias notebooks y un programa principal.

Los directorios presentes son los siguientes:
- data: Contiene los conjuntos de datos y datos de cache.
- logs: Contiene registros de ejecuciones del programa
- models: Contiene la grilla de modelos y el binario del mejor modelo de train dumpeado con la librería pickle.
- notebooks: Contiene notebooks de las 3 etapas mostradas en la sección de solución de la página: Clasificación, Segmentación y Aplicación.
- results: Contiene tablas de resultados de clasificación y segmentación.
- scripts: Contiene el script usado para cambiar el formato del conjunto de datos Algarve's al formato del proyecto. 
Además se encuentran notebooks para entrenar al modelo neuronal profundo, la baseline, y el modelo zero-shot utilizado para clasificar
de forma artificial el conjunto de datos de Algarve's fragmentado.
- src: Aquí se encuentra el programa principal de entrenamiento estructurado con arquitectura de pipeline (ingestión, transformación, entrenamiento y evaluación).

### Conjuntos de datos:
Los conjuntos de datos utilizados en el proyecto son:
- Pool-dectection: https://github.com/yacine-benbaccar/Pool-Detection/tree/master
- Algarve's Dataset: https://www.kaggle.com/datasets/cici118/swimming-pool-detection-algarves-landscape
- Unidad de Información Geográfica: https://intgis.montevideo.gub.uy/sit/mapserv/data/fotos_2021/J-29-C-5-N-3.jpg

### Script de Entrenamiento:
Para reproducir las pruebas de entrenamiento implementadas en el directorio src,
pueden ser de ayuda los siguientes comandos:

- Mostrar descripción de flags:

```
'python main.py --help
```

- Entrenar modelo en dirección de entrenamiento por defecto (conjunto de Pool-Detection)

```
'python main.py --train
```

- Evaluar modelo en dirección de evaluación por defecto (conjunto de Algarve's fragmentado)

```
'python main.py --predict
```

- Utilizar caché de features luego de entrenar/evaluar:

```
py main.py --train --cache_features
```

- Probar entrenamiento con grilla de tamaño 1:

```
py main.py --train --small_grid
```
