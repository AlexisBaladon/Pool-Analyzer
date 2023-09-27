## Participants:

- Alexis Baladón
- Ignacio Viscardi
- Facundo Pedreira

## Webpage:

The webpage was developed using Angular and is hosted in:

https://alexisbaladon.pages.fing.edu.uy/timag-presentation

## Program Instructions:

To install dependencies, it is necessary to run the following command:
'pip install -r requirements.txt'

This software is divided in multiple notebooks and a main program.

This is the description of each directory:
- data: Contains the datasets and cache.
- logs: Contains a register of prorgam executions.
- models: Contains the model hyperparameter grid and the best model binary file.
- notebooks: Contains notebook from each 3 stages of the final solution: Classification, Segmentation and application.
- results: Contains tables with results of classification and segmentation.
- scripts: Contains the script used to change the format of the Algarve's Dataset to the expected project format, and the notebook used to fragment and classify each fragment of the dataset. Aditionally, there are scripts to train the optimal and baseline model.
- src: Contains the main training program which was built using a pipeline architecture (ingestion, transformation, training and evaluation).

### Datasets:
The datasets used in the project are:
- Pool-dectection: https://github.com/yacine-benbaccar/Pool-Detection/tree/master
- Algarve's Dataset: https://www.kaggle.com/datasets/cici118/swimming-pool-detection-algarves-landscape
- Unidad de Información Geográfica: https://intgis.montevideo.gub.uy/sit/mapserv/data/fotos_2021/J-29-C-5-N-3.jpg

### Training Script:
To reproduce the training tests implemented in src, the following commands will be of aid:

- Show flags description:

```
'python main.py --help
```

- Train the model in the default training dataset (yacine)

```
'python main.py --train
```

- Evaluate the trained model using the default evaluation dataset (fragmented Algarve's)

```
'python main.py --predict
```

- Use cached features:

```
py main.py --train --cache_features
```

- Train with a small and fast grid:

```
py main.py --train --small_grid
```
