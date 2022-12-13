# TimeCereal
TimeCereal - A tool to interpolate, visualize and interact with time series.
Prototypical application.

## Dependencies
A `Python 3.10` interpreter is necessary to execute this prototype. Install all requirements with:

`pip install -r src/requirements.txt`

## Usage
Navigate to `src` and enter the command

`python main.py --name NAME [ --train TRAIN] [ --train_ae TRAIN] [ --sparsen SPARSEN] [ --path PATH]`

where
* `--name`: Dataset name
* `--train`: `True` oder `False`, default: `False`. If `True`, retrain all models.
* `--train_ae`:  `True` oder `False`, default: `False`. If `True`, retrain autoencoder only.
* `--sparsen`:  `True` oder `False`, default: `False`. If `True`, remove randomly chosen chunks from the training data.
* `--path`: Optional. All results will be saved among `RESULTS/name`. If you wish to create a subfolder in this directory, set this parameter.
