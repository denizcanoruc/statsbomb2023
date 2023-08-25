# statsbomb2023


## Installation

You can install a development version directly from GitHub. This requires [Poetry](https://python-poetry.org/).

```sh
# Clone the repository
$ git clone git://github.com/denizcanoruc/statsbomb2023.git
$ cd statsbomb2023
# Create a virtual environment
$ python -m venv .venv
$ source venv/bin/activate
# Install the package and its dependencies
$ poetry install
```

## Getting started

<details>
<summary><b>STEP 1: Obtain StatsBomb 360 data.</b></summary>

The clean data is stored on "/cw/dtaidata/ml/2019-DTAISportsAnalyticsLab/soccer-events-statsbomb/ENG-2223-360/clean_data", so we will use local loader to load the data. 

```bash
python statsbomb2023 load-data \
  sqlite://$(pwd)/stores/database.sql \
  --getter="local" \
  --root="/cw/dtaidata/ml/2019-DTAISportsAnalyticsLab/soccer-events-statsbomb/ENG-2223-360/clean_data" \
  --competition-id="2" \
  --season-id="235"
```

With this command we only load 2022/2023 season. We can also load 2021/2022 season with season id 108.

</details>

<details>
<summary><b>STEP 2: Create train and test data.</b></summary>

Now we will extract all passes from the data, create a feature representation and assign a label to each pass. It is enough to compute features and labels that are required for soccermap: Start location, end location, speed, freeze frame 360 for features and success for label.

```bash
python statsbomb2023 create-dataset \
  sqlite://$(pwd)/stores/database.sql \
  $(pwd)/stores/datasets/sb23/train \
  $(pwd)/config/dataset/train_sb23.yaml \
  --xfn="startlocation" \
  --xfn="endlocation" \
  --xfn="speed" \
  --xfn="freeze_frame_360" \
  --yfn="success"
```

```bash
python statsbomb2023 create-dataset \
  sqlite://$(pwd)/stores/database.sql \
  $(pwd)/stores/datasets/sb23/test \
  $(pwd)/config/dataset/test_sb23.yaml \
  --xfn="startlocation" \
  --xfn="endlocation" \
  --xfn="speed" \
  --xfn="freeze_frame_360" \
  --yfn="success"
```

_(this will take ~2 hours to run)_

</details>

<details>
<summary><b>STEP 3: Train model components.</b></summary>

All models are dynamically instantiated from a hierarchical configuration file managed by the [Hydra](https://github.com/facebookresearch/hydra) framework. The main config is available in [config/config.yaml](./config/config.yaml) and a set of example configurations for training specific models is available in [config/experiment](./config/experiment). The experiment configs allow you to overwrite parameters from the main config and allow you to easily iterate over new model configurations! You can run a chosen experiment config with:

```bash
python statsbomb2023 train \
  $(pwd)/config \
  $(pwd)/stores/datasets/sb23/train \
  experiment="soccermap"
```

Experiments are tracked using [MLFlow](https://mlflow.org/). You can view the results of your experiments by running `mlflow ui --backend-store-uri stores/model` in the root directory of the project and browsing to <http://localhost:5000>.

</details>

<details>
<summary><b>STEP 4: Test the model.</b></summary>

Once you have trained the model, you can test it. Therefore, specify a dataset (right now we have a single match due to the test.yaml we used) and the run ID of the trained model. The run IDs are printed after training a component or can be found in the MLFlow UI.

```bash
python statsbomb2023 test \
  $(pwd)/config \
  $(pwd)/stores/datasets/st23/test \
  runs:/fd27d72de6644565bf75769d37cc7feb
```

</details>


