"""Command line interface."""
import json
import tempfile

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import hydra
import mlflow
import numpy as np
import typer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from statsbomb2023.common import utils
from statsbomb2023.common.config import logger
from statsbomb2023.common.databases import Database, connect
from statsbomb2023.modules import SoccerMapModule, train_module, test_module
from statsbomb2023.common.features import all_features
from statsbomb2023.common.datasets import PassesDataset
from statsbomb2023.common.labels import all_labels



# Initialize Typer CLI app
app = typer.Typer()


def parse_config(config_path: Path, overrides: Optional[List[str]] = None) -> DictConfig:
    """Parse a config file."""
    overrides = [] if overrides is None else overrides
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base="1.2"):
        cfg = hydra.compose(config_name="config", return_hydra_config=True, overrides=overrides)
        # A couple of optional utilities:
        # - disabling python warnings
        # - easier access to debug mode
        # - forcing debug friendly configuration
        # You can safely get rid of this line if you don't want those
        utils.extras(cfg)
        # Pretty print config using Rich library
        if cfg.get("print_config"):
            utils.print_config(cfg, resolve=True)
        return cfg

@app.command()
def load_data(
    database: str = typer.Argument(
        ...,
        show_default=False,
        help=(
            "The URL of the database to use for storing the data. "
            + "For example, 'sqlite:///data.db'. Currently, only SQLite and HDF "
            + "are supported."
        ),
    ),
    getter: str = typer.Option(
        "remote",
        help="Load data from the 'remote' API or from a 'local' directory.",
        rich_help_panel="Data source",
    ),
    root: Optional[Path] = typer.Option(
        None,
        show_default=False,
        help="The root directory of the data. Only used when --getter is 'local'.",
        rich_help_panel="Data source",
    ),
    competition_id: Optional[int] = typer.Option(
        None,
        show_default=False,
        help="Retrieve the data for a specific competition",
        rich_help_panel="Data selection",
    ),
    season_id: Optional[int] = typer.Option(
        None,
        show_default=False,
        help="Retrieve the data for a specific season",
        rich_help_panel="Data selection",
    ),
    game_id: Optional[int] = typer.Option(
        None,
        show_default=False,
        help="Retrieve the data for a specific game",
        rich_help_panel="Data selection",
    ),
) -> None:
    """Load raw StatsBomb event data from the remote API or a local folder.

    The data is downloaded, converted to the SPADL format and stored in a database.

    The data to load can be specified by combining the --competition_id,
    --season_id and --game_id options. If none of these parameters is
    specified, all available data will be processed.

    To load non-public data from the API,
    authentication is required. Authentication can be done by setting
    environment variables named `SB_USERNAME` and `SB_PASSWORD` to your
    login credentials.
    """
    # Init database connection
    logger.info(f"Instantiating database connection.")
    db = connect(database, mode="a")
    # Import data
    logger.info("Starting data import!")
    db.import_data(
        getter=getter,
        root=root,
        competition_id=competition_id,
        season_id=season_id,
        game_id=game_id,
    )
    db.close()
    logger.info("✅ Saved raw data to database.")


@app.command()
def create_dataset(
    database: str = typer.Argument(
        ...,
        show_default=False,
        help="The URL of the database to use. For example, 'sqlite:///data.db'.",
    ),
    dataset_fp: Path = typer.Argument(
        ...,
        show_default=False,
        help="The directory where the dataset should be stored.",
    ),
    config_fp: Path = typer.Argument(
        ...,
        show_default=False,
        help="The path to the config file specifying which competitions, seasons and games to include.",
    ),
    xfn: Optional[List[str]] = typer.Option(
        None,
        show_default=False,
        help="Name of a feature generator to apply.",
    ),
    yfn: Optional[List[str]] = typer.Option(
        None,
        show_default=False,
        help="Name of a label generator to apply.",
    ),
) -> None:
    """Extract features and labels for training models."""
    # Load raw data
    logger.info("Instantiating database connection.")
    db = connect(database, mode="r")
    # Create dataset
    logger.info("Starting dataset creation!")
    xfns = [] if xfn is None else xfn
    yfns = [] if yfn is None else yfn
    if len(xfns) == 0 and len(yfns) == 0:
        # generate all features and labels
        xfns = all_features
        yfns = all_labels
    dataset = PassesDataset(
        path=dataset_fp,
        xfns=xfns,
        yfns=yfns,
        load_cached=False,
    )
    dataset.create(db, OmegaConf.to_object(OmegaConf.load(config_fp)))
    logger.info("✅ Created the dataset at %s.", str(dataset_fp))


@app.command()
def train(
    config_fp: Path = typer.Argument(..., show_default=False, help="Configuration file."),
    dataset_fp: Path = typer.Argument(
        ...,
        show_default=False,
        help="The directory where the training dataset is stored.",
    ),
    dataset_test_fp: Optional[Path] = typer.Option(
        None,
        show_default=False,
        help="The directory where the evaluation dataset is stored.",
    ),
    overrides: Optional[List[str]] = typer.Argument(
        None,
        show_default=False,
        help="Overrides for the configuration file.",
    ),
):
    """Train a model component."""
    # Parse config file
    cfg = parse_config(config_path=config_fp, overrides=overrides)
    # Load dataset
    logger.info("Instantiating training dataset")
    dataset_train = partial(PassesDataset, path=dataset_fp)
    # Instantiote model
    logger.info(f"Instantiating model component <{cfg['module']['_target_']}>")
    module: pl.LightningModule = hydra.utils.instantiate(cfg.module, _convert_="partial")
    # Setup callbacks
    train_cfg = OmegaConf.to_object(cfg.get("train_cfg", DictConfig({})))
    utils.instantiate_callbacks(train_cfg)
    utils.instantiate_loggers(train_cfg)
    # Train model
    logger.info("Starting training!")
    mlflow.set_experiment(experiment_name=cfg.get("experiment_name", cfg.experiment_name))
    with mlflow.start_run() as run:
        # Log config
        with tempfile.TemporaryDirectory() as tmpdirname:
            fp = Path(tmpdirname)
            OmegaConf.save(config=cfg, f=fp / "config.yaml")
            mlflow.log_artifact(str(fp / "config.yaml"))
        train_module(module, dataset_train, optimized_metric=cfg.get("optimized_metric"), **train_cfg)
        mlflow.pytorch.log_model(module, "model")
        # Evaluate model on test set, using the best model achieved during training
        if cfg.get("test_after_training", True) and dataset_test_fp is not None:
            logger.info("Evaluating model on test set")
            dataset_test = partial(PassesDataset, path=dataset_test_fp)
            metrics = component.test_module(module,dataset_test, **cfg.get("test_cfg", {}))
            mlflow.log_metrics(
                {f"test/{key}": val for key, val in utils.nested_to_record(metrics).items()}
            )
            logger.info(f"Test metrics: {json.dumps(metrics, indent=4, sort_keys=True)}")
    logger.info("✅ Finished training. Model saved with ID %s", run.info.run_id)


@app.command()
def test(
    config_fp: Path = typer.Argument(..., show_default=False, help="Configuration file."),
    dataset_fp: Optional[Path] = typer.Argument(
        None,
        show_default=False,
        help="The directory where the evaluation dataset is stored.",
    ),
    model_uri: str = typer.Argument(
        ...,
        show_default=False,
        help="The URI of the model to test.",
    ),
    overrides: Optional[List[str]] = typer.Argument(
        None,
        show_default=False,
        help="Overrides for the configuration file.",
    ),
):
    """Evaluate a model component."""
    cfg = parse_config(config_path=config_fp, overrides=overrides)
    logger.info("Instantiating test dataset.")
    dataset_test = partial(PassesDataset, path=dataset_fp)
    logger.info("Loading model component")
    module: pl.LightningModule = mlflow.pytorch.load_model(model_uri + "/model")
    logger.info("Starting evaluation!")
    metrics = test_module(module, dataset_test, **cfg.get("test_cfg", {}))
    logger.info(f"Test metrics: {json.dumps(metrics, indent=4, sort_keys=True)}")
    logger.info("✅ Finished evaluation.")


'''
@app.command()
def predict(
    config_fp: Path = typer.Argument(..., show_default=False, help="Configuration file."),
    model_uri: str = typer.Argument(
        ...,
        show_default=False,
        help="The URI of the model to use.",
    ),
    game_id: int = typer.Argument(
        ...,
        show_default=False,
        help="The ID of the game for which to produce predictions.",
    ),
    overrides: Optional[List[str]] = typer.Argument(
        None,
        show_default=False,
        help="Overrides for the configuration file.",
    ),
):
    cfg = parse_config(config_path=config_fp, overrides=overrides)
    logger.info("Loading model component")
    component: UnxpassComponent = load_model(model_uri + "/component")
    output_dir = (
        Path(cfg.stores_dir) / "surfaces" / component.component_name / model_uri.split("/")[-1]
    )
    if (output_dir / f"game_{game_id}.npz").exists():
        logger.info(f"Surfaces for game {game_id} already exist. Skipping.")
        return
    # Load raw data
    logger.info(f"Instantiating database connection <{cfg.database._target_}>")
    db = connect(cfg.database, "r")
    # Create datasets
    logger.info(f"Instantiating test dataset <{cfg.dataset.loader._target_}>")
    dataset_test: Dataset = hydra.utils.instantiate(
        cfg.dataset.loader, path=Path(cfg.dataset.store) / "test"
    )
    logger.info("Starting prediction!")
    if isinstance(component, UnxPassXGBoostComponent):
        surfaces = component.predict_surface(
            dataset_test,
            game_id=game_id,
            db=db,
            x_bins=int(104 / 4),
            y_bins=int(68 / 4),
            result=None,
        )
    elif isinstance(component, UnxPassPytorchComponent):
        surfaces = component.predict_surface(dataset_test, game_id=game_id)
    else:
        raise NotImplementedError(f"Surface prediction is not implemented for {component_name}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"game_{game_id}", **surfaces)
    logger.info("✅ Finished training.")


@app.command()
def predict_value(
    config_fp: Path,
    component_name: str,
    result: str,
    offensive_run_id: str,
    defensive_run_id: str,
    game_id: int,
    overrides: Optional[List[str]] = typer.Argument(None),
):
    cfg = parse_config(config_path=config_fp, overrides=overrides)
    # Load raw data
    logger.info(f"Instantiating database connection <{cfg.database._target_}>")
    db: Database = hydra.utils.instantiate(cfg.database)
    # Create datasets
    logger.info(f"Instantiating test dataset <{cfg.dataset.loader._target_}>")
    dataset_test: Dataset = hydra.utils.instantiate(
        cfg.dataset.loader, path=Path(cfg.dataset.store) / "test"
    )
    logger.info(f"Instantiating model component <{cfg.component[component_name]._target_}>")
    component: UnxpassComponent = hydra.utils.get_class(
        cfg.component[component_name]._target_
    ).load("runs:/" + offensive_run_id, "runs:/" + defensive_run_id)
    logger.info("Starting prediction!")
    surfaces = component.predict_surface(
        dataset_test,
        game_id=game_id,
        db=db,
        x_bins=int(104 / 4),
        y_bins=int(68 / 4),
        result=result,
    )
    output_dir = Path(cfg.stores_dir) / "surfaces" / component_name / result / offensive_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"game_{game_id}", **surfaces)
    logger.info("✅ Finished training.")
'''

if __name__ == "__main__":
    app()