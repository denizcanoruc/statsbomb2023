"""Implements the labels used in each compoment."""
from functools import reduce
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from socceraction.spadl.utils import add_names
from socceraction.vaep.labels import concedes, scores

from statsbomb2023.common.databases import Database


_pass_like = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
]


def success(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether an action was executed successfully.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'success' and a row for each action set to
        True if the action was completed successfully.
    """
    res = pd.DataFrame()
    res["success"] = actions["result_name"] == "success"
    return res


def _get_xg(action):
    if action["type_name"] == "shot":
        return action["extra"]["shot"]["statsbomb_xg"]
    return 0


def scores_xg(actions: pd.DataFrame, nr_actions: int = 10) -> pd.DataFrame:
    """Determine the xG value generated by the team possessing the ball within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores_xg' and a row for each action set to
        the total xG value generated by the team possessing the ball within the
        next x actions.
    """
    shots = actions.apply(_get_xg, axis=1)
    y = pd.concat([shots, actions["team_id"]], axis=1)
    y.columns = ["shot", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "shot"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    # removing opponent shots
    for i in range(1, nr_actions):
        y.loc[(y["team_id+%d" % i] != y["team_id"]), "shot+%d" % i] = 0

    # combine multiple shots in possession
    # see https://fbref.com/en/expected-goals-model-explained
    y["sum"] = 1
    y["scores_xg"] = 1 - y[["sum", "shot"] + ["shot+%d" % i for i in range(1, nr_actions)]].apply(
        lambda shots: reduce(lambda agg, xg: agg * (1 - xg), shots), axis=1
    )
    return y[["scores_xg"]]


def concedes_xg(actions: pd.DataFrame, nr_actions: int = 10) -> pd.DataFrame:
    """Determine the xG value conceded by the team possessing the ball within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes_xg' and a row for each action set to
        the total xG value conceded by the team possessing the ball within the
        next x actions.
    """
    shots = actions.apply(_get_xg, axis=1)
    y = pd.concat([shots, actions["team_id"]], axis=1)
    y.columns = ["shot", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "shot"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    # removing created shots
    for i in range(1, nr_actions):
        y.loc[(y["team_id+%d" % i] == y["team_id"]), "shot+%d" % i] = 0

    # combine multiple shots in possession
    # see https://fbref.com/en/expected-goals-model-explained
    y["sum"] = 1
    y["concedes_xg"] = 1 - y[
        ["sum", "shot"] + ["shot+%d" % i for i in range(1, nr_actions)]
    ].apply(lambda shots: reduce(lambda agg, xg: agg * (1 - xg), shots), axis=1)
    return y[["concedes_xg"]]


def receiver(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine the player who received the ball.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'receiver' and a row for each action set to
        the player who received the ball.
    """
    receivers = []
    passes = actions[actions.type_name.isin(_pass_like)]
    for (game_idx, action_idx), action in passes.iterrows():
        if action["freeze_frame_360"] is None:
            continue
        pass_options = np.array(
            [
                (t["x"], t["y"])
                for t in action["freeze_frame_360"]
                if t["teammate"] and not t["actor"]
            ]
        )
        if len(pass_options) == 0:
            continue
        ball_coo = np.array([action["end_x"], action["end_y"]])
        dist = np.sqrt(np.sum((pass_options - ball_coo) ** 2, axis=1))
        idx_receiver = np.argmin(dist)
        for idx_teammate, o in enumerate(pass_options):
            receivers.append(
                {
                    "game_id": game_idx,
                    "action_id": action_idx,
                    "pass_option_id": idx_teammate,
                    "receiver": idx_teammate == idx_receiver,
                }
            )
    return pd.DataFrame(
        receivers, columns=["game_id", "action_id", "pass_option_id", "receiver"]
    ).set_index(["game_id", "action_id", "pass_option_id"])


all_labels = [scores, scores_xg, concedes, concedes_xg, success]  # , receiver]


def get_labels(
    db: Database,
    game_id: int,
    yfns: List[Callable] = all_labels,
    actionfilter: Optional[Callable] = None,
):
    """Apply a list of label generators.

    Parameters
    ----------
    db : Database
        The database with raw data.
    game_id : int
        The ID of the game for which features should be computed.
    yfns : List[Callable], optional
        The label generators.
    actionfilter : Callable, optional
        A function that filters the actions to be used.
    nb_prev_actions : int, optional
        The number of previous actions to be included in a game state.

    Returns
    -------
    pd.DataFrame
        A dataframe with the labels.
    """
    game_actions = add_names(db.actions(game_id))
    if actionfilter is None:
        idx = pd.Series([True] * len(game_actions), index=game_actions.index)
    else:
        idx = actionfilter(game_actions)
    try:
        df_labels = pd.concat(
            # TODO: move .set_index to socceraction label generators
            [fn(game_actions.reset_index()).set_index(game_actions.index).loc[idx] for fn in yfns],
            axis=1,
        )
    except Exception:
        df_labels = pd.concat(
            # TODO: move .set_index to socceraction label generators
            [fn(game_actions).loc[idx] for fn in yfns],
            axis=1,
        )
    return df_labels