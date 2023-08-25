"""Implements the SoccerMap architecture."""
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


team_name = {23: 'Watford', 
    101: 'Leeds United', 
    35: 'Brighton & Hove Albion', 
    56: 'Norwich City', 
    31: 'Crystal Palace', 
    33: 'Chelsea', 
    37: 'Newcastle United', 
    40: 'West Ham United', 
    1: 'Arsenal', 
    38: 'Tottenham Hotspur', 
    25: 'Southampton', 
    34: 'Burnley', 
    22: 'Leicester City', 
    36: 'Manchester City', 
    24: 'Liverpool', 
    93: 'Brentford', 
    46: 'Wolverhampton Wanderers', 
    29: 'Everton', 
    39: 'Manchester United', 
    59: 'Aston Villa', 
    43: 'Nottingham Forest', 
    28: 'AFC Bournemouth', 
    55: 'Fulham'}

team_index = {23: 0, 
    101: 1, 
    35: 2, 
    56: 3, 
    31: 4, 
    33: 5, 
    37: 6, 
    40: 7, 
    1: 8, 
    38: 9, 
    25: 10, 
    34: 11, 
    22: 12, 
    36: 13, 
    24: 14, 
    93: 15, 
    46: 16, 
    29: 17, 
    39: 18, 
    59: 19, 
    43: 20, 
    28: 21, 
    55: 22}

class EmbeddingLayer(nn.Module):
    """ Fully-connected layer for creating embeddings for the one-hot encoded 
        game state input (home/away, reamining time, goal difference and passing team.)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(39, 30)
        self.fc2 = nn.Linear(30, 21)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    

class MixingLayer(nn.Module):
    """ The 2D-fully-concolutional layer for mixing the embedded game state data with 
        original 9-channel soccermap input data.

        8-channel tesnor input is created with embeded gaem state data by copying it 
        along the dimensions of the soccermap data. Then, these to tensirs concatinated along 
        channel axis. This final tensor fully-convolved with kernels (1x1), firstly to higher channel
        size, the to smaller chanel size of 9 (original input size of soccermap).
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(30, 16, kernel_size=(1, 1), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(16, 9, kernel_size=(1, 1), stride=1, padding="valid")

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        return x
    

class ToGameStateSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        pass_team_id = sample["possession_team_id_a0"]
        time_past = sample["time_seconds_overall_a0"]
        goal_diff = sample["goalscore_diff"]
        
        # create ohe goal difference label
        goal_diff_ohe = np.zeros((7))
        if goal_diff <= -3:
            goal_diff_ohe[0] = 1
        elif goal_diff == -2:
            goal_diff_ohe[1] = 1
        elif goal_diff == -1:
            goal_diff_ohe[2] = 1
        elif goal_diff == 0:
            goal_diff_ohe[3] = 1
        elif goal_diff == 1:
            goal_diff_ohe[4] = 1
        elif goal_diff == 2:
            goal_diff_ohe[5] = 1
        else:
            goal_diff_ohe[6] = 1


        # create ohe team feature
        pass_team_ohe = np.zeros((23))
        pass_team_ohe[team_index[pass_team_id]] = 1

        # create home away label
        is_home = np.expand_dims(sample["is_home_a0"], axis=0)

        # create ohe remainig seconds label
        rem_sec_ohe = np.zeros((8))
        time_past =  np.clip((time_past),0,5400)
        if time_past < 900:
            rem_sec_ohe[0] = 1
        elif 900 <= time_past < 1800:
            rem_sec_ohe[1] = 1
        elif 1800 <= time_past < 2700:
            rem_sec_ohe[2] = 1
        elif 2700 <= time_past < 3600:
            rem_sec_ohe[3] = 1
        elif 3600 <= time_past < 4500:
            rem_sec_ohe[4] = 1
        elif 4500 <= time_past < 4800:
            rem_sec_ohe[5] = 1
        elif 4800 <= time_past < 5100:
            rem_sec_ohe[6] = 1
        elif 4800 <= time_past <= 5400:
            rem_sec_ohe[7] = 1
        else:
            raise ValueError("Remaining second is out of bounds.")
        
        game_state_vector = np.concatenate((is_home,goal_diff_ohe,rem_sec_ohe,pass_team_ohe))


        start_x, start_y, end_x, end_y = (
            sample["start_x_a0"],
            sample["start_y_a0"],
            sample["end_x_a0"],
            sample["end_y_a0"],
        )
        speed_x, speed_y = sample["speedx_a02"], sample["speedy_a02"]
        frame = pd.DataFrame.from_records(sample["freeze_frame_360_a0"])

        # Location of the player that passes the ball
        # passer_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
        # Location of the ball
        ball_coo = np.array([[start_x, start_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])
        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(
            -1, 2
        )
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        # Output
        matrix = np.zeros((9, self.y_bins, self.x_bins))

        # CH 1: Locations of attacking team
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1

        # CH 3: Distance to ball
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        # CH 8-9: Ball speed
        matrix[7, y0_ball, x0_ball] = speed_x
        matrix[8, y0_ball, x0_ball] = speed_y

        # Mask
        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = np.array([[end_x, end_y]])
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1

        if "receiver" in sample:
            target = int(sample["receiver"]) if not math.isnan(sample["receiver"]) else -1
            return (
                torch.from_numpy(game_state_vector).float(),
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        return (
            torch.from_numpy(game_state_vector).float(),
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([1]).float(),
        )