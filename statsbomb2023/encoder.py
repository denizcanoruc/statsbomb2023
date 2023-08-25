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

class CombinedEncoder(nn.Module):
    def __init__(self, latent_dim = 16):
        super().__init__()    
        
        self.fc1 = nn.Linear(32 + 23, latent_dim * 4)
        self.fc2 = nn.Linear(latent_dim * 4, latent_dim * 2)
        self.fc3 = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim = 16):
        super().__init__()    

        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()    

        # home away
        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 16)
        self.fc13 = nn.Linear(16, 1)

        # remaining time
        self.fc21 = nn.Linear(128, 64)
        self.fc22 = nn.Linear(64, 16)
        self.fc23 = nn.Linear(16, 8)

        # goal difference
        self.fc31 = nn.Linear(128, 64)
        self.fc32 = nn.Linear(64, 16)
        self.fc33 = nn.Linear(16, 7)

    def forward(self, x):

        # home away
        ha = F.relu(self.fc11(x))
        ha = F.relu(self.fc12(ha))
        ha = F.sigmoid(self.fc13(ha))

        # remaining time
        rt = F.relu(self.fc21(x))
        rt = F.relu(self.fc22(rt))
        rt = self.fc23(rt)

        # goal difference
        gd = F.relu(self.fc31(x))
        gd = F.relu(self.fc32(gd))
        gd = self.fc33(gd)

        return ha, rt, gd


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 8, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv3 = nn.Conv2d(16, 8, kernel_size=(3, 3), stride=1, padding="valid")

        self.fc1 = nn.Linear(528 , 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    

 

class ToSoccerEncoderTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self):
        self.y_bins = 68
        self.x_bins = 104

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        pass_team_id = sample["possession_team_id_a0"]
        psm = np.expand_dims(pd.DataFrame.from_records(sample["psm"]), axis=0)
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
        is_home = sample["is_home_a0"]

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

        # create pass label
        end_x, end_y = (
            sample["end_x_a0"],
            sample["end_y_a0"],
        )
        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = np.array([[end_x, end_y]])
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1
        
        # concat psm with result of the pass
        psm_with_mask = np.concatenate((psm,mask))

        return (
            torch.from_numpy(psm_with_mask).float(),
            torch.from_numpy(pass_team_ohe).float(),
            torch.tensor(np.expand_dims(is_home, axis=0)).float(),
            torch.tensor(rem_sec_ohe).float(),
            torch.tensor(goal_diff_ohe).float()
            )