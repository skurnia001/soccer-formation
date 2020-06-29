import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment


def load_dataset(form_data_file, frame_segment_file):
    new_df = pd.read_csv(form_data_file)
    dff = pd.read_csv(frame_segment_file)
    df2 = new_df.merge(dff)
    segment_ids = df2['segment_id'].unique()
    return df2, segment_ids


def scale_pos(pos):
    """
    min-max scale the coordinates
    x: 0.0~1.0
    y: 0.0~0.7

    Args:
        pos (np.ndarray, shape (10, 2)): the coordinates of the field players

    Returns:
        pos_scaled (np.ndarray, shape (10, 2)): the scaled coordinates of the field players
    """

    x_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    y_scaler = MinMaxScaler(feature_range=(0.0, 0.7))

    x_scaled = x_scaler.fit_transform(pos.T[0].reshape(-1, 1))
    y_scaled = y_scaler.fit_transform(pos.T[1].reshape(-1, 1))

    pos_scaled = np.hstack((x_scaled, y_scaled))

    return pos_scaled


def standardize_pos(pos):
    """
    normalize the coordinates

    Args:
        pos (np.ndarray, shape (10, 2)): the coordinates of the field players

    Returns:
        pos_std (np.ndarray, shape (10, 2)): the standardized coordinates of the field players
    """

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_std = x_scaler.fit_transform(pos.T[0].reshape(-1, 1))
    y_std = y_scaler.fit_transform(pos.T[1].reshape(-1, 1))

    pos_std = np.hstack((x_std, y_std))

    return pos_std





def to_pos(df2):
    player_ids = df2['player_id'].unique()
    player_ids_A = player_ids[player_ids < 128]
    player_ids_B = player_ids[player_ids > 128]
    frame_num = df2['frame_id'].max() - df2['frame_id'].min() + 1
    pos_A = np.zeros((frame_num, 10, 2))
    pos_B = np.zeros((frame_num, 10, 2))

    for i, player_id in enumerate(player_ids_A):
        pos_A[:, i] = df2[df2['player_id'] == player_id][['x_pos', 'y_pos']].values
    for f in range(frame_num):
        pos_A[f] = standardize_pos(pos_A[f])

    for i, player_id in enumerate(player_ids_B):
        pos_B[:, i] = df2[df2['player_id'] == player_id][['x_pos', 'y_pos']].values
    for f in range(frame_num):
        pos_B[f] = standardize_pos(pos_B[f])
    return pos_A, player_ids_A, pos_B, player_ids_B


def compute_form_summary(pos):
    """
    1. compute the mean coordinates of 10 roles
    2. create the role assignment matrix

    Args:
        pos (np.ndarray): the coordinates of the field players after pre-processing

    Returns:
        form_summary (np.ndarray): the mean coordinates of each role
        assign_mat (np.ndarray): the role assignment matrix
    """

    # Corner case of only 1 frame
    if pos.shape[0] == 1:
        return pos.copy().reshape(10,2)

    pos_role = pos.copy()

    # Bialkowski's method
    # https://ieeexplore.ieee.org/document/7492601
    for i in range(2):  # one iteration is enough
        # compute a multivariate normal random variable for each role
        rvs = []
        for role in range(10):
            mean = [np.mean(pos_role[:, role, 0]), np.mean(pos_role[:, role, 1])]
            cov = np.cov(pos_role[:, role, 0], pos_role[:, role, 1])
            rvs.append(multivariate_normal(mean, cov))

        # formulate a cost matrix based on the log probability of each position
        # assign the labels to the players using the cost matrix
        assign_mat = np.zeros((10, 10))
        for f in range(len(pos)):
            cost_mat = np.zeros((10, 10))
            for role in range(10):
                cost_mat[role] = rvs[role].logpdf(pos_role[f])

            # exucute the Hungarian method to assign the role label to each player
            row_ind, col_ind = linear_sum_assignment(-cost_mat)
            pos_role[f] = pos_role[f, col_ind]
            # add counts the role assignment matrix
            for row, col in zip(row_ind, col_ind):
                assign_mat[row, col] += 1

        # compute the coordinates of 10 roles
        form_summary = np.mean(pos_role, axis=0)

        # min-max scaling
        # x: 0.0~1.0
        # y: 0.0~0.7
        form_summary = scale_pos(form_summary)

        return form_summary


def compute_similarity(form1, form2):
    """
    compute the similarity between two formations

    Args:
        form 1 (np.ndarray, shape (10, 2)): the coordinate of 10 players
        form 2 (np.ndarray, shape (10, 2))

    Returns:
        similarity (float); the similarity between form1 and form2
    """

    # min-max scaling
    # x: 0.0~1.0
    #  y: 0.0~0.7
    form1 = scale_pos(form1)
    form2 = scale_pos(form2)

    # create the similarity matrix
    sim_mat = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            norm = np.linalg.norm(form1[i] - form2[j], 2)  # L2 norm
            m = 1 - ((norm ** 2) / (1 / 3))
            sim_mat[i, j] = np.max([m, 0])

    # comute the similarity via the Hungarian method
    row_ind, col_ind = linear_sum_assignment(-sim_mat)
    similarity = sim_mat[row_ind, col_ind].sum() / 10

    return similarity, col_ind


def load_formations():
    """
    Returns:
        forms (list, shape (25, 10, 3)): the 2d array of positional data of position names for 25 templete formations
        form_labels (list): the list of the names of 25 templete formations
        forwards (list): the list of the names of offensive roles
        defenders (list): the lsit of the names of defensive roles
    """

    # define the formations
    f_343 = np.array(
        [[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [5, 2, 'RM'], [5, 4, 'RCM'], [5, 6, 'LCM'], [5, 8, 'LM'],
         [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_352 = np.array(
        [[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [4, 4, 'RCDM'], [4, 6, 'LCDM'], [6, 2, 'RM'], [6, 5, 'CAM'],
         [6, 8, 'LM'], [8, 4, 'RST'], [8, 6, 'LST']])

    f_41212 = np.array(
        [[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 5, 'CDM'], [5, 3, 'RCM'], [5, 7, 'LCM'],
         [6, 5, 'CAM'], [8, 4, 'RST'], [8, 6, 'LST']])

    f_4231 = np.array(
        [[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 4, 'RCDM'], [4, 6, 'LCDM'], [6, 2, 'RM'],
         [6, 8, 'LM'], [6, 5, 'CAM'], [8, 5, 'ST']])

    f_442 = np.array(
        [[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [6, 2, 'RM'], [5, 4, 'RCM'], [5, 6, 'LCM'],
         [6, 8, 'LM'], [8, 4, 'RST'], [8, 6, 'LST']])

    f_4123 = np.array(
        [[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 5, 'CDM'], [5, 3, 'RCM'], [5, 7, 'LCM'],
         [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_4213 = np.array(
        [[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 4, 'RCDM'], [4, 6, 'LCDM'], [6, 5, 'CAM'],
         [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_541 = np.array(
        [[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [3, 1, 'RWB'], [3, 9, 'LWB'], [5, 4, 'RCM'], [5, 6, 'LCM'],
         [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_532 = np.array(
        [[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [3, 1, 'RWB'], [3, 9, 'LWB'], [5, 2, 'RCM'], [5, 5, 'CM'],
         [5, 8, 'LCM'], [8, 4, 'RST'], [8, 6, 'LST']])

    forms = [f_343, f_352,
             f_41212, f_4231, f_442, f_4123, f_4213,
             f_541, f_532
             ]

    form_labels = ["3-4-3", "3-5-2",
                   "4-1-2-1-2", "4-2-3-1", "4-4-2", "4-2-1-3", "4-1-2-3",
                   "5-4-1", "5-3-2"
                   ]

    return forms, form_labels


def visualize_form_summary(form_summary):
    """
    1. visualize the formation summary
    2. show the role assignment matrix
    3. suggest several formations similar to the formation summary

    Args:
        form_summary (np.ndarray, shape (10, 2)): the mean coordinates of each role
        assign_mat (np.ndarray, shape (10, 10)): the role assignment matrix
        player_ids (list): the list of the id of 10 field players
    """

    # load the data of the formation templetes
    forms, form_labels = load_formations()

    # compute the similarities between the formation summary and each formation templet
    sims = []
    assigns = []
    for form in forms:
        sim, assign = compute_similarity(form_summary, form)
        sims.append(sim)
        assigns.append(assign)

    return sims


def formation(form_data_file, frame_segment_file):
    df, segment_ids = load_dataset(form_data_file, frame_segment_file)
    forms, forms_label = load_formations()

    form_arr = np.zeros((segment_ids.shape[0] - 1, 20, 4))
    sims_arr = np.zeros((segment_ids.shape[0] - 1, len(forms_label)*2, 3))
    # TODO: finish the formation
    for i, df_segment in enumerate(df.groupby('segment_id')):
        if df_segment[0] == -1:
            continue

        pos_A, player_ids_A, pos_B, player_ids_B = to_pos(pd.DataFrame(df_segment[1]))

        form_summary = compute_form_summary(pos_A)
        form_arr[i - 1, :10, :] = np.hstack(
            (np.full(10, df_segment[0]).reshape(-1, 1), player_ids_A.reshape(-1, 1), form_summary))

        sims = visualize_form_summary(form_summary)
        sims_arr[i - 1, :len(forms_label), :] = np.hstack((np.full(len(forms_label), df_segment[0]).reshape(-1, 1), np.full(len(forms_label), 0).reshape(-1, 1),
                                             np.array(sims).reshape(-1, 1)))

        form_summary = compute_form_summary(pos_B)
        form_arr[i - 1, 10:, :] = np.hstack(
            (np.full(10, df_segment[0]).reshape(-1, 1), player_ids_B.reshape(-1, 1), form_summary))

        sims = visualize_form_summary(form_summary)
        sims_arr[i - 1, len(forms_label):, :] = np.hstack((np.full(len(forms_label), df_segment[0]).reshape(-1, 1), np.full(len(forms_label), 1).reshape(-1, 1),
                                             np.array(sims).reshape(-1, 1)))

    form_arr = form_arr.reshape((segment_ids.shape[0] - 1) * 20, 4)
    sims_arr = sims_arr.reshape(((segment_ids.shape[0] - 1) * len(forms_label) * 2, 3))

    form_df = pd.DataFrame(form_arr, columns=['segment_id', 'player_id', 'x_pos', 'y_pos']).astype({'segment_id':np.int32, 'player_id':np.int16, 'x_pos':np.float64, 'y_pos':np.float64})
    sims_df = pd.DataFrame(sims_arr, columns=['segment_id', 'team_id', 'sim_score']).astype({'segment_id':np.int32, 'team_id':np.int16, 'sim_score':np.float64})
    sims_df = pd.concat([sims_df, pd.DataFrame(forms_label * (segment_ids.shape[0] - 1) * 2, columns=['formation'])], axis=1)

    form_df.to_csv("data//processed//formation.csv", float_format='%.6f', index=False)
    sims_df.to_csv("data//processed//sims_score.csv", float_format='%.6f', index=False)

    return form_df, sims_df
