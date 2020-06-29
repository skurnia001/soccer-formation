import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def preprocessing(data_file_path, col_name, dtypes, dataset_type, file_suffix, **kwargs):
    """
        Need parameter~
        +lazy option
    """
    # https://gitlab.com/grains2/slicing-and-dicing-soccer/-/blob/master/SoccER_DatasetDescription.pdf
    if dataset_type == 'SoccER':
        form_data_df = pd.read_csv(data_file_path, delim_whitespace=True, header=None, names=col_name, dtype=dtypes)

        player_ids = form_data_df['player_id'].unique()

        x_means = []
        for player_id in player_ids:
            x_means.append(form_data_df['x_pos'][form_data_df['player_id'] == player_id].mean())

        # removing both teams' goalkeeper by removing players with lowest and highest mean x_pos
        player_ids = np.delete(player_ids, np.argsort(x_means)[[0, -1]])

        form_data_df = form_data_df[form_data_df['player_id'].isin(player_ids)]

        # Parse XML file
        root = ET.parse(kwargs['xml_file_path']).getroot()

        label_collected = ['BallPossession', 'BallOut', 'Goal', 'Foul']
        bef_frame_id = 0  # be careful in the difference in indexing of the segment_arr and frame_id (off by one)
        segment_id = 0
        team_id = -1
        segment_arr = np.zeros(form_data_df['frame_id'].max() + 1, dtype=np.int32)
        for track_node in root.findall('track'):
            if track_node.attrib['label'] in label_collected:
                box_node = track_node.find('box')

                track_label = track_node.attrib['label']
                frame_id = int(box_node.attrib['frame'])

                if team_id == -1:
                    segment_arr[bef_frame_id:frame_id - 1] = -1
                else:
                    segment_arr[bef_frame_id:frame_id - 1] = segment_id
                bef_frame_id = frame_id - 1

                if track_label == 'BallPossession':
                    team_id_attrib = box_node.find("attribute[@name='teamId']")
                    team_id_now = int(team_id_attrib.text)
                    if team_id != team_id_now:
                        segment_id += 1
                        team_id = team_id_now
                else:
                    team_id = -1
                # in this particular datasets, for every goal occuring in frame x, frame x+600 will be missing.
                # Thus, frame x+600 is derived from frame x+599 for dataset continuity.
                if track_label == 'Goal':
                    tmp = form_data_df[form_data_df['frame_id'] == frame_id + 599].copy()
                    tmp['frame_id'] = frame_id + 600
                    form_data_df = form_data_df.append(tmp)

        # Corner case of the last annotation
        if team_id == -1:
            segment_arr[bef_frame_id] = -1
        else:
            segment_arr[bef_frame_id] = segment_id

        # Set the rest as invalid segment
        segment_arr[bef_frame_id + 1:] = -1

        frame_segment_arr = np.array([np.arange(1, form_data_df['frame_id'].max() + 2), segment_arr], dtype=np.int32).T

        frame_segment_df = pd.DataFrame(frame_segment_arr, columns=['frame_id', 'segment_id'])

        frame_segment_df.to_csv('data//processed//segment.csv', index=False)

        # End of XML file parsing

        # Since we add new rows, it is necessary to sort the rows (at least to make the dataset tidy)
        form_data_df.sort_values('frame_id', kind='stable', inplace=True, ignore_index=True)

        # If no new rows is added, only reindexing is needed.
        # form_data_df.index = range(len(form_data_df.index))

        form_data_df.to_csv('data//processed//form_data.csv', float_format='%.6f', index=False)

        return frame_segment_df, form_data_df
    else:
        raise NotImplementedError
