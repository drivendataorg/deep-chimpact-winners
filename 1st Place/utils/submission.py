# GLOBAL VARIABLES
import numpy as np
import pandas as pd
import os


DISTANCES = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
             5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
             13.5, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
             23.0, 24.0, 25.0]

DISTANCES     = np.array(DISTANCES)


# SUBMISSION HELPER
def path2info(row):
    data = row['image_path'].split(os.sep)[-1]
    video_id = data.split('-')[0]
    time     = data.split('-')[1].replace('.png','')
    row['video_id'] = video_id
    row['time']     = int(time)
    return row

def get_shortest_distance(row, zero_distance=False):
    distances = DISTANCES.copy()
    if not zero_distance:
        distances = distances[1:]
    col = 'distance'
    try:
        d = row[col]
    except:
        col = 'pred'
        d = row[col]
    r   = np.abs(distances-d)
    idx = np.argmin(r)
    new_d    = distances[idx]
    row[col] = new_d
    return row