import os
import pandas as pd

from CONST.path_const import root_path

def get_raw_data_from_file(filename: str, suffix: str = '.xlsx'):
    dataPath = os.path.join(root_path, 'data', f'{filename}{suffix}')
    if suffix == '.xlsx':
        data = pd.read_excel(dataPath)
    elif suffix == '.pkl':
        data = pd.read_pickle(dataPath)
    else:
        raise FileNotFoundError(f"No file's end with {suffix}")
    
    return data

