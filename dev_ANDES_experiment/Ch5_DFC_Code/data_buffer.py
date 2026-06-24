import os
import numpy as np
import pandas as pd

#==================== data input buffer ======================================#
class DataBuffer:

  def __init__(self, device='cpu'):
    self.device = device
    self.list_of_darrays = []
    self.list_of_tarrays = []

    # print data buffer properties when created
    print("DataBuffer is created.")

  def read_data_from(self, input_path: str, 
                     header_row_number: int, 
                     points_per_day: int, 
                     column_names_to_read: list[str], 
                     data_format: str):
    # Assert that the input path exists
    assert os.path.exists(input_path), f"Input path {input_path} does not exist."

    # Read data from the input path
    print(f"DataBuffer.read_data_from(): Reading data from {input_path}...")
    for dirpath, _, filenames in os.walk(input_path):
      for filename in filenames:
        # skip hidden files
        if filename.startswith('.'):
          continue 
        # skip backup files
        if filename.endswith('.bak'):
          continue
        # read data as a dataframe from csv or xls (currently supports data from NEM(AEMO) or Elia)
        filepath = os.path.join(dirpath, filename)
        if data_format == 'NEM_csv':
          df = pd.read_csv(filepath, header=header_row_number, parse_dates=["DATETIME"])\
                 .sort_values(by=["DATETIME"])
          power_base = 1 # NEMweb data should be preprocessed to be normalized already
        elif data_format == 'Elia_xls':
          df = pd.read_excel(filepath, header=header_row_number, parse_dates=["DateTime"], date_format='%d/%m/%Y %H:%M')\
                 .sort_values(by=["DateTime"])
          power_base = df.iloc[4, 6] # WARNING: position of Power Basis, hard-coded for Elia xls data
        elif data_format == 'DeepComp_csv':
          df = pd.read_csv(filepath, header=header_row_number, parse_dates=["Time"], date_format='%d/%m/%Y %H:%M')\
                 .sort_values(by=["Time"])
          power_base = df[column_names_to_read[-1]].max()
        else:
          raise ValueError(f"DataBuffer.read_data_from(): Failed to read {filename} using {data_format} format.")

        pfpm      = df[column_names_to_read[1:]].to_numpy()
        timestamp = df[column_names_to_read[0]].to_numpy()

        try:
          pfpm_days      = pfpm.reshape(-1, points_per_day, pfpm.shape[1])
          timestamp_days = timestamp.reshape(-1, points_per_day)
        except:
          print(f'DataBuffer.read_data_from(): Skipped {filename} (failed to reshape it as {(-1, points_per_day, len(column_names_to_read))})')
          continue
        
        # remove the days that contain NaN; 
        wanted_days_bool = ~np.isnan(pfpm_days).any(axis=(1, 2))
        pfpm_days        = pfpm_days[wanted_days_bool] / power_base # normalize the data
        timestamp_days   = timestamp_days[wanted_days_bool]
        
        pfpm_array       = pfpm_days.reshape(-1, pfpm_days.shape[2])
        timestamp_array  = timestamp_days.reshape(-1)
        
        self.list_of_darrays.append(pfpm_array) 
        self.list_of_tarrays.append(timestamp_array)

  def prepare_data(self) -> np.ndarray:
    darray = np.concatenate(self.list_of_darrays)
    tarray = np.concatenate(self.list_of_tarrays)
    
    df = pd.DataFrame(darray, columns=['Pf', 'Pm'])
    df.insert(0, 't', tarray)
    df = df.sort_values(by=['t'])
    return df.to_numpy()
