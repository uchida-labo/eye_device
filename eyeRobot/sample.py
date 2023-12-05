import os

situation_name = 'calibration_sample'
date_path = '1205'
dir_path_excel = 'C:\\Users\\admin\\Desktop\\data\\' + situation_name + '\\excel_data\\' + date_path
dir_path_video = 'C:\\Users\\admin\\Desktop\\data\\' + situation_name + '\\video_data\\' + date_path

os.makedirs(dir_path_excel)
os.makedirs(dir_path_video)