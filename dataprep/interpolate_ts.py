'''
some ts of kinect images have been manually obtained
this script will interpolate the missing ts for other images
'''

import os
from datetime import datetime, timedelta
import numpy as np
base_date = datetime(1970, 1, 1)

def get_ms_from_str(ts_str):
    t=[float(t) for t in ts_str.split(':')]
    ms=t[0]*3600*1000+t[1]*60*1000+t[2]*1000
    return ms


path=r'D:\hand_depth_dataset\kinect'
files=os.listdir(path)
for f in files:
    visual_ts_path=os.path.join(path,f,'visual_ts.txt')
    with open(visual_ts_path, 'r') as file:
        lines = file.readlines()
        visual_ts=[x.strip() for x in lines]
        f1,ts1=visual_ts[0].split(' ')
        f2,ts2=visual_ts[1].split(' ')
        ms1=get_ms_from_str(ts1)
        ms2=get_ms_from_str(ts2)
        #read all files from the dir
        all_files=os.listdir(os.path.join(path,f,'color'))
        all_files.sort()
        idx1=all_files.index(f1+'.jpg')
        idx2=all_files.index(f2+'.jpg')
        ts_list=[np.nan]*len(all_files)
        ts_list[idx1]=ms1
        ts_list[idx2]=ms2
        ts_list=np.array(ts_list)

        valid_indices = np.where(~np.isnan(ts_list))[0]
        valid_values = ts_list[valid_indices]
        missing_indices = np.where(np.isnan(ts_list))[0]
        interpolated_values = np.interp(missing_indices, valid_indices, valid_values)
        ts_list[missing_indices] = interpolated_values

        










