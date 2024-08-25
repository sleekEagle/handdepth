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


path=r'D:\hand_depth_dataset\canon'
files=os.listdir(path)
for f in files:
    print('processing',f)
    visual_ts_path=os.path.join(path,f,'visual_ts.txt')
    int_ts_path=os.path.join(path,f,'interpolated_ts.txt')
    with open(visual_ts_path, 'r') as file:
        lines = file.readlines()
        visual_ts=[x.strip() for x in lines]
        f1,ts1=visual_ts[0].split(' ')
        f2,ts2=visual_ts[1].split(' ')
        ms1=get_ms_from_str(ts1)
        ms2=get_ms_from_str(ts2)
        #read all files from the dir
        all_files=os.listdir(os.path.join(path,f))
        all_files = [k for k in all_files if k.endswith('jpg')]
        all_files = [f.split('.')[0].zfill(5)+'.jpg' for f in all_files]
        all_files.sort()
        idx1=all_files.index(f1+'.jpg')
        idx2=all_files.index(f2+'.jpg')

        grad=(ms2-ms1)/(idx2-idx1)
        arange=np.arange(0,len(all_files))
        inter_vals=ms1+grad*(arange-idx1)

        #write the interpolates ts to file
        interpolated_ts_dict = dict(zip(all_files, inter_vals))
        with open(int_ts_path, 'w') as file:
            for file_name, ts in interpolated_ts_dict.items():
                file.write(f"{file_name} {ts}\n")















