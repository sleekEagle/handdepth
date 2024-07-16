'''
some ts of kinect images have been manually obtained
this script will interpolate the missing ts for other images
'''

import os


path=r'D:\hand_depth_dataset\kinect'
files=os.listdir(path)
for f in files:
    visual_ts_path=os.path.join(path,f,'visual_ts.txt')
    with open(visual_ts_path, 'r') as file:
        lines = file.readlines()
        visual_ts=[x.strip() for x in lines]
        f1,ts1=visual_ts[0].split(' ')
        f2,ts2=visual_ts[1].split(' ')
        