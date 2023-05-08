# This should only be run once to obtain the mean RTs for each task. 
# The values are saved to a csv file (so they can easily be referenced)
# The csv is then used when analyze_lev1.py is used with rt_centered

import glob
import pandas as pd
import numpy as np


root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS'
tasks = ['stroop', 'ANT',  'stopSignal', 'twoByTwo',
                 'discountFix', 'DPX', 'motorSelectiveStop','CCTHot', 'WATT3']


rt_mns = {}

for task in tasks:
    events_files = glob.glob(
        f'{root}/sub-s*/ses-[0-9]/func/*{task}*tsv'
    )
    sub_mn_rts = []
    for events_file in events_files:
        events_file_pd = pd.read_csv(events_file, sep='\t')
        sub_mn_rts.append(events_file_pd.response_time.mean())
    rt_mns[task] = [np.mean(sub_mn_rts)]

rt_mns_df = pd.DataFrame.from_dict(rt_mns)
rt_mns_df.to_csv('/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/'
    'utils_lev1/rt_mns.csv', index=False)