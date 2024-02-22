#!/usr/bin/env python

import glob
from pathlib import Path

def get_subids(root):
    subdirs = glob.glob(f'{root}/s*/')
    subid = [val[-4:-1] for val in subdirs]
    return subid


# tasks = ['stroop', 'ANT',  'CCTHot', 'stopSignal', 'twoByTwo',
#         'WATT3', 'discountFix', 'DPX', 'motorSelectiveStop']
tasks = ['ANT']


batch_stub = ('/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code'
              '/run_stub.batch')
root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS'


# Do not change these values without asking Jeanette first
rt_mapping = {
    'stroop': ['rt_centered'],
    'ANT': ['rt_centered'],
    'CCTHot': ['no_rt'], 
    'stopSignal':['rt_centered'], 
    'twoByTwo': ['rt_centered'], 
    'WATT3': ['no_rt'],
    'discountFix': ['rt_centered'], 
    'DPX': ['rt_centered'], 
    'motorSelectiveStop': ['rt_centered']
}
# rt_mapping = {
#     'stroop': ['no_rt'],
#     'ANT': ['no_rt'],
#     'CCTHot': ['no_rt'], 
#     'stopSignal':['no_rt'], 
#     'twoByTwo': ['no_rt'], 
#     'WATT3': ['no_rt'],
#     'discountFix': ['no_rt'], 
#     'DPX': ['no_rt'], 
#     'motorSelectiveStop': ['no_rt']
# }

subids = get_subids(root)

for task in tasks:
    batch_root = Path(f'/oak/stanford/groups/russpold/data/uh2/aim1/'
                      f'derivatives/output_ANT_noderivs/{task}_lev1_output/batch_files/')
    batch_root.mkdir(parents=True, exist_ok=True)
    rt_options = rt_mapping[task]
    for rt_inc in rt_options:
        batch_file = (f'{batch_root}/task_{task}_rtmodel_{rt_inc}_simplified.batch')   
        with open(batch_stub) as infile, open(batch_file, 'w') as outfile:
            for line in infile:
                line = line.replace('JOBNAME', f"{task}_{rt_inc}")
                outfile.write(line)
            for sub in subids:
                outfile.write(
                    f"echo /oak/stanford/groups/russpold/data/uh2/aim1/"
                    f"analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --simplified_events --omit_deriv \n"
                    f"/oak/stanford/groups/russpold/data/uh2/aim1/"
                    f"analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --simplified_events --omit_deriv \n")

print('Done!')