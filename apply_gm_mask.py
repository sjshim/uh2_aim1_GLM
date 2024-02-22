from nilearn.image import load_img, math_img, resample_to_img
from nilearn.datasets import load_mni152_gm_mask
import glob
import os

aim1_output_dir = '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/output_new_nort/'
maps = sorted([i for i in glob.glob(aim1_output_dir+'*_lev2*/*/randomise*nii.gz') if 'gm' not in i])
gm_mask = load_mni152_gm_mask(resolution=2.2)

gm_img = load_img(gm_mask)

for f_name in maps:
    if 'tfce' not in f_name:
        out_name = f_name.replace('.nii.gz', '_gm-masked.nii.gz')
        print(out_name)
        resampled_gm_mask = resample_to_img(gm_mask, f_name, interpolation='nearest')
        math_img('data * gm_mask', data = f_name, gm_mask = resampled_gm_mask).to_filename(out_name)