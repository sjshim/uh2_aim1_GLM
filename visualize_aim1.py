import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from nilearn import plotting
from nilearn import image
import matplotlib

matplotlib.use("Agg")

bids = "/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/output_new_ANT/"

tfce = False
tstat = True
combine = True

tasks = ['ANT']

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
# Define slices
slices = np.linspace(-30, 70, 5)
for task in tasks:
    rtmodel = rt_mapping[task][0]
    print(task, rtmodel)
    visual_dir = bids + f"{task}_lev2_output/visualizations"
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    contrasts = sorted(
        glob.glob(bids + f"{task}_lev2_output/{task}*")
    )
    start = "contrast_"
    end = "_rtmodel"
    contrast_names = [c[c.find(start) + len(start): c.rfind(end)]
                    for c in contrasts]
    if (rtmodel == "no_rt") & ("response_time" in contrast_names):
        contrast_names.remove("response_time")
    contrast_names = sorted(list(set(contrast_names)))
    if tfce:
        f, axes = plt.subplots(
            len(contrast_names), 1, figsize=(20, len(contrast_names) * 5), squeeze=False
        )
        plt.suptitle(task, fontsize=20)
        for idx, contrast in enumerate(contrast_names):
            print(task, contrast)
            contrast_map = glob.glob(
                bids
                + f"{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel_{rtmodel}*/randomise_output_model_one_sampt_tfce_corrp_fstat1.nii.gz"
            )[0]
            display = plotting.plot_stat_map(
                contrast_map,
                cmap="bwr",
                axes=axes[idx][0],
                display_mode="z",
                cut_coords=slices,
                title=contrast,
            )
            display.close()
        pdf = visual_dir + f"/task-{task}_{rtmodel}_visualization_tfce.pdf"
        plt.subplots_adjust(hspace=0)
        f.savefig(pdf)
    if tstat:
        f, axes = plt.subplots(
            len(contrast_names), 1, figsize=(20, len(contrast_names) * 5), squeeze=False
        )
        plt.suptitle(task, fontsize=20)
        for idx, contrast in enumerate(contrast_names):
            contrast_map = glob.glob(
                bids
                + f"{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel_{rtmodel}*/randomise_output_model_one_sampt_tstat1_gm-masked.nii.gz"
            )[0]
            display = plotting.plot_stat_map(
                contrast_map,
                cmap="cold_hot",
                axes=axes[idx][0],
                display_mode="z",
                cut_coords=slices,
                title=contrast,
            )
            display.close()
        pdf = visual_dir + f"/task-{task}_{rtmodel}_visualization_tstat_gm-masked.pdf"
        plt.subplots_adjust(hspace=0)
        f.savefig(pdf)
    if combine:
        f, axes = plt.subplots(
            len(contrast_names), 1, figsize=(20, len(contrast_names) * 5), squeeze=False
        )
        plt.suptitle(task, fontsize=20)
        for idx, contrast in enumerate(contrast_names):
            mask_file = glob.glob(
                bids
                + f"{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel_{rtmodel}*/randomise_output_model_one_sampt_tfce_corrp_fstat1.nii.gz"
            )[0]
            mask_img = image.load_img(mask_file)
            mask = image.binarize_img(mask_img, threshold=0.95)
            contrast_map = glob.glob(
                bids
                + f"{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel_{rtmodel}*/randomise_output_model_one_sampt_tstat1.nii.gz"
            )[0]
            contrast_img = image.load_img(contrast_map)
            combined_map = image.math_img(
                "img1 * img2", img1=contrast_img, img2=mask)
            display = plotting.plot_stat_map(
                combined_map,
                cmap="cold_hot",
                axes=axes[idx][0],
                display_mode="z",
                cut_coords=slices,
                title=contrast,
            )
            display.close()
        pdf = (
            visual_dir
            + f"/task-{task}_{rtmodel}_visualization_tstat-tfce-thresholded.pdf"
        )
        plt.subplots_adjust(hspace=0)
        f.savefig(pdf)
