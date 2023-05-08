#!/usr/bin/bash

all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim1/BIDS/derivatives/output/*lev1_output/batch_files/*simplified.batch)

for cur_batch in ${all_batch}
do
  sbatch ${cur_batch}
done
