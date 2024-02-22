import pandas as pd

# Load the data
design_matrix_df = pd.read_csv('/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/output_new_nort/CCTHot_lev1_output/task_CCTHot_rtmodel_no_rt/simplified_events/sub-061_task-CCTHot_design-matrix.csv')

TR = 0.68  # repetition time in seconds

# Prepare an empty dataframe for the simplified design matrix
simplified_df = pd.DataFrame(columns=['onset', 'duration', 'amplitude', 'regressor'])

for regressor in design_matrix_df.columns:
    is_previous_zero = True  # to track the start of non-zero segments
    
    for index, value in enumerate(design_matrix_df[regressor]):
        if value != 0:
            if is_previous_zero:  # this is the start of a non-zero segment
                onset = index * TR
                start_idx = index  # store start index to calculate amplitude later
            is_previous_zero = False
        elif not is_previous_zero:  # this is the end of a non-zero segment
            duration = (index - start_idx) * TR
            amplitude = design_matrix_df[regressor].iloc[start_idx:index].mean()
            
            # Append to the simplified dataframe using loc
            simplified_df.loc[len(simplified_df)] = [onset, duration, amplitude, regressor]
            
            is_previous_zero = True

# Handle cases where the last value in a regressor is non-zero
for regressor in design_matrix_df.columns:
    last_value = design_matrix_df[regressor].iloc[-1]
    if last_value != 0:
        duration = (len(design_matrix_df) - start_idx) * TR
        amplitude = design_matrix_df[regressor].iloc[start_idx:].mean()
        simplified_df.loc[len(simplified_df)] = [onset, duration, amplitude, regressor]

# Save the results
simplified_df.to_csv('simplified_design_matrix.csv', index=False)
