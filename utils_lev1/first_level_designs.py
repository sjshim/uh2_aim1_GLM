from nilearn.glm.first_level import compute_regressor
import numpy as np
import pandas as pd


def get_mean_rt(task):
    """
    Grabs precomputed mean RT from 
    /oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/utils_lev1/rt_mns.csv
    """
    mn_rt_file = ('/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/'
        'utils_lev1/rt_mns.csv')
    mn_rts = pd.read_csv(mn_rt_file)
    mn_rt_task = mn_rts[task].values[0]
    return mn_rt_task


def make_regressor_and_derivative(n_scans, tr, events_df, add_deriv,
                   amplitude_column=None, duration_column=None,
                   onset_column=None, subset=None, demean_amp=False, 
                   cond_id = 'cond'):
    """ Creates regressor and derivative using spm + derivative option in
        nilearn's compute_regressor
        Input:
          n_scans: number of scans
          tr: time resolution in seconds
          events_df: events data frame
          add_deriv: "yes"/"no", whether or not derivatives of regressors should
                     be included
          amplitude_column: Required.  Amplitude column from events_df
          duration_column: Required.  Duration column from events_df
          onset_column: optional.  if not specified "onset" is the default
          subset: optional.  Boolean for subsetting rows of events_df
          demean_amp: Whether amplitude should be mean centered
          cond_id: Name for regressor that is created.  Note "cond_derivative" will
            be assigned as name to the corresponding derivative
        Output:
          regressors: 2 column pandas data frame containing main regressor and derivative
    """
    if subset == None:
        events_df['temp_subset'] = True
        subset = 'temp_subset == True'
    if onset_column == None:
        onset_column = 'onset'
    if amplitude_column == None or duration_column == None:
        print('Must enter amplitude and duration columns')
        return
    if amplitude_column not in events_df.columns:
        print("must specify amplitude column that exists in events_df")
        return
    if duration_column not in events_df.columns:
        print("must specify duration column that exists in events_df")
        return

    reg_3col = events_df.query(subset)[[onset_column, duration_column, amplitude_column]]
    reg_3col = reg_3col.rename(
        columns={duration_column: "duration",
        amplitude_column: "modulation"})
    if demean_amp:
        reg_3col['modulation'] = reg_3col['modulation'] - \
        reg_3col['modulation'].mean()
    if add_deriv == 'deriv_yes':
        hrf_model = 'spm + derivative'
    else:
        hrf_model= 'spm'
    
    regressor_array, regressor_names = compute_regressor(
        np.transpose(np.array(reg_3col)),
        hrf_model,
        #this deals with fmriprep slice timing setting
        np.arange(n_scans)*tr+tr/2,
        con_id=cond_id
    ) 
    regressors =  pd.DataFrame(regressor_array, columns=regressor_names)  
    return regressors, reg_3col


def define_nuisance_trials(events_df, task):
    """
    Splits junk trials into omission, commission and too_fast, with the exception
    of twoByTwo where too_fast also includes first trial of block
    Note, these categories do not apply to WATT3 or CCTHot
    inputs: 
        events_df: the pandas events data frame
        task: The task name
    output:
        too_fast, omission, commission: indicators for each junk trial type
    """
    if task in ['DPX', 'stroop']:
        omission = (events_df.key_press == -1)
        commission = ((events_df.key_press != events_df.correct_response) &
                      (events_df.key_press != -1) &
                      (events_df.response_time >= .2))
        too_fast = (events_df.response_time < .2) 
    if task in ['ANT']:
        omission = ((events_df.key_press == -1) & (events_df.trial_id == 'stim'))
        commission = ((events_df.key_press != events_df.correct_response) &
                    (events_df.key_press != -1) &
                    (events_df.response_time >= .2) &
                    (events_df.trial_id == 'stim'))
        too_fast = ((events_df.response_time < .2) & (events_df.trial_id == 'stim'))
    if task in ['twoByTwo']:
        omission = (events_df.key_press == -1)
        commission = ((events_df.key_press != events_df.correct_response) &
                      (events_df.key_press != -1) & 
                      (events_df.response_time >= .2))
        too_fast = ((events_df.response_time < .2) |
                    (events_df.first_trial_of_block == 1))
    if task in ['stopSignal']:
        omission = ((events_df.trial_type == 'go') &
                    (events_df.key_press == -1))
        commission = ((events_df.trial_type == 'go') &
                      (events_df.key_press != events_df.correct_response) &
                      (events_df.response_time >= .2))
        too_fast = ((events_df.trial_type == 'go') &
                    (events_df.key_press != -1) &
                    (events_df.response_time < .2))
    if task in ['motorSelectiveStop']:
        trial_type_list = ['crit_go', 'noncrit_nosignal', 'noncrit_signal']
        omission = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.key_press == -1))
        commission = ((events_df.trial_type.isin(trial_type_list)) &
                      (events_df.key_press != events_df.correct_response) &
                      (events_df.response_time >= .2))
        too_fast = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.key_press != -1) &
                    (events_df.response_time < .2))
    if task in ['discountFix']:  
        omission = (events_df.key_press == -1)
        commission = 0*omission
        too_fast = (events_df.response_time < .2)
    events_df['omission'] = 1 * omission
    events_df['commission'] = 1 * commission
    events_df['too_fast'] = 1 * too_fast
    percent_junk = np.mean(omission | commission | too_fast)
    return events_df, percent_junk

def rename_columns(df, prefix):
    onset_column = 'onset' if 'onset' in df.columns else 'button_onset'
    renamed_columns = {}
    for col in df.columns:
        if col == onset_column:
            renamed_columns[col] = f"{prefix}_{onset_column}"
        else:
            renamed_columns[col] = f"{prefix}_{col}"
    return df.rename(columns=renamed_columns)

def merge_dataframes(df1, df2, prefix1, prefix2):
    onset_col_1 = f"{prefix1}_onset" if f"{prefix1}_onset" in df1.columns else f"{prefix1}_button_onset"
    onset_col_2 = f"{prefix2}_onset" if f"{prefix2}_onset" in df2.columns else f"{prefix2}_button_onset"
    return df1.merge(df2, left_on=onset_col_1, right_on=onset_col_2, how='outer')

def make_basic_stroop_desmat(
    events_file, add_deriv, regress_rt, n_scans, tr, 
    confound_regressors
):
    """Creates basic stroop regressors (and derivatives) 
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk =  define_nuisance_trials(events_df, 'stroop')
    events_df['constant_1_column'] = 1 
    events_df['incongruent'] = 0
    events_df.loc[events_df.trial_type == 'incongruent', 'incongruent'] = 1
    events_df['congruent'] = 0
    events_df.loc[events_df.trial_type == 'congruent', 'congruent'] = 1
    subset_main_regressors = 'too_fast == 0 and commission == 0 and omission == 0 and onset > 0' 

    too_fast_regressor, too_fast_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor, omission_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor, commission_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'commission'
        )
    congruent, congruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="congruent", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id ='congruent'
    )
    incongruent, incongruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="incongruent", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='incongruent'
    )
    design_matrix = pd.concat([congruent, incongruent,
        too_fast_regressor, omission_regressor, commission_regressor, confound_regressors], axis=1)
    
    too_fast_3col = rename_columns(too_fast_3col, 'too_fast')
    omission_3col = rename_columns(omission_3col, 'omission')
    commission_3col = rename_columns(commission_3col, 'commission')
    congruent_3col = rename_columns(congruent_3col, 'congruent')
    incongruent_3col = rename_columns(incongruent_3col, 'incongruent')

    design_matrix_3col = merge_dataframes(congruent_3col, incongruent_3col, 'congruent', 'incongruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, too_fast_3col, 'congruent', 'too_fast')
    design_matrix_3col = merge_dataframes(design_matrix_3col, omission_3col, 'congruent', 'omission')
    design_matrix_3col = merge_dataframes(design_matrix_3col, commission_3col, 'congruent', 'commission')

    # Optionally, sort the resulting dataframe by 'all_task_onset' for better clarity
    design_matrix_3col.sort_values(by='congruent_onset', inplace=True)

    contrasts = {
        "stroop_incong_minus_cong": "incongruent - congruent",
        "stroop_cong_v_baseline": "congruent",
        "stroop_incong_v_baseline":"incongruent",
        "task": "1/2*incongruent + 1/2*congruent" #Sunjae Check this for me!
        }
    if regress_rt == 'rt_centered':
        mn_rt = get_mean_rt('stroop')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, design_matrix_3col

def make_basic_ant_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic ANT regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    # stick cue regressors (onset: cue onset, duration: 0.5) at the beginning of each trial
    # stick probe regressors (onset: stim onset, duration: 1) at the end of each trial
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'ANT')
    subset_main_regressors = ('too_fast == 0 and commission == 0'
                              'and omission == 0 and onset > 0 and trial_id == "stim"') 
    events_df['constant_1_column'] = 1
    events_df['constant_0.5_column'] = 0.5

    events_df['cue_parametric'] = 0
    events_df.loc[events_df.cue == 'double', 'cue_parametric'] = 1
    events_df.loc[events_df.cue == 'spatial', 'cue_parametric'] = -1

    events_df['congruency_parametric'] = 0
    events_df.loc[events_df.flanker_type == 'incongruent', 'congruency_parametric'] = 1
    events_df.loc[events_df.flanker_type == 'congruent', 'congruency_parametric'] = -1

    events_df['cue_congruency_interaction'] = events_df.cue_parametric.values *\
                                              events_df.congruency_parametric.values 
    events_df['next_flanker_type'] = events_df['flanker_type'].shift(-1)
    events_df['previous_cue_type'] = events_df['cue'].shift(1) 

    too_fast_regressor, too_fast_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset="onset > 0 and trial_id == 'stim'", demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor, omission_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="onset > 0 and trial_id == 'stim'", demean_amp = False, cond_id = 'omission'
        )
    commission_regressor, commission_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="onset > 0 and trial_id == 'stim'", demean_amp = False, cond_id = 'commission'
        )
    cue_double_congruent, cue_double_congruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_0.5_column",
        subset="onset > 0 and cue == 'double' and next_flanker_type == 'congruent'",
        demean_amp=False, cond_id='cue_double_congruent'
    )
    cue_double_incongruent, cue_double_incongruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_0.5_column",
        subset="onset > 0 and cue == 'double' and next_flanker_type == 'incongruent'",
        demean_amp=False, cond_id='cue_double_incongruent'
    )
    cue_spatial_congruent, cue_spatial_congruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_0.5_column",
        subset="onset > 0 and cue == 'spatial' and next_flanker_type == 'congruent'",
        demean_amp=False, cond_id='cue_spatial_congruent'
    )
    cue_spatial_incongruent, cue_spatial_incongruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_0.5_column",
        subset="onset > 0 and cue == 'spatial' and next_flanker_type == 'incongruent'",
        demean_amp=False, cond_id='cue_spatial_incongruent'
    )
    probe_double_congruent, probe_double_congruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset="onset > 0 and previous_cue_type == 'double' and flanker_type == 'congruent'",
        demean_amp=False, cond_id='probe_double_congruent'
    )
    probe_double_incongruent, probe_double_incongruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset="onset > 0 and previous_cue_type == 'double' and flanker_type == 'incongruent'",
        demean_amp=False, cond_id='probe_double_incongruent'
    )
    probe_spatial_congruent, probe_spatial_congruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset="onset > 0 and previous_cue_type == 'spatial' and flanker_type == 'congruent'",
        demean_amp=False, cond_id='probe_spatial_congruent'
    )
    probe_spatial_incongruent, probe_spatial_incongruent_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset="onset > 0 and previous_cue_type == 'spatial' and flanker_type == 'incongruent'",
        demean_amp=False, cond_id='probe_spatial_incongruent'
    )

    # version 2 regressors
    # cue_double, cue_double_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="constant_1_column", duration_column="constant_0.5_column",
    #     subset="onset > 0 and cue == 'double'", 
    #     demean_amp=False, cond_id='cue_double'
    # )
    # cue_spatial, cue_spatial_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="constant_1_column", duration_column="constant_0.5_column",
    #     subset="onset > 0 and cue == 'spatial'", 
    #     demean_amp=False, cond_id='cue_spatial'
    # )
    # probe_congruent, probe_congruent_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="constant_1_column", duration_column="constant_1_column",
    #     subset=subset_main_regressors + " and flanker_type == 'congruent'", 
    #     demean_amp=False, cond_id='probe_congruent'
    # )
    # probe_incongruent, probe_incongruent_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="constant_1_column", duration_column="constant_1_column",
    #     subset=subset_main_regressors + " and flanker_type == 'incongruent'", 
    #     demean_amp=False, cond_id='probe_incongruent'
    # )

    # version 1 regressors
    # cue_parametric, cue_parametric_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="cue_parametric", duration_column="constant_1_column",
    #     subset=subset_main_regressors, demean_amp = True, cond_id = 'cue_parametric'
    # )
    # congruency_parametric, congruency_parametric_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="congruency_parametric", duration_column="constant_1_column",
    #     subset=subset_main_regressors, demean_amp=True, cond_id='congruency_parametric'
    # )
    # cue_congruency_interaction, interaction_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="cue_congruency_interaction", duration_column="constant_1_column",
    #     subset=subset_main_regressors, demean_amp=True, cond_id='interaction'
    # )
    # all_trials, all_trials_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
    #     amplitude_column="constant_1_column", duration_column="constant_1_column",
    #     subset=subset_main_regressors, demean_amp=False, cond_id='task'
    # )
    #cue parametric at onset of cue
    # cue_regressor, cue_3col = make_regressor_and_derivative(
    #     n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
    #     amplitude_column="constant_1_column", duration_column="constant_1_column",
    #     subset="trial_id != 'stim' and onset > 0", demean_amp=False, cond_id='cue'
    # )
    design_matrix = pd.concat([
        cue_double_congruent, cue_double_incongruent, cue_spatial_congruent, cue_spatial_incongruent,
        probe_double_congruent, probe_double_incongruent, probe_spatial_congruent, probe_spatial_incongruent,
        too_fast_regressor, omission_regressor, commission_regressor, confound_regressors
    ], axis=1)
    
    too_fast_3col = rename_columns(too_fast_3col, 'too_fast')
    omission_3col = rename_columns(omission_3col, 'omission')
    commission_3col = rename_columns(commission_3col, 'commission')
    cue_double_congruent_3col = rename_columns(cue_double_congruent_3col, 'cue_double_congruent')
    cue_double_incongruent_3col = rename_columns(cue_double_incongruent_3col, 'cue_double_incongruent')
    cue_spatial_congruent_3col = rename_columns(cue_spatial_congruent_3col, 'cue_spatial_congruent')
    cue_spatial_incongruent_3col = rename_columns(cue_spatial_incongruent_3col, 'cue_spatial_incongruent')
    probe_double_congruent_3col = rename_columns(probe_double_congruent_3col, 'probe_double_congruent')
    probe_double_incongruent_3col = rename_columns(probe_double_incongruent_3col, 'probe_double_incongruent')
    probe_spatial_congruent_3col = rename_columns(probe_spatial_congruent_3col, 'probe_spatial_congruent')
    probe_spatial_incongruent_3col = rename_columns(probe_spatial_incongruent_3col, 'probe_spatial_incongruent')
    
    # Merge dataframes and ensure the correct columns are being used
    design_matrix_3col = merge_dataframes(cue_double_congruent_3col, cue_double_incongruent_3col, 'cue_double_congruent', 'cue_double_incongruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, cue_spatial_congruent_3col, 'cue_double_congruent', 'cue_spatial_congruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, cue_spatial_incongruent_3col, 'cue_double_congruent', 'cue_spatial_incongruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, probe_double_congruent_3col, 'cue_double_congruent', 'probe_double_congruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, probe_double_incongruent_3col, 'cue_double_congruent', 'probe_double_incongruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, probe_spatial_congruent_3col, 'cue_double_congruent', 'probe_spatial_congruent')
    design_matrix_3col = merge_dataframes(design_matrix_3col, probe_spatial_incongruent_3col, 'cue_double_congruent', 'probe_spatial_incongruent')

    # Optionally, sort the resulting dataframe by 'all_task_onset' for better clarity
    design_matrix_3col.sort_values(by='cue_double_congruent_onset', inplace=True)

    contrasts = {
                    'cue_parametric': '0.25*(cue_double_congruent+cue_double_incongruent+probe_double_congruent+probe_double_incongruent)'+
                    '-0.25*(cue_spatial_congruent+cue_spatial_incongruent+probe_spatial_congruent+probe_spatial_incongruent)',
                    'congruency_parametric': '0.25*(cue_double_incongruent+cue_spatial_incongruent+probe_double_incongruent+probe_spatial_incongruent)'+
                    '-0.25*(cue_double_congruent+cue_spatial_congruent+probe_double_congruent+probe_spatial_congruent)',
                    'task': 'cue_double_congruent+cue_double_incongruent+cue_spatial_congruent+cue_spatial_incongruent'+
                    '+probe_double_congruent+probe_double_incongruent+probe_spatial_congruent+probe_spatial_incongruent'
                }
    if regress_rt == 'rt_centered':
        mn_rt = get_mean_rt('ANT')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt, rt_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
            amplitude_column="response_time_centered", duration_column="constant_1_column",
            subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        )
        design_matrix = pd.concat([design_matrix, rt], axis=1)  # Use rt DataFrame here
        contrasts["response_time"] = "response_time"

    if regress_rt == 'rt_uncentered':
        rt, rt_3col = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
            amplitude_column="response_time", duration_column="constant_1_column",
            subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        )
        design_matrix = pd.concat([design_matrix, rt], axis=1)  # Use rt DataFrame here
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, design_matrix_3col


def make_basic_ccthot_desmat(events_file, add_deriv, regress_rt, 
    n_scans, tr, confound_regressors
):
    """Creates basic CCTHot regressors (and derivatives)
       Input:
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return:
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    # no junk trial definition for this task
    percent_junk = 0
    events_df['constant_1_column'] = 1  
    end_round_idx = events_df.index[events_df.trial_id == 'ITI']
    # shift by 1 to next trial start, ignoring the last feedback trial
    start_round_idx = [0] + [x+1 for x in end_round_idx[:-1]]
    assert len(end_round_idx) == len(start_round_idx)
    events_df['trial_start'] = False
    events_df.loc[start_round_idx, 'trial_start'] = True

    trial_durs = []
    for start_idx, end_idx in zip(start_round_idx, end_round_idx):
            # Note, this automatically excludes the ITI row
        trial_durs.append(
            events_df.iloc[start_idx:end_idx]
                            ['duration'].sum()
        )
    events_df['trial_duration'] = np.nan
    events_df.loc[start_round_idx, 'trial_duration'] = trial_durs
    
    all_task, all_task_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="trial_duration",
        subset='trial_start==True and onset > 0', demean_amp=False, cond_id='task'
    )
    events_df['button_onset'] = events_df.onset+events_df.response_time
    pos_draw, pos_draw_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="EV", duration_column="constant_1_column",
        onset_column='button_onset',
        subset='trial_start==True', demean_amp=True, 
        cond_id='positive_draw'
    )
    events_df['absolute_loss_amount'] = np.abs(events_df.loss_amount)
    neg_draw, neg_draw_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="absolute_loss_amount", duration_column="constant_1_column",
        onset_column='button_onset',
        subset="action=='draw_card' and clicked_on_loss_card==1 and onset > 0", demean_amp=True, 
        cond_id='negative_draw'
    )
    trial_gain, trial_gain_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="gain_amount", duration_column="trial_duration",
        subset="trial_start==True and gain_amount == gain_amount and onset > 0", demean_amp=True, 
        cond_id='trial_gain'
    )
    trial_loss, trial_loss_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="absolute_loss_amount", duration_column="trial_duration",
        subset="trial_start==True and onset > 0 and absolute_loss_amount == absolute_loss_amount", demean_amp=True, 
        cond_id='trial_loss'
    )
    design_matrix = pd.concat([all_task, pos_draw, neg_draw, trial_gain, 
        trial_loss, confound_regressors], axis=1)

    all_task_3col = rename_columns(all_task_3col, 'all_task')
    pos_draw_3col = rename_columns(pos_draw_3col, 'pos_draw')
    neg_draw_3col = rename_columns(neg_draw_3col, 'neg_draw')
    trial_gain_3col = rename_columns(trial_gain_3col, 'trial_gain')
    trial_loss_3col = rename_columns(trial_loss_3col, 'trial_loss')

    design_matrix_3col = merge_dataframes(all_task_3col, pos_draw_3col, 'all_task', 'pos_draw')
    design_matrix_3col = merge_dataframes(design_matrix_3col, neg_draw_3col, 'all_task', 'neg_draw')
    design_matrix_3col = merge_dataframes(design_matrix_3col, trial_gain_3col, 'all_task', 'trial_gain')
    design_matrix_3col = merge_dataframes(design_matrix_3col, trial_loss_3col, 'all_task', 'trial_loss')

    # Optionally, sort the resulting dataframe by 'all_task_onset' for better clarity
    design_matrix_3col.sort_values(by='all_task_onset', inplace=True)

    contrasts = {'task': 'task',
                'trial_loss': 'trial_loss',
                'trial_gain': 'trial_gain',
                'positive_draw': 'positive_draw',
                'negative_draw': 'negative_draw'
                }
    if regress_rt != 'no_rt':
        print('RT cannot be modeled for this task')
    return design_matrix, contrasts, percent_junk, design_matrix_3col


def make_basic_stopsignal_desmat(events_file, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic stop signal regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'stopSignal')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and '
                            'omission == 0 and onset > 0')
    events_df['constant_1_column'] = 1  
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'commission'
        )
    go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'go'", 
        demean_amp=False, cond_id='go'
    )
    stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'stop_success'", 
        demean_amp=False, cond_id='stop_success'
    )
    stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'stop_failure'", 
        demean_amp=False, cond_id='stop_failure'
    )
    design_matrix = pd.concat([go, stop_success, stop_failure, too_fast_regressor, 
        omission_regressor, commission_regressor, confound_regressors], axis=1)
    contrasts = {'go': 'go', 
                    'stop_success': 'stop_success',
                    'stop_failure': 'stop_failure',
                    'stop_success-go': 'stop_success-go',
                    'stop_failure-go': 'stop_failure-go',
                    'stop_success-stop_failure': 'stop_success-stop_failure',
                    'stop_failure-stop_success': 'stop_failure-stop_success',
                    'task': '.333*go + .333*stop_failure + .333*stop_success'
                  }
    if regress_rt == 'rt_centered':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        mn_rt = get_mean_rt('stopSignal')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


def make_basic_two_by_two_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic two by two regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requested.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'twoByTwo')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and '
                            'omission == 0 and onset > 0')
    events_df['constant_1_column'] = 1  
    events_df.trial_type = ['cue_'+c if c is not np.nan else 'task_'+t
                            for c, t in zip(events_df.cue_switch,
                                            events_df.task_switch)]
    events_df.trial_type.replace('cue_switch', 'task_stay_cue_switch',
                                inplace=True)

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'omission'
        )
    task_switch_900 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .900 and trial_type == 'task_switch'", 
        demean_amp=False, cond_id='task_switch_900'
    )
    task_stay_cue_switch_900 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .900 and trial_type == 'task_stay_cue_switch'",
        demean_amp=False, cond_id='task_stay_cue_switch_900'
    )
    cue_stay_900 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .900 and trial_type == 'cue_stay'", 
        demean_amp=False, cond_id='cue_stay_900'
    )
    task_switch_100 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .100 and trial_type == 'task_switch'", 
        demean_amp=False, cond_id='task_switch_100'
    )
    task_stay_cue_switch_100 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .100 and trial_type == 'task_stay_cue_switch'", 
        demean_amp=False, cond_id='task_stay_cue_switch_100'
    )
    cue_stay_100 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .100 and trial_type == 'cue_stay'", 
        demean_amp=False, cond_id='cue_stay_100'
    )
    design_matrix = pd.concat([task_switch_900, task_stay_cue_switch_900, 
        cue_stay_900, task_switch_100, task_stay_cue_switch_100, cue_stay_100, 
        too_fast_regressor, commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {'task_switch_cost_900': 'task_switch_900-task_stay_cue_switch_900',
                    'cue_switch_cost_900': 'task_stay_cue_switch_900-cue_stay_900',
                    'task_switch_cost_100': 'task_switch_100-task_stay_cue_switch_100',
                    'cue_switch_cost_100': 'task_stay_cue_switch_100-cue_stay_100',
                    'task_switch_cost': '(.5*task_switch_900+.5*task_switch_100)-'
                                        '(.5*task_stay_cue_switch_900+'
                                        '.5*task_stay_cue_switch_100)', 
                    'cue_switch_cost': '(.5*task_stay_cue_switch_900+'
                                       '.5*task_stay_cue_switch_100)'
                                       '-(.5*cue_stay_900+.5*cue_stay_100)', 
                    'task': '1/6*task_switch_900 + 1/6*task_switch_100 +' 
                            '1/6*task_stay_cue_switch_900 +'
                            ' 1/6*task_stay_cue_switch_100 + '
                            '1/6*cue_stay_900 + 1/6*cue_stay_100'}
    if regress_rt == 'rt_centered':
        mn_rt = get_mean_rt('twoByTwo')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


def make_basic_watt3_desmat(events_file, add_deriv, regress_rt, 
    n_scans, tr, confound_regressors
):
    """Creates basic WATT3 regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    # no junk trial definition for this task
    percent_junk = 0
    events_df['constant_1_column'] = 1  
    events_df[['practice_main', 'with_without', 'not_using']] = \
        events_df.condition.str.split(expand=True, pat='_')
    events_df.with_without = events_df.with_without.replace('without', -1)
    events_df.with_without = events_df.with_without.replace('with', 1)
    end_round_idx = events_df.index[events_df.trial_id == 'ITI']
    # shift by 1 to next trial start, ignoring the last ITI
    start_round_idx = [0] + [x+1 for x in end_round_idx[:-1]]
    assert len(end_round_idx) == len(start_round_idx)
    events_df['trial_start'] = False
    events_df.loc[start_round_idx, 'trial_start'] = True

    trial_durs = []
    for start_idx, end_idx in zip(start_round_idx, end_round_idx):
            # Note, this automatically excludes the ITI row
        trial_durs.append(
            events_df.iloc[start_idx:end_idx]
                            ['duration'].sum()
        )
    events_df['trial_duration'] = np.nan
    events_df.loc[start_round_idx, 'trial_duration'] = trial_durs
    
    all_task, all_task_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="trial_duration",
        subset="trial_start==True and onset > 0 and practice_main == 'PA'", demean_amp=False, cond_id='task'
    )
    task_parametric, task_parametric_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="with_without", duration_column="trial_duration",
        subset="trial_start==True and onset > 0", demean_amp=True, cond_id='task_parametric'
    )
    practice, practice_3col = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df = events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="trial_duration",
        subset="trial_start==True and onset > 0 and practice_main == 'UA'", demean_amp=False, cond_id="practice"
    )
    
    design_matrix = pd.concat([all_task, task_parametric, practice, confound_regressors], axis=1)
    all_task_3col = rename_columns(all_task_3col, 'all_task')
    task_parametric_3col = rename_columns(task_parametric_3col, 'task_parametric')
    practice_3col = rename_columns(practice_3col, 'practice')

    design_matrix_3col = merge_dataframes(all_task_3col, task_parametric_3col, 'all_task', 'task_parametric')
    design_matrix_3col = merge_dataframes(design_matrix_3col, practice_3col, 'all_task', 'practice')

    # Optionally, sort the resulting dataframe by 'all_task_onset' for better clarity
    design_matrix_3col.sort_values(by='all_task_onset', inplace=True)

    contrasts = {'task':'task',
                 'task_parametric':'task_parametric',
                 'practice':'practice'
                 }
    
    if regress_rt != 'no_rt':
        print('RT cannot be modeled for this task')
    return design_matrix, contrasts, percent_junk, design_matrix_3col


def make_basic_discount_fix_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic discount fix regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'discountFix')
    #commission and omission are all 0s by definition
    subset_main_regressors = ('too_fast == 0 and key_press != -1')
    events_df['constant_1_column'] = 1  
    events_df['choice_parametric'] = -1
    events_df.loc[events_df.trial_type == 'larger_later',
                  'choice_parametric'] = 1

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    task = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset='too_fast == 0', demean_amp=False, 
        cond_id='task'
    )
    choice = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="choice_parametric", duration_column="constant_1_column",
        subset='too_fast == 0', demean_amp=True, 
        cond_id='choice'
    )
    design_matrix = pd.concat([task, choice, too_fast_regressor, 
        confound_regressors], axis=1)
    contrasts = {'task': 'task',
                 'choice': 'choice'}
    if regress_rt == 'rt_centered':
        mn_rt = get_mean_rt('discountFix')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


def make_basic_dpx_desmat(events_file, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
    ):
    """Creates basic dpx regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'DPX')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and '
                            'omission == 0')
    events_df['constant_1_column'] = 1  
    percent_junk = np.mean(events_df['too_fast'])
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'omission'
        )
    AX = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'AX'", 
        demean_amp=False, cond_id='AX'
    )
    AY = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'AY'", 
        demean_amp=False, cond_id='AY'
    )
    BX = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'BX'", 
        demean_amp=False, cond_id='BX'
    )
    BY = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'BY'", 
        demean_amp=False, cond_id='BY'
    )
    design_matrix = pd.concat([AX, AY, BX, BY, 
        too_fast_regressor, commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {'AX': 'AX',
                 'BX': 'BX',
                 'AY': 'AY',
                 'BY': 'BY',
                 'task': '.25*AX + .25*BX + .25*AY + .25*BY',
                 'AY-BY': 'AY-BY', 
                 'BX-BY': 'BX-BY'}
    if regress_rt == 'rt_centered':
        mn_rt = get_mean_rt('DPX')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


def make_basic_motor_selective_stop_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
    ):
    """Creates basic Motor selective stop regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'motorSelectiveStop')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and omission == 0')
    events_df['constant_1_column'] = 1  
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'omission'
        )
    crit_go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_go'", 
        demean_amp=False, cond_id='crit_go'
    )
    crit_stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_stop_success'",
        demean_amp=False, cond_id='crit_stop_success'
    )
    crit_stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_stop_failure'", 
        demean_amp=False, cond_id='crit_stop_failure'
    )
    noncrit_signal = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'noncrit_signal'", 
        demean_amp=False, cond_id='noncrit_signal'
    )
    noncrit_nosignal = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'noncrit_nosignal'", 
        demean_amp=False, cond_id='noncrit_nosignal'
    )
    design_matrix = pd.concat([crit_go, crit_stop_success, crit_stop_failure,
        noncrit_signal, noncrit_nosignal,too_fast_regressor, 
        commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {'crit_go': 'crit_go',
                 'crit_stop_success': 'crit_stop_success',
                 'crit_stop_failure': 'crit_stop_failure',
                 'noncrit_signal': 'noncrit_signal',
                 'noncrit_nosignal': 'noncrit_nosignal',
                 'crit_stop_success-crit_go': 'crit_stop_success-crit_go', 
                 'crit_stop_failure-crit_go': 'crit_stop_failure-crit_go', 
                 'crit_stop_success-crit_stop_failure': 'crit_stop_success-crit_stop_failure',
                 'crit_go-noncrit_nosignal': 'crit_go-noncrit_nosignal',
                 'noncrit_signal-noncrit_nosignal': 'noncrit_signal-noncrit_nosignal',
                 'crit_stop_success-noncrit_signal': 'crit_stop_success-noncrit_signal',
                 'crit_stop_failure-noncrit_signal': 'crit_stop_failure-noncrit_signal',
                 'task': '.2*crit_go + .2*crit_stop_success +'
                         '.2*crit_stop_failure + .2*noncrit_signal + .2*noncrit_nosignal'
                 }
    if regress_rt == 'rt_centered':
        rt_subset = subset_main_regressors + " and trial_type!='crit_stop_success'"
        mn_rt = get_mean_rt('motorSelectiveStop')
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt_subset = subset_main_regressors + " and trial_type!='crit_stop_success'"
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

make_task_desmat_fcn_dict = {
        'stroop': make_basic_stroop_desmat,
        'ANT': make_basic_ant_desmat,
        'CCTHot': make_basic_ccthot_desmat,
        'stopSignal': make_basic_stopsignal_desmat,
        'twoByTwo': make_basic_two_by_two_desmat,
        'WATT3': make_basic_watt3_desmat,
        'discountFix': make_basic_discount_fix_desmat,
        'DPX': make_basic_dpx_desmat,
        'motorSelectiveStop': make_basic_motor_selective_stop_desmat
    }

