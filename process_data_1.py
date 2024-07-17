import os
import pandas as pd
import numpy as np

from copy import deepcopy
import sys

import bz2
import pickle
import _pickle as cPickle
from tqdm import tqdm
import seaborn as sns
import math
# import cateyes as ce

import json
from scipy.signal import medfilt
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import warnings
import numbers

import logging
logger = logging.getLogger(__name__)

# These lines allow me to see logging.info messages in my jupyter cell output
logger.handlers.clear() # clear from history
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)

plt.style.use('ggplot')

class subject_data():

    def __init__(self, subject_data_folder,
                 cbPatient = False, analyze_gaze = False,
                 offset_car_xz = False,
                 analysis_parameters_file = 'default_analysis_parameters.json',
                 use_cached_data = False,
                 gaze_contingent=False,
                 estimate_head_trackers = False,
                 session_num='S001'):

        self.analysis_parameters = json.load(open(analysis_parameters_file))

        self.sub_id = os.path.split(subject_data_folder)[-1]
        self.cbPatient = cbPatient
        self.data_parent_folder = os.path.split(subject_data_folder)[:-1][0]
        self.data_folder = subject_data_folder
        self.analyze_gaze = analyze_gaze
        self.raw_gaze_data = False
        self.gaze_contingent = gaze_contingent
        self.session_num = session_num
        self.estimate_head_trackers = estimate_head_trackers
        #self.update_tracker_directories_using_ppid(subject_folder)


        if not use_cached_data:
            self.experiment_settings = False  # Making the attribute known
            self.read_experiment_settings_from_file()  # self.experiment_settings is now set
            self.experiment_settings['figure_out_folder'] = self.get_figure_out_folder('figs')
            self.experiment_settings['analyze_gaze'] = analyze_gaze

        self.results = False
        self.resultsLeft = False # placeholders for data of CB patients separated by turn
        self.resultsRight = False
        self.time_series_results = False

        self.has_gaze_in_head = False
        self.has_head_pose = False

        self.pickle_dir = 'pickled-data' 

        if self.session_num != 'S001':
            self.pickle_path = os.path.join(self.pickle_dir, f'{self.sub_id}_{self.session_num}_pickle_dict.pbz2')
            if estimate_head_trackers:
                self.pickle_path = os.path.join(self.pickle_dir, f'{self.sub_id}_{self.session_num}_pickle_dict_estimate_head_track.pbz2')
        else: 
            self.pickle_path = os.path.join(self.pickle_dir, f'{self.sub_id}_pickle_dict.pbz2')
            if estimate_head_trackers:
                self.pickle_path = os.path.join(self.pickle_dir, f'{self.sub_id}_pickle_dict_estimate_head_track.pbz2')


        if os.path.isdir(self.pickle_dir) is False:
            os.makedirs(self.pickle_dir)

        self.trial_dict = False
        self.use_cached_data = use_cached_data

        if use_cached_data:

            f = bz2.BZ2File(self.pickle_path, "rb")
            print('self.pickle_path: ',self.pickle_path)
            pickle_dict = cPickle.load(f)
            f.close()

            self.trial_dict = pickle_dict['trial_dict']
            # self.analysis_parameters = pickle_dict['analysis_parameters']
            self.results = pickle_dict['results']

            self.has_gaze_in_head = pickle_dict['has_gaze_in_head']
            self.has_head_pose = pickle_dict['has_head_pose']

        else:

            self.read_trial_results_from_file()

            if '_cyclopeangaze_headPositionTracker_location_0' in self.results.columns:
                row = self.results.iloc[0]
                head_tracker_path = row['_cyclopeangaze_headPositionTracker_location_0'].split('/')[2:]
                t1_path = os.path.join(self.data_parent_folder, row['ppid'],*head_tracker_path)
                
                # replace "tracker" folder with "estimated_head_trackers" folder if using estimated head trackers
                if self.estimate_head_trackers: 
                    t1_path = t1_path.replace('trackers', 'estimated_head_trackers')                    
                
                head_tracker_df = pd.read_csv(t1_path)

                # Some subs have head tackers that lack a head pose matrix
                if len(head_tracker_df.iloc[0].filter(regex='4x4')) > 0:
                    self.has_head_pose = True

            if analyze_gaze:
                # self.import_raw_gaze_data() returns false if not found
                self.raw_gaze_data = self.import_raw_gaze_data() 
            else:
                self.raw_gaze_data = None

            if type(self.raw_gaze_data) is pd.DataFrame:
                self.has_gaze_in_head = True
            else:
                self.has_gaze_in_head = False

            if 'divergence' in self.results.columns:
                self.results = self.results.drop(['divergence'], axis=1)

            if offset_car_xz:
                logger.info('Adding car xz offset ' + str(offset_car_xz) + ' for ' + self.sub_id)
                self.experiment_settings['offset_car_xz'] = offset_car_xz
            else:
                self.experiment_settings['offset_car_xz'] = None
                #self.calc_divergence_for_all_trials()  # necessary to update after changing position

            self.results.reset_index(drop=True)
        
            

    def get_pl_export_folder(self, session):

        session_parent_folder = os.path.join(self.data_folder, session, 'PupilData')

        session_exports_folder_list = []
        [session_exports_folder_list.append(name) for name in os.listdir(session_parent_folder) if name[0] != '.'];
        pupil_session_string = session_exports_folder_list[-1]

        if not 'exports' in os.listdir(os.path.join(session_parent_folder, pupil_session_string)):
            logger.info('\nNo export folder found in Pupil Labs session folder. No gaze data imported.')
            return False

        exports_parent_folder = os.path.join(session_parent_folder, pupil_session_string, 'exports')

        exports_folder_list = []
        [exports_folder_list.append(name) for name in os.listdir(exports_parent_folder) if name[0] != '.'];

        if len(exports_folder_list) == 0:
            logger.info('\nNo subfolder found in found in Pupil Labs Export folder. No gaze data imported.')
            return False

        exports_folder_path = os.path.join(exports_parent_folder, exports_folder_list[-1])

        return exports_folder_path

    def import_raw_gaze_data(self):
        '''
        :param pupil_session_idx:  the index of the pupil session to use with respect to the list of folders in the
        # session directoy.  Typically, these start at 000 and go up from there.

        :return: gaze positions
        '''        
        def return_gaze_path(session):
            exports_folder = self.get_pl_export_folder(session)
    
            if exports_folder == False:
                self.raw_gaze_data = False
                return False
    
            else: return os.path.join(exports_folder, 'gaze_positions.csv')
        
        pupil_gazepositions_filepath = return_gaze_path(self.session_num)
        
        if pupil_gazepositions_filepath == False: # skip gaze data if exports doesn't exists
            return False

        # Defaults to the most recent pupil export folder (highest number)
        gaze_positions = pd.read_csv(pupil_gazepositions_filepath)
        
        if os.path.exists(os.path.join(self.data_folder, 'S002', 'PupilData')):
            pupil_gazepositions_filepath = return_gaze_path('S002')
            if pupil_gazepositions_filepath != False:
                new_gaze_positions = pd.read_csv(pupil_gazepositions_filepath)
                gaze_positions = pd.concat((gaze_positions, new_gaze_positions), axis=0)
        return gaze_positions

    def get_gaze_data_slice(self, start_time, end_time):

        # right now this only works if there is one exports file otherwise it throws a bool error
        firstIdx = list(map(lambda i: i > start_time, self.raw_gaze_data['gaze_timestamp'])).index(True)
        try:
            lastIdx = list(map(lambda i: i > end_time, self.raw_gaze_data['gaze_timestamp'])).index(True)
        except:
            end_time =  self.raw_gaze_data['gaze_timestamp'].values[-1]
            lastIdx = list(map(lambda i: i >= end_time, self.raw_gaze_data['gaze_timestamp'])).index(True)

        slice_of_data = self.raw_gaze_data.iloc[firstIdx:lastIdx + 1] # changed this to work for double session data -AG
        return slice_of_data

    def separate_calibration_trials(self):
        
        self.calibration_trial_results = self.results.loc[self.results['trialType'] == 'CalibrationAssessment']
        self.results = self.results.loc[self.results['trialType'] != 'CalibrationAssessment'] # now results will not have calibration trials
        
    def get_figure_out_folder(self, subfolder):
        # creates the dir subfolder/subID

        # make directory to save figures
        savePlotsPath = os.path.join(subfolder, self.sub_id)

        if os.path.isdir(savePlotsPath):
            pass
        else:
            os.makedirs(savePlotsPath)

        return savePlotsPath

    def update_tracker_directories_using_ppid(self, subject_folder):
        '''
        Update tracker directories in self.results.
        '''
        # make filename the ppid

        self.results['ppid'] = subject_folder
        list_carTransform_split = [verts.split('/') for verts in self.results['simplecar_carTransformMatrix_location_0']]
        list_roadVerts_split = [verts.split('/') for verts in self.results['road_vertices_location_0']]
        list_roadTransform_split = [verts.split('/') for verts in self.results['roadTransformMat_location_0']]

        if '_cyclopeangaze_headPositionTracker_location_0' in self.results.columns:
            list_headTransform_split = [verts.split('/') for verts in self.results['_cyclopeangaze_headPositionTracker_location_0']]
            trackers = [list_carTransform_split, list_roadVerts_split, list_roadTransform_split, list_headTransform_split]
        else:
            trackers = [list_carTransform_split, list_roadVerts_split, list_roadTransform_split]

        # takes care of case where ppid is changed after data collection
        for tracker_list in trackers:
            for i in range(len(tracker_list)):
                tracker_list[i][2] = 'S00' + str(self.results['session_num'].iloc[i])
                #print(tracker_list[i][-1])#[-7:-4])
                trial_num = str(self.results['trial_num'].iloc[i])
                if len(trial_num) == 2:
                    trial_num = '0' + trial_num
                if len(trial_num) == 1:
                    trial_num = '00' + trial_num

                tracker_list[i][-1] = tracker_list[i][-1].replace(tracker_list[i][-1][-7:-4], trial_num)
        
        filepaths = []
        for tracker_list in trackers:
            filepaths.append([os.path.join('carSettings',*row[1:]).replace('\\','/') for row in tracker_list])
        
        self.results['simplecar_carTransformMatrix_location_0'] = filepaths[0]
        self.results['road_vertices_location_0'] = filepaths[1]
        self.results['roadTransformMat_location_0'] = filepaths[2]

        if '_cyclopeangaze_headPositionTracker_location_0' in self.results.columns:
            self.results['_cyclopeangaze_headPositionTracker_location_0'] = filepaths[3]
        
    def read_experiment_settings_from_file(self):

        settings_path = os.path.join(self.data_folder, self.session_num, 'session_info', 'settings.json')

        with open(settings_path) as json_data:
            self.experiment_settings = json.load(json_data)

    def read_trial_results_from_file(self):

        trial_data_path = os.path.join(self.data_folder, self.session_num)
        self.results = pd.read_csv(os.path.join(trial_data_path, 'trial_results.csv'))
        if 'trialType' not in self.results.columns:
            # add trial type column to avoid throwing error
            self.results.loc[:,'trialType'] = np.nan
        
        self.separate_calibration_trials()
        
        self.update_tracker_directories_using_ppid(subject_folder=self.sub_id) # this fixes naming and trackers in case name was changed after data collection

        # drop any rows that have nans in trial results (reset trials) and report number
        num_reset_trials = len(self.results['turn_direction'].dropna()) - len(self.results['turn_direction'])

        if num_reset_trials != 0:
            indices_wo_nans = np.where(self.results['turn_direction'].notnull())[0]
            self.results = self.results.iloc[indices_wo_nans, :]
            print('Number of off-road reset trials for participant ' + self.sub_id + ': ' + str(abs(num_reset_trials)))

        # this is where we could change self.results to have only left or right turns
        # we want to run through everything twice, with all left, then all right turns

        if self.cbPatient == True:
            gb = self.results.groupby('turn_direction')
            groups = [gb.get_group(x) for x in gb.groups]
            if groups[0]['turn_direction'].iloc[0] == 'left':
                self.resultsLeft, self.resultsRight = groups[0].copy(), groups[1].copy()
            else: 
                self.resultsRight, self.resultsLeft = groups[0].copy(), groups[1].copy()
            
            if os.path.isdir(os.path.join(self.experiment_settings['figure_out_folder'], 'LEFT')): # folders to save plots separately
                pass
            else:
                os.makedirs(os.path.join(self.experiment_settings['figure_out_folder'],'LEFT'))
                os.makedirs(os.path.join(self.experiment_settings['figure_out_folder'],'RIGHT'))
        
            if 'trialType' not in self.resultsLeft.columns:
                self.resultsLeft.loc[:,'trialType'] = np.nan
                self.resultsRight.loc[:,'trialType'] = np.nan
                
    def get_trial_from_results(self, trial_results_row_in, analyze_gaze = False):

        '''
        in: a row from the subject trial dataframe (e.g. sub_all_trial_data)
        out: a trial_data object

        Useful when iterating over rows in the dataframe to compute new variables

        '''

        if self.use_cached_data:
            trial = self.trial_dict[trial_results_row_in.trial_num]
            trial.subject_data = self
            return trial

        # return trial_data(trial_results_row_in, self.experiment_settings, self.data_folder)
        return trial_data(trial_results_row_in, self, analyze_gaze)

    def get_trial_from_index(self, trial_index, analyze_gaze = False):

        '''
        in: an index to a row from the subject trial dataframe (e.g. sub_all_trial_data)
        out: a trial_data object

        Useful when iterating over rows in the dataframe to compute new variables

        '''

        if self.use_cached_data:
            trial = self.trial_dict[self.results.iloc[trial_index].trial_num]
            trial.subject_data = self
            return trial

        return trial_data(self.results.iloc[trial_index], self, analyze_gaze)

    def plot_car_trajectory_all_trials(self, plot_gaze_data=False):

        logger.info('Plotting all car trajectories for ' + self.sub_id)
        
        # Plot car trajectory
        for trialIndex, trial_results_row in self.results.iterrows():
            a_trial = self.get_trial_from_results(trial_results_row)
            
            if a_trial.results['trialType'] != 'CalibrationAssessment':
                a_trial.plot_car_trajectory(showFig=False, saveFig=True, flip_left_turns=True, saveFolder=a_trial.results['turn_direction'],plot_gaze_data=plot_gaze_data)

    def plot_mean_car_trajectory(self,
                                 trial_results_slice,
                                 showFig=True,
                                 saveFig=True,
                                 interp_time_seconds = 5,
                                 interp_res_seconds = (1 / 90),
                                 saveFolder = ''):


        interp_car_x_list = []
        interp_car_z_list = []

        for trialIndex, trial_results_row in trial_results_slice.iterrows():

            a_trial = self.get_trial_from_results(trial_results_row)

            (interp_timestamps, interp_car_x, interp_car_z) = a_trial.interpolate_car_pos_roadspace(interp_time_seconds=5, \
                                                                                                    interp_res_seconds=(
                                                                                                                1 / 90),
                                                                                                    flip_left_turns=True)

            interp_car_x_list.append(interp_car_x)
            interp_car_z_list.append(interp_car_z)

        mean_car_x = np.mean(interp_car_x_list, 0)
        mean_car_z = np.mean(interp_car_z_list, 0)

        plt.ioff()  # prevents the fig from showing if in notebook mode

        plt.figure(figsize=(8, 8))
        ax = plt.subplot()

        # Load the first trial to get the road trajectory
        a_trial = self.get_trial_from_results(trial_results_slice.iloc[0])

        a_trial.plot_road(ax, flip_left_turns=True)

        for rowIdx in np.arange((np.shape(interp_car_x_list)[0])):
            ax.plot(interp_car_x_list[rowIdx], interp_car_z_list[rowIdx], 'k', linewidth=0.5)

        ax.plot(mean_car_x, mean_car_z, 'r', linewidth=1.5)
        ax.plot(mean_car_x[0], mean_car_z[0], 'o')

        ax.text(.5, .5, self.sub_id + '\n' + \
                'Density: ' + str(a_trial.results['contrast']) + '\n' + \
                'Radius: ' + str(a_trial.results['turn_radius']), \
                horizontalalignment='center', \
                verticalalignment='center', transform=ax.transAxes, fontsize=15)

        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Vertical Distance (m)')

        ax.axis("equal")  # sets the x/y axes scales equal

        plt.ion()  # prevents the fig from showing if in notebook mode

        if saveFig:

            plotFigSubPath = os.path.join(self.experiment_settings['figure_out_folder'], saveFolder, 'mean_trajectories')
            fig_name = 'mean_trajectory_' + 'D-' + str(a_trial.results['contrast']) + '_R-' + str(
                a_trial.results['turn_radius'])

            if os.path.isdir(plotFigSubPath) is False:
                os.makedirs(plotFigSubPath)

            plt.savefig(os.path.join(plotFigSubPath, fig_name + '.png'))

        if showFig:
            plt.show()
        else:
            plt.close()
    
    
    
    def plot_time_vs_steer_bias_all_trials(self, time_start=0, interp_time_seconds=7.37, interp_res_seconds=(1 / 90), saveFolder='', turn_dir=None, collapse_across_OF=False, showFig=False, saveFig=False):

        '''
        By default this function returns a figures with subplots for all radii. If you only want
        a figure with one radius, pass it data that has one radius only.
        '''
        
        logger.info('Plotting time vs. steering bias for ' + self.sub_id)
        
        gb = self.results.groupby(['turn_direction'])
        if turn_dir == 'left':
             data = gb.get_group('left')
        elif turn_dir == 'right':
             data = gb.get_group('right')
        else: data = self.results
        
        placeholder_df = pd.DataFrame()
        
        for group_key, trial_results_slice in data.groupby(['turn_radius']):
            self.time_series_results = pd.DataFrame() # clear out data frame
            for row_idx, row in trial_results_slice.iterrows(): # have to group by contrast after
                a_trial = self.get_trial_from_results(row)   
                a_trial.plot_time_vs_steer_bias(showFig=False, saveFig=False, interp_time_seconds=interp_time_seconds)
            # average over radii and set placeholder df
            copy = self.time_series_results.copy(deep=True)
            copy = copy.groupby(by=copy.columns,axis=1).mean()
            copy.loc[:,'turn_radius'] = group_key[0]
            placeholder_df = pd.concat((placeholder_df, copy))
        
        self.time_series_results = placeholder_df.copy(deep=True)        

        if saveFig or showFig:
            
            str_graph_title = 'Effect of Optic Flow on Steering Bias over Time \nParticipant ' + str(self.results['ppid'].values[0])
            mean_time_series_df = self.time_series_results.groupby(by=self.time_series_results.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
            
            radiusList = np.unique(np.array(mean_time_series_df['turn_radius'].values,dtype=np.float64))
    
            # plots the steering bias over time
            idx = 0
            fontsize=14
            fig, ax = plt.subplots(1, 3, figsize=(14,6))
            plt.suptitle(str_graph_title, fontsize=fontsize+2)
            colors = sns.color_palette("bright")
        
            optic_flow_density = ['Low', 'Medium', 'High']
        
            for group_key, slices in mean_time_series_df.groupby(['turn_radius']):
                # finding only divergence by optic flow columns
                cols = np.array(slices.columns)
                index_chop_at_ppid = slices.columns.get_loc('turn_radius')
                cols = cols[1:index_chop_at_ppid]
                optic_flow_lvs = []
                for i in range(len(cols)):
                    optic_flow_lvs.append(cols[i][-3:])
                plt.ioff()
                if collapse_across_OF:
                    x = slices.iloc[:,1:len(cols)+1].mean(axis=1)
                    y = slices['interp_timestamps']
                    ax[idx].plot(x, y, linewidth=1.5, color='r') #label=optic_flow_lvs[contrast-1]) # get just density from title
    
                else:
                    for contrast in range(1, len(cols)+1):
                        x = slices.iloc[:,contrast]
                        y = slices['interp_timestamps']
                        ax[idx].plot(x, y, linewidth=1.5, label=optic_flow_density[contrast-1], color=colors[contrast-1]) #label=optic_flow_lvs[contrast-1]) # get just density from title
    
                ax[idx].axvline(x=2, ls = '--', color='k') # plots vertical dotted line
                ax[idx].set_ylabel('Time (sec)', fontsize=fontsize)
                ax[idx].set_xlabel('Distance to Inner Road Edge (m)', fontsize=fontsize)
                ax[idx].set_xlim(-0.2, 4.2)
                ax[idx].set_ylim(time_start, interp_time_seconds)
                ax[idx].vlines(x=0,ymin=np.min(y), ymax=np.max(y), color='k')
                ax[idx].vlines(x=4,ymin=np.min(y), ymax=np.max(y), color='k')
                ax[idx].set_title('Radius: ' + str(radiusList[idx]) + 'm', fontsize=fontsize)
                if idx == 1:
                    ax[idx].legend(title='Flow Density',fontsize='large', title_fontsize='large', loc='lower right')
                idx = idx + 1
            
            plt.tight_layout()
        
        if saveFig:
            save_path = str_graph_title.split(',')[0].replace(' ','-') + '-Time-' + str(time_start) + 'to' + str(interp_time_seconds) + '(s)'
            if collapse_across_OF:
                save_path = save_path + '-CollapseAcrossOF'
            plt.savefig(os.path.join(save_path.replace('\n', '') + '.png'), dpi = 300)
            print('File saved at: ', os.path.join(save_path.replace('\n', '') + '.png'))

        if showFig:
            plt.show()
        else:
            plt.close()

    
    
    
    def plot_mean_trajectory_for_all_conditions(self, showFig=False, interp_time_seconds=5, interp_res_seconds=(1 / 90),saveFolder='', turn_dir=None):

        logger.info('Plotting mean car trajectories for ' + self.sub_id)
        
        gb = self.results.groupby(['turn_direction'])
        if turn_dir == 'left':
             data = gb.get_group('left')
        elif turn_dir == 'right':
             data = gb.get_group('right')
        else: data = self.results
             
        for group_key, trial_results_slice in data.groupby(['contrast', 'turn_radius']):
            self.plot_mean_car_trajectory(trial_results_slice, showFig=False, interp_time_seconds=interp_time_seconds,
                                          interp_res_seconds=(1 / 90),saveFolder=saveFolder)

    def calc_divergence_for_all_trials(self):

        self.calc_divergence_over_segment_for_all_trials(start_percent=0, end_percent=100)

    def calc_divergence_over_segment_for_all_trials(self, start_percent = 0, end_percent = 100):

        logger.info('Calculating divergence for ' + self.sub_id)

        # lists that will be n trials long
        mean_div = []
        mean_abs_div = []
        mean_div_from_inner_road_edge = []

        if start_percent == 0 and end_percent == 100:
            label_suffix = ''
        else:
            label_suffix = '_' + str(start_percent) + '_' + str(end_percent)
        
        # Calculate mean divergence from road center
        for trialIndex, trial_results_row in self.results.iterrows():

            a_trial = self.get_trial_from_results(trial_results_row)
            a_trial.calc_mean_divergence_over_segment(start_percent=start_percent, end_percent=end_percent)

    def calc_mean_gaze_behavior_over_segment_for_all_trials(self, start_percent=0, end_percent=100):


        logger.info('Calculating calc_mean_gaze_behavior_over_segment for ' + self.sub_id)

        if start_percent == 0 and end_percent == 100:
            label_suffix = ''
        else:
            label_suffix = '_' + str(start_percent) + '_' + str(end_percent)

        # Calculate mean divergence from road center
        for trialIndex, trial_results_row in self.results.iterrows():
            a_trial = self.get_trial_from_results(trial_results_row)
            a_trial.calc_mean_gaze_behavior_over_segment(start_percent=start_percent, end_percent=end_percent)



    def offset_contrast_for_each_radius(self, magnitude=-0.05):
        '''
        Creates a new column, 'contrasts_offset_by_radius'.
        Prevents error bars from overlapping when a values is plotted by contrast (xaxis) and radius (lines)
        '''

        contrastList = np.array(self.results['contrast'].values, dtype=np.float64)
        radiusList = np.array(self.results['turn_radius'].values, dtype=np.float64)
        # drop nans from calibration assessment
        radiusList = radiusList[~np.isnan(radiusList)]
        contrastList = contrastList[~np.isnan(contrastList)]
        
        offsets_rad = [-magnitude, 0, magnitude]
        np.linspace(np.mean(contrastList) - magnitude, np.mean(contrastList) + magnitude, len(contrastList))

        for count, radiusValue in enumerate(np.unique(radiusList)):
            idx = np.where(radiusList == radiusValue)[0]
            contrastList[idx] = contrastList[idx] + offsets_rad[count]

        nans_for_cal_trials = np.repeat(np.nan, len(np.where(self.results['trialType']=='CalibrationAssessment')[0]))
        self.results['contrasts_offset_by_radius'] = np.concatenate((nans_for_cal_trials,contrastList),axis=0)
        
    def plot_for_contrast_x_radius(self, varName, yLabel, showFig=False, saveFig=True, size=(6, 8), saveFolder = '', turn_dir = None):

        logger.info('plot_for_contrast_x_radius: Plotting ' + varName + ' for ' + self.sub_id)

        if varName not in self.results.columns:
            logger.error('plot_for_contrast_x_radius: Variable ' + varName + ' not in <subject_data>.results')
            return None

        if 'offset_contrast_for_each_radius' not in self.results.columns:
            self.offset_contrast_for_each_radius()

        plt.ioff()

        the_fig = plt.figure(figsize=size)
        ax = plt.subplot()

        sns.set(style="ticks", rc={"lines.linewidth": 3})
        
        gb = self.results.groupby(['turn_direction'])
        if turn_dir == 'left':
             data = gb.get_group('left')
        elif turn_dir == 'right':
             data = gb.get_group('right')
        else: data = self.results
        
        sns.lineplot(

            data=data, x="contrasts_offset_by_radius", y=varName, hue="turn_radius", err_style="bars",
            errorbar=("ci", 95),
            palette="bright")
        #sns.scatterplot(self.results,x="contrasts_offset_by_radius",y=varName,hue="turn_radius",palette='bright')

            # data=self.results, x="contrasts_offset_by_radius", y=varName, hue="turn_radius", err_style="bars")#,
            #errorbar=("ci", 95),
            #palette="bright"
        #)

        ax.legend(loc='best', title='Turn Radius')
        ax.set_ylabel(yLabel)
        ax.set_xlabel('Optic Flow Density')

        xticks = np.unique(self.results['contrast'])
        ax.set_xticks(xticks[~np.isnan(xticks)])
        ax.set_xticklabels(['Zero', 'Low', 'High'])
        plt.ion()

        if saveFig:

            plotFigSubPath = os.path.join(self.experiment_settings['figure_out_folder'], saveFolder)
            fig_name = self.sub_id + '_' + varName + '_contrast_x_radius'

            if os.path.isdir(plotFigSubPath) is False:
                os.makedirs(plotFigSubPath)

            plt.savefig(os.path.join(plotFigSubPath, fig_name + '.png'), bbox_inches='tight')

        if showFig:
            plt.show()

        return the_fig
        
    def write_fake_car_data(self):

        for trialIndex, trial_results_row in self.results.iterrows():

            this_trial = self.get_trial_from_results(trial_results_row)
            this_trial.calculate_road_edges()
            this_trial.calculate_road_vertices_in_world()

            self.data_parent_folder = os.path.join(*os.path.split(self.data_folder)[:-1])

            full_path_to_car_data = trial_results_row['simplecar_carTransformMatrix_location_0'].split('/')
            fake_data_parent_folder = full_path_to_car_data[1:-1]
            car_data_filename = full_path_to_car_data[-1]

            ###########################################################################
            # first, align the car with the left edge of the road
            fake_data_parent_folder[-1] = 'fakecartrackers_car_on_left_road_edge'
            fake_car_data_parent_folder = os.path.join(self.data_parent_folder,*fake_data_parent_folder)

            if os.path.isdir(fake_car_data_parent_folder) is False:
                            os.makedirs(fake_car_data_parent_folder)

            vert_timestamps_linear = np.linspace(this_trial.car_data['time'].iloc[0], 
                                            this_trial.car_data['time'].iloc[-1], 
                                            len(this_trial.road_vertices));

            interp_road_x_world = np.interp(this_trial.car_data['time'], 
                                  vert_timestamps_linear,
                                  this_trial.road_vertices['roadedge_left_x_world'])

            interp_road_z_world = np.interp(this_trial.car_data['time'], 
                                  vert_timestamps_linear,
                                  this_trial.road_vertices['roadedge_left_z_world'])


            this_trial.car_data['pos_x'] = interp_road_x_world
            this_trial.car_data['pos_z'] = interp_road_z_world
            this_trial.car_data['simplecar_4x4_R0C3'] = interp_road_x_world
            this_trial.car_data['simplecar_4x4_R2C3'] = interp_road_z_world

            this_trial.car_data.to_csv(os.path.join(fake_car_data_parent_folder,car_data_filename))

            ###########################################################################
            # first, align the car with the right edge of the road
            fake_data_parent_folder[-1] = 'fakecartrackers_car_on_right_road_edge'
            fake_car_data_parent_folder = os.path.join(self.data_parent_folder,*fake_data_parent_folder)

            if os.path.isdir(fake_car_data_parent_folder) is False:
                            os.makedirs(fake_car_data_parent_folder)

            this_trial.car_data['pos_x'] = this_trial.road_vertices['roadedge_right_x_world']
            this_trial.car_data['pos_z'] = this_trial.road_vertices['roadedge_right_z_world']
            this_trial.car_data['simplecar_4x4_R0C3'] = this_trial.road_vertices['roadedge_right_x_world']
            this_trial.car_data['simplecar_4x4_R2C3'] = this_trial.road_vertices['roadedge_right_z_world']

            this_trial.car_data.to_csv(os.path.join(fake_car_data_parent_folder,car_data_filename))

    def calc_s2s_rmse_wheel_angle_for_all_trials(self):

        # Plot car trajectory
        for trialIndex, trial_results_row in self.results.iterrows():

            a_trial = self.get_trial_from_results(trial_results_row)
            a_trial.calc_rmse()

            # # plot the time series of wheel angle
            # if trialIndex == 0:
            #     plt.figure()
            #     plt.plot(np.array(a_trial.car_data['time']), np.array(a_trial.car_data['wheelAngle']), 'o')
            #     plt.title('Trial ' + str(trialIndex+1) + ' Wheel Angle vs. Time')
            #     plt.xlabel('Time (sec)')
            #     plt.ylabel('Wheel Angle (deg)')
            #     plt.show()

    def calc_mean_gaze_az_rel_car_for_all_trials(self, turn_dir=None):

        print(f'Calculating gaze az_rel_car for all trials.')

        mean_az_rel_car = []
        median_az_rel_car = []

        gb = self.results.groupby(['turn_direction'])
        if turn_dir == 'left':
             data = gb.get_group('left')
        elif turn_dir == 'right':
             data = gb.get_group('right')
        else: data = self.results
        
        # for trialIndex, trial_results_row in data.iterrows():
        for trialIndex, trial_results_row in tqdm(self.results.iterrows(), desc="Processing: " + self.sub_id,
                                                  unit='trials', total=len(self.results)):

            # print(f'Processing: {trialIndex}')
            a_trial = self.get_trial_from_results(trial_results_row, analyze_gaze=True)

            if not hasattr(a_trial, 'processed_gaze_data'):
                print('Processed gaze data not present.')
                mean_az_rel_car.append(np.nan)

            else:

                median_az = np.median(a_trial.processed_gaze_data['gaze_az_rel_car'])
                median_az_rel_car.append(median_az)

                mean_az = np.mean(a_trial.processed_gaze_data['gaze_az_rel_car'])
                mean_az_rel_car.append(mean_az)

                # a_trial.results['mean_gaze_az_rel_car'] = mean_az
                # a_trial.results['median_gaze_az_rel_car'] = median_az

        print(f'Done calculating gaze az_rel_car for all trials.')
        # self.results['mean_az_rel_car'] = mean_az_rel_car
        # self.results['median_az_rel_car'] = median_az_rel_car

    def save_experiment_trials_to_pickle(self, interp_sec = 5):
        '''
        This does not save any calibration trials to the pickle.
        This does not update your subject results file.
        That will require checking if the column exists and, if not, updating it.
        '''

        trial_dict ={}
        # for trialIndex, trial_results_row in self.results.iterrows():
        for trialIndex, trial_results_row in tqdm(self.results.iterrows(), desc="Processing: " + self.sub_id, unit='trials', total=len(self.results)):
            a_trial = self.get_trial_from_results(trial_results_row, analyze_gaze=self.analyze_gaze)

            a_trial.interpolate_divergence_roadspace(interp_time_seconds=interp_sec)
            a_trial.calc_rmse()

            a_trial.subject_data = None
            trial_dict[a_trial.results.trial_num] = a_trial

        trial_dict['subject_results'] = self.results
        trial_dict['time_series_results'] = self.time_series_results

        pickle_dict = {'trial_dict': trial_dict,
                       'analysis_parameters': self.analysis_parameters,
                       'results': trial_dict['subject_results'],
                       'experiment_settings': self.experiment_settings,
                       'has_gaze_in_head': self.has_gaze_in_head,
                       'has_head_pose': self.has_head_pose
                       }
        
        
        with bz2.BZ2File(self.pickle_path, "wb") as f:
            cPickle.dump(pickle_dict, f)

    def plot_mean_gaze_rel_car(self, saveFig=True, showFig=False, saveFolder='gaze_rel_car'):

        from matplotlib.ticker import StrMethodFormatter

        plt.ioff()  # prevents the fig from showing if in notebook mode

        plt.rcParams.update({'font.size': 15})

        the_fig, ax = plt.subplots(figsize=(8, 8))

        font = {'family': 'sans',
                'weight': 'normal',
                'size': 15,
                }

        l_turns = self.results[self.results['turn_direction'] == 'left']
        r_turns = self.results[self.results['turn_direction'] == 'right']

        h1 = sns.pointplot(data=l_turns, x="contrast", y="mean_gaze_az_rel_car", hue='turn_radius', errorbar='ci',
                           dodge=0.1)
        h2 = sns.pointplot(data=r_turns, x="contrast", y="mean_gaze_az_rel_car", hue='turn_radius', errorbar='ci',
                           dodge=0.2, linestyles=':')

        plt.hlines([0], -10, 10)
        plt.ylim(-20, 20)
        plt.xlim(-.5, 2.5)

        # plt.legend(shadow=True)
        # plt.xlabel('o.f. density')
        plt.xlabel('flow density', fontdict=font)
        plt.ylabel('gaze azimuth relative to car', fontdict=font)
        plt.suptitle('left turns (solid lines) and right turns (dotted)', fontdict=font)

        ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        ax.set_xticklabels(['only rotational', 'low', 'high'])

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), prop=font)

        if saveFig:

            plotFigSubPath = os.path.join(self.experiment_settings['figure_out_folder'], saveFolder)
            fig_name = self.sub_id + '_' 'gaze_rel_car'

            if os.path.isdir(plotFigSubPath) is False:
                os.makedirs(plotFigSubPath)

            plt.savefig(os.path.join(plotFigSubPath, fig_name + '.png'), bbox_inches='tight')

            print(os.path.join(plotFigSubPath, fig_name + '.png'))

        plt.ion()  # prevents the fig from showing if in notebook mode

        if showFig:
            plt.show()
        else:
            plt.close()

        return the_fig


    def gaze_rel_heading_polar_plot(self, saveFig=True, showFig=False, saveFolder='gaze_rel_car'):
        plt.rcParams.update({'font.size': 40})

        the_fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))

        ax.set_theta_zero_location("N")

        line_style = '-'

        for trialIndex, trial_results_row in self.results.iterrows():

            a_trial = self.get_trial_from_results(trial_results_row)

            plts = a_trial.processed_gaze_data['pupilLabsTimeStamp']

            a_trial.processed_gaze_data['pupil_rel_time'] = plts - plts[0]

            az_rad = np.deg2rad(a_trial.processed_gaze_data['gaze_az_rel_car_filtered'])
            t = a_trial.processed_gaze_data['pupil_rel_time']

            if trial_results_row['turn_radius'] == 35:
                line_color = 'r'
            elif trial_results_row['turn_radius'] == 55:
                line_color = 'g'
            elif trial_results_row['turn_radius'] == 75:
                line_color = 'b'

            ax.plot(az_rad, t, ls=line_style, c=line_color, alpha=0.3)

        ax.plot([0, 0], [0, 15], ':k', lw=3)
        ax.set_rmax(np.max(t))
        ax.grid(True)
        ax.set_thetamin(-60)
        ax.set_thetamax(60)
        ax.set_theta_direction(-1)

        if saveFig:

            plotFigSubPath = os.path.join(self.experiment_settings['figure_out_folder'], saveFolder)
            fig_name = self.sub_id + '_' 'gaze_rel_heading_polar'

            if os.path.isdir(plotFigSubPath) is False:
                os.makedirs(plotFigSubPath)

            plt.savefig(os.path.join(plotFigSubPath, fig_name + '.png'), bbox_inches='tight')
            print('Generating' + str(os.path.join(plotFigSubPath, fig_name + '.png')))

        plt.ion()  # prevents the fig from showing if in notebook mode

        if showFig:
            plt.show()
        else:
            plt.close()

        plt.rcParams.update({'font.size': 15})

        return the_fig

    def draw_full_roadway(self):

        plt.style.use('default')

        plt.figure(figsize=(8, 8))
        ax = plt.subplot()
        ax.axis("equal")

        for idx in np.arange(0, len(self.results) - 1):
            tr = self.get_trial_from_index(idx)
            ax, f_prev, l_prev = tr.draw_road_in_world(ax)

            tr2 = self.get_trial_from_index(idx + 1)
            ax, f_cur, l_cur = tr2.draw_road_in_world(ax)

            ax.plot([f_cur[0], l_prev[0]], [f_cur[1], l_prev[1]], 'k')
            ax.plot([f_cur[2], l_prev[2]], [f_cur[3], l_prev[3]], 'k')

        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)

        plt.axis('off')
        plt.grid('off')

        fig_name = self.sub_id + '_' '_roadway.png'

        plt.savefig(os.path.join(self.experiment_settings['figure_out_folder'], fig_name), bbox_inches='tight', dpi=300,transparent=True)

        plt.style.use('ggplot')


class trial_data():

    def __init__(self, trial_results_row_in, subject_data, analyze_gaze = False):
            
        self.left_eye_only = False
        self.subject_data = subject_data
        subject_data_folder = subject_data.data_folder
        experiment_settings = subject_data.experiment_settings

        self.data_parent_folder = os.path.join(*os.path.split(subject_data_folder)[:-1])
        self.data_folder = subject_data_folder

        self.results = deepcopy(trial_results_row_in)
        self.interpolated_div_and_timestamps = False
        # self.results = trial_results_row_in
        self.experiment_settings = experiment_settings

        # makes path correct, if folder name was changed after data collection
        if self.results['trialType'] != 'CalibrationAssessment': # only for non-calibration trial data
            
            # make sure correct tracker is directed to correct session number
            trial_results_row_in['road_vertices_location_0'].split('/')[2:][0] = self.results['session_num']
            self.road_vertices = pd.read_csv(os.path.join(self.data_parent_folder, self.results['ppid'], *trial_results_row_in['road_vertices_location_0'].split('/')[2:]))
            self.road_data = pd.read_csv(os.path.join(self.data_parent_folder, self.results['ppid'], *trial_results_row_in['roadTransformMat_location_0'].split('/')[2:]))
            self.car_data = pd.read_csv(os.path.join(self.data_parent_folder, self.results['ppid'], *trial_results_row_in['simplecar_carTransformMatrix_location_0'].split('/')[2:]))

            if '_cyclopeangaze_headPositionTracker_location_0' in trial_results_row_in.keys():
                
                path = os.path.join(self.data_parent_folder, self.results['ppid'], *trial_results_row_in['_cyclopeangaze_headPositionTracker_location_0'].split('/')[2:])

                if self.subject_data.estimate_head_trackers:
                    # replace "tracker" folder with "estimated_head_trackers" folder if using estimated head trackers
                    path = path.replace('trackers', 'estimated_head_trackers')   
                
                self.head_tracker = pd.read_csv(path)

            if self.experiment_settings['offset_car_xz']:
                self.offset_car_position()

        self.pl_timestamps = False
        self.raw_gaze_data = False
        self.processed_gaze_data = False

        # Gaze data
        self.experiment_settings['analyze_gaze'] = analyze_gaze

        # self.subject_data.raw_gaze_data will be False if no gaze data has been assigned
        if analyze_gaze and self.subject_data.has_gaze_in_head and self.subject_data.has_head_pose:
            self.initialize_gaze_data(self.results)

        if self.results['trialType'] != 'CalibrationAssessment':  # only for non-calibration trial data
            self.calculate_car_position_in_road()
            
    def initialize_gaze_data(self, trial_results_row_in):

        self.pl_timestamps = pd.read_csv(os.path.join(self.data_parent_folder, self.results['ppid'],*trial_results_row_in['time_sync_pupilTimeStamp_location_0'].split('/')[2:]))
        start = float(self.pl_timestamps['pupilLabsTimeStamp'].head(1).iloc[0])
        end = float(self.pl_timestamps['pupilLabsTimeStamp'].tail(1).iloc[0])

        self.raw_gaze_data = self.subject_data.get_gaze_data_slice(start, end)
        
        if self.results['trialType'] != 'CalibrationAssessment': # only for non-calibration trial data            
        
            self.processed_gaze_data = deepcopy(self.raw_gaze_data)
            self.processed_gaze_data.rename(columns={"gaze_timestamp": "pupilLabsTimeStamp"}, inplace=True)
            self.processed_gaze_data.sort_values(by='pupilLabsTimeStamp', inplace=True)
            self.processed_gaze_data = self.merge_unity_data_with_processed_gaze_data()

            if hasattr(self, 'head_tracker') == False:
                return

            self.processed_gaze_data = self.calculate_gaze_in_world() # Slowww!
            #self.processed_gaze_data = self.processed_gaze_data.loc[self.processed_gaze_data['gaze_confidence']>self.subject_data.analysis_parameters['pl_confidence_threshold']]

            self.calc_head_pos_in_road()
            
            # calculate gaze_dir_in_road_az and gaze_dir_in_road_el
            # gaze_dir_in_road_az_filtered, and gaze_dir_in_road_el_filtered
            # gaze_dir_in_road_az_vel_filtered, and gaze_dir_in_road_el_vel_filtered
            self.calc_gaze_in_road()
            
            # calcualte gaze_az_rel_car, gaze_el_rel_car
            self.calc_gaze_azimuth_relative_to_car()
            self.calc_gaze_elevation_rel_head()
            
            self.processed_gaze_data['pupil_rel_time'] = self.processed_gaze_data['pupilLabsTimeStamp']

            self.processed_gaze_data['gaze_dir_in_road_az_filtered'] = self.filter_spherical_coords('gaze_dir_in_road_az')
            self.processed_gaze_data['gaze_dir_in_road_el_filtered'] = self.filter_spherical_coords('gaze_dir_in_road_el')

            
            self.processed_gaze_data['gaze_az_rel_car_filtered'] = self.filter_spherical_coords('gaze_az_rel_car')
            self.processed_gaze_data['gaze_az_rel_car_vel'] = self.calc_spherical_vel('gaze_az_rel_car_filtered')
            self.processed_gaze_data['gaze_az_rel_car_vel_filtered'] = self.calc_filtered_spherical_vel(
                'gaze_az_rel_car_filtered')

            
            self.processed_gaze_data['gaze_el_rel_head_filtered'] = self.filter_spherical_coords('gaze_el_rel_head')
            self.processed_gaze_data['gaze_el_rel_head_vel'] = self.calc_spherical_vel('gaze_el_rel_head_filtered')
            self.processed_gaze_data['gaze_el_rel_head_vel_filtered'] = self.calc_filtered_spherical_vel(
                'gaze_el_rel_head_filtered')

            self.results['mean_gaze_az_rel_car'] = np.nanmean(self.processed_gaze_data['gaze_az_rel_car_filtered'])
            self.results['median_gaze_az_rel_car'] = np.nanmedian(self.processed_gaze_data['gaze_az_rel_car_filtered'])

            self.update_subject_results('mean_gaze_az_rel_car', self.results['mean_gaze_az_rel_car'])
            self.update_subject_results('median_gaze_az_rel_car', self.results['median_gaze_az_rel_car'])

            self.calc_mean_gaze_behavior_over_segment(30,70)
            
        else:
            self.processed_gaze_data = deepcopy(self.raw_gaze_data)
            self.processed_gaze_data.rename(columns={"gaze_timestamp": "pupilLabsTimeStamp"}, inplace=True)
            self.processed_gaze_data.sort_values(by='pupilLabsTimeStamp', inplace=True)
            
    def merge_unity_data_with_processed_gaze_data(self):

        # Some variable name fixing to preserve meaning after the merge
        # processed gaze data
        self.processed_gaze_data = self.processed_gaze_data.reset_index()
        column_names = self.processed_gaze_data.columns
        column_names = ['gaze_confidence' if 'confidence' in c else c for idx, c in enumerate(column_names)]
        column_names = ['eye_' + c if 'norm_pos' in c else c for idx, c in enumerate(column_names)]
        column_names = ['eye_' + c if 'base_data' in c else c for idx, c in enumerate(column_names)]
        self.processed_gaze_data.columns = column_names
        mergedDF = self.processed_gaze_data.reset_index()

        # head data
        if hasattr(self,'head_tracker'):
            head_tracker = deepcopy(self.head_tracker)
            column_names = head_tracker.columns
            column_names = ['head_' + c[15:] if '_cyclopeangaze_' in c else c for idx, c in enumerate(column_names)]
            column_names = ['head_' + c if 'pos' in c else c for idx, c in enumerate(column_names)]
            head_tracker.columns = column_names
            if 'time' in head_tracker.columns:
                head_tracker = head_tracker.drop('time',axis=1)
            head_tracker['pupilLabsTimeStamp'] = self.pl_timestamps['pupilLabsTimeStamp']
            
            new_head_tracker = pd.DataFrame()
            for colname, col in head_tracker.items():
                new_head_tracker[colname] = np.interp(mergedDF['pupilLabsTimeStamp'], head_tracker['pupilLabsTimeStamp'], col)
                    
            mergedDF = pd.merge(new_head_tracker, mergedDF, on='pupilLabsTimeStamp',
                                how='right', sort=True)

        # car data
        car_data = deepcopy(self.car_data)
        column_names = car_data.columns
        # column_names = ['unity_time' if 'time' in c else c for idx, c in enumerate(column_names)]
        column_names = ['car_' + c if 'pos' in c else c for idx, c in enumerate(column_names)]
        column_names = ['car_' + c if 'distFromRoadCenter' in c else c for idx, c in enumerate(column_names)]
        column_names = ['car_' + c if 'signedDistFromRoadCenter' in c else c for idx, c in enumerate(column_names)]
        car_data.columns = column_names
        car_data = car_data.drop('time',axis=1)
        car_data['pupilLabsTimeStamp'] = self.pl_timestamps['pupilLabsTimeStamp']
        
        # linearly interpolate car data here before merge
        new_car_data = pd.DataFrame()
        for colname, col in car_data.items():
            new_car_data[colname] = np.interp(mergedDF['pupilLabsTimeStamp'], car_data['pupilLabsTimeStamp'], col)
            
        #mergedDF = pd.merge(mergedDF, car_data, on='pupilLabsTimeStamp', how='outer',sort=True)
        mergedDF = pd.merge(mergedDF, new_car_data, on='pupilLabsTimeStamp', how='outer',sort=True)
        
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
            # mergedDF = mergedDF.interpolate(method='linear', downcast='infer')
        mergedDF = mergedDF.drop(['index'],axis=1)
        mergedDF = mergedDF.drop(['level_0'],axis=1)
        
        return mergedDF



    def calc_gaze_in_road(self):

        inv_road_mat = np.linalg.inv(self.get_matrix_from_row('_4x4', self.road_data.iloc[0]))

        # Set translation to 0 - this matrix will rotate, but not translate.
        inv_road_mat[:3,3] = 0

        gaze_dir_xyz_roadspace = [np.dot(inv_road_mat, np.hstack([xyz, 1]))[:3] for xyz in 
                            self.processed_gaze_data.filter(regex='giw_dir_*').values]

        x,y,z = zip(*gaze_dir_xyz_roadspace)

        gir_az = np.rad2deg([np.arctan2(x,z) for x,y,z in gaze_dir_xyz_roadspace])
        gir_el = np.rad2deg([np.arctan2(y,z) for x,y,z in gaze_dir_xyz_roadspace])

        self.processed_gaze_data['gaze_dir_in_road_az'] = gir_az
        self.processed_gaze_data['gaze_dir_in_road_el']  = gir_el
        
        self.processed_gaze_data['gaze_dir_in_road_x'] = x
        self.processed_gaze_data['gaze_dir_in_road_y'] = y
        self.processed_gaze_data['gaze_dir_in_road_z'] = z


    def get_matrix_from_row(self, varName, row):

        mat_4x4 = np.empty([4, 4])
        for r in range(4):
            for c in range(4):
                mat_4x4[r, c] = row['{}_R{}C{}'.format(varName, r, c)]

        return mat_4x4

    def offset_car_position(self):
        '''
        I think this saves the coordinates of the offset (i.e. where the driver is in WORLD coordinates)
        '''

        offset_pos_xyzw = []
        offset_x, offset_z = self.experiment_settings['offset_car_xz']

        for index, rowData in self.car_data.iterrows():
            offset_pos_xyzw.append(np.dot(self.get_matrix_from_row('simplecar_4x4', rowData), [offset_x, 0, offset_z, 1]))

        x, y, z, w = zip(*offset_pos_xyzw)

        self.car_data['offset_pos_x'] = x # these are saved in world coordinates
        self.car_data['offset_pos_y'] = y
        self.car_data['offset_pos_z'] = z

    def calculate_car_position_in_road(self):

        '''
        Adds columns ['pos_x_roadspace','pos_y_roadspace','pos_z_roadspace'] to self.car_data

        '''

        # Calc inverse road matrix
        inv_road_mat = np.linalg.inv(self.get_matrix_from_row('_4x4', self.road_data.iloc[0]))

        # Transform the car's position into road space
        if 'offset_pos_x' in self.car_data.columns:
            car_xyz_roadspace = [np.dot(inv_road_mat, np.hstack([xyz, 1]))[:3] for xyz in
                                  self.car_data[['offset_pos_x', 'offset_pos_y', 'offset_pos_z']].values]
        else:
            car_xyz_roadspace = [np.dot(inv_road_mat, np.hstack([xyz, 1]))[:3] for xyz in
                                  self.car_data[['pos_x', 'pos_y', 'pos_z']].values]

        car_xyz_roadspace_df = pd.DataFrame(car_xyz_roadspace,
                                            columns=['pos_x_roadspace', 'pos_y_roadspace', 'pos_z_roadspace'])

        self.car_data = self.car_data.merge(car_xyz_roadspace_df, left_index=True, right_index=True, validate='1:1')

    def calculate_road_edges(self):
        '''
        Adds columns:
            self.road_vertices['road_right_X']
            self.road_vertices['road_right_Z']
            self.road_vertices['road_left_X']
            self.road_vertices['road_left_Z']

        '''

        road_vertices = self.road_vertices

        # Calculate road edges
        try:
            road_width = self.experiment_settings['road_width']
        except:
            road_width = self.experiment_settings['road_halfWidth']
            
        self.road_vertices['road_right_X'] = road_vertices['xVertex_position'] + road_vertices['xNormal'] * \
                                             road_width
        self.road_vertices['road_right_Z'] = road_vertices['zVertex_position'] + road_vertices['zNormal'] * \
                                             road_width
        self.road_vertices['road_left_X'] = road_vertices['xVertex_position'] - road_vertices['xNormal'] * \
                                            road_width
        self.road_vertices['road_left_Z'] = road_vertices['zVertex_position'] - road_vertices['zNormal'] * \
                                            road_width

        # logger.info('Added trial_data[road_vertices][[\'road_right_X\',\'road_right_Z\',\'road_left_X\',\'road_left_Z\']]')

    def plot_road(self, ax, flip_left_turns=True):

        # Check to see if we have the data we need.  If not, calculate it!
        if 'road_right_X' not in self.road_vertices.columns:
            self.calculate_road_edges()

        # Before plotting, we will multiply x values by flip_left
        if flip_left_turns and self.results['turn_direction'] == 'left':
            flip_left = -1
        else:
            flip_left = 1

        plt.ioff()  # prevents the fig from showing if in notebook mode
        ax.plot(self.road_vertices['xVertex_position'] * flip_left, self.road_vertices['zVertex_position'],
                label='Road Center', linestyle='--')
        ax.plot(self.road_vertices['road_right_X'] * flip_left, self.road_vertices['road_right_Z'], 'k-')
        ax.plot(self.road_vertices['road_left_X'] * flip_left, self.road_vertices['road_left_Z'], 'k-',
                label='Road Edge')
        plt.ion()  # prevents the fig from showing if in notebook mode

        return ax

    def calc_head_pos_in_road(self):

        inv_road_mat = np.linalg.inv(self.get_matrix_from_row('_4x4', self.road_data.iloc[0]))

        head_pos_xyz_roadspace = [np.dot(inv_road_mat, np.hstack([xyz, 1]))[:3] for xyz in 
                                self.processed_gaze_data.filter(regex='head_pos_._world_space').values]

        x,y,z = zip(*head_pos_xyz_roadspace)

        self.processed_gaze_data['head_pos_x_road_space']  = x
        self.processed_gaze_data['head_pos_y_road_space']  = y
        self.processed_gaze_data['head_pos_z_road_space']  = z

    def plot_car_in_road_space(self, ax, flip_left_turns=True):

        # Check to see if we have the data we need.  If not, calculate it!
        if 'pos_x_roadspace' not in self.car_data.columns:
            self.calculate_car_position_in_road()

        # Before plotting, we will multiply x values by flip_left
        if flip_left_turns and self.results['turn_direction'] == 'left':
            flip_left = -1
        else:
            flip_left = 1

        plt.ioff()  # prevents the fig from showing if in notebook mode
        ax.plot(self.car_data['pos_x_roadspace'] * flip_left, self.car_data['pos_z_roadspace'], label="Car's Position", color='b', linewidth=3)
        ax.plot(self.car_data['pos_x_roadspace'].iloc[0] * flip_left, self.car_data['pos_z_roadspace'].iloc[0], 'o')
        plt.ion()  # prevents the fig from showing if in notebook mode

        return ax

    def plot_car_trajectory(self, plot_gaze_data = False, showFig=False, saveFig=True, flip_left_turns=True, showLegend=False, saveFolder=''):
        
        fontsize=25
        
        def get_gaze_vector_at_unity_time(utime,mag=20,):

            g_data = self.processed_gaze_data[self.processed_gaze_data.unity_time == utime]

            if( len(g_data)>1):
                g_data = g_data.iloc[-1]

            gaze_origin = g_data.filter(regex='head_pos_._road_space')
            gaze_dir = g_data.filter(regex='gaze_dir_in_road_(x|y|z)')
            gazepoint_in_road_xyz = gaze_origin.values.flatten() + gaze_dir.values.flatten() * mag

            return [gaze_origin.values.flatten(), gazepoint_in_road_xyz.flatten()]

        # Check to see if we have the data we need.  If not, calculate it!
        if 'road_right_X' not in self.road_vertices.columns:
            self.calculate_road_edges()

        if 'pos_x_roadspace' not in self.car_data.columns:
            self.calculate_car_position_in_road()

        if 'nearest_road_vertex_x_roadspace' not in self.car_data.columns:
            # Necessary for showing nearest point on road
            self.calc_divergence()

        plt.ioff()  # prevents the fig from showing if in notebook mode

        fig, ax = plt.subplots(figsize=(8,8))

        self.plot_road(ax, flip_left_turns)
        self.plot_car_in_road_space(ax, flip_left_turns)

        for f in np.linspace(0, len(self.car_data) - 1, 10, dtype=int):

            car_xz = self.car_data[['pos_x_roadspace', 'pos_z_roadspace']].iloc[f]
            nearest_road_vert_xz = self.car_data[['nearest_road_vertex_x_roadspace','nearest_road_vertex_z_roadspace']].iloc[f]
            divergence = self.car_data['divergence'].iloc[f]

            if flip_left_turns == True and self.results['turn_direction'] == 'left':
                car_xz.iloc[0] = -car_xz.iloc[0]
                nearest_road_vert_xz.iloc[0] = -nearest_road_vert_xz.iloc[0]


            xs, ys = zip(car_xz, nearest_road_vert_xz)
            ax.plot(xs, ys, 'r')
            ax.text(nearest_road_vert_xz[0] + 1, nearest_road_vert_xz[1] + 1, '{:.2f}'.format(divergence), fontsize=fontsize, color='red')

        ax.text(.5, .5, 'Density: ' + str(self.results['contrast']) + '\n' + \
                'Radius: ' + str(self.results['turn_radius']) + '\n' + \
                self.results['turn_direction'] + ' turn', \
                horizontalalignment='center', \
                verticalalignment='center', transform=ax.transAxes, fontsize=fontsize)

        ax.set_xlabel('Horizontal Distance (m)', fontsize=fontsize)
        ax.set_ylabel('Vertical Distance (m)', fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

        ax.axis("equal")  # sets the x/y axes scales equal

        if showLegend:
            ax.legend()

        plt.ion()

        
        if plot_gaze_data == True:

            for f in np.linspace(0, len(self.car_data) - 1, 10, dtype=int):

                origin_xyz, endpoint_xyz = get_gaze_vector_at_unity_time(self.car_data.iloc[f].time)
                if flip_left_turns == True and self.results['turn_direction'] == 'left':
                    plt.plot([-origin_xyz[0],-endpoint_xyz[0]],[origin_xyz[2],endpoint_xyz[2]], 'g')
                else:
                    plt.plot([origin_xyz[0],endpoint_xyz[0]],[origin_xyz[2],endpoint_xyz[2]], 'g')

        if saveFig:
            

            # make directory to save the figures
            figSubFolderStr = 'D-' + str(self.results['contrast']) + '_R-' + str(self.results['turn_radius'])
            plotFigSubPath = []
            plotFigSubPath.append(os.path.join(self.experiment_settings['figure_out_folder'], saveFolder, figSubFolderStr))
            plotFigSubPath.append(os.path.join(self.experiment_settings['figure_out_folder'], figSubFolderStr))
            
            for path in plotFigSubPath:
                if os.path.isdir(path) is False:
                    os.makedirs(path)
                    
                plt.savefig(os.path.join(path, '_B' + str(self.results['block_num']) + \
                                    '_T' + str(self.results['trial_num_in_block']) + '.png'),dpi=300 )
                    
        if showFig:
            plt.show()
        else:
            plt.close()

        return fig,ax

    def plot_time_vs_steer_bias(self,
                             showFig=True,
                             saveFig=True,
                             interp_time_seconds=None,
                             interp_res_seconds = (1 / 90), saveFolder=''):

        # if running one trial at a time, you'll plot over course of whole trial 
        if interp_time_seconds==None:
            interp_time_seconds=self.results['end_time']-self.results['start_time']
                
        (interp_timestamps, interp_div) = self.interpolate_divergence_roadspace(interp_time_seconds=interp_time_seconds, \
                                                                                                        interp_res_seconds=(
                                                                                                                    1 / 90),
                                                                                                        flip_left_turns=True)
        placeholder_df = pd.DataFrame({'interp_timestamps': interp_timestamps})
        #placeholder_df['turn_radius'] = str(self.results['turn_radius'])
    
        try:
            div_from_inner_road_edge = [self.experiment_settings['road_width'] - mag for mag in interp_div]
        except:
            div_from_inner_road_edge = [self.experiment_settings['road_halfWidth'] - mag for mag in interp_div]
        
        placeholder_df['mean_div_magnitude_density_' + str(self.results['contrast'])] = div_from_inner_road_edge

        # log information to time_series results and concatenate to save for all radii values
        self.subject_data.time_series_results = pd.concat((self.subject_data.time_series_results, placeholder_df),axis=1)
        
        if saveFig or showFig:
            plt.figure(figsize=(5,5))
            ax = plt.subplot()
            # plot average divergences by optic flow on same graph
            ax.plot(div_from_inner_road_edge[3:], interp_timestamps[3:], linewidth=1.5, label=str(self.results['contrast']))
            
            ax.axvline(x=2, ls = '--', color='k') # plots vertical dotted line
            ax.axvline(x=0, ls = '-', color='k') # plots vertical line
            ax.axvline(x=4, ls = '-', color='k') # plots vertical line
            ax.set_xlim(-0.1,4.1)
            ax.text(0.1,0.92, self.subject_data.sub_id + '\n' + \
                    'Radius: ' + str(self.results['turn_radius']) + '\n' + saveFolder, \
                    horizontalalignment='center', \
                    verticalalignment='center', transform=ax.transAxes, fontsize=15)
        
            ax.set_xlabel('Distance from Inner Road Edge (m)')
            ax.set_ylabel('Time (sec)')
            ax.legend(loc='best', title='Optic Flow \nDensity')
        
            # ax.axis("equal")  # sets the x/y axes scales equal
        
        if saveFig:
            plotFigSubPath = os.path.join(self.experiment_settings['figure_out_folder'], saveFolder, 'time_vs_steer_bias')
            fig_name = 'time_vs_steer_bias_' +  'R-' + str(self.results['turn_radius']) +'-TrialNum-' + str(self.results.trial_num_in_block)
    
            if os.path.isdir(plotFigSubPath) is False:
                os.makedirs(plotFigSubPath)
    
            plt.savefig(os.path.join(plotFigSubPath, fig_name + '.png'))
    
        if showFig:
            plt.show()
        else:
            plt.close()
        
        
        
        
    def plot_azimuth_elevation_by_trial_car_space(self):
        # plot the tracked gaze from beginning of trial to end. Azimuth on x, elevation on y.
        
        fig, ax = plt.subplots()
        ax.set_title('Gaze in car space during single turn. Trial #' + \
                     str(self.results['trial_num_in_block']), fontsize=18)
        ax.set_xlabel('Azimuth (degrees from car forward direction)')
        ax.set_ylabel('Elevation (degrees)')
        plt.text(0.95, 0.95, 'Density: ' + str(self.results['contrast']) + '\n' + \
                'Radius: ' + str(self.results['turn_radius']) + '\n' + \
                self.results['turn_direction'] + ' turn', \
                horizontalalignment='right', \
                verticalalignment='top', transform = ax.transAxes)
            
        x = self.processed_gaze_data['gaze_az_rel_car']
        y = self.processed_gaze_data['gaze_el_rel_head']
        
        
        data = pd.concat((x,y), axis=1)
        conf_thres = self.subject_data.analysis_parameters['pl_confidence_threshold']
        x = data[self.processed_gaze_data['gaze_confidence'] > conf_thres].iloc[:,0] 
        y = data[self.processed_gaze_data['gaze_confidence'] > conf_thres].iloc[:,1] 
        
        #ax.set_ylim(-0.002, 0.002)
        ax.plot(x, y, 'k-')
        plt.axhline(0, color='b', linestyle='--', linewidth = 2)
        ax.plot(x.iloc[0], y.iloc[0], 'go', label='Turn start')
        ax.plot(x.iloc[-1], y.iloc[-1], 'ro', label='Turn end')
        ax.axis("equal")
        plt.legend(loc='upper left')
        plt.show()

    def calc_time_in_trial(self):
        self.car_data['time_in_trial'] = self.car_data['time'] - self.car_data['time'][0]

    def interpolate_divergence_roadspace(self, interp_time_seconds=5, interp_res_seconds=(1 / 90), flip_left_turns=True):

        '''
        returns tuple (interp_timestamps, car_x, car_z)
        interp_time are the timestamps
        where car_x and car_z are numpy arrays of interpolated car position data
        '''

        # Check to see if we have the data we need.  If not, calculate it!
        if 'time_in_trial' not in self.car_data.columns:
            self.calc_time_in_trial()

        if 'divergence' not in self.car_data.columns or 'abs_divergence' not in self.car_data.columns:
            self.calc_divergence()

        interp_timestamps = np.arange(0, interp_time_seconds, interp_res_seconds)
        # interpolate signed bias data to average over optic flow conditions
        car_divergence = np.interp(interp_timestamps, self.car_data['time_in_trial'], self.car_data['divergence'])

        # if flip_left_turns and self.results['turn_direction'] == 'left':
        #     car_divergence = car_divergence * -1
        interpolated_data = np.concatenate((interp_timestamps.reshape(-1,1),car_divergence.reshape(-1,1)),axis=1)
        self.interpolated_div_and_timestamps = pd.DataFrame(interpolated_data, columns=['interp_timestamps', 'interp_divergence'])
        return (interp_timestamps, car_divergence)
    
    def interpolate_car_pos_roadspace(self, interp_time_seconds=5, interp_res_seconds=(1 / 90), flip_left_turns=True):

        '''
        returns tuple (interp_timestamps, car_x, car_z)
        interp_time are the timestamps
        where car_x and car_z are numpy arrays of interpolated car position data
        '''

        # Check to see if we have the data we need.  If not, calculate it!
        if 'time_in_trial' not in self.car_data.columns:
            self.calc_time_in_trial()

        if 'road_right_X' not in self.road_vertices.columns:
            self.calculate_road_edges()

        if 'pos_x_roadspace' not in self.car_data.columns:
            self.calculate_car_position_in_road()

        interp_timestamps = np.arange(0, interp_time_seconds, interp_res_seconds)
        car_x = np.interp(interp_timestamps, self.car_data['time_in_trial'], self.car_data['pos_x_roadspace'])
        car_z = np.interp(interp_timestamps, self.car_data['time_in_trial'], self.car_data['pos_z_roadspace'])

        if flip_left_turns and self.results['turn_direction'] == 'left':
            car_x = car_x * -1

        return (interp_timestamps, car_x, car_z)

    def calc_divergence(self):

        def calc_lane_divergence_interp(row_in, road_vertices_in, road_data_in, num_interp_points = 2000):
            # This function calculates the distance from the car to the nearest road vertex
            # ...but interpolates between road vertices for greater accuracy.

            xs = road_vertices_in['xVertex_position']
            xs = np.interp(np.linspace(0, len(road_vertices_in), num_interp_points), np.arange(len(xs)), xs)

            zs = road_vertices_in['zVertex_position']
            zs = np.interp(np.linspace(0, len(road_vertices_in), num_interp_points), np.arange(len(zs)), zs)

            dist_to_each_vertex = np.sqrt((row_in['pos_x_roadspace']  - xs) ** 2 \
                                          + (row_in['pos_z_roadspace'] - zs) ** 2)

            divergence = np.min(dist_to_each_vertex)            
            idx = np.argmin(dist_to_each_vertex)
            
            # convert road to carspace
            road_mat = self.get_matrix_from_row('_4x4', self.road_data.iloc[0])
            car_mat = self.get_matrix_from_row('simplecar_4x4', row_in)
            
            # car mat needs offset too, if there is one 
            if 'offset_pos_x' in row_in.keys():
                car_mat[0,3] = row_in['offset_pos_x'] # calculated in offset_car_position
                car_mat[2,3] = row_in['offset_pos_z']
            
            road_vert_xyzw = np.hstack([xs[idx], 0 ,zs[idx], 1]) # vertex in road space
            road_vert_in_worldspace = np.dot(road_mat,road_vert_xyzw) # vertex in world space
            road_vert_in_carspace = np.dot(np.linalg.inv(car_mat), road_vert_in_worldspace) # vertex in car space
                 
            
            # Negative when car is towards outside turn
            if self.results['turn_direction'] == 'left' and road_vert_in_carspace[0] < 0:
                divergence = - divergence
            elif road_vert_in_carspace[0] > 0 and self.results['turn_direction'] == 'right':
                divergence = - divergence                
                
            nearest_road_vertex_xyz_roadspace = (xs[idx], 0, zs[idx])
            
            return (nearest_road_vertex_xyz_roadspace, divergence)

        if 'pos_x_roadspace' not in self.car_data.columns:
            self.calculate_car_position_in_road()

        # this takes a low of time...
        # goes through each row and calculates distance from all interpolated pts to find closest pt
        out = self.car_data.apply(lambda row: calc_lane_divergence_interp(row, self.road_vertices, self.road_data), axis=1)
        (nearest_road_vertex_xyz_roadspace, divergence) = zip(*out)
        nearest_road_vertex_xyz_roadspace = np.array(nearest_road_vertex_xyz_roadspace)

        self.car_data['divergence'] = divergence
        self.car_data['abs_divergence'] = np.abs(self.car_data['divergence'])

        self.car_data['nearest_road_vertex_x_roadspace'] = nearest_road_vertex_xyz_roadspace[:,0]
        self.car_data['nearest_road_vertex_y_roadspace'] = nearest_road_vertex_xyz_roadspace[:,1]
        self.car_data['nearest_road_vertex_z_roadspace'] = nearest_road_vertex_xyz_roadspace[:,2]

        self.calc_mean_divergence_over_segment()

    def calc_mean_gaze_behavior_over_segment(self, start_percent=0, end_percent=100):

        if 'divergence' not in self.car_data.columns:
            self.calc_divergence()

        num_frames = len(self.car_data['divergence'])
        start_idx = int(np.floor(num_frames * (start_percent / 100)))
        end_idx = int(np.ceil(num_frames * (end_percent / 100)))

        if end_idx > num_frames: end_idx = num_frames

        if start_percent == 0 and end_percent == 100:
            label_suffix = ''
        else:
            label_suffix = '_' + str(start_percent) + '_' + str(end_percent)

        start_time = self.car_data['time'].iloc[start_idx]
        end_time = self.car_data['time'].iloc[end_idx]

        startIdx = list(map(lambda i: i >= start_time, self.processed_gaze_data['unity_time'])).index(True)
        endIdx = list(map(lambda i: i > end_time, self.processed_gaze_data['unity_time'])).index(True)-1

        # suppress the warnings here for nanmean, just means if all gaze data was poor all data was dropped
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.results['gaze_az_rel_car_filtered' + label_suffix] = np.nanmean(self.processed_gaze_data['gaze_az_rel_car_filtered'].iloc[startIdx:endIdx])
            self.results['gaze_az_rel_car_filtered' + label_suffix] = np.nanmean(self.processed_gaze_data['gaze_az_rel_car_filtered'].iloc[startIdx:endIdx])
            self.results['gaze_az_rel_car_filtered' + label_suffix + '_std_dev'] = np.nanstd(self.processed_gaze_data['gaze_az_rel_car_filtered'].iloc[startIdx:endIdx])
            self.results['gaze_az_rel_car_filtered' + label_suffix + '_std_dev'] = np.nanstd(self.processed_gaze_data['gaze_az_rel_car_filtered'].iloc[startIdx:endIdx])
            
            if 'gaze_el_rel_head_filtered' not in self.processed_gaze_data.keys():
                self.calc_gaze_elevation_rel_head()
                self.processed_gaze_data['gaze_el_rel_head_filtered'] = self.filter_spherical_coords('gaze_el_rel_head')
                
            self.results['gaze_el_rel_head_filtered' + label_suffix] = np.nanmean(self.processed_gaze_data['gaze_el_rel_head_filtered'].iloc[startIdx:endIdx])
            self.results['gaze_el_rel_head_filtered' + label_suffix] = np.nanmean(self.processed_gaze_data['gaze_el_rel_head_filtered'].iloc[startIdx:endIdx])
            self.results['gaze_el_rel_head_filtered' + label_suffix + '_std_dev'] = np.nanstd(self.processed_gaze_data['gaze_el_rel_head_filtered'].iloc[startIdx:endIdx])
            self.results['gaze_el_rel_head_filtered' + label_suffix + '_std_dev'] = np.nanstd(self.processed_gaze_data['gaze_el_rel_head_filtered'].iloc[startIdx:endIdx])
        
        self.update_subject_results('gaze_el_rel_head_filtered' + label_suffix, self.results['gaze_el_rel_head_filtered' + label_suffix])
        self.update_subject_results('gaze_el_rel_head_filtered' + label_suffix, self.results['gaze_el_rel_head_filtered' + label_suffix])
        self.update_subject_results('gaze_el_rel_head_filtered' + label_suffix + '_std_dev', self.results['gaze_el_rel_head_filtered' + label_suffix + '_std_dev'])
        self.update_subject_results('gaze_el_rel_head_filtered' + label_suffix + '_std_dev', self.results['gaze_el_rel_head_filtered' + label_suffix + '_std_dev'])
        
        self.update_subject_results('gaze_az_rel_car_filtered' + label_suffix, self.results['gaze_az_rel_car_filtered' + label_suffix])
        self.update_subject_results('gaze_az_rel_car_filtered' + label_suffix, self.results['gaze_az_rel_car_filtered' + label_suffix])
        self.update_subject_results('gaze_az_rel_car_filtered' + label_suffix + '_std_dev', self.results['gaze_az_rel_car_filtered' + label_suffix + '_std_dev'])
        self.update_subject_results('gaze_az_rel_car_filtered' + label_suffix + '_std_dev', self.results['gaze_az_rel_car_filtered' + label_suffix + '_std_dev'])


    def calc_mean_divergence_over_segment(self, start_percent = 0, end_percent = 100):

        if 'divergence' not in self.car_data.columns:
            self.calc_divergence()
       
        # recalculate mean divergence from inner road edge 
        try:
            div_from_inner_road_edge = self.experiment_settings['road_width'] - self.car_data['divergence']
            
        except: 
            div_from_inner_road_edge = self.experiment_settings['road_halfWidth'] - self.car_data['divergence']
            
        self.car_data['div_from_inner_road_edge'] = div_from_inner_road_edge

        num_frames = len( self.car_data['divergence'] )
        start_idx = int(np.floor( num_frames * (start_percent/100)))
        end_idx = int(np.ceil(num_frames * (end_percent/100)))

        if end_idx > num_frames: end_idx = num_frames

        if start_percent == 0 and end_percent == 100:
            label_suffix = ''
        else:
            label_suffix = '_' + str(start_percent) + '_' + str(end_percent)

        self.results['mean_divergence' + label_suffix] = np.mean(self.car_data['divergence'].iloc[start_idx:end_idx])
        self.results['mean_div_from_inner_road_edge' + label_suffix] = np.mean(self.car_data['div_from_inner_road_edge'].iloc[start_idx:end_idx])
        self.results['mean_abs_divergence' + label_suffix] = np.mean(self.car_data['abs_divergence'].iloc[start_idx:end_idx])

        self.update_subject_results('mean_divergence' + label_suffix, self.results['mean_divergence' + label_suffix])
        self.update_subject_results('mean_div_from_inner_road_edge' + label_suffix, self.results['mean_div_from_inner_road_edge' + label_suffix])
        self.update_subject_results('mean_abs_divergence' + label_suffix, self.results['mean_abs_divergence' + label_suffix])

   
    def calculate_road_vertices_in_world(self):
        
        road_transform = self.get_matrix_from_row('_4x4',self.road_data.iloc[0])
        
        road_transform = self.get_matrix_from_row('_4x4',self.road_data.iloc[0])
        roadedge_left_xyzw_world = self.road_vertices.apply(lambda row: np.dot(road_transform, [row['road_left_X'], 0, row['road_left_Z'], 1]),axis=1)
        roadedge_right_xyzw_world = self.road_vertices.apply(lambda row: np.dot(road_transform, [row['road_right_X'], 0, row['road_right_Z'], 1]),axis=1)

        roadedge_left_x_world, roadedge_left_y_world, roadedge_left_z_world, w = zip(*roadedge_left_xyzw_world)
        roadedge_right_x_world, roadedge_right_y_world, roadedge_right_z_world, w = zip(*roadedge_right_xyzw_world)

        self.road_vertices['roadedge_left_x_world'] = roadedge_left_x_world
        self.road_vertices['roadedge_left_z_world'] = roadedge_left_z_world

        self.road_vertices['roadedge_right_x_world'] = roadedge_right_x_world
        self.road_vertices['roadedge_right_z_world'] = roadedge_right_z_world

    def filter_spherical_coords(self, gaze_pos_column, medFiltSize=False, meanfiltsize=False, vel_threshold=False,
                                confidence_threshold=False):

        df_in = self.processed_gaze_data
        filt_pos_data = df_in[gaze_pos_column].values

        if medFiltSize == False:
            medFiltSize = self.subject_data.analysis_parameters['spherical_pos_median_filt_size']

        if meanfiltsize == False:
            meanfiltsize = self.subject_data.analysis_parameters['spherical_pos_mean_filt_size']

        if confidence_threshold == False:
            confidence_threshold = self.subject_data.analysis_parameters['pl_confidence_threshold']

        # Remove low confidence samples
        below_conf_idx = np.where(df_in['gaze_confidence'] < confidence_threshold)[0]
        # pct = 100 * (len(below_conf_idx) / len(filt_pos_data))
        # print(f'PCT below conf: {pct:.2f}')
        filt_pos_data[below_conf_idx] = np.nan

        # Filter gaze position
        filt_pos_data = pd.Series(filt_pos_data)
        filt_pos_data.dropna(inplace=True) # drop na so rolling filter works more accurately
        filt_pos_data = filt_pos_data.rolling(medFiltSize, center=True).median()
        filt_pos_data = filt_pos_data.rolling(meanfiltsize, center=True).mean()
        filt_pos_data.dropna(inplace=True)
        # interpolate back to original indices so that it matches better, but do this after outliers are removed
        #filt_pos_data = filt_pos_data[(np.abs(stats.zscore(filt_pos_data)) < 3)]
       
        # plt.plot(filt_pos_data, label=gaze_pos_column)
        # plt.ylim(-40,40)
        # filt_pos_data = np.interp(np.arange(len(df_in)), filt_pos_data.index, filt_pos_data)
        # if self.left_eye_only:
        #     plt.legend(title=self.subject_data.sub_id + ', Left Eye Only')
        #     plt.savefig(r"D:\Arianna's backup\Gaze-Figures\\" + self.subject_data.sub_id + '-LeftEyeOnly.png')
        # else:
        #     plt.legend(title=self.subject_data.sub_id)
        #     plt.savefig(r"D:\Arianna's backup\Gaze-Figures\\" + self.subject_data.sub_id + '.png')
        return filt_pos_data
    
    

    def calc_spherical_vel(self, gaze_pos_column, time_column='pupilLabsTimeStamp'):

        # Unfiltered velocity signal
        rel_time = self.processed_gaze_data[time_column] - self.processed_gaze_data[time_column][0]

        with np.errstate(divide='ignore', invalid='ignore'):
            # silences "RuntimeWarning: invalid value encountered in true_divide"
            vel = np.hstack([0, np.abs(np.diff(self.processed_gaze_data[gaze_pos_column])) / np.diff(rel_time)])

        return vel

    def calc_filtered_spherical_vel(self, gaze_pos_column, time_column='pupilLabsTimeStamp', medFiltSize=False,
                                    meanfiltsize=False):

        if medFiltSize == False:
            medFiltSize = self.subject_data.analysis_parameters['spherical_vel_median_filt_size']

        if meanfiltsize == False:
            meanfiltsize = self.subject_data.analysis_parameters['spherical_vel_mean_filt_size']

        df_in = self.processed_gaze_data

        # Unfiltered velocity signal
        rel_time = df_in[time_column] - df_in[time_column][df_in[time_column].first_valid_index()]

        with np.errstate(divide='ignore', invalid='ignore'):
            # silences "RuntimeWarning: invalid value encountered in true_divide"
            df_in[gaze_pos_column + '_vel'] = np.hstack(
                [0, np.abs(np.diff(df_in[gaze_pos_column])) / np.diff(rel_time)])

        # Assume a step size that is the median value of time diffs
        # step_size = np.nanmedian(np.diff(rel_time))
        # EDIT FROM AG:
        # replacing with line that removes zeros (timestep repeats) and calculates average of remaining values for avg step size. Otherwise step size can be zero which throws error
        step_size = np.nanmean(np.diff(rel_time)[np.where(np.diff(rel_time)!=0)[0]])
        if np.isnan(step_size):
            # happens if a trial has many nans throughout, so I'll hardcode the normal step size that we see with almost all trials. 
            # This is a RARE occurrence. Only 1 trial in 24 participants so far
            step_size = 0.003205
            # add a debug statement -AG
        t_i = np.arange(rel_time.values[rel_time.first_valid_index()], rel_time.values[rel_time.last_valid_index()] + 0.2, step_size) 
        
        df_in[df_in['gaze_confidence']<self.subject_data.analysis_parameters['pl_confidence_threshold']] = np.nan
        az = df_in[gaze_pos_column] 

        # Linear interpolation
        az_interp = np.interp(t_i, rel_time, az)

        with np.errstate(divide='ignore', invalid='ignore'):
            az_interp_vel = np.hstack([0, np.abs(np.diff(az_interp)) / np.diff(t_i)])

        # filtering
        az_vel_restored = np.interp(rel_time, t_i, az_interp_vel)
        az_vel_restored = pd.Series(az_vel_restored).rolling(meanfiltsize, center=True).mean()
        az_vel_restored = az_vel_restored.rolling(medFiltSize, center=True).median()

        return az_vel_restored

  

    def calc_gaze_azimuth_relative_to_car(self):

        if self.left_eye_only:
            self.processed_gaze_data = self.calculate_gaze_in_world()
            self.calc_head_pos_in_road()
            self.calc_gaze_in_road()
        
        self.processed_gaze_data[self.processed_gaze_data['gaze_confidence']<self.subject_data.analysis_parameters['pl_confidence_threshold']] = np.nan # necessary in gaze this is run before elevation calcs

        def car_space_to_world_space(row_in):
            '''
            Default world direction is forward in world space
            '''

            # Get columns with 'cyclopeangaze_4x4' in the column name
            car_mat = row_in.filter(regex='simplecar_4x4')
            
            # first, set forward in world space (fic)
            car_fic_dir_xyzw = [0, 0, 1, 1]

            # rotate forward with car in world.  now it is forward in car space
            # To rotate the direction without translating it, set translation component to 0.
            car_4x4 = np.reshape(car_mat.values, [4, 4])
            car_4x4[:3, 3] = 0
            car_fiw_dir_xyz = np.dot(car_4x4, car_fic_dir_xyzw)[:3]

            return car_fiw_dir_xyz

        # Car forward in world 
        car_forward_dir_in_world_xyz = self.processed_gaze_data.apply(lambda row: car_space_to_world_space(row),axis=1)

        car_forward_dir_in_world_xyz = np.stack(car_forward_dir_in_world_xyz)
        #self.processed_gaze_data[['car_forward_dir_in_world_x','car_forward_dir_in_world_y','car_forward_dir_in_world_z']] = car_forward_dir_in_world_xyz
        
        car_forward_dir_in_world_xyzw = np.hstack([car_forward_dir_in_world_xyz,np.ones([len(car_forward_dir_in_world_xyz),1])])

        # Transform car forward from world space to road space
        inv_road_mat = np.linalg.inv(self.get_matrix_from_row('_4x4', self.road_data.iloc[0]))
        inv_road_mat[:3, 3] = 0
        car_forward_in_road_xyzw = np.dot(inv_road_mat,car_forward_dir_in_world_xyzw.T).T
        car_forward_in_road_xyz = car_forward_in_road_xyzw[:,:3]
        car_forward_in_road_az = np.rad2deg([np.arctan2(x,z) for x,y,z in car_forward_in_road_xyz])

        self.processed_gaze_data['gaze_az_rel_car'] = self.processed_gaze_data['gaze_dir_in_road_az'] - car_forward_in_road_az # gaze azimuth data
        self.processed_gaze_data['car_orientation_in_road_az'] = car_forward_in_road_az
        




        
    def calc_gaze_elevation_rel_head(self):
        
        # # filter by confidence 
        self.processed_gaze_data[self.processed_gaze_data['gaze_confidence']<self.subject_data.analysis_parameters['pl_confidence_threshold']] = np.nan

        # APPROACH 2:
        def world_space_to_car_space(row_in):
            
            # Get columns with 'cyclopeangaze_4x4' in the column name
            car_mat = row_in.filter(regex='simplecar_4x4')
            car_4x4 = np.reshape(car_mat.values, [4, 4])
            car_4x4[:3,3] = 0
            
            # TESTING VALUES: 1) Constant straight ahead in car space 2) Constant 45 degrees down in car space 
            # homogenous_giw_xyz = np.dot(car_4x4, np.array([0,0,1,1]))[:3]    # 1
            # homogenous_giw_xyz = np.dot(car_4x4, np.array([0,-1,1,1]))[:3]   # 2
            homogenous_giw_xyz = row_in[['giw_dir_x', 'giw_dir_y', 'giw_dir_z']].values   # REAL DATA
            homogenous_giw_xyz[1] *= -1 # because pupil labs has an opposite y coordinate system
            
            homogenous_giw_xyzw = np.hstack([homogenous_giw_xyz, 1])
            
            gaze_in_car_xyz = np.dot(np.linalg.inv(np.array(car_4x4, dtype=float)), homogenous_giw_xyzw)[:3]
            return gaze_in_car_xyz
        
        gaze_in_car_space_xyz = self.processed_gaze_data.apply(lambda row: world_space_to_car_space(row),axis=1)
        self.processed_gaze_data[['gaze_in_car_space_x','gaze_in_car_space_y','gaze_in_car_space_z']] = [[x,y,z] for x,y,z in gaze_in_car_space_xyz]
        
        self.processed_gaze_data['gaze_el_rel_head'] =  [np.rad2deg(np.arctan2(y,np.abs(z))) for x,y,z in gaze_in_car_space_xyz]
        




    def calculate_gaze_in_world(self):
        
        def eih_to_giw_row(row_in):

            # Get columns with 'head_4x4' in the column name
            head_mat = row_in.filter(regex='head_4x4')
            head_4x4 = np.reshape(head_mat.values,[4,4])
            
            # To rotate the direction without translating it, set translation component to 0. 
            head_4x4[:3,3] = 0

            # Get EIH in homogenous coordinates
            if (row_in.filter(regex='gaze_normal0').isna().sum()  + row_in.filter(regex='gaze_normal1').isna().sum() == 6):
                giw_dict = {'pupilLabsTimeStamp': row_in['pupilLabsTimeStamp'],
                    'giw_dir_x': np.nan, 
                    'giw_dir_y': np.nan,
                    'giw_dir_z': np.nan,
                    'giw_az': np.nan,
                    'giw_el': np.nan}

            else:

                # Average over the two mono gaze vectors
                eih_xyz = np.nanmean([row_in.filter(regex='gaze_normal0'), row_in.filter(regex='gaze_normal1')],axis=0)

                eih_xyz = eih_xyz / np.linalg.norm(eih_xyz)
                
                # Convert to GIW using homogenous coordinates
                eih_direction_xyzw = np.hstack([eih_xyz,1])
                
                # this rotates eye in head with the head. Does NOT translate
                giw_direction_xyz = np.dot(head_4x4,eih_direction_xyzw)[:3]
                
                giw_az = np.rad2deg(np.arctan2(giw_direction_xyz[0],giw_direction_xyz[2]))
                giw_el = np.rad2deg(np.arctan2(giw_direction_xyz[1],giw_direction_xyz[2]))
                
                
    #             x,y,z = zip(*head_pos_xyz_roadspace)

                giw_dict = {'pupilLabsTimeStamp':row_in['pupilLabsTimeStamp'],
                            'giw_dir_x': giw_direction_xyz[0], 
                            'giw_dir_y': giw_direction_xyz[1],
                            'giw_dir_z': giw_direction_xyz[2],
                            'giw_az': giw_az,
                            'giw_el': giw_el}
                
            return giw_dict

        
        giw_dict_list = self.processed_gaze_data.apply(lambda row: eih_to_giw_row(row),axis=1)
        giw_df = pd.DataFrame.from_records( giw_dict_list)
        
        # EDIT AG: added the following line so we can replace gaze info if we choose to run this function again
        try:
            self.processed_gaze_data = self.processed_gaze_data.drop(['giw_dir_x', 'giw_dir_y','giw_dir_z'],axis=1)
        except:
            pass
        
        # both pupil labs timestamp columns are the same so the "outer" command in merge doesn't make a difference -AG
        merged_df_with_gaze = pd.merge(giw_df, self.processed_gaze_data, on='pupilLabsTimeStamp', how='outer',sort=True)

        return merged_df_with_gaze




    def calc_rmse(self):

        wheel_angle = self.car_data[['time', 'wheelAngle']]
        wheel_angle = wheel_angle[
            wheel_angle['wheelAngle'].diff() != 0]  # commenting this line plots even repeats of wheel angle

        # there are some repeats, so remove the rows where diff is zero
        wheel_angles = wheel_angle['wheelAngle']
        s2s_rmse_wheel_angle = np.sqrt(np.mean(wheel_angles[~np.isnan(wheel_angles)].diff() ** 2))

        # Set the trial results
        self.results['s2s_rmse_wheel_angle'] = s2s_rmse_wheel_angle
        self.update_subject_results('s2s_rmse_wheel_angle',s2s_rmse_wheel_angle)




    def update_subject_results(self,column_name,value):
        #row_label = str(self.subject_data.results.index[self.subject_data.results['trial_num'] == self.results.trial_num])
        row_label = self.subject_data.results.index[self.subject_data.results['trial_num'] == self.results.trial_num][0]
        self.subject_data.results.at[row_label, column_name] = value





    def draw_road_in_world(self, ax_in):

        m = self.get_matrix_from_row('_4x4', self.road_data.iloc[0])
        self.calculate_road_edges()
        self.calculate_road_vertices_in_world()

        lx = self.road_vertices['roadedge_left_x_world']
        lz = self.road_vertices['roadedge_left_z_world']
        rx = self.road_vertices['roadedge_right_x_world']
        rz = self.road_vertices['roadedge_right_z_world']

        ax_in.plot(lx, lz, 'r')
        ax_in.plot(rx, rz, 'r')

        return ax_in, [lx.iloc[0], lz.iloc[0], rx.iloc[0], rz.iloc[0]], [lx.iloc[-1], lz.iloc[-1], rx.iloc[-1], rz.iloc[-1]]




    def get_dataframe_for_modeling(self):
        '''
        This function pulls together all the relevant data for each trial that will be useful
        when modeling steering behavior from optic flow fields. Ths work is in collaboration 
        with Oliver Layton.
        '''
        
        modeling_data = pd.DataFrame()

        # world index
        interp_frames = np.interp(self.pl_timestamps['pupilLabsTimeStamp'].values, self.processed_gaze_data['pupilLabsTimeStamp'].values, self.processed_gaze_data['world_index'].values).astype(int)
        interp_frames = interp_frames.astype(float)
        interp_frames[np.where(interp_frames == 0)[0]] = np.nan # replace any rows that were nans originally in gaze data back to nans 
        modeling_data['frame_num'] = interp_frames
        
        # timestamps
        modeling_data['unity_time'] = self.car_data['time']
        modeling_data['pupilLabsTime'] = self.pl_timestamps['pupilLabsTimeStamp'] 
        
        # car steering data
        modeling_data['signedWheelAngle'] = self.car_data['wheelAngle']
        modeling_data['carSignedDistFromRoadCenter'] = self.car_data['signedDistFromRoadCenter']
        
        # car position and orientation data in ROAD SPACE
        car_orientation = pd.DataFrame(columns=['car_orientation_in_road_x','car_orientation_in_road_y','car_orientation_in_road_z'])
        car_position = pd.DataFrame(columns=['car_pos_in_road_x','car_pos_in_road_y','car_pos_in_road_z'])
        for row_idx, row in self.car_data.iterrows():
            transform_car_to_world = row.filter(regex='simplecar_4x4').values.reshape(4,4) # car to world
            transform_world_to_road = np.linalg.inv(self.road_data.values.reshape(4,4)) # world to road
            transform_car_in_road = np.dot(transform_car_to_world, transform_world_to_road) # the top left 3x3 matrix is rotation matrix only 
            
            # transform the forward direction of the car in car space ([0,0,1]) to orientation in road space
            car_orientation_in_road_xyz = np.dot(transform_car_in_road[:3,:3], [0,0,1]).reshape(1,3)
            car_orientation = pd.concat((car_orientation, pd.DataFrame(car_orientation_in_road_xyz,columns=car_orientation.columns)),axis=0)
        
            car_position_in_road_xyz = np.dot(transform_world_to_road, np.hstack([row[['pos_x','pos_y','pos_z']].values,1]))[:3].reshape(1,3)
            car_position = pd.concat((car_position, pd.DataFrame(car_position_in_road_xyz,columns=car_position.columns)),axis=0)
        
        modeling_data = pd.concat((modeling_data, car_orientation.reset_index(drop=True), car_position.reset_index(drop=True)),axis=1)
        
        return modeling_data
    
    



def run_single_subject_gd(subject_data_folder, cbPatient = True, analyze_gaze = True, use_cached_data = False, gaze_contingent=False, estimate_head_trackers=False, session_num='S001'):

    this_sub = subject_data(subject_data_folder, cbPatient=cbPatient, analyze_gaze=analyze_gaze,
                            offset_car_xz=(-.4304, 0), use_cached_data=use_cached_data, gaze_contingent=gaze_contingent, estimate_head_trackers=estimate_head_trackers, session_num=session_num)

    if not use_cached_data:

        try:
            interp_time_seconds = np.round((this_sub.experiment_settings['turn_arc_length'] + 2 *
                                            this_sub.experiment_settings['length_straight_road_at_start']) \
                                           / this_sub.experiment_settings['speed_baseline'], 2)
        except: 
            interp_time_seconds = np.round((this_sub.experiment_settings['turn_arc_length'] + 2 *
                                        this_sub.experiment_settings['tracked_road_straight_leg_length']) \
                                       / this_sub.experiment_settings['speed_baseline'], 2)

        # this_sub.calc_divergence_over_segment_for_all_trials(0, 2)  # straight part only
        # this_sub.calc_divergence_over_segment_for_all_trials(30, 70)  # straight part only


        # this_sub.results['mean_div_from_inner_road_edge_30_70_adjusted'] = this_sub.results[
        #                                                                         'mean_div_from_inner_road_edge_30_70'] - \
        #                                                                     this_sub.results[
        #                                                                         'mean_div_from_inner_road_edge_0_2']

        this_sub.save_experiment_trials_to_pickle(interp_time_seconds)







#%%

if __name__ == "__main__":
    
    path_to_raw_data = 'raw_data'

    people = ['youngerDriver1']
    
    for person in people:
        
        run_single_subject_gd(os.path.join(path_to_raw_data + person),
                        analyze_gaze=True,
                        use_cached_data=False) # use_cached_data = False generates the pickle file from scratch
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            