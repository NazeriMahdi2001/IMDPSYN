import numpy as np
import configparser, ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from source.commons import floor_decimal
from source.tabulate_scenario import create_table
from source.compute_probabilities import compute_intervals
from source.commons import writeFile

class Abstraction:
    def __init__(self, dynamics, config_file):
        self.dynamics = dynamics

        config = configparser.ConfigParser()
        config.read(config_file)

        # Dimension of the state and control space
        self.stateDimension = int(config['DEFAULT']['stateDimension'])
        self.controlDimension = int(config['DEFAULT']['controlDimension'])
        
        # Domain of the state space
        self.stateLowerBound = self.parse_list(config['DEFAULT']['stateLowerBound'])
        self.stateUpperBound = self.parse_list(config['DEFAULT']['stateUpperBound'])

        # Domain of the control space
        self.controlLowerBound = self.parse_list(config['DEFAULT']['controlLowerBound'])
        self.controlUpperBound = self.parse_list(config['DEFAULT']['controlUpperBound'])

        # Domain of the goal set
        self.goalLowerBound = self.parse_list(config['DEFAULT']['goalLowerBound'])
        self.goalUpperBound = self.parse_list(config['DEFAULT']['goalUpperBound'])

        # Domain of the critical set
        self.criticalLowerBound = self.parse_list(config['DEFAULT']['criticalLowerBound'])
        self.criticalUpperBound = self.parse_list(config['DEFAULT']['criticalUpperBound'])

        # Resolution of the state space : size of each abstract cell
        self.stateResolution = self.parse_list(config['DEFAULT']['stateResolution'])

        # Number of samples in each abstract cell
        self.numGridSamples = self.parse_list(config['DEFAULT']['numGridSamples']).astype(int)

        # Number of divisions in each dimension of the state space for when I want to find if there exists a control input such that the next state of the nominal system is inside the target set of an abstract state
        self.numDivisions = int(config['DEFAULT']['numDivisions'])

        # Number of abstract cells in each dimension
        self.absDimension = np.ceil((self.stateUpperBound - self.stateLowerBound) / self.stateResolution).astype(int)

        self.numNoiseSamples= int(config['DEFAULT']['numNoiseSamples'])

    def find_abs_state(self, state):
        # Find the abstract state of a continuous state
        return np.floor((state - self.stateLowerBound) / self.stateResolution).astype(int)

    def if_within(self, state, lowerBound, upperBound):
        return np.all(state >= lowerBound) and np.all(state <= upperBound)

    def parse_list(self, value):
        return np.array(ast.literal_eval(value))

    def generate_samples(self):
        # sample record is an n-d array of lists, where each list contains the samples that reach the corresponding abstract state
        sample_record = np.empty(self.absDimension, dtype=object)
        for index, _ in np.ndenumerate(sample_record):
            sample_record[index] = []

        # samples from the state space
        state_linspaces = [np.linspace(0, bound, int(samples), endpoint=False) for bound, samples in
                           zip(self.stateResolution, self.numGridSamples[0])]
        state_grid = np.moveaxis(np.meshgrid(*state_linspaces, copy=False), 0, -1)

        # samples from the control space
        control_linspaces = [np.linspace(lower, upper, int(samples), endpoint=True) for lower, upper, samples in
                             zip(self.controlLowerBound, self.controlUpperBound, self.numGridSamples[1])]
        control_grid = np.moveaxis(np.meshgrid(*control_linspaces, copy=False), 0, -1)

        # generate samples
        for abs_state in tqdm(np.ndindex(tuple(self.absDimension)), desc="Processing abstract states", total=np.prod(self.absDimension)):
            abs_state_lower_bound = self.stateLowerBound + self.stateResolution * np.array(abs_state)

            for state_index in np.ndindex(state_grid.shape[:-1]):
                # pick a sample state
                state = abs_state_lower_bound + state_grid[state_index]
                best_control = {}
                # iterate over all control inputs
                for control_index in np.ndindex(control_grid.shape[:-1]):
                    control = control_grid[control_index]
                    next_state = np.array(self.dynamics.set_state(*state).update_dynamics(control))

                    if not self.if_within(next_state, self.stateLowerBound, self.stateUpperBound):
                        continue

                    abs_next_state = self.find_abs_state(next_state)
                    abs_next_lower_bound = self.stateLowerBound + self.stateResolution * abs_next_state
                    abs_next_upper_bound = abs_next_lower_bound + self.stateResolution

                    next_state_distance_to_border = np.minimum(np.array(next_state - abs_next_lower_bound), np.array(abs_next_upper_bound - next_state))
                    next_state_freedom = np.min(next_state_distance_to_border / ((self.stateResolution) @ self.dynamics.max_jacobian(control).T))

                    # keep the control input that gives the maximum freedom - which is the maximum distance to the border
                    abs_next_state = tuple(abs_next_state)
                    if abs_next_state in best_control:
                        if best_control[abs_next_state][1] < next_state_freedom:
                            best_control[abs_next_state] = (control, next_state_freedom)
                    else:
                        best_control[abs_next_state] = (control, next_state_freedom)


                # record best control inputs for each forward reached abstract state
                for reachable_state in best_control:
                    next_state = np.array(self.dynamics.set_state(*state).update_dynamics(best_control[reachable_state][0]))
                    abs_next_state = self.find_abs_state(next_state)
                    sample_record[tuple(abs_next_state)].append((state, best_control[reachable_state][0], next_state))
                    # print(state, best_control[reachable_state][0], next_state, self.find_abs_state(state), abs_next_state)
                
            # break # uncomment this line to test the code with a single abstract state
            
        self.record = sample_record

        print("Samples are generated")

    def find_actions(self):
        self.actions = np.empty(self.absDimension, dtype=object)
        for index, _ in np.ndenumerate(self.actions):
            self.actions[index] = []

        half_resolution = np.array(self.stateResolution) / (self.numDivisions * 2)
        state_linspaces = [np.linspace(lower, upper, self.numDivisions, endpoint=True) for lower, upper in zip(half_resolution, self.stateResolution - half_resolution)]
        state_grid = np.moveaxis(np.meshgrid(*state_linspaces, copy=False), 0, -1)
        
        for abs_state_index, _ in np.ndenumerate(self.record):
            abs_state_lower_bound = self.stateLowerBound + self.stateResolution * np.array(abs_state_index)
            abs_state_upper_bound = abs_state_lower_bound + self.stateResolution

            predecessors = {}
            for sample in self.record[abs_state_index]:
                key = tuple(self.find_abs_state(sample[0]))
                if key not in predecessors:
                    predecessors[key] = []
                predecessors[key].append(sample)

            for pre_state_index in predecessors:
                if len(predecessors[pre_state_index]) < self.numGridSamples[0][0] * self.numGridSamples[0][1]:
                    continue
                pre_state_lower_bound = self.stateLowerBound + self.stateResolution * np.array(pre_state_index)
                pre_state_upper_bound = pre_state_lower_bound + self.stateResolution
                target_size = np.ones((int(self.numDivisions ** self.stateDimension), 2)) * -1e3

                # print(pre_state_index, abs_state_index, len(predecessors[pre_state_index]))
                for sample in predecessors[pre_state_index]:
                    next_state_dist_to_border = np.minimum(np.array(sample[2] - abs_state_lower_bound), np.array(abs_state_upper_bound - sample[2]))
                    grid_index = -1

                    for idx in np.ndindex(state_grid.shape[:-1]):
                        grid_index += 1
                        point = state_grid[idx] + pre_state_lower_bound
                        pre_state_dist_to_border = np.abs(sample[0] - point) + half_resolution

                        delta_f = pre_state_dist_to_border @ np.array(self.dynamics.max_jacobian(sample[1])).T
                        target_size[grid_index, :] = np.maximum(target_size[grid_index, :], next_state_dist_to_border - delta_f)
                

                if np.min(target_size) > 0:
                    #print(f'For every continuous state in {np.array(pre_state_index)}, these exist a control input such that the next state of the nominal system is inside target set of abstract state {abs_state_index}')
                    #print(f'i = {np.array(pre_state_index)}, j = {np.array(abs_state_index)}, c_i = {np.array(pre_state_lower_bound)} to {np.array(pre_state_upper_bound)}, c_j = {np.array(abs_state_lower_bound)} to {np.array(abs_state_upper_bound)}, t_i_to_j = {np.array(abs_state_lower_bound + np.min(target_size, axis=0))} to {np.array(abs_state_upper_bound) - np.min(target_size, axis=0)}')
                    self.actions[pre_state_index].append([pre_state_index, abs_state_index, abs_state_lower_bound + np.min(target_size, axis=0), abs_state_upper_bound - np.min(target_size, axis=0)])
                # plt.figure()
                # freedom_grid = target_size[:, 0].reshape((self.numDivisions, self.numDivisions))
                # plt.imshow(freedom_grid, extent=(pre_state_lower_bound[0], pre_state_upper_bound[0], pre_state_lower_bound[1], pre_state_upper_bound[1]), origin='lower', cmap='viridis', alpha=0.6)
                # plt.colorbar(label='Freedom')
                # plt.xlabel('State Dimension 1')
                # plt.ylabel('State Dimension 2')
                # plt.title(f'Freedom for transition from {pre_state_index} to {abs_state_index}')
                # plt.savefig(f'plot{pre_state_index}{abs_state_index}.png', dpi=500)

    def generate_noise_samples(self, noiseAmplitude=0.1):
        self.noise_samples = np.random.uniform(np.zeros(self.stateDimension), self.stateResolution*noiseAmplitude, (self.numNoiseSamples, self.stateDimension))

    def find_transitions(self):
        self.partition = {
            'state_variables': ['x','y'],
            'dim': self.stateDimension,
            'lb': self.stateLowerBound,
            'ub': self.stateUpperBound,
            'regions_per_dimension': self.absDimension,
        }
        self.partition['size_per_region'] = self.stateResolution
        
        self.partition['goal_idx'] = set()
        self.partition['unsafe_idx'] = set()

        # iterate over all abstract states
        for abs_state_index, _ in np.ndenumerate(self.record):
            abs_state_lower_bound = self.stateLowerBound + self.stateResolution * np.array(abs_state_index)
            abs_state_upper_bound = abs_state_lower_bound + self.stateResolution
            
            if len(self.goalUpperBound) > 0:
                if self.if_within(abs_state_lower_bound, self.goalLowerBound, self.goalUpperBound) and self.if_within(abs_state_upper_bound, self.goalLowerBound, self.goalUpperBound):
                    self.partition['goal_idx'].add(abs_state_index)
            
            if len(self.criticalLowerBound) > 0:
                if self.if_within(abs_state_lower_bound, self.criticalLowerBound, self.criticalUpperBound) or self.if_within(abs_state_upper_bound, self.criticalLowerBound, self.criticalUpperBound):
                    self.partition['unsafe_idx'].add(abs_state_index)

        # Every partition element also has an integer identifier
        iterator = itertools.product(*map(range, np.zeros(self.partition['dim'], dtype=int), self.partition['regions_per_dimension'])) # why it was self.partition['regions_per_dimension'] + 1?
        self.partition['tup2idx'] = {tup: idx for idx,tup in enumerate(iterator)}
        self.partition['idx2tup'] = {tup: idx for idx, tup in enumerate(iterator)}
        self.partition['nr_regions'] = len(self.partition['tup2idx'])

        # The probability table is an N+1 x 2 table, with N the number of samples. The first column contains the lower bound
        # probability, and the second column the upper bound.
        probability_table = np.zeros((self.numNoiseSamples+1, 2))

        # We specify the probability with which a probability interval is wrong (i.e., 1 minus the confidence probability)
        inverse_confidence = 0.05

        P_low, P_upp = create_table(N=self.numNoiseSamples, beta=inverse_confidence, kstep=1, trials=0, export=False)
        probability_table = np.column_stack((P_low, P_upp))

        self.transitions = np.empty(self.absDimension, dtype=object)
        for index, _ in np.ndenumerate(self.transitions):
            self.transitions[index] = []

        for index, _ in np.ndenumerate(self.transitions):
            for action in self.actions[index]:
                target_lb = action[2]
                target_ub = action[3]

                clusters = {
                    'lb': self.noise_samples + target_lb,
                    'ub': self.noise_samples + target_ub,
                    'value': np.ones(self.numNoiseSamples)
                }
                
                output = compute_intervals(Nsamples=self.numNoiseSamples, inverse_confidence=inverse_confidence, partition=self.partition, clusters=clusters, probability_table=probability_table, debug=False)
                self.transitions[index].append(output)

        print("Transitions are found")

    def create_IMDP(self, timebound=np.inf, problem_type='reachavoid', foldername='/root/IMDP/prism'):
        print('\nExport abstraction as PRISM model...')

        timespec = ""
        if timebound != np.inf:
            timespec = 'F<=' + str(timebound) + ' '
        else:
            timespec = 'F '

        if problem_type == 'avoid':
            specification = 'Pminmax=? [' + timespec + ' "failed" ]'
        else:
            specification = 'Pmaxmin=? [' + timespec + ' "reached" ]'

        # Write specification file
        writeFile(foldername + "/abstraction.pctl", 'w', specification)

        ##############################

        # Define tuple of state variables (for header in PRISM state file)
        state_var_string = ['(' + ','.join([f'x{i+1}' for i in range(self.stateDimension)]) + ')']

        state_file_header = ['0:(' + ','.join([str(-3)] * self.stateDimension) + ')',
                             '1:(' + ','.join([str(-2)] * self.stateDimension) + ')',
                             '2:(' + ','.join([str(-1)] * self.stateDimension) + ')']

        state_file_content = []

        for abs_state in itertools.product(*map(range, self.absDimension)):
            state_id = self.partition['tup2idx'][abs_state] + 3
            state_representation = str(state_id) + ':' + str(abs_state).replace(' ', '')
            state_file_content.append(state_representation)

        state_file_string = '\n'.join(state_var_string + state_file_header + state_file_content)

        # Write content to file
        writeFile(foldername + "/abstraction.sta", 'w', state_file_string)

        label_head = ['0="init" 1="deadlock" 2="reached" 3="failed"'] + \
                     ['0: 1 3'] + ['1: 1 3'] + ['2: 2']

        label_body = ['' for i in range(self.record.size)]

        for abs_state in itertools.product(*map(range, self.absDimension)):
            state_id = self.partition['tup2idx'][abs_state]
            substring = str(state_id + 3) + ': 0'

            # Check if region is a deadlock state
            if self.actions[abs_state] == []:
                substring += ' 1'

            # Check if region is in goal set
            if abs_state in self.partition['goal_idx']:
                substring += ' 2'
            elif abs_state in self.partition['unsafe_idx']:
                substring += ' 3'

            label_body[state_id] = substring

        label_full = '\n'.join(label_head) + '\n' + '\n'.join(label_body)

        # Write content to file
        writeFile(foldername + "/abstraction.lab", 'w', label_full)

        ##############################

        nr_choices_absolute = -1
        nr_transitions_absolute = 0
        transition_file_list = ''
        head = 3

        for index, _ in np.ndenumerate(self.transitions):
            if index in self.partition['goal_idx']:
                # print(' ---- Skip',index,'because it is a goal region')
                continue
            if index in self.partition['unsafe_idx']:
                # print(' ---- Skip',index,'because it is a critical region')
                continue
            
            if self.transitions[index] != []:
                choice = -1
                for action in self.transitions[index]:
                    choice += 1
                    nr_choices_absolute += 1
                    print(action)
                    for trnasition_ind in range(len(action['successor_idxs'])):
                        transition_file_list += str(self.partition['tup2idx'][index] + head) + ' ' + str(choice) + ' ' + \
                        str(int(action['successor_idxs'][trnasition_ind]) + head) + ' ' + str(action['interval_strings'][trnasition_ind]) + \
                        ' a_' + str(nr_choices_absolute) + '\n'
                        nr_transitions_absolute += 1
                    
                    transition_file_list += str(self.partition['tup2idx'][index] + head) + ' ' + str(choice) + \
                    ' 0 ' + str(action['outOfPartition_interval_string']) + \
                    ' a_' + str(nr_choices_absolute) + '\n'
                    nr_transitions_absolute += 1
                
            else: 
                new_transitions = str(self.partition['tup2idx'][index] + head) + ' 0 ' + str(self.partition['tup2idx'][index] + head) + ' [1.0,1.0]\n'
                transition_file_list += new_transitions
                nr_choices_absolute += 1
                nr_transitions_absolute += 1
                    
        
        size_states = self.record.size + head
        size_choices = nr_choices_absolute + head
        size_transitions = nr_transitions_absolute + head
        header = str(size_states) + ' ' + str(size_choices) + ' ' + str(size_transitions) + '\n'
        firstrow = '0 0 0 [1.0,1.0]\n1 0 1 [1.0,1.0]\n2 0 2 [1.0,1.0]\n'
        writeFile(foldername + "/abstraction.tra", 'w', header + firstrow + transition_file_list)
