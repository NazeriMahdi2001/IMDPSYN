import numpy as np
import matplotlib.pyplot as plt
import configparser, ast

def parse_list(value):
    return np.array(ast.literal_eval(value))

config_file = 'models/pendulum.conf'
config = configparser.ConfigParser()
config.read(config_file)

# Dimension of the state and control space
stateDimension = int(config['DEFAULT']['stateDimension'])
controlDimension = int(config['DEFAULT']['controlDimension'])

# Domain of the state space
stateLowerBound = parse_list(config['DEFAULT']['stateLowerBound'])
stateUpperBound = parse_list(config['DEFAULT']['stateUpperBound'])

# Domain of the goal set
goalLowerBound = parse_list(config['DEFAULT']['goalLowerBound'])
goalUpperBound = parse_list(config['DEFAULT']['goalUpperBound'])

# Domain of the critical set
criticalLowerBound = parse_list(config['DEFAULT']['criticalLowerBound'])
criticalUpperBound = parse_list(config['DEFAULT']['criticalUpperBound'])

stateResolution = parse_list(config['DEFAULT']['stateResolution'])

prismIntervalVector = np.genfromtxt('prismPRISM_interval_vector.csv')
prismIntervalVector = prismIntervalVector[3:]
prismIntervalVector = prismIntervalVector.reshape(np.round(((stateUpperBound - stateLowerBound) / stateResolution)).astype(int)).T

plt.imshow(prismIntervalVector, extent=(stateLowerBound[0], stateUpperBound[0], stateUpperBound[1], stateLowerBound[1]), aspect='auto', cmap='gray')
plt.colorbar(label='Interval Value')
plt.xlabel('State Dimension 1')
plt.ylabel('State Dimension 2')
plt.title('Prism Interval Vector Heatmap')

plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], stateResolution[0]))
plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], stateResolution[1]))
plt.gca().invert_yaxis()

plt.savefig('prism_interval_vector_heatmap.png', dpi=500)