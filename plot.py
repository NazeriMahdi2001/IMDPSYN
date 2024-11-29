import numpy as np
import matplotlib.pyplot as plt
import configparser, ast

policy = {}
with open('prismPRISM_interval_policy.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split('=')
        value = value.split('_')[1]
        key = tuple(ast.literal_eval(key))
        policy[key] = int(value)

trans = {}
with open('prism/abstraction.sta', 'r') as file:
    next(file)  # Skip the header line "(x1,x2)"
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = int(key)
            value = tuple(ast.literal_eval(value))
            trans[key] = value

def parse_list(value):
    return np.array(ast.literal_eval(value))

config_file = 'models/dintegrator.conf'
# config_file = 'models/pendulum.conf'
# config_file = 'models/car2d.conf'
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

plt.imshow(prismIntervalVector, extent=(stateLowerBound[0], stateUpperBound[0], stateUpperBound[1], stateLowerBound[1]), aspect='auto', cmap='jet')
plt.colorbar(label='Interval Value')
plt.xlabel('State Dimension 1')
plt.ylabel('State Dimension 2')

plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], 4*stateResolution[0]))
plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], 4*stateResolution[1]))
plt.gca().invert_yaxis()


plt.savefig('prism_interval_vector_heatmap.png', dpi=500)

plt.figure(figsize=[10, 10])
plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], stateResolution[0]*4))
plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], stateResolution[1]*4))
plt.xlim(stateLowerBound[0], stateUpperBound[0])
plt.ylim(stateLowerBound[1], stateUpperBound[1])
plt.xlabel('State Dimension 1')
plt.ylabel('State Dimension 2')
plt.grid(True)

init = (25, 32)
for _ in range(100):
    if init in policy:
        next = policy[init]
        if next in trans:
            next = trans[next]
            # draw an arrow from init to next
            a = stateLowerBound + np.array(init) * stateResolution + stateResolution / 2
            b = stateLowerBound + np.array(next) * stateResolution + stateResolution / 2
            plt.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], head_width=0.2, head_length=0.05, fc='b', ec='b')
            init = next

# add the goal set
goalLowerBound = np.array(goalLowerBound)
goalUpperBound = np.array(goalUpperBound)
plt.fill_between([goalLowerBound[0], goalUpperBound[0]], goalLowerBound[1], goalUpperBound[1], color='g', alpha=0.2)

# criticalLowerBound = np.array(criticalLowerBound)
# criticalUpperBound = np.array(criticalUpperBound)
# plt.fill_between([criticalLowerBound[0], criticalUpperBound[0]], criticalLowerBound[1], criticalUpperBound[1], color='r', alpha=0.2)

plt.savefig('trajectory.png', dpi=500)