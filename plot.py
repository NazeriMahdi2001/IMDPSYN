import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import configparser, ast
from models.pendulum import InvertedPendulum
from models.dintegrator import DoubleIntegrator
from models.car2d import Robot2D

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

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
system = DoubleIntegrator()


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
numDivisions = int(config['DEFAULT']['numDivisions'])
noiseLevel = float(config['DEFAULT']['noiseLevel'])

prismIntervalVector = np.genfromtxt('prismPRISM_interval_vector.csv')
prismIntervalVector = prismIntervalVector[3:]
prismIntervalVector = prismIntervalVector.reshape(np.round(((stateUpperBound - stateLowerBound) / stateResolution)).astype(int)).T

plt.imshow(prismIntervalVector, extent=(stateLowerBound[0], stateUpperBound[0], stateUpperBound[1], stateLowerBound[1]), aspect='auto', cmap='coolwarm')
plt.colorbar(label='Interval Value')
plt.xlabel('State Dimension 1')
plt.ylabel('State Dimension 2')

plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], 4*stateResolution[0]))
plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], 4*stateResolution[1]))
plt.gca().invert_yaxis()


plt.savefig('prism_interval_vector_heatmap.png', dpi=500, bbox_inches='tight')

plt.figure(figsize=[10, 10])
plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], stateResolution[0]))
plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], stateResolution[1]))
plt.xlim(stateLowerBound[0], stateUpperBound[0])
plt.ylim(stateLowerBound[1], stateUpperBound[1])
plt.xlabel('State Dimension 1')
plt.ylabel('State Dimension 2')
plt.grid(True)

x = np.array([5, -0.84])
init = tuple(((x - stateLowerBound) // stateResolution).astype(int))
for _ in range(100):
    if init in policy:
        next = policy[init]
        if next in trans:
            next = trans[next]
            # draw an arrow from init to next
            a = stateLowerBound + np.array(init) * stateResolution + stateResolution / 2
            b = stateLowerBound + np.array(next) * stateResolution + stateResolution / 2
            plt.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], head_width=0.2, head_length=0.05, fc='b', ec='b').set_zorder(10)
            init = next

# add the goal set
goalLowerBound = np.array(goalLowerBound)
goalUpperBound = np.array(goalUpperBound)
plt.fill_between([goalLowerBound[0], goalUpperBound[0]], goalLowerBound[1], goalUpperBound[1], color='g', alpha=0.2)

for i in range(len(criticalLowerBound)):
    clb = np.array(criticalLowerBound[i])
    cub = np.array(criticalUpperBound[i])
    plt.fill_between([clb[0], cub[0]], clb[1], cub[1], color='r', alpha=0.2)

plt.savefig('trajectory.png', dpi=500, bbox_inches='tight')

plt.figure(figsize=[10, 10])
plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], stateResolution[0]))
plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], stateResolution[1]))
plt.xlim(stateLowerBound[0], stateUpperBound[0])
plt.ylim(stateLowerBound[1], stateUpperBound[1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)

init = tuple(((x - stateLowerBound) // stateResolution).astype(int))
plt.scatter(x[0], x[1], c='blue', s=50, marker='s').set_zorder(11)
voxelResolution = stateResolution / numDivisions
for _ in range(100):
    if init in policy:
        next = policy[init]
        if next in trans:
            next = trans[next]
            # draw an arrow from init to next

            policy_filename = f'policy/policy_{init}_{next}.npy'
            refined_policy = np.load(policy_filename)

            residue = x - init * stateResolution - stateLowerBound
            ind = (residue // voxelResolution).astype(int)
            control = refined_policy[*ind, :]
            nx = system.set_state(*x).update_dynamics(control) + np.random.uniform(-0.5*stateResolution*noiseLevel, 0.5*stateResolution*noiseLevel, stateDimension)
            plt.scatter(x[0], x[1], c='black', s=5).set_zorder(11)
            plt.arrow(x[0], x[1], nx[0] - x[0], nx[1] - x[1], head_width=0, head_length=0, fc='blue', ec='blue').set_zorder(10)
            
            init = tuple(((nx - stateLowerBound) // stateResolution).astype(int))
            x = nx

plt.scatter(x[0], x[1], c='blue', s=50).set_zorder(11)
# add the goal set
goalLowerBound = np.array(goalLowerBound)
goalUpperBound = np.array(goalUpperBound)
plt.fill_between([goalLowerBound[0], goalUpperBound[0]], goalLowerBound[1], goalUpperBound[1], color='g', alpha=0.2)

for i in range(len(criticalLowerBound)):
    clb = np.array(criticalLowerBound[i])
    cub = np.array(criticalUpperBound[i])
    plt.fill_between([clb[0], cub[0]], clb[1], cub[1], color='r', alpha=0.2)

plt.savefig('trajectory_refined.png', dpi=500, bbox_inches='tight')