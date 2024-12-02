import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import configparser, ast
from models.pendulum import InvertedPendulum
from models.dintegrator import DoubleIntegrator
from models.car2d import Robot2D
from models.drone import Drone
from mpl_toolkits.mplot3d import Axes3D

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

# config_file = 'models/dintegrator.conf'
# config_file = 'models/pendulum.conf'
# config_file = 'models/car2d.conf'
config_file = 'models/drone.conf'
system = Drone()


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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
norm = prismIntervalVector / np.max(prismIntervalVector)
colors = plt.cm.hot(norm)
ax.voxels(prismIntervalVector > 0, facecolors=colors, edgecolor='k')
plt.savefig('3dplot.png')