from models.pendulum import InvertedPendulum
from models.dintegrator import DoubleIntegrator
from models.car2d import Robot2D
from models.drone import Drone
from models.dubins import DubinsCar
from source.abstraction import Abstraction

# prism_executable='/home/mnazeri/IMDPSYN/prism-4.8.1-linux64-x86/bin/prism'
prism_executable='/Users/mbk-24-0253/Desktop/IMDPSYN/prism-4.8.1-mac64-arm/bin/prism'
foldername='/Users/mbk-24-0253/Desktop/IMDPSYN/prism'

# abstraction = Abstraction(DoubleIntegrator(), '/Users/mbk-24-0253/Desktop/IMDPSYN/models/dintegrator.conf')
# abstraction = Abstraction(InvertedPendulum(), '/Users/mbk-24-0253/Desktop/IMDPSYN/models/pendulum.conf')
abstraction = Abstraction(Robot2D(), '/Users/mbk-24-0253/Desktop/IMDPSYN/models/car2d.conf')
# abstraction = Abstraction(Drone(), '/home/mnazeri/IMDPSYN/models/drone.conf')
# abstraction = Abstraction(DubinsCar(), '/home/mnazeri/IMDPSYN/models/dubins.conf')

abstraction.generate_noise_samples()
abstraction.generate_samples()
abstraction.find_actions()
abstraction.find_transitions()
abstraction.create_IMDP(foldername=foldername)
abstraction.solve_iMDP(foldername=foldername, prism_executable=prism_executable)
