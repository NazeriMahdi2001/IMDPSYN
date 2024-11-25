from models.pendulum import InvertedPendulum
from models.dintegrator import DoubleIntegrator
from source.abstraction import Abstraction

prism_executable='/home/mnazeri/IMDPSYN/prism-4.8.1-linux64-x86/bin/prism'
foldername='/home/mnazeri/IMDPSYN/prism'

abstraction = Abstraction(DoubleIntegrator(), '/home/mnazeri/IMDPSYN/models/dintegrator.conf')
abstraction.generate_noise_samples()
abstraction.generate_samples()
abstraction.find_actions()
abstraction.find_transitions()
abstraction.create_IMDP(foldername=foldername)
abstraction.solve_iMDP(foldername=foldername, prism_executable=prism_executable)
