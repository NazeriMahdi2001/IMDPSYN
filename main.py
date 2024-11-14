from models.pendulum import InvertedPendulum
from source.abstraction import Abstraction

prism_executable='/root/IMDP/prism-4.8.1-linux64-x86/bin/prism'
foldername='/root/IMDP/prism'

abstraction = Abstraction(InvertedPendulum(), '/root/IMDP/models/pendulum.conf')
abstraction.generate_noise_samples()
abstraction.generate_samples()
abstraction.find_actions()
abstraction.find_transitions()
abstraction.create_IMDP(foldername=foldername)
abstraction.solve_iMDP(foldername=foldername, prism_executable=prism_executable)
