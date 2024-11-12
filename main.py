from models.pendulum import InvertedPendulum
from source.abstraction import Abstraction


abstraction = Abstraction(InvertedPendulum(), '/root/IMDP/models/pendulum.conf')
abstraction.generate_noise_samples()
abstraction.generate_samples()
abstraction.find_actions()
abstraction.find_transitions()
abstraction.create_IMDP()
