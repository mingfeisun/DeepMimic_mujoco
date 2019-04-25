# Intro

Mujoco version of [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html): 
* No C++ codes --> pure python
* No bullet engine --> Mujoco engine
* No PPO --> TRPO-based 

Examples: 

* Walk (play MoCap data):
<img src="docs/walk.gif" alt="walk" width="400px"/>

* Spinkick (play MoCap data):
<img src="docs/spinkick.gif" alt="spinkick" width="400px"/>

* Dance_b (play MoCap data):
<img src="docs/dance.gif" alt="dance" width="400px"/>

* Stand up straight (training via TRPO):
<img src="docs/standup.gif" alt="standup" width="400px"/>

# Install
* Mujoco: Download mujoco200 and put it in the ~/.mujoco/ folder (mjkey.txt should also be in this folder). Then install mujoco-py:
``` bash 
python3 -m pip install mujoco-py
```

* python3 modules: python dependencies
``` bash
python3 -m pip installl gym
python3 -m pip install tensorflow-gpu
python3 -m pip install pyquaternion
python3 -m pip install joblib
```

* MPI & MPI4PY: mpi for parrellel training
``` bash 
sudo apt-get install openmpi-bin openmpi-common openssh-client libopenmpi-dev
python3 -m pip install mpi4py
```

# Usage
* Testing examples:
``` bash
python3 dp_env_v3.py # play a mocap
python3 env_torque_test.py # torque control with p-controller
```

* Gym env

Before training a policy:
**Modify the step in dp_env_v3.py to set up correct rewards for the task**. Use **dp_env_v3.py** as the training env.

Training a policy that makes the agent stands up straight:
``` bash
python3 trpo.py
```
Running a policy:
``` bash
python3 trpo.py --task evaluate --load_model_path XXXX # for evaluation
# e.g., python3 trpo.py --task evaluate --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0
```

# Acknowledge

This repository is based on code accompanying the SIGGRAPH 2018 paper:
"DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills".
The framework uses reinforcement learning to train a simulated humanoid to imitate a variety
of motion skills from mocap data.
Project page: https://xbpeng.github.io/projects/DeepMimic/index.html
