# Deepbots Tutorial

This tutorial is on how to use the deepbots framework. We will recreate the 
[CartPole](https://gym.openai.com/envs/CartPole-v0/) problem in Webots, step-by-step and solve it with the 
[Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) Reinforcement Learning (RL) 
algorithm, using [PyTorch](https://pytorch.org/) as our neural network backend library.

The complete example can be found on the [deepworlds](https://github.com/aidudezzz/deepworlds/) directory. 


## Prerequisites

Before starting, several prerequisites should be met.

1. [Install Webots](https://cyberbotics.com/doc/guide/installing-webots)
    - [Windows](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-windows)
    - [Linux](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux)
    - [macOS](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-macos)
2. [Install Python version 3.X](https://www.python.org/downloads/) (please refer to 
[Using Python](https://cyberbotics.com/doc/guide/using-python) to select proper Python version for your system) 
3. Follow the [Using Python](https://cyberbotics.com/doc/guide/using-python) guide
4. [Using PyCharm IDE](https://cyberbotics.com/doc/guide/using-your-ide#pycharm)
5. [Install PyTorch](https://pytorch.org/get-started/locally/)

## CartPole Tutorial
<!--TODO add screenshots-->
Now we are ready to start working on the CartPole problem.

##### Creating the project  
1. Open Webots and on the *menu bar* click *Wizards -> New Project Directory* 
2. Select a directory of your choice
3. On world settings **all** boxes should be ticked
4. Give your world a name, e.g. "cartPoleWorld.wbt"
5. Press Finish
6. Download the <a id="raw-url" href="https://raw.githubusercontent.com/aidudezzz/deepbots-tutorial/master/CartPoleRobot.wbo">Download FILE</a>

You should end up with: *screenshot*

![cartpole axis](/images/cartPoleWorldAxes.png)
