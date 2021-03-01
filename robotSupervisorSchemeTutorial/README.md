# CartPole Beginner Robot-Supervisor Scheme Tutorial

![Solved cartpole demonstration](/robotSupervisorSchemeTutorial/images/cartPoleWorld.gif)

This tutorial shows the creation of the [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem using the updated
version of the [*deepbots framework*](https://github.com/aidudezzz/deepbots), utilizing the 
[robot-supervisor scheme](https://github.com/aidudezzz/deepbots#combined-robot-supervisor-scheme) which combines the
gym environment and the robot controller in one script, forgoing the emitter and receiver communication.

The first parts of the tutorial are identical to the 
[original CartPole tutorial](https://github.com/aidudezzz/deepbots-tutorials/tree/master/cartPoleTutorial) that uses the 
[emitter-receiver scheme](https://github.com/aidudezzz/deepbots#emitter---receiver-scheme), so one can follow either
one, depending on their use-case. Mainly, if you desire to set up a more complicated example that might use multiple
robots or similar, refer to the emitter-receiver tutorial to get started.

Keep in mind that the tutorial is very detailed and many parts can be completed really fast by an 
experienced user. The tutorial assumes no familiarity with the [Webots](https://cyberbotics.com/) simulator.

We will recreate the [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem step-by-step in 
[Webots](https://cyberbotics.com/), and solve it with the 
[Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) 
Reinforcement Learning (RL) algorithm, using [PyTorch](https://pytorch.org/) as our neural network backend library.

We will focus on creating the project, the world and the controller script and how to use the *deepbots framework*.
For the purposes of the tutorial, a basic implementation of the PPO algorithm is provided. For guides on how to 
construct a custom robot, please visit the official Webots 
[tutorial](https://cyberbotics.com/doc/guide/tutorial-6-4-wheels-robot). 

You can check the complete example [here](/robotSupervisorSchemeTutorial/full_project) with all the scripts and nodes 
used in this tutorial.
The CartPole example is available on the [deepworlds](https://github.com/aidudezzz/deepworlds/) repository.


## Prerequisites

_Please note that this tutorial targets the newest deepbots release (0.1.3) which is currently in development,
you can install the dev version with this command:_

_`pip install -i https://test.pypi.org/simple/ deepbots`_

Before starting, several prerequisites should be met. Follow the [installation section on the deepbots framework main 
repository](https://github.com/aidudezzz/deepbots#installation).

For this tutorial you will also need to [install PyTorch](https://pytorch.org/get-started/locally/) 
(no CUDA/GPU support needed for this tutorial).


## CartPole
### Creating the project

Now we are ready to start working on the *CartPole* problem. First of all, we should create a new project.

1. Open Webots and on the menu bar, click *"Wizards -> New Project Directory..."*\
    ![New project menu option](/robotSupervisorSchemeTutorial/images/newProjectMenuScreenshot.png)
2. Select a directory of your choice
3. On world settings **all** boxes should be ticked\
    ![World settings](/robotSupervisorSchemeTutorial/images/worldSettingsScreenshot.png)
4. Give your world a name, e.g. "cartPoleWorld.wbt"
5. Press Finish

You should end up with:\
![Project created](/robotSupervisorSchemeTutorial/images/projectCreatedScreenshot.png)


### Adding a *supervisor robot* node in the world

First of all we will download the *CartPole robot node* definition that is supplied for the purposes of the tutorial.
 
1. Right-click on
[this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/robotSupervisorSchemeTutorial/full_project/controllers/robotSupervisorController/CartPoleRobot.wbo) 
and click *Save link as...* to download the CartPole robot definition 
2. Save the .wbo file in a directory of your choice, where you can easily find it later.

Now that the project and the starting world are created, we are going to create our *robot*, that also has *supervisor*
privileges. Later, we will add the *controller* script, through which we will be able to handle several 
aspects of the simulation needed for RL, but also control the robot with the actions produced by the 
RL agent.
 
1. Click on the *Add a new object or import an object* button\
![Add new object button](/robotSupervisorSchemeTutorial/images/addNewObjectButtonScreenshot.png)
2. Click on *Import...* on the bottom right of the window\
![Add Robot node](/robotSupervisorSchemeTutorial/images/importRobotNodeScreenshot.png)
3. Locate the .wbo file downloaded earlier, select it and click *Open*
4. Now on the left side of the screen, under the *Rectangle Arena* node, you can see the *Robot* node
5. Double click on the *Robot* node to expand it
6. Scroll down to find the *supervisor* field and set it to TRUE\
![Set supervisor to TRUE](/robotSupervisorSchemeTutorial/images/setSupervisorTrueScreenshot.png)
7. Click *Save*\
![Click save button](/robotSupervisorSchemeTutorial/images/clickSaveButtonScreenshot.png)


### Adding the controllers

Now we will create the controller script needed that contains the environment and the robot controls.
Then we are going to assign the *robot controller* script to the *robot* node created before.

Creating the *robotSupervisorController* script:
1. On the *menu bar*, click *"Wizards -> New Robot Controller..."*\
![New robot controller](/robotSupervisorSchemeTutorial/images/newControllerMenuScreenshot.png)
2. On *Language selection*, select *Python*
3. Give it the name "*robotSupervisorController*"
4. Press *Finish* 

*If you are using an external IDE:    
1. Un-tick the "open ... in Text Editor" boxes and press *Finish*
2. Navigate to the project directory, inside the *controllers/robotSupervisorController/* directory
3. Open the controller script with your IDE

The new Python controller script should be created and opened in Webots text editor looking like this:\
![New robot controller](/robotSupervisorSchemeTutorial/images/newControllerCreated.png)

Assigning the *robotSupervisorController* to the *robot* node *controller* field:
1. Expand the *robot* node created earlier and scroll down to find the *controller* field
2. Click on the *controller* field and press the "*Select...*" button below\
![New robot controller](/robotSupervisorSchemeTutorial/images/assignSupervisorController1Screenshot.png)
3. Find the "*robotSupervisorController*" controller from the list and click it\
![New robot controller](/robotSupervisorSchemeTutorial/images/assignSupervisorController2Screenshot.png)
4. Click *OK*
5. Click *Save*

### Code overview

Before delving into writing code, we take a look at the general workflow of the framework. We will create a class 
that inherits a *deepbots framework* class and write implementations for several key methods, specific for the 
*CartPole* problem.

We will be implementing the basic methods *get_observations*, *get_reward*, *is_done* and *reset*, used for RL based 
on the [OpenAI Gym](https://gym.openai.com/) framework logic, that will be contained in the *robotSupervisorController*. 
These methods will compose the *environment* for the RL algorithm. The *robotSupervisorController* will also contain the 
RL *agent*, that will receive *observations* and output *actions*.

The *robot controller* will gather data from the *robot's* sensors and pack it to compose the *observation* for the 
agent using the *get_observations* method that we will implement. Then, using the *observation* the *agent* will 
perform a forward pass and return an *action*. Then the *robot controller* will use the *action* with the 
*apply_action* method, which will perform the *action* on the *robot*. 
This closes the loop that repeats until a termination condition is met, defined in the *is_done* method. 


### Writing the script

Now we are ready to start writing the *robotSupervisorController* script.
It is recommended to delete the contents of the script that were automatically created. 

### RobotSupervisor controller script

In this script we will import the *RobotSupervisor* class from the *deepbots framework* and inherit it into our own 
*CartPoleRobot* class. Then, we are going to implement the various basic framework methods:
1. *get_observations* which will create the *observation* for our agent in each step
2. *get_reward* which will return the reward for agent for each step
3. *is_done* which will look for the episode done condition
4. *solved* which will look for a condition that shows that the agent is fully trained and able to solve the problem 
   adequately (note that this method is not required by the framework, we just add it for convenience)
5. *get_default_observation* which is used by the *reset* method that the framework implements
6. *apply_action* which will take the action provided by the agent and apply it to the robot by setting its 
   motors' speeds
8. dummy implementations for *get_info* and *render* required by the *gym.Env* class that is inherited

Before we start coding, we should add two scripts, one that contains the RL PPO agent, 
and the other containing utility functions that we are going to need.

Save both files inside the project directory, under Controllers/robotSupervisorController/
1. Right-click on [this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/robotSupervisorSchemeTutorial/full_project/controllers/robotSupervisorController/PPO_agent.py) and click *Save link as...* to download the PPO agent
2. Right-click on [this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/robotSupervisorSchemeTutorial/full_project/controllers/robotSupervisorController/utilities.py) and click *Save link as...* to download the utilities script

Starting with the imports, first we are going to need the *RobotSupervisor* class and then
a couple of utility functions, the PPO agent implementation, the gym spaces to define the action and observation spaces
and finally numpy, which is installed as a dependency of the libraries we already installed.
```python
from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from utilities import normalizeToRange, plotData
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np
```
Then we define our class, inheriting the imported one.
```python
class CartpoleRobot(RobotSupervisor):
    def __init__(self):
        super().__init__()
```
Then we set up the observation and action spaces.

The observation space makes up the agent's (or the neural network's) input size and 
values and is defined by the table below:

Num | Observation | Min | Max
----|-------------|-----|----
0 | Cart Position z axis | -0.4 | 0.4
1 | Cart Velocity | -Inf | Inf
2 | Pole Angle | -1.3 rad | 1.3 rad
3 | Pole Velocity at Tip | -Inf | Inf

The action space defines the outputs of the neural network, which are 2. One for the forward movement
and one for the backward movement of the robot. 

```python
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        self.action_space = Discrete(2)
```
We then get a reference to the robot node, initialize the pole sensor, get a reference for the pole endpoint and 
initialize the wheel motors.
```python
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.positionSensor = self.getDevice("polePosSensor")
        self.positionSensor.enable(self.timestep)

        self.poleEndpoint = self.getFromDef("POLE_ENDPOINT")
        self.wheels = []
        for wheelName in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(wheelName)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)
```
Finally, we initialize several variables used during training. Note that the `self.stepsPerEpisode` is set to `200` 
based on the problem's definition. This concludes the `__init__()` method.
```python
        self.stepsPerEpisode = 200  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
```        

After the initialization we start implementing the various methods needed. We start with the `get_observations()`
method, which creates the agent's input from various information observed from the Webots world and returns it. We use
the `normalizeToRange()` utility method to normalize the values into the `[-1.0, 1.0]` range.

We will start by getting the CartPole robot node position and velocity on the z axis. The z axis is the direction of 
its forward/backward movement. We then read the position sensor value that returns the angle off vertical of the pole.
Finally, we get the pole tip velocity from the poleEndpoint node we defined earlier.

(mind the indentation, the following methods belong to the *CartpoleRobot* class)
```python
    def get_observations(self):
        # Position on z axis
        cartPosition = normalizeToRange(self.robot.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on z axis
        cartVelocity = normalizeToRange(self.robot.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Pole angle off vertical
        poleAngle = normalizeToRange(self.positionSensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # Angular velocity x of endpoint
        endpointVelocity = normalizeToRange(self.poleEndpoint.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cartPosition, cartVelocity, poleAngle, endpointVelocity]
```
Now for something simpler, we will define the `get_reward()` method, which simply returns
`1` for each step. Usually reward functions are more elaborate, but for this problem it is simply
defined as the agent getting a +1 reward for each step it manages to keep the pole from falling.

```python
    def get_reward(self, action=None):
        return 1
```
Moving on, we define the *is_done()* method, which contains the episode termination conditions:
- Episode terminates if episode score is over 195
- Episode terminates if the pole has fallen beyond an angle which can be realistically recovered (+-15 degrees)
- Episode terminates if the robot hit the walls by moving into them, which is calculated based on its position on z axis
```python
    def is_done(self):
        if self.episodeScore > 195.0:
            return True

        poleAngle = round(self.positionSensor.getValue(), 2)
        if abs(poleAngle) > 0.261799388:  # 15 degrees off vertical
            return True

        cartPosition = round(self.robot.getPosition()[2], 2)  # Position on z axis
        if abs(cartPosition) > 0.39:
            return True
        
        return False
```
We separate the *solved* condition into another method, the `solved()` method, because it requires different handling.
The *solved* condition depends on the agent completing consecutive episodes successfully, consistently. We measure this,
by taking the average episode score of the last 100 episodes and checking if it's over 195.
```python
    def solved(self):
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False
```
For this tutorial we use the default implementation of reset, which simply requires us to return a starting observation.
We do this in the `get_default_observation()` method which simply returns a zero vector.
```python
    def get_default_observation(self):
        return [0.0 for _ in range(self.observation_space.shape[0])]
```
We will now define the `apply_action()` method which gets the action that the agent outputs and turns it into physical
motion of the robot. For this tutorial we use a discrete action space, and thus the agent outputs an integer that is
either `0` or `1` denoting forward or backward motion using the robot's motors.
```python
    def apply_action(self, action):
        action = int(action[0])

        if action == 0:
            motorSpeed = 5.0
        else:
            motorSpeed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motorSpeed)
```

Lastly, we add dummy implementations of `get_info()` and `render()` methods, because in this example they are not 
actually used, but are required by the framework and gym in the background.

```python
    def render(self, mode='human'):
        print("render() is not used")

    def get_info(self):
        return None
```
That's the *CartpoleRobot* class complete. Now all that's left, is to add (outside the class scope, mind the 
indentation) the code that runs the RL loop.

### RL Training Loop

Finally, it all comes together inside the RL training loop. Now we initialize the RL agent and create the 
*CartPoleRobot* class object with which it gets trained to solve the problem, maximizing the reward received
by our reward function and achieve the solved condition defined.

First we initialize a supervisor object and then initialize the PPO agent, providing it with the observation and action
spaces. Note that we extract the number 4 as numberOfInputs and number 2 as numberOfActorOutputs from the gym spaces,
because the algorithm implementation expects integers for these arguments to initialize the neural network's input and
output neurons.

```python
env = CartpoleRobot()
agent = PPOAgent(numberOfInputs=env.observation_space.shape[0], numberOfActorOutputs=env.action_space.n)
```

Then we set the `solved` flag to `false`. This flag is used to terminate the training loop and signifies whether 
the solved condition is met.
```python
solved = False
```
Before setting up the RL loop, we define an episode counter and a limit for the number of episodes to run.
```python
episodeCount = 0
episodeLimit = 2000
```
Now we define the outer loop which runs the number of episodes we just defined and resets the world to get the 
starting observation. We also reset the episode score to zero.

(please be mindful of the indentation on the following code, because we are about to define several levels of nested
loops and ifs)
```python
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episodeCount < episodeLimit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episodeScore = 0
```

Inside the outer loop defined above we define the inner loop which runs for the course of an episode. This loop
runs for a maximum number of steps defined by the problem. Here, the RL agent - environment loop takes place.

We start by calling the `agent.work()` method, by providing it with the current observation, which for the first step
is the zero vector returned by the `reset()` method. The `reset()` method actually uses the `get_default_observation()`
method we defined earlier. The `work()` method implements the forward pass of the agent's 
actor neural network, providing us with the next action. As the comment suggests the PPO algorithm implements 
exploration by sampling the probability distribution the agent outputs from its actor's softmax output layer.

```python
    for step in range(env.stepsPerEpisode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selectedAction, actionProb = agent.work(observation, type_="selectAction")
``` 

The next part contains the call to the `step()` method. This method calls most of the methods we implemented earlier 
(`get_observation()`, `get_reward()`, `is_done()` and `get_info()`), steps the Webots controller and applies the action 
that the agent selected on the robot with the `apply_action()` method we defined. 
Step returns the new observation, the reward for the previous 
action and whether the episode is terminated (info is not implemented in this example).

Then, we create the `Transition`, which is a named tuple that contains, as the name suggests, the transition between
the previous `observation` (/state) to the `newObservation` (/newState). This is needed by the agent for its training 
procedure, so we call the agent's `storeTransition()` method to save it to the buffer. Most RL algorithms require a 
similar procedure and have similar methods to do it.

```python
        # Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
        # the done condition
        newObservation, reward, done, info = env.step([selectedAction])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
        agent.storeTransition(trans)
```

Finally, we check whether the episode is terminated and if it is, we save the episode score, run a training step
for the agent giving the number of steps taken in the episode as batch size, check whether the problem is solved
via the `solved()` method and break.

If not, we add the step reward to the `episodeScore` accumulator, save the `newObservation` as `observation` and loop 
onto the next episode step.

```python
        if done:
            # Save the episode's score
            env.episodeScoreList.append(env.episodeScore)
            agent.trainStep(batchSize=step)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episodeScore += reward  # Accumulate episode reward
        observation = newObservation  # observation for next step is current step's newObservation
```

This is the inner loop complete and now we add a print statement and increment the episode counter to finalize the outer
loop.

(note that the following code snippet is part of the outer loop)
```python
    print("Episode #", episodeCount, "score:", env.episodeScore)
    episodeCount += 1  # Increment episode counter
```

With the outer loop complete, this completes the training procedure. Now all that's left is the testing loop which is a
barebones, simpler version of the training loop. First we print a message on whether the task is solved or not (i.e. 
reached the episode limit without satisfying the solved condition) and call the `reset()` method. Then, we create a 
`while True` loop that runs the agent's forward method, but this time selecting the action with the max probability
out of the actor's softmax output, eliminating exploration/randomness. Finally, the `step()` method is called, but 
this time we keep only the observation it returns to keep the environment - agent loop running.

```python
if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")
observation = env.reset()
while True:
    selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
    observation, _, _, _ = env.step([selectedAction])
```

### Conclusion

Now with the coding done you can click on the *Run the simulation* button and watch the training run!
 
![Run the simulation](/robotSupervisorSchemeTutorial/images/clickPlay.png)\
Webots allows to speed up the simulation, even run it without graphics, so the training shouldn't take long, at 
least to see the agent becoming visibly better at moving under the pole to balance it. It takes a while for it to 
achieve the *solved* condition, but when it does, it becomes quite good at balancing the pole! You can even apply forces 
in real time by pressing Alt - left-click and drag on the robot or the pole.

That's it for this tutorial! :)

**_We welcome you to leave comments and feedback for the tutorial on the relevant 
[discussions page](https://github.com/aidudezzz/deepbots-tutorials/discussions/12) or to open an issue for any 
problem you find in it!_**

![Solved cartpole demonstration](/robotSupervisorSchemeTutorial/images/cartPoleWorld.gif)
