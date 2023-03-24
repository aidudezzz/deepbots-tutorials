# CartPole Beginner Robot-Supervisor Scheme Tutorial

![Solved cartpole demonstration](/robotSupervisorSchemeTutorial/images/cartPoleWorld.gif)

This tutorial explains how to use the [*deepbots framework*](https://github.com/aidudezzz/deepbots) by setting 
up a simple problem. We will use the 
[robot-supervisor scheme](https://github.com/aidudezzz/deepbots#combined-robot-supervisor-scheme) 
which combines the gym environment and the robot controller in one script, forgoing the emitter and receiver communication.
 
This tutorial can get you started for use cases that use a single robot, which should cover most needs.
If you desire to set up a more complicated example that might use multiple robots or similar, refer to the 
[emitter-receiver tutorial](https://github.com/aidudezzz/deepbots-tutorials/tree/master/emitterReceiverSchemeTutorial) 
to get started.

Keep in mind that the tutorial is very detailed and many parts can be completed really fast by an 
experienced user. The tutorial assumes no familiarity with the [Webots](https://cyberbotics.com/) simulator.

We will recreate the [CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) problem step-by-step in 
[Webots](https://cyberbotics.com/), and solve it with the 
[Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) 
Reinforcement Learning (RL) algorithm, using [PyTorch](https://pytorch.org/) as our neural network backend library.

We will focus on creating the project, the world and the controller scripts and how to use the *deepbots framework*.
For the purposes of the tutorial, a basic implementation of the PPO algorithm, together with a custom CartPole robot 
node contained within the world are supplied. For guides on how to construct a custom robot, please visit the official Webots 
[tutorial](https://cyberbotics.com/doc/guide/tutorial-6-4-wheels-robot). 

You can check the complete example [here](/robotSupervisorSchemeTutorial/full_project) with all the scripts and nodes 
used in this tutorial.
The CartPole example is available (with some added code for plots/monitoring and keyboard controls) on the 
[deepworlds](https://github.com/aidudezzz/deepworlds/) repository.

## Prerequisites

Before starting, several prerequisites should be met. Follow the [installation section on the deepbots framework main 
repository](https://github.com/aidudezzz/deepbots#installation).

For this tutorial you will also need to [install PyTorch](https://pytorch.org/get-started/locally/) 
(no CUDA/GPU support needed for this simple example as the very small neural networks used are sufficient to solve the task).


## CartPole
### Creating the project

Now we are ready to start working on the *CartPole* problem. First of all, we should create a new project.

1. Open Webots and on the menu bar, click *"File -> New -> New Project Directory..."*\
    ![New project menu option](/emitterReceiverSchemeTutorial/images/1_new_proj_menu.png)
2. Select a directory of your choice
3. On world settings **all** boxes should be ticked\
    ![World settings](/emitterReceiverSchemeTutorial/images/2_world_settings.png)
4. Give your world a name, e.g. "cartpole_world.wbt"
5. Press Finish

You should end up with:\
![Project created](/emitterReceiverSchemeTutorial/images/3_project_created.png)


### Adding a *supervisor robot* node in the world

First of all we will download the *CartPole robot node* definition that is supplied for the purposes of the tutorial, 
we will later add it into the world.
 
1. Right-click on
[this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/robotSupervisorSchemeTutorial/full_project/controllers/robotSupervisorController/cartpole_robot_definition.txt) 
and click *Save link as...* to download the CartPole robot definition 
2. Save the .txt file in a directory of your choice
3. Navigate to the directory and open the downloaded file with a text editor
4. Select everything and copy it

Now we need to add the *CartPole robot* into the world:

1. Navigate to your project's directory, open `/worlds` and edit the `.wbt` world file with a text editor
2. Navigate to the end of the file and paste the contents of the `cartpole_robot_definition.txt` file we downloaded earlier
3. Now all you need to do is reload the world and the robot will appear in it!
![Reload button](/robotSupervisorSchemeTutorial/images/4_reload_world.png)

Ignore the warning that appears in the console as we will later add a *robot controller* script to control the robot.

Now that the project and the starting world are created and the robot added to the world, we need to also give *supervisor*
privileges to the robot, so it can get various values from the simulation and also control it.
This will be done through the *robot controller* script which we will add later. Through this script we will be able to 
handle several aspects of the simulation needed for RL, but also control the robot with the actions produced by the RL agent.

Now let's go ahead and give the *supervisor* privileges to the robot:

_(Make sure the simulation is stopped and reset to its original state, by pressing the pause button and then the reset button)_

1. Double-click on the new *Robot* node to expand it
2. Scroll down to find the *supervisor* field and set it to TRUE\
![Set supervisor to TRUE](/robotSupervisorSchemeTutorial/images/5_set_supervisor_true.png)
3. Click *Save*\
![Click save button](/robotSupervisorSchemeTutorial/images/6_click_save_button.png)\
*If the save button is grayed out, move the camera a bit in the 3D view and it should be enabled*


### Adding the controllers

Now we will create the controller script needed that contains the environment and the robot controls.
Then we are going to assign the *robot controller* script to the *robot* node created before.

Creating the *robot supervisor controller* script:
1. On the *menu bar*, click *"File -> New -> New Robot Controller..."*\
![New robot controller](/robotSupervisorSchemeTutorial/images/7_new_controller_menu.png)
2. On *Language selection*, select *Python*
3. Give it the name "*robot_supervisor_controller*"
4. Press *Finish* 

*If you are using an external IDE:    
1. Un-tick the "open ... in Text Editor" boxes and press *Finish*
2. Navigate to the project directory, inside the *controllers/controllerName/* directory
3. Open the controller script with your IDE

The new Python controller script should be created and opened in Webots text editor looking like this:\
![New robot controller](/robotSupervisorSchemeTutorial/images/8_new_controllers_created.png)

Assigning the *robotSupervisorController* to the *robot* node *controller* field:
1. Expand the *robot* node and scroll down to find the *controller* field
2. Click on the *controller* field and press the "*Select...*" button below\
![New robot controller](/robotSupervisorSchemeTutorial/images/9_assign_supervisor_controller_1.png)
3. Find the "*robot supervisor controller*" controller from the list and click it\
![New robot controller](/robotSupervisorSchemeTutorial/images/10_assign_supervisor_controller_2.png)
4. Click *OK*
5. Click *Save*

### Code overview

Before delving into writing code, we take a look at the general workflow of the framework. We will create a class 
that inherits a *deepbots framework* class and write implementations for several key methods, specific for the 
*CartPole* problem.

We will be implementing the basic methods *get_observations*, *get_default_observation*, *get_reward*, *is_done* and *solved*, 
used for RL, based on the [Gym](https://www.gymlibrary.dev/) framework logic, that will be contained in the 
*robot supervisor controller*. 
These methods will compose the *environment* for the RL algorithm. Within the *robot supervisor controller*, below the 
environment class we will also add the RL *agent* that will receive *observations* and output *actions* and the 
RL training loop.

The *robot supervisor controller* will also gather data from the *robot's* sensors and pack it to compose the *observation* 
for the agent using the *get_observations* method that we will implement. Then, using the *observation* the *agent* will 
perform a forward pass and return an *action*. Then the *robot supervisor controller* will use the *action* with the 
*apply_action* method, which will perform the *action* on the *robot*. 
This closes the loop that repeats until a termination condition is met, defined in the *is_done* method. 


### Writing the script

Now we are ready to start writing the *robot supervisor controller* script.
It is recommended to delete the contents of the script that were automatically generated. 

### Robot supervisor controller script

In this script we will import the *RobotSupervisorEnv* class from the *deepbots framework* and inherit it into our own 
*CartpoleRobot* class. Then, we are going to implement the various basic framework methods:
1. *get_observations* which will create the *observation* for our agent in each step
2. *get_default_observation* which is used by the *reset* method that the framework implements
3. *get_reward* which will return the reward for agent for each step
4. *is_done* which will look for the episode done condition
5. *solved* which will look for a condition that shows that the agent is fully trained and able to solve the problem 
   adequately (note that this method is not required by the framework, we just add it for convenience)
6. *apply_action* which will take the action provided by the agent and apply it to the robot by setting its 
   motors' speeds
7. dummy implementations for *get_info* and *render* required to have a complete Gym environment

Before we start coding, we should add two scripts, one that contains the RL PPO agent, 
and the other containing utility functions that we are going to need.

Save both files inside the project directory, under controllers/robot_supervisor_controller/
1. Right-click on [this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/robotSupervisorSchemeTutorial/full_project/controllers/robot_supervisor_controller/PPO_agent.py) and click *Save link as...* to download the PPO agent
2. Right-click on [this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/robotSupervisorSchemeTutorial/full_project/controllers/robot_supervisor_controller/utilities.py) and click *Save link as...* to download the utilities script

Starting with the imports, first we are going to need the *RobotSupervisorEnv* class and then
a couple of utility functions, the PPO agent implementation, the gym spaces to define the action and observation spaces
and finally numpy, which is installed as a dependency of the libraries we already installed.
```python
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np
```
Then we define our class, inheriting the imported one.
```python
class CartpoleRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
```
Then we set up the observation and action spaces.

The observation space makes up the agent's (or the neural network's) input size and 
values and is defined by the table below:

Num | Observation | Min | Max
----|-------------|-----|----
0 | Cart Position x axis | -0.4 | 0.4
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
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)
        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")

        self.wheels = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)
```
Finally, we initialize several variables used during training. Note that the `self.steps_per_episode` is set to `200` 
based on the problem's definition. This concludes the `__init__()` method.
```python
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
```

After the initialization we start implementing the various methods needed. We start with the `get_observations()`
method, which creates the agent's input from various information observed from the Webots world and returns it. We use
the `normalize_to_range()` utility method to normalize the values into the `[-1.0, 1.0]` range.

We will start by getting the cartpole robot nodes position and velocity on the x-axis. The x-axis is the direction of 
its forward/backward movement. We then read the position sensor value that returns the angle off vertical of the pole.
Finally, we get the pole tip velocity from the poleEndpoint node we defined earlier. The values are packed in a list
and returned.

(mind the indentation, the following methods belong to the *CartpoleRobot* class)
```python
    def get_observations(self):
        # Position on x-axis
        cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x-axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Pole angle off vertical
        pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cart_position, cart_velocity, pole_angle, endpoint_velocity]
```

Let's also define the *get_defaults_observation()* that is used internally by deepbots when a new training episode starts:
```python
    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]
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
- Episode terminates if the robot hit the walls by moving into them, which is calculated based on its position on x-axis
```python
    def is_done(self):
        if self.episode_score > 195.0:
            return True

        pole_angle = round(self.position_sensor.getValue(), 2)
        if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
            return True

        cart_position = round(self.robot.getPosition()[0], 2)  # Position on x-axis
        if abs(cart_position) > 0.39:
            return True

        return False
```
We separate the *solved* condition into another method, the `solved()` method, because it requires different handling.
The *solved* condition depends on the agent completing consecutive episodes successfully, consistently. We measure this,
by taking the average episode score of the last 100 episodes and checking if it's over 195.
```python
    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False
```

To complete the Gym environment, we add dummy implementations of `get_info()` and `render()` methods, 
because in this example they are not actually used, but are required by the framework and gym in the background.

```python
    def get_info(self):
        return None

    def render(self, mode='human'):
        pass
```

Lastly, this controller actually controls the robot itself, so we need to define the `apply_action()` method which gets the 
action that the agent outputs and turns it into physical motion of the robot. For this tutorial we use a discrete action space, 
and thus the agent outputs an integer that is either `0` or `1` denoting forward or backward motion using the robot's motors.
```python
    def apply_action(self, action):
        action = int(action[0])

        if action == 0:
            motor_speed = 5.0
        else:
            motor_speed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)
```
That's the *CartpoleRobot* class complete that now contains all required methods to run an RL training loop and also get 
information from the simulation and the robot sensors, but also control the robot! 
Now all that's left, is to add (outside the class scope, mind the indentation) the code that runs the RL loop.

### RL Training Loop

Finally, it all comes together inside the RL training loop. Now we initialize the RL agent and create the 
*CartPoleSupervisor* class object, i.e. the RL environment, with which the agent gets trained to solve the problem, 
maximizing the reward received by our reward function and achieve the solved condition defined.

**Note that popular frameworks like [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) contain the 
RL training loop within their *learn* method or similar. Frameworks like *sb3* are fully compatible with *deepbots*, as 
*deepbots* defines Gym environments and interfaces them with Webots saving you a lot of trouble, which then can be 
supplied to frameworks like *sb3* or any other RL framework of your choice.**

For this tutorial we follow a more hands-on approach to get a better grasp of how RL works. Also feel free to check out 
the simple PPO agent implementation we provide. 

First we create a supervisor object and then initialize the PPO agent, providing it with the observation and action
spaces. Note that we extract the number 4 as _number_of_inputs_ and number 2 as _number_of_actor_outputs_ from the gym spaces,
because the algorithm implementation expects integers for these arguments to initialize the neural network's input and
output neurons.

```python
env = CartpoleRobot()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)
```

Then we set the `solved` flag to `false`. This flag is used to terminate the training loop when the solved condition is met.
```python
solved = False
```
Before setting up the RL loop, we define an episode counter and a limit for the number of episodes to run.
```python
episode_count = 0
episode_limit = 2000
```
Now we define the outer loop which runs the number of episodes we just defined and resets the world to get the 
starting observation. We also reset the episode score to zero.

_(please be mindful of the indentation on the following code, because we are about to define several levels of nested
loops and ifs)_
```python
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0
```

Inside the outer loop defined above we define the inner loop which runs for the course of an episode. This loop
runs for a maximum number of steps defined by the problem. Here, the RL agent - environment loop takes place.

We start by calling the `agent.work()` method, by providing it with the current observation, which for the first step
is the zero vector returned by the `reset()` method, through the `get_default_observation()` method we defined. 
The `reset()` method actually uses the `get_default_observation()` method we defined earlier. 
The `work()` method implements the forward pass of the agent's actor neural network, providing us with the next action. 
As the comment suggests the PPO algorithm implements exploration by sampling the probability distribution the agent 
outputs from its actor's softmax output layer.

```python
    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
``` 

The next part contains the call to the `step()` method which is defined internally in deepbots. This method calls most of 
the methods we implemented earlier (`get_observation()`, `get_reward()`, `is_done()` and `get_info()`), steps the Webots 
controller and applies the action that the agent selected on the robot with the `apply_action()` method we defined. 
Step returns the new observation, the reward for the previous action and whether the episode is 
terminated (info is not implemented in this example).

Then, we create the `Transition`, which is a named tuple that contains, as the name suggests, the transition between
the previous `observation` (or `state`) to the `new_observation` (or `new_state`). This is needed by the agent for its training 
procedure, so we call the agent's `store_transition()` method to save it to its buffer. Most RL algorithms require a 
similar procedure and have similar methods to do it.

```python
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)
```

Finally, we check whether the episode is terminated and if it is, we save the episode score, run a training step
for the agent giving the number of steps taken in the episode as batch size, and check whether the problem is solved
via the `solved()` method and break.

If not, we add the step reward to the `episode_score` accumulator, save the `new_observation` as `observation` and loop 
onto the next episode step.

**Note that in more realistic training procedures, the training step might not run for each episode. Depending on the problem
you might need to run the training procedure multiple times per episode or once per multiple episodes. This is set as `n_steps` 
or similar in frameworks like *sb3*. Moreover, changing the batch size along with `n_steps` might influence greatly the 
training results and whether the agent actually converges to a solution, and consequently are crucial parameters.**

```python
        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation
```

This is the inner loop complete, and now we add a print statement and increment the episode counter to finalize the outer
loop.

(note that the following code snippet is part of the outer loop)
```python
    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter
```

With the outer loop complete, this completes the training procedure. Now all that's left is the testing loop which is a
barebones, simpler version of the training loop. First we print a message on whether the task is solved or not (i.e. 
reached the episode limit without satisfying the solved condition) and call the `reset()` method. Then, we create a 
`while True` loop that runs the agent's forward method, but this time selecting the action with the max probability
out of the actor's softmax output, eliminating exploration/randomness. Finally, the `step()` method is called, but 
this time we keep only the observation it returns to keep the environment - agent loop running. If the *done* flag is true, we 
reset the environment to start over.

```python
if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()
```

### Conclusion

Now with the coding done you can click on the *Run the simulation* button and watch the training run!
 
![Run the simulation](/robotSupervisorSchemeTutorial/images/11_click_play.png)\
Webots allows to speed up the simulation, even run it without graphics, so the training shouldn't take long, at 
least to see the agent becoming visibly better at moving under the pole to balance it. It takes a while for it to 
achieve the *solved* condition, but when it does, it becomes quite good at balancing the pole! You can even apply forces 
in real time by pressing Alt - left-click and drag on the robot or the pole.

That's it for this tutorial! :)

**_We welcome you to leave comments and feedback for the tutorial on the relevant 
[discussions page](https://github.com/aidudezzz/deepbots-tutorials/discussions/12?sort=new) or to open an issue for any 
problem you find in it!_**

![Solved cartpole demonstration](/robotSupervisorSchemeTutorial/images/cartPoleWorld.gif)
