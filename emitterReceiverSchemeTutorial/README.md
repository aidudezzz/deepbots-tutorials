# CartPole Emitter-Receiver Scheme Tutorial

![Solved cartpole demonstration](/emitterReceiverSchemeTutorial/images/cartPoleWorld.gif)

This tutorial explains how to use the [*deepbots framework*](https://github.com/aidudezzz/deepbots) by setting 
up a simple problem. We will use the 
[emitter-receiver scheme](https://github.com/aidudezzz/deepbots#emitter---receiver-scheme) which is appropriate for
more complicated use-cases, such as setting up multiple robots. For simple use cases of a single robot, please use the 
[robot-supervisor scheme tutorial](https://github.com/aidudezzz/deepbots-tutorials/tree/master/robotSupervisorSchemeTutorial).


~~Moreover, we will implement a custom reset procedure, which is not actually needed as seen in the robot-supervisor 
scheme tutorial, but might be useful for some use-cases.~~ 

_Since the original tutorial was written, Webots was updated and now provides 
[more options to reset the world](https://cyberbotics.com/doc/reference/supervisor?tab-language=python#resetreload-matrix), 
and thus the old custom reset procedure is not needed anymore.
The reset method provided by the deepbots framework is enough for most use-cases. Of course, deepbots design philosophy allows
you to override the reset method and provide your own implementation that fits your problem best. We do it ourselves in some of 
our examples in [deepworlds](https://github.com/aidudezzz/deepworlds)! You can check out a pretty involved custom reset method 
that entirely overrides the built-in reset method 
[here.](https://github.com/aidudezzz/deepworlds/blob/f3f286d5c3df5ca858745a40111a2834001e15e7/examples/find_and_avoid_v2/controllers/robot_supervisor_manager/find_and_avoid_v2_robot_supervisor.py#L713-L802)_

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

You can check the complete example [here](/emitterReceiverSchemeTutorial/full_project) with all the scripts and nodes used
in this tutorial.
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


### Adding a *robot node* and a *supervisor robot* node in the world

First of all we will download the *CartPole robot node* definition that is supplied for the purposes of the tutorial, 
we will later add it into the world.
 
1. Right-click on
[this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/emitterReceiverSchemeTutorial/full_project/controllers/supervisor_controller/cartpole_robot_definition.txt)
and click *Save link as...* to download the CartPole robot definition 
2. Save the .txt file in a directory of your choice
3. Navigate to the directory and open the downloaded file with a text editor
4. Select everything and copy it

Now we need to add the *CartPole robot* into the world:

1. Navigate to your project's directory, open `/worlds` and edit the `.wbt` world file with a text editor
2. Navigate to the end of the file and paste the contents of the `cartpole_robot_definition.txt` file we downloaded earlier
3. Now all you need to do is reload the world and the robot will appear in it!
![Reload button](/emitterReceiverSchemeTutorial/images/4_reload_world.png)

The *CartPole robot node* comes ready along with the communication devices needed (an emitter and a receiver) to send and receive data.
Ignore the warning that appears in the console as we will later add a *robot controller* script to control the robot.

Now that the *CartPole robot node* is added, we are going to create another special kind of *robot*, called
a *supervisor*. Later, we will add the *supervisor controller* script, through which we will be able to handle several 
aspects of the simulation.

_(Make sure the simulation is stopped and reset to its original state, by pressing the pause button and then the reset button)_

1. Click on the *Add a new object or import an object* button\
![Add new object button](/emitterReceiverSchemeTutorial/images/5_add_new_object.png)
(If button is grayed out, left-click on the last node on the tree (the robot node) so the *add* button becomes active again.)
2. Expand the *Base nodes* and left-click on *Robot*\
![Add Robot node](/emitterReceiverSchemeTutorial/images/6_add_robot_node.png)
3. Click *Add*. Now on the left side of the screen, under the previously added *Robot* node, you can see a new *Robot* node
4. Click on the new *Robot* node and set its DEF  field below to "SUPERVISOR". From now on we are going to refer to this robot
as the *supervisor*
5. Double-click on the new *Robot* node to expand it
6. Scroll down to find the *supervisor* field and set it to TRUE\
![Set supervisor to TRUE](/emitterReceiverSchemeTutorial/images/7_set_supervisor_true.png)
7. On the *children* field, right-click and select *Add new*
8. Expand the *Base nodes* and find *Emitter*
9. Select it and on the lower right press *Add*
10. Repeat from step 7, but this time add the *Receiver* node
11. Click *Save*\
![Click save button](/emitterReceiverSchemeTutorial/images/8_click_save_button.png)


### Adding the controllers

Now we will create the two basic controller scripts needed to control the *supervisor* and the *robot* nodes.
Then we are going to assign the *supervisor controller* script to the *supervisor* node and the *robot controller*
to the *robot* created before.
Note that the *CartPole robot* node is going to be loaded into the world through the *supervisor controller* script
later, but we still need to create its controller.

Creating the *supervisor controller* and *robot controller* scripts:
1. On the *menu bar*, click *"File -> New -> New Robot Controller..."*\
![New robot controller](/emitterReceiverSchemeTutorial/images/9_new_controller_menu.png)
2. On *Language selection*, select *Python*
3. Give it the name "*supervisor_controller*"
4. Press *Finish* 
5. Repeat from step 1, but on step 3 give the name "*robot controller*"

*If you are using an external IDE:    
1. Un-tick the "open ... in Text Editor" boxes and press *Finish*
2. Navigate to the project directory, inside the *Controllers/controllerName/* directory
3. Open the controller script with your IDE

Two new Python controller scripts should be created and opened in Webots text editor looking like this:\
![New robot controller](/emitterReceiverSchemeTutorial/images/10_new_controllers_created.png)

_(Make sure the simulation is stopped and reset to its original state, by pressing the pause button and then the reset button)_

Assigning the *supervisor_controller* to the *supervisor robot* node *controller* field:
1. Expand the *supervisor* node created earlier and scroll down to find the *controller* field
2. Click on the *controller* field and press the "*Select...*" button below\
![New robot controller](/emitterReceiverSchemeTutorial/images/11_assign_supervisor_controller_1.png)
3. Find the "*supervisor_controller*" controller from the list and select it\
![New robot controller](/emitterReceiverSchemeTutorial/images/12_assign_supervisor_controller_2.png)
4. Click *OK*
5. Click *Save*

Follow the same steps for the *robot* and the *robot controller* created earlier.

### Code overview

Before delving into writing code, we take a look at the general workflow of the framework. We will create two classes 
that inherit the *deepbots framework* classes and write implementations for several key methods, specific for the 
*CartPole* problem.

We will implement the basic methods *get_observations*, *get_default_observation*, *get_reward*, *is_done* and *solved*, 
used for RL, based on the [Gym](https://www.gymlibrary.dev/) framework logic, that will be contained in the 
*supervisor controller*. 
These methods will compose the *environment* for the RL algorithm. Within the *supervisor controller*, below the environment 
class we will also add the RL *agent* that will receive *observations* and output *actions* and the RL training loop.

For the *robot controller* script we will implement a couple of basic methods to enable it to send and receive data from 
the *supervisor*. We will initialize the motors and the pole position sensor. The motors will have their speeds set depending
on the action the robot receives from the *supervisor* at each simulation step. The position sensor data will be sent to the 
*supervisor* for it to compose the *agent's* observation.

The following diagram loosely defines the general workflow:\
![deepbots workflow](/emitterReceiverSchemeTutorial/images/13_workflow_diagram.png)

The *robot controller* will gather data from the *robot's* sensors and send it to the *supervisor controller*. The 
*supervisor controller* will use the data received and extra data to compose the *observation* for the agent. Then, 
using the *observation* the *agent* will perform a forward pass and return an *action*. Then the *supervisor controller* 
will send the *action* back to the *robot controller*, which will perform the *action* on the *robot*. This closes the 
loop, that repeats until a termination condition is met, defined in the *is_done* method. 


### Writing the scripts

Now we are ready to start writing the *robot controller* and *supervisor controller* scripts.
It is recommended to delete all the contents of the two scripts that were automatically generated. 

### Robot controller script

First, we will start with the more simple *robot controller* script. In this script we will import the *CSVRobot*
class from the *deepbots framework* and inherit it into our own *CartPoleRobot* class. Then, we are going to
implement the two basic framework methods *create_message* and *use_message_data*. The former gathers data from the 
*robot's* sensors and packs it into a string message to be sent to the *supervisor controller* script. The latter 
unpacks messages sent by the *supervisor* that contain the next action, and uses the data to move the *CartPoleRobot* 
forward and backward by setting the motor's speeds.

The only import we are going to need is the *CSVRobot* class.
```python
from deepbots.robots.controllers.csv_robot import CSVRobot
```
Then we define our class, inheriting the imported one.
```python
class CartpoleRobot(CSVRobot):
    def __init__(self):
        super().__init__()
```
Then we initialize the position sensor that reads the pole's angle, needed for the agent's observation.
```python
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)
```
Finally, we initialize the four motors completing our `__init__()` method.
```python
        self.wheels = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)
```
After the initialization method is done we move on to the `create_message()` method implementation, used to pack the 
value read by the sensor into a string, so it can be sent to the *supervisor controller*.

_(mind the indentation, the following two methods belong to the *CartpoleRobot* class)_

```python
    def create_message(self):
        # Read the sensor value, convert to string and save it in a list
        message = [str(self.position_sensor.getValue())]
        return message
```
Finally, we implement the `use_message_data()` method, which unpacks the message received by the 
*supervisor controller*, that contains the next action. Then we implement what the *action* actually means for the
*CartPoleRobot*, i.e. moving forward and backward using its motors.
```python
        def use_message_data(self, message):
        action = int(message[0])  # Convert the string message into an action integer

        if action == 0:
            motor_speed = 5.0
        elif action == 1:
            motor_speed = -5.0
        else:
            motor_speed = 0.0

        # Set the motors' velocities based on the action received
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)
```
That's the *CartpoleRobot* class complete. Now all that's left, is to add (outside the class scope, mind the 
indentation) the code that runs the controller.

```python
# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
```

Now the *robot controller* script is complete! We move on to the *supervisor controller* script.

### Supervisor controller script
Before we start coding, we should add two scripts, one that contains the RL PPO agent, 
and the other containing utility functions that we are going to need.

Save both files inside the project directory, under Controllers/supervisorController/
1. Right-click on [this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/emitterReceiverSchemeTutorial/full_project/controllers/supervisorController/PPO_agent.py) and click *Save link as...* to download the PPO agent
2. Right-click on [this link](https://raw.githubusercontent.com/aidudezzz/deepbots-tutorials/master/emitterReceiverSchemeTutorial/full_project/controllers/supervisorController/utilities.py) and click *Save link as...* to download the utilities script

Now for the imports, we are going to need the numpy library, the deepbots CSVSupervisorEnv class, the PPO agent and the
utilities.

```python
import numpy as np
from deepbots.supervisor.controllers.csv_supervisor_env import CSVSupervisorEnv
from PPO_agent import PPOAgent, Transition
from utilities import normalize_to_range
```

Then we define our class inheriting the imported one, also defining the observation and action spaces.
Here, the observation space is basically the number of the neural network's inputs, so its defined simply as an
integer. 

Num | Observation | Min | Max
----|-------------|-----|----
0 | Cart Position z axis | -0.4 | 0.4
1 | Cart Velocity | -Inf | Inf
2 | Pole Angle | -1.3 rad | 1.3 rad
3 | Pole Velocity at Tip | -Inf | Inf

The action space defines the outputs of the neural network, which are 2. One for the forward movement
and one for the backward movement of the robot. 

```python
class CartPoleSupervisor(CSVSupervisorEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = 4  # The agent has 4 inputs
        self.action_space = 2  # The agent can perform 2 actions
```
Then we initialize the `self.robot` variable which will hold a reference to the *CartPole robot* node.

We also get a reference for the *pole endpoint* node, which is a child node of the *CartPole robot node* and is going
to be useful for getting the pole tip velocity. 
```python
        self.robot = self.getFromDef("ROBOT")
        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.message_received = None  # Variable to save the messages received from the robot
```
Finally, we initialize several variables used during training. Note that the `self.steps_per_episode` is set to `200` 
based on the problem's definition. Feel free to change the `self.episode_limit` variable.

```python
        self.episode_count = 0  # Episode counter
        self.episode_limit = 10000  # Max number of episodes allowed
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
```        

Now it's time for us to implement the base environment methods that a regular Gym environment uses and
most RL algorithm implementations (agents) expect.
These base methods are *get_observations()*, *get_reward()*, *is_done()*, *get_info()* and *render()*. Additionally, we are 
going to implement the *get_default_observation()* method which is used internally by deepbots and the *solved()* method
that will help us determine when to stop training.
 
Let's start with the `get_observations()` method, which builds the agent's observation (i.e. the neural network's input) 
for each step. This method also normalizes the values in the [-1.0, 1.0] range as appropriate, using the 
`normalize_to_range()` utility method.

We will start by getting the *CartPole robot* node position and velocity on the x-axis. The x-axis is the direction of 
its forward/backward movement. We will also get the pole tip velocity from the *poleEndpoint* node we defined earlier.
```python
    def get_observations(self):
        # Position on x-axis, first (0) element of the getPosition vector
        cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x-axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)
```
Now all it's missing is the pole angle off vertical, which will be provided by the robot sensor. We don't have access to
sensor values of other nodes, so the robot needs to actually send the value.
To get it, we will need to call the `handle_receiver()` method which deepbots provides to get the message sent by the robot 
into the `self.messageReceived` variable. The message received, as defined into the robot's `create_message()` method, is a 
string which, here, gets converted back into a single float value. 

```python
        # Update self.message_received received from robot, which contains pole angle
        self.message_received = self.handle_receiver()
        if self.message_received is not None:
            pole_angle = normalize_to_range(float(self.message_received[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            # Method is called before self.message_received is initialized
            pole_angle = 0.0
```

Finally, we return a list containing all four values we created earlier.

```python
        return [cart_position, cart_velocity, pole_angle, endpoint_velocity]
```

Let's also define the *get_defaults_observation()* that is used internally by deepbots when a new training episode starts:
```python
    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space)]
```

Now for something simpler, we will define the `get_reward()` method, which simply returns
`1` for each step. Usually reward functions are more elaborate, but for this problem it is simply
defined as the agent getting a +1 reward for each step it manages to keep the pole from falling.

```python
    def get_reward(self, action=None):
        return 1
```

Moving on, we define the *is_done()* method, which contains the episode termination conditions:
- Episode terminates if the pole has fallen beyond an angle which can be realistically recovered (+-15 degrees)
- Episode terminates if episode score is over 195
- Episode terminates if the robot hit the walls by moving into them, which is calculated based on its position on x-axis

```python
    def is_done(self):
        if self.message_received is not None:
            pole_angle = round(float(self.message_received[0]), 2)
        else:
            # method is called before self.message_received is initialized
            pole_angle = 0.0
        if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
            return True

        if self.episode_score > 195.0:
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

Lastly, we add a dummy implementation of `get_info()` and `render()` methods, because in this example they are not actually 
used, but is required to define a proper Gym environment.

```python
    def get_info(self):
        return None

    def render(self, mode="human"):
        pass
```

This concludes the *CartPoleSupervisor* class, that now contains all required methods to run an RL training loop!

### RL Training Loop

Finally, it all comes together inside the RL training loop. Now we initialize the RL agent and create the 
*CartPoleSupervisor* class object, i.e. the RL environment, with which the agent gets trained to solve the problem, 
maximizing the reward received by our reward function and achieve the solved condition defined.

**Note that popular frameworks like [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) contain the 
RL training loop within their *learn* method or similar. Frameworks like *sb3* are fully compatible with *deepbots*, as 
*deepbots* defines Gym environments and interfaces them with Webots saving you a lot of trouble, which then can be 
supplied to frameworks like *sb3*.**

For this tutorial we follow a more hands-on approach to get a better grasp of how RL works. Also feel free to check out 
the simple PPO agent implementation we provide. 

First we create a supervisor object and then initialize the PPO agent, providing it with the observation and action
spaces.

```python
env = CartPoleSupervisor()
agent = PPOAgent(env.observation_space, env.action_space)
```

Then we set the `solved` flag to `False`. This flag is used to terminate the training loop.
```python
solved = False
```

Now we define the outer training loop which runs the number of episodes defined in the supervisor class
and resets the world to get the starting observation. We also reset the episode score to zero.

_(please be mindful of the indentation on the following code, because we are about to define several levels of nested
loops and ifs)_
```python
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and env.episode_count < env.episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0
```

Inside the outer loop defined above we define the inner loop which runs for the course of an episode. This loop
runs for a maximum number of steps defined by the problem. Here, the RL agent - environment loop takes place.

We start by calling the `agent.work()` method, by providing it with the current observation, which for the first step
is the zero vector returned by the `reset()` method, through the `get_default_observation()` method we defined. 
The `work()` method implements the forward pass of the agent's actor neural network, providing us with the next action. 
As the comment suggests the PPO algorithm implements exploration by sampling the probability distribution the 
agent outputs from its actor's softmax output layer.

```python
    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
``` 

The next part contains the call to the `step()` method which is defined internally in deepbots. This method calls most of 
the methods we implemented earlier (`get_observation()`, `get_reward()`, `is_done()` and `get_info()`), steps the Webots 
controller and sends the action that the agent selected to the robot for execution. Step returns the new observation, 
the reward for the previous action and whether the episode is terminated (info is not implemented in this example).

Then, we create the `Transition`, which is a named tuple that contains, as the name suggests, the transition between
the previous `observation` (or `state`) to the `new_observation` (or `new_state`). This is needed by the agent for its training 
procedure, so we call the agent's `store_transition()` method to save it to its buffer. Most RL algorithms require a 
similar procedure and have similar methods to do it.

```python
        # Step the env to get the current selected_action's reward, the new observation and whether we reached
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
    print("Episode #", env.episode_count, "score:", env.episode_score)
    env.episode_count += 1  # Increment episode counter
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
 
![Run the simulation](/emitterReceiverSchemeTutorial/images/14_click_play.png)\
Webots allows to speed up the simulation, even run it without graphics, so the training shouldn't take long, at 
least to see the agent becoming visibly better at moving under the pole to balance it. It takes a while for it to 
achieve the *solved* condition, but when it does, it becomes quite good at balancing the pole! You can even apply forces 
in real time by pressing Alt - left-click and drag on the robot or the pole.

That's it for this tutorial! :)

**_We welcome you to leave comments and feedback for the tutorial on the relevant 
[discussions page](https://github.com/aidudezzz/deepbots-tutorials/discussions/15?sort=new) or to open an issue for any 
problem you find in it!_**

![Solved cartpole demonstration](/emitterReceiverSchemeTutorial/images/cartPoleWorld.gif)
