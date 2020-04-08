# Deepbots Tutorial (WIP)

This tutorial explains how to use the *deepbots framework* by setting up a simple problem. 
We will recreate the [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem in [Webots](https://cyberbotics.com/), 
step-by-step and solve it with the [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) 
Reinforcement Learning (RL) algorithm, using [PyTorch](https://pytorch.org/) as our neural network backend library.

This tutorial will focus on the controller scripts and how to use the *deepbots framework*. The robot node definition is 
supplied for the tutorial. For guides on how to construct a robot, please visit the official 
Webots [tutorial](https://cyberbotics.com/doc/guide/tutorial-6-4-wheels-robot). 

The complete example can be found on the [deepworlds](https://github.com/aidudezzz/deepworlds/) repository. 


## Prerequisites

Before starting, several prerequisites should be met.

1. [Install Webots](https://cyberbotics.com/doc/guide/installing-webots)
    - [Windows](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-windows)
    - [Linux](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux)
    - [macOS](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-macos)
2. [Install Python version 3.X](https://www.python.org/downloads/) (please refer to 
[Using Python](https://cyberbotics.com/doc/guide/using-python#introduction) to select the proper Python version for 
your system) 
3. Follow the [Using Python](https://cyberbotics.com/doc/guide/using-python) guide
4. [Using PyCharm IDE](https://cyberbotics.com/doc/guide/using-your-ide#pycharm)
5. [Install PyTorch](https://pytorch.org/get-started/locally/)
6. Install deepbots  (Add link)


## CartPole
### Creating the project

Now we are ready to start working on the *CartPole* problem. First of all, we should create a new project.

1. Open Webots and on the menu bar, click *"Wizards -> New Project Directory..."*\
    ![New project menu option](/images/newProjectMenuScreenshot.png)
2. Select a directory of your choice
3. On world settings **all** boxes should be ticked\
    ![World settings](/images/worldSettingsScreenshot.png)
4. Give your world a name, e.g. "cartPoleWorld.wbt"
5. Press Finish

You should end up with:\
![Project created](/images/projectCreatedScreenshot.png)


### Adding a *supervisor robot node* in the world

Now that the project and the starting world are created, we are going to create a special kind of *robot*, 
a *supervisor*. Later, we will add the *supervisor controller script*, through which we will be able to handle several 
aspects of the simulation needed for RL (e.g. resetting).
 
1. Click on the *Add a new object or import an object* button\
![Add new object button](/images/addNewObjectButtonScreenshot.png)
2. Click on *Base nodes -> Robot*\
![Add Robot node](/images/addRobotNodeScreenshot.png)
3. Click *Add*. Now on the left side of the screen, under the *Rectangle Arena* node, you can see the *Robot* node
4. Click on the *Robot* node and set its DEF  field below to "supervisor" to make it easily distinguishable
4. Double click on the *Robot* node to expand it
5. Scroll down to find the *supervisor* field and set it to TRUE\
![Set supervisor to TRUE](/images/setSupervisorTrueScreenshot.png)
6. Click *Save*\
![Click save button](/images/clickSaveButtonScreenshot.png)


### Adding the controllers

Now we will create the two basic controller scripts needed to control the *supervisor* and the *robot* nodes.
Then we are going to assign the *supervisor controller script* to the *supervisor robot node* created before.
Note that the *CartPole robot node* is going to be loaded into the world through the *supervisor controller script* 
later, but we still need to create its controller.

Creating the *supervisor controller* and *robot controller* scripts:
1. On the *menu bar*, click *"Wizards -> New Robot Controller..."*\
![New robot controller](/images/newControllerMenuScreenshot.png)
2. On *Language selection*, select *Python*
3. Give it the name "*supervisorController*"*
4. Press *Finish* 
5. Repeat from step 1, but on step 3 give the name "*robotController*"

*If you are using an external IDE:    
1. Un-tick the "open ... in Text Editor" boxes and press *Finish*
2. Navigate to the project directory, inside the *Controllers/controllerName/* directories
3. Open the controller script with your IDE

Two new Python controller scripts should be created and opened in Webots text editor looking like this:\
![New robot controller](/images/newControllerMenuScreenshot.png)

Assigning the *supervisorController* to the *supervisor robot node* *controller* field:
1. Expand the *supervisor Robot node* created earlier and scroll down to find the *controller* field
2. Click on the *controller* field and press the "*Select...*" button below\
![New robot controller](/images/assignSupervisorController1Screenshot.png)
3. Find the "*supervisorController*" controller from the list and click it\
![New robot controller](/images/assignSupervisorController2Screenshot.png)
4. Click *OK*
5. Click *Save*\
![Click save button](/images/clickSaveButtonScreenshot.png)


### Downloading the CartPole robot node

The *CartPole robot node* definition is supplied for the purposes of the tutorial.
 
1. Right-click on [this link](/CartPoleRobot.wbo) and click *Save link as...* to download the CartPole robot 
definition 
2. Save the .wbo file inside the project directory, under Controllers/supervisorController/


### Code overview

Before delving into writing code, we take a look at the general workflow of the framework. We will create two classes 
that inherit the *deepbots framework* classes and write implementations for several key methods, specific for the 
*CartPole* problem.

We will be implementing the basic methods *get_observations*, *get_reward*, *is_done* and *reset*, used for RL based 
on the [OpenAI Gym](https://gym.openai.com/) framework logic, that will be contained in the *supervisor controller*. 
These methods will compose the *environment* for the RL algorithm. The *supervisor controller* will also contain the 
RL *agent*, that will receive *observations* and output *actions*.

We will also be implementing methods that will be used by the *handle_emitter* and *handle_receiver* methods on the 
*robot controller* to send and receive data between the *robot* and the *supervisor*.

The following diagram loosely defines the general workflow of the framework:\
![deepbots workflow](/images/workflowDiagram.png)

The *robot controller* will gather data from the *robot*'s sensors and send it to the *supervisor controller*. The 
*supervisor controller* will use the data received and extra data to compose the *observation* for the agent. Then, 
using the *observation* the *agent* will perform a forward pass and return an *action*. Then the *supervisor controller* 
will send the *action* to the *robot controller*, which will perform the *action* on the *robot*. This closes the loop, 
that repeats until a termination condition is met, defined in the *is_done* method. 


### Writing the scripts

First, we will write the *robot controller script*. In this script we will import the *RobotEmitterReceiverCSV*
class from the *deepbots framework* and inherit it into our own *CartPoleRobot* class. Then, we are going to
implement the two basic framework methods *create_message* and *use_message_data*. The former gathers data from the 
*Robot*'s sensors and packs it into a string message to be sent to the *supervisor controller* script. The latter 
unpacks messages sent by the *supervisor* that contain the next action, and uses the data to move the *CartPoleRobot* 
forward and backward.


### Robot controller script

The only import we are going to need is the *RobotEmitterReceiverCSV* class.
```python
from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
```
Then we define our class, inheriting the imported one.
```python
class CartpoleRobot(RobotEmitterReceiverCSV):
    def __init__(self):
        super().__init__()
```
Then we initialize the position sensor, that reads the pole's angle, needed for the agent's observation.
```python
        self.positionSensor = self.robot.getPositionSensor("polePosSensor")
        self.positionSensor.enable(self.get_timestep())
```
Finally, we initialize the four motors completing our `__init__()` method.
```python
        self.wheel1 = self.robot.getMotor('wheel1')  # Get the wheel handle
        self.wheel1.setPosition(float('inf'))  # Set starting position
        self.wheel1.setVelocity(0.0)  # Zero out starting velocity
        self.wheel2 = self.robot.getMotor('wheel2')
        self.wheel2.setPosition(float('inf'))
        self.wheel2.setVelocity(0.0)
        self.wheel3 = self.robot.getMotor('wheel3')
        self.wheel3.setPosition(float('inf'))
        self.wheel3.setVelocity(0.0)
        self.wheel4 = self.robot.getMotor('wheel4')
        self.wheel4.setPosition(float('inf'))
        self.wheel4.setVelocity(0.0)
```
After the initialization method is done we move on to the `create_message()` method implementation, used to pack the 
value read by the sensor into a string, so it can be sent to the *supervisor controller*.
```python
    def create_message(self):
        # Read the sensor value, convert to string and move it in a list
        message = [str(self.positionSensor.getValue())]
        return message
```
Finally, we implement the `use_message_data()` method, which unpacks the message received by the 
*supervisor controller*, that contains the next action. Then we implement what the *action* actually means for the
*CartPoleRobot*, i.e. moving forward and backward using its motors.
```python
    def use_message_data(self, message):
        action = int(message[0])  # Convert the string message into an action integer

        if action == 0:
            motorSpeed = 5.0
        elif action == 1:
            motorSpeed = -5.0
        else:
            motorSpeed = 0.0
        
        # Set the motors' velocities based on the action received
        self.wheels1.setVelocity(motorSpeed)
        self.wheels2.setVelocity(motorSpeed)
        self.wheels3.setVelocity(motorSpeed)
        self.wheels4.setVelocity(motorSpeed)
```

And that's it for the *robot controller script*!

### Supervisor controller script
