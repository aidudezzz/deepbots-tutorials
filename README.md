# Deepbots Tutorial (WIP)

This tutorial is on how to use the deepbots framework. We will recreate the 
[CartPole](https://gym.openai.com/envs/CartPole-v0/) problem in [Webots](https://cyberbotics.com/), 
step-by-step and solve it with the [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) 
Reinforcement Learning (RL) algorithm, using [PyTorch](https://pytorch.org/) as our neural network backend library.

The complete example can be found on the [deepworlds](https://github.com/aidudezzz/deepworlds/) repository. 


## Prerequisites

Before starting, several prerequisites should be met.

1. [Install Webots](https://cyberbotics.com/doc/guide/installing-webots)
    - [Windows](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-windows)
    - [Linux](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux)
    - [macOS](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-macos)
2. [Install Python version 3.X](https://www.python.org/downloads/) (please refer to 
[Using Python](https://cyberbotics.com/doc/guide/using-python#introduction) to select proper Python version for your system) 
3. Follow the [Using Python](https://cyberbotics.com/doc/guide/using-python) guide
4. [Using PyCharm IDE](https://cyberbotics.com/doc/guide/using-your-ide#pycharm)
5. [Install PyTorch](https://pytorch.org/get-started/locally/)
6. Install deepbots  (Add link)

## CartPole Tutorial
Now we are ready to start working on the CartPole problem. First of all, we should create a new project.

### Creating the project
1. Open Webots and on the menu bar, click *"Wizards -> New Project Directory..."*\
    ![New project menu option](/images/newProjectMenuScreenshot.png)
2. Select a directory of your choice
3. On world settings **all** boxes should be ticked\
    ![World settings](/images/worldSettingsScreenshot.png)
4. Give your world a name, e.g. "cartPoleWorld.wbt"
5. Press Finish

You should end up with:\
![Project created](/images/projectCreatedScreenshot.png)

### Adding a *supervisor* robot in the world
<!---1. Right-click on [this link](/CartPoleRobot.wbo) and click *Save link as...* to download the CartPole robot 
definition 
2. Save the .wbo file inside the project directory--> 

Now that the project and the starting world are created, we are going to create a special kind of robot, 
a *supervisor*. Through the *supervisor controller script* we will be able to handle several aspects of the 
simulation needed for RL (e.g. resetting).
 
Adding the *supervisor* robot node:
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
Then we are going to assign the *supervisor* controller script to the *supervisor Robot* created before.
The *CartPole Robot* is going to be loaded into the world through the *supervisor* controller script later, but
we still need to create its controller.

Creating the *supervisor* and *robot controller scripts*:
1. On the *menu bar*, click *"Wizards -> New Robot Controller..."*\
![New robot controller](/images/newControllerMenuScreenshot.png)
2. On *Language selection*, select *Python*
3. Give it the name "*supervisorController*" 
4. Press *Finish* 
5. Repeat from step 1, but on step 3 give the name "*robotController*"

Two new Python controller scripts should be created and opened in Webots text editor looking like this:\
![New robot controller](/images/newControllerMenuScreenshot.png)
    
*If you are using an external IDE:    
1. Un-tick the "open ... in Text Editor" box and then go to step 4.
2. Navigate to the project directory, inside the *Controllers/controllerName/* directories
3. Open the controller scripts with your IDE

Assigning the *supervisorController* as *supervisor* controller:
1. Expand the *supervisor Robot* created earlier and scroll down to find the *controller* field
2. Click on the *controller* field and press the "*Select...*" button below\
![New robot controller](/images/assignSupervisorController1Screenshot.png)
3. Find the "*supervisorController*" controller from the list and click it\
![New robot controller](/images/assignSupervisorController2Screenshot.png)
4. Click *OK*
5. Click *Save*

   
### Writing the scripts