from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np


class CartpoleRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        self.action_space = Discrete(2)

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
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

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

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        return 1

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

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        action = int(action[0])

        if action == 0:
            motor_speed = 5.0
        else:
            motor_speed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)


env = CartpoleRobot()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)

solved = False
episode_count = 0
episode_limit = 2000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

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
