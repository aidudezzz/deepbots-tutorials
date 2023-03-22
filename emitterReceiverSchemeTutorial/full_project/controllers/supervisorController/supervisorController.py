import numpy as np
from deepbots.supervisor.controllers.csv_supervisor_env import CSVSupervisorEnv
from PPOAgent import PPOAgent, Transition
from utilities import normalize_to_range


class CartPoleSupervisor(CSVSupervisorEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = 4  # The agent has 4 inputs
        self.action_space = 2  # The agent can perform 2 actions

        self.robot = self.getFromDef("ROBOT")
        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.message_received = None  # Variable to save the messages received from the robot

        self.episode_count = 0  # Episode counter
        self.episode_limit = 10000  # Max number of episodes allowed
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        # Position on x-axis, first (0) element of the getPosition vector
        cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x-axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        # Update self.message_received received from robot, which contains pole angle
        self.message_received = self.handle_receiver()
        if self.message_received is not None:
            pole_angle = normalize_to_range(float(self.message_received[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            # Method is called before self.message_received is initialized
            pole_angle = 0.0

        return [cart_position, cart_velocity, pole_angle, endpoint_velocity]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space)]

    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        return 1

    def is_done(self):
        if self.message_received is not None:
            pole_angle = round(float(self.message_received[0]), 2)
        else:
            # method is called before self.message_received is initialized
            pole_angle = 0.0
        if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical
            return True

        if self.episode_score > 195.0:
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

    def render(self, mode="human"):
        pass


supervisor = CartPoleSupervisor()
agent = PPOAgent(supervisor.observation_space, supervisor.action_space)

solved = False
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.episode_count < supervisor.episode_limit:
    observation = supervisor.reset()  # Reset robot and get starting observation
    supervisor.episode_score = 0

    for step in range(supervisor.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selectedAction, actionProb = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached
        # the done condition
        newObservation, reward, done, info = supervisor.step([selectedAction])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            supervisor.episode_score_list.append(supervisor.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = supervisor.solved()  # Check whether the task is solved
            break

        supervisor.episode_score += reward  # Accumulate episode reward
        observation = newObservation  # observation for next step is current step's newObservation

    print("Episode #", supervisor.episode_count, "score:", supervisor.episode_score)
    supervisor.episode_count += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = supervisor.reset()
while True:
    selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
    observation, _, _, _ = supervisor.step([selectedAction])
