from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV


class CartpoleRobot(RobotEmitterReceiverCSV):
	def __init__(self):
		super().__init__()
		self.positionSensor = self.robot.getPositionSensor("polePosSensor")
		self.positionSensor.enable(self.get_timestep())
		self.wheel1 = self.robot.getMotor('wheel1')	 # Get the wheel handle
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
		
	def create_message(self):
		# Read the sensor value, convert to string and save it in a list
		message = [str(self.positionSensor.getValue())]
		return message
	
	def use_message_data(self, message):
		action = int(message[0])  # Convert the string message into an action integer

		if action == 0:
			motorSpeed = 5.0
		elif action == 1:
			motorSpeed = -5.0
		else:
			motorSpeed = 0.0
		
		# Set the motors' velocities based on the action received
		self.wheel1.setVelocity(motorSpeed)
		self.wheel2.setVelocity(motorSpeed)
		self.wheel3.setVelocity(motorSpeed)
		self.wheel4.setVelocity(motorSpeed)


# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
