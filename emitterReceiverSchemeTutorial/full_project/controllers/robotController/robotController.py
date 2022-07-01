from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV


class CartpoleRobot(RobotEmitterReceiverCSV):
    def __init__(self):
        super().__init__()
        self.positionSensor = self.robot.getDevice("polePosSensor")
        self.positionSensor.enable(self.timestep)
        self.wheels = []
        for wheelName in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.robot.getDevice(wheelName)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

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
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motorSpeed)


# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
