from deepbots.robots.controllers.csv_robot import CSVRobot


class CartpoleRobot(CSVRobot):
    def __init__(self):
        super().__init__()
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)
        self.wheels = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

    def create_message(self):
        # Read the sensor value, convert to string and save it in a list
        message = [str(self.position_sensor.getValue())]
        return message

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


# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
