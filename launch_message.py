class LaunchMessage:
    def __init__(self) -> None:
        self.message = ""

    def update(self, message):
        self.message = message

    def get(self):
        return self.message


launch_message_instance = LaunchMessage()