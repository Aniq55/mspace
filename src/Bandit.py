class Bandit:
    def __init__(self, id, arm_list):
        self.id = id
        self.arm_list = arm_list
        self.belief = None # update belief at the end of each play
        
    def play(self, arm):
        pass