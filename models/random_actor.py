class RandomActor():
    def __init__(self, env):
        self.env = env

    def get_action(self, states):
        assert len(states.shape) == 1, "can't work with batches"
        return self.env.action_space.sample()