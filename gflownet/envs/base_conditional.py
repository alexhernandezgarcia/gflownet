from gflownet.envs.base import GFlowNetEnv
from abc import abstractmethod

class GFlowNetCondEnv(GFlowNetEnv):
    def __init__(self, **kwardgs):
        super().__init__(**kwargs)
        self.conditional = True
    
    @abstractmethod
    def reset_condition(self, condition=None):
        """
        Resets the condition of the environment.
        """
        pass

    def reset(self, env_id=None, condition=None):
        self.reset_condition(condition=condition)
        super().reset(env_id=env_id)

    @abstractmethod
    def get_condition_dataloader(self, batch_size, train=True):
        pass
    