from abc import abstractmethod
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv

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

    @abstractmethod
    def statebatch2policy(self, states: List[List], conditions: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states and conditions into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        pass

    @abstractmethod
    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"], conditions: TensorType["batch", "condition_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        pass

    @abstractmethod
    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"],  conditions: TensorType["batch", "condition_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states and conditions in torch "GFlowNet format" for the proxy.
        """
        pass
    
    def reward_torchbatch(
        self, 
        states: TensorType["batch", "state_dim"],
        conditions: TensorType["batch", "condition_dim"],
        done: TensorType["batch"] = None
    ):
        """
        Computes the rewards of a batch of states in "GFlownet format"
        """
        if done is None:
            done = torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
        states_proxy = self.statetorch2proxy(states[done, :], conditions[done, :])
        reward = torch.zeros(done.shape[0], dtype=self.float, device=self.device)
        if states[done, :].shape[0] > 0:
            reward[done] = self.proxy2reward(self.proxy(states_proxy))
        return reward