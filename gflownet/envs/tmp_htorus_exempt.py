class HybridTorus(GFlowNetEnv):
    """
    Continuous (hybrid: discrete and continuous) hyper-torus environment in which the
    action space consists of the selection of which dimension d to increment and of the
    angle of dimension d. The trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle (in radians and within [0, 2 *
    pi]) at each dimension and the number of actions.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.

    state_space_atol: float
        Tolerance for comparing states similarity.
    """

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: each state is
        a vector of length n_dim where each value is an angle in radians. The n_actions
        item is removed.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return tfloat(states, device=self.device, float_type=self.float)[:, :-1]

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: if
        policy_encoding_dim_per_angle >= 2, then the state (angles) is encoded using
        trigonometric components.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, float_type=self.float, device=self.device)
        if (
            self.policy_encoding_dim_per_angle is None
            or self.policy_encoding_dim_per_angle < 2
        ):
            return states
        step = states[:, -1]
        code_half_size = self.policy_encoding_dim_per_angle // 2
        int_coeff = (
            torch.arange(1, code_half_size + 1).repeat(states.shape[-1] - 1).to(states)
        )
        encoding = (
            torch.repeat_interleave(states[:, :-1], repeats=code_half_size, dim=1)
            * int_coeff
        )
        return torch.cat(
            [torch.cos(encoding), torch.sin(encoding), torch.unsqueeze(step, 1)],
            dim=1,
        )

    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state. Angles are converted into degrees in [0, 360]
        """
        angles = np.array(state[:-1])
        angles = angles * 180 / np.pi
        angles = str(angles).replace("(", "[").replace(")", "]").replace(",", "")
        n_actions = str(int(state[-1]))
        return angles + " | " + n_actions

    def readable2state(self, readable: str) -> List:
        """
        Converts a human-readable string representing a state into a state as a list of
        positions. Angles are converted back to radians.
        """
        # Preprocess
        pattern = re.compile(r"\s+")
        readable = re.sub(pattern, " ", readable)
        readable = readable.replace(" ]", "]")
        readable = readable.replace("[ ", "[")
        # Process
        pair = readable.split(" | ")
        angles = [np.float32(el) * np.pi / 180 for el in pair[0].strip("[]").split(" ")]
        n_actions = [int(pair[1])]
        return angles + n_actions

    def copy(self):
        return deepcopy(self)

    def get_grid_terminating_states(
        self, n_states: int, n_dim: Optional[int] = None
    ) -> List[List]:
        """
        Samples n terminating states by sub-sampling the state space as a grid, where
        each dimension is sampled uniformly in [0, 2 * pi]. The number of points per
        dimension is determined by the number of terminating states to sample, such that
        the total number of points is at least n_states and at most 2 ** n_dim.

        Parameters
        ----------
        n_states : int
            The minimun number of terminating states to sample.
        n_dim : int, optional
            The number of dimensions in the state space. If None, the number of dimensions
            of the environment is used. Passed to the function to allow for conditional
            environments with different number of dimensions (see cond_ctorus).

        Returns
        -------
        states : list
            A list of sampled terminating states.
        """
        if n_dim is None:
            n_dim = self.n_dim
        # Compute the number of points per dimension
        n_per_dim = int(np.ceil(n_states ** (1 / n_dim)))
        # linspace on a circle (accounting for 0 == 2pi)
        linspace = np.linspace(0, 2 * np.pi, n_per_dim + 1)[:-1]
        angles = np.meshgrid(*[linspace] * n_dim)
        angles = np.stack(angles).reshape((n_dim, -1)).T
        states = np.concatenate(
            (angles, self.length_traj * np.ones((angles.shape[0], 1))), axis=1
        ).tolist()
        return states

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None, n_dim=None
    ) -> List[List]:
        """
        Samples n_states terminating states uniformly in the state space.
        The angles are sampled uniformly in [0, 2 * pi].
        The number of steps is set to the length of the trajectory.

        Parameters
        ----------
        n_states : int
            The number of terminating states to sample.
        seed : int
            Random seed for the sampling.
        n_dim : int, optional
            The number of dimensions in the state space. If None, the number of
            dimensions of the environment is used. Passed to the function to allow for
            conditional environments with different number of dimensions (see
            cond_ctorus).

        Returns
        -------
        states : list
            A list of sampled terminating states.
        """
        if n_dim is None:
            n_dim = self.n_dim
        rng = np.random.default_rng(seed)
        angles = rng.uniform(low=0.0, high=(2 * np.pi), size=(n_states, n_dim))
        states = np.concatenate(
            (angles, self.length_traj * np.ones((n_states, 1))), axis=1
        )
        return states.tolist()

    def fit_kde(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        kernel: str = "gaussian",
        bandwidth: float = 0.1,
    ):
        r"""
        Fits a Kernel Density Estimator on a batch of samples.

        The samples are previously augmented in order to account for the periodic
        aspect of the sample space.

        Parameters
        ----------
        samples : tensor
            A batch of samples in proxy format.
        kernel : str
            An identifier of the kernel to use for the density estimation. It must be a
            valid kernel for the scikit-learn method
            :py:meth:`sklearn.neighbors.KernelDensity`.
        bandwidth : float
            The bandwidth of the kernel.
        """
        samples_aug = self.augment_samples(samples)
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples_aug)
        return kde

    def plot_reward_samples(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        samples_reward: TensorType["batch_size", "state_proxy_dim"],
        rewards: TensorType["batch_size"],
        min_domain: float = -np.pi,
        max_domain: float = 3 * np.pi,
        alpha: float = 0.5,
        dpi: int = 150,
        max_samples: int = 500,
        **kwargs,
    ):
        """
        Plots the reward contour alongside a batch of samples.

        The samples are previously augmented in order to visualise the periodic aspect
        of the sample space. It is assumed that the rewards are sorted from left to
        right (first) and top to bottom of the grid of samples.

        Parameters
        ----------
        samples : tensor
            A batch of samples from the GFlowNet policy in proxy format. These samples
            will be plotted on top of the reward density.
        samples_reward : tensor
            A batch of samples containing a grid over the sample space, from which the
            reward has been obtained. Ignored by this method.
        rewards : tensor
            The rewards of samples_reward. It should be a vector of dimensionality
            n_per_dim ** 2 and be sorted such that the each block at rewards[i *
            n_per_dim:i * n_per_dim + n_per_dim] correspond to the rewards at the i-th
            row of the grid of samples, from top to bottom.
        min_domain : float
            Minimum value of the domain to keep in the plot.
        max_domain : float
            Maximum value of the domain to keep in the plot.
        alpha : float
            Transparency of the reward contour.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        max_samples : int
            Maximum of number of samples to include in the plot.
        """
        if self.n_dim != 2:
            return None
        rewards = torch2np(rewards)
        n_per_dim = int(np.sqrt(rewards.shape[0]))
        assert n_per_dim**2 == rewards.shape[0]
        # Augment rewards to apply periodic boundary conditions
        rewards = rewards.reshape((n_per_dim, n_per_dim))
        rewards = np.tile(rewards, (3, 3))
        # Create mesh grid from samples_reward
        x = np.linspace(-2 * np.pi, 4 * np.pi, 3 * n_per_dim)
        y = np.linspace(-2 * np.pi, 4 * np.pi, 3 * n_per_dim)
        x_coords, y_coords = np.meshgrid(x, y)
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot reward contour
        h = ax.contourf(x_coords, y_coords, rewards, alpha=alpha)
        ax.axis("scaled")
        fig.colorbar(h, ax=ax)
        ax.plot([0, 0], [0, 2 * np.pi], "-w", alpha=alpha)
        ax.plot([0, 2 * np.pi], [0, 0], "-w", alpha=alpha)
        ax.plot([2 * np.pi, 2 * np.pi], [2 * np.pi, 0], "-w", alpha=alpha)
        ax.plot([2 * np.pi, 0], [2 * np.pi, 2 * np.pi], "-w", alpha=alpha)
        # Randomize and subsample samples
        random_indices = np.random.permutation(samples.shape[0])[:max_samples]
        samples = samples[random_indices, :]
        # Augment samples
        samples_aug = self.augment_samples(samples, exclude_original=True)
        ax.scatter(
            samples_aug[:, 0], samples_aug[:, 1], alpha=1.5 * alpha, color="white"
        )
        ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha)
        # Set axes limits
        ax.set_xlim([min_domain, max_domain])
        ax.set_ylim([min_domain, max_domain])
        # Set ticks and labels
        ticks = [0.0, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]
        labels = ["0.0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{3}$", f"$2\pi$"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.grid()
        # Set tight layout
        plt.tight_layout()
        return fig

    def plot_kde(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        kde,
        alpha: float = 0.5,
        dpi=150,
        colorbar: bool = True,
        **kwargs,
    ):
        """
        Plots the density previously estimated from a batch of samples via KDE over the
        entire sample space.

        Parameters
        ----------
        samples : tensor
            A batch of samples containing a grid over the sample space. These samples
            are used to plot the contour of the estimated density.
        kde : KDE
            A scikit-learn KDE object fit with a batch of samples.
        alpha : float
            Transparency of the density contour.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        """
        if self.n_dim != 2:
            return None
        # Create mesh grid from samples
        n_per_dim = int(np.sqrt(samples.shape[0]))
        assert n_per_dim**2 == samples.shape[0]
        x_coords = samples[:, 0].reshape((n_per_dim, n_per_dim))
        y_coords = samples[:, 1].reshape((n_per_dim, n_per_dim))
        # Score samples with KDE and reshape
        Z = np.exp(kde.score_samples(samples)).reshape((n_per_dim, n_per_dim))
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot KDE
        h = ax.contourf(x_coords, y_coords, Z, alpha=alpha)
        ax.axis("scaled")
        if colorbar:
            fig.colorbar(h, ax=ax)
        # Set ticks and labels
        ticks = [0.0, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]
        labels = ["0.0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{3}$", f"$2\pi$"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Set tight layout
        plt.tight_layout()
        return fig

    @staticmethod
    def augment_samples(samples: np.array, exclude_original: bool = False) -> np.array:
        """
        Augments a batch of samples by applying the periodic boundary conditions from
        [0, 2pi) to [-2pi, 4pi) for all dimensions.
        """
        samples_aug = []
        for offsets in itertools.product(
            [-2 * np.pi, 0.0, 2 * np.pi], repeat=samples.shape[-1]
        ):
            if exclude_original and all([offset == 0.0 for offset in offsets]):
                continue
            samples_aug.append(
                np.stack(
                    [samples[:, dim] + offset for dim, offset in enumerate(offsets)],
                    axis=-1,
                )
            )
        samples_aug = np.concatenate(samples_aug, axis=0)
        return samples_aug

    def process_data_set(self, samples, progress=False):
        """
        Process dataset loaded from a file inside Buffer init
        """
        if hasattr(samples, "tolist"):
            return samples.tolist()
        return samples
