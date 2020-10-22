def generate_mdp_env(num_states, num_actions, rewards, costs,
                         transition_seed=None, training_seed=None):
        """
        Construct an arbitrary MDP (by selecting a transition probability matrix
        over the space of all such matrices of given dimension).
        """
        np.random.seed(transition_seed)
    
        states  = list(range(num_states))
        actions = list(range(num_actions))
    
        probs = {}
        for elem in product(states, actions):
            dist = np.random.random(num_states)
            probs[elem] = dist / np.sum(dist)
    
        def transition_matrix(state, action):
            return probs[(state, action)]
    
    
        rewards_fn = functions['rewards'][rewards]
        costs_fn   = functions['costs'][costs]
    
        np.random.seed(training_seed)
    
        env = envs.MDPEnv(states, actions, transition_matrix, rewards_fn, costs_fn)
    
        return states, actions, env


class IOManager:
    """
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def to_stdout(self, msg):
        print(msg)

    def save_npy(self, filename, array):
        np.save(os.path.join(self.output_dir, filename), array)

    def save_plt(self, filename, fig, **kwargs):
        fig.savefig(os.path.join(self.output_dir, filename), **kwargs) 


class ExperimentManager:
    """
    Given an environment and an agent, do training and store the results.
    """

    def __init__(self, env, agent, io_manager, **kwargs):
        """
        Params
        ------
        env        : an OpenAI gym environment (instantiated)
        agent      : an agent (instantiated)
        io_manager : an IOManager (instantiated)
        kwargs     : dict 
         - width (default = 100)
           The width of the simple moving average window for computation of the
           reward+cost ratios.
         - io_interval (default = 10000)
           How frequently the agent should print provisional data to stdout and
           log to save files.
         - n_steps (default = 500000)
           Number of training steps.
         - agent_name (default = 'UnspecifiedAgentName')
           How to label the agent in plaintext.
         - logging (default = True)
           Whether the experiment will save ratios to a file.
         - plotting (default = False)
           Whether the experiment will save plot summaries of the ratios.
         - stdouting (default = True)
           Uhether the experiment will print provisional info to stdout.
        """
        self.env   = env
        self.agent = agent
        self.io    = io_manager

        defaults = {
            'width':               100,
            'print_interval':   10_000,
            'n_steps':         500_000,
            'agent_name':      'UnspecifiedAgentName',
            'logging':         True,
            'plotting':        False,
            'stdouting':       True,
        }
        defaults.update(kwargs)
        self.update(defaults)

    def update(self, **kwargs):
        """
        Sends every specified keyword argument to an attribute of the object.
        That is, if foo=bar is passed into the kwargs, then after running this
        method, self.foo == bar
        """
        self.__dict__.update(kwargs)

    def stdout_callback(self, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        print out training information to stdout during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'ratio': None,
            'step':  None,
        }
        defaults.update(kwargs)

        output_message = ' '.join([
            f'{self.agent_name}',
            f'timestep: {kwargs["step"]:7d}',
            f'(rho={kwargs["ratio"]:.2f}, state={self.agent.state}, action={self.agent.action})'
        ]))

        self.io.to_stdout(output_message)
    
    def logger_callback(self, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        log training information to a specified log output during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'ratios':     None,  # will break if unchanged, that's good
        }
        defaults.update(kwargs)

        filename = f"{self.agent_name}_ratios.npy"

        self.io.save_npy(filename, kwargs['ratios'])
    
    def plot_callback(self, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        plot training information to a specified output directory during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'ratios':     None,  # will break if unchanged, that's good
            'xlabel':     'Step',
            'ylabel':     'Ratio'
        }
        defaults.update(kwargs)

        filename = f"{kwargs['agent_name']}_ratios.png"

        fig, ax = plt.subplots()

        ax.plot(np.arange(len(kwargs['ratios'])), np.array(kwargs['ratios']))
        ax.set_xlabel(kwargs['xlabel'])
        ax.set_ylabel(kwargs['ylabel'])

        self.io.save_plt(filename, fig)
    
    def train(self):
        """
        Train a predefined agent on an initialized environment for a specified
        number of steps. Returns the agent's ratio at each step of the training.
        """
        self.env.reset()
    
        ratios = []
        rewards, costs = deque(maxlen=self.width), deque(maxlen=self.width)
    
        for step in range(self.steps):
            # First, process the agent and the environment
            action = self.agent.sample_action(self.env.state)
            next_state, (reward, cost), _, _ = self.env.step(action)
            self.agent.update((reward, cost), next_state)

            # Next, process the rewards and costs signals
            rewards.append(reward)
            costs.append(cost)
            ratios.append(np.mean(rewards) / np.mean(costs))
    
            if step % self.print_interval == 0:  # I/O callbacks
                if self.stdouting:
                    self.stdout_callback(step=step, ratio=ratios[-1])
    
                if self.logging:
                    self.logger_callback(step=step, ratios=ratios)
    
        
        if self.logging:  # Final I/O callback
            self.logger_callback(ratios=ratios)

        if self.plotting:  # Plotting callback
            self.plot_callback(ratios=ratios)

        return ratios
