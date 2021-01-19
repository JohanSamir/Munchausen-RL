"""Compact implementation of a DQN agent

Specifically, we implement the following components:

  * prioritized replay
  * huber_loss
  * mse_loss
  * double_dqn
  * noisy
  * dueling

Details in: 
"Human-level control through deep reinforcement learning" by Mnih et al. (2015).
"Noisy Networks for Exploration" by Fortunato et al. (2017).
"Deep Reinforcement Learning with Double Q-learning" by Hasselt et al. (2015).
"Dueling Network Architectures for Deep Reinforcement Learning" by Wang et al. (2015).
"""

import functools
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
import utils
from scipy.special import logsumexp




def mse_loss(targets, predictions):
  return jnp.mean(jnp.power((targets - (predictions)),2))


#@functools.partial(jax.jit, static_argnums=(7,8,9,10,11,12,13))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, cumulative_gamma,double_dqn, mse_inf,tau,alpha,clip_value_min,num_actions):
  """Run the training step."""
  #print('ACTIONS 555',num_actions)
  def loss_fn(model, target, mse_inf):
    q_values = jax.vmap(model, in_axes=(0))(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)

    if mse_inf:
      loss = jnp.mean(jax.vmap(mse_loss)(target, replay_chosen_q))
    else:
      loss = jnp.mean(jax.vmap(dqn_agent.huber_loss)(target, replay_chosen_q))
    return loss


  grad_fn = jax.value_and_grad(loss_fn)

  if double_dqn:
    #target = target_DDQN(optimizer, target_network, next_states, rewards,  terminals, cumulative_gamma)
    target=target_m_dqn(optimizer,target_network,states,next_states,actions,rewards,terminals,
                cumulative_gamma,tau,alpha,clip_value_min,num_actions)
  else:
    target = dqn_agent.target_q(target_network, next_states, rewards,  terminals, cumulative_gamma) 

  loss, grad = grad_fn(optimizer.target, target, mse_inf)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss

def target_m_dqn(model, target_network, states, next_states, actions,rewards, terminals, 
                cumulative_gamma,tau,alpha,clip_value_min,num_actions):

  replay_action_one_hot = jax.nn.one_hot(actions, num_actions)
  #print('replay_action_one_hot:',replay_action_one_hot.shape,replay_action_one_hot)

  #Neurla Networks -> Target[Target] y Expected [Online]
  q_state_values = jax.vmap(target_network, in_axes=(0))(states).q_values
  q_state_values = jnp.squeeze(q_state_values)
  #replay_qt_max = jnp.argmax(q_state_values, axis=1).reshape(q_state_values.shape[0],1)
  replay_qt_max = jnp.amax(q_state_values, axis=1).reshape(q_state_values.shape[0],1)

  next_q_values = jax.vmap(target_network, in_axes=(0))(next_states).q_values
  next_q_values = jnp.squeeze(next_q_values)
  #replay_next_qt_max = jnp.argmax(next_q_values, axis=1).reshape(next_q_values.shape[0],1)
  replay_next_qt_max = jnp.amax(next_q_values, axis=1).reshape(next_q_values.shape[0],1)


  # Equa.1 [Johan]
  #A
  #print('next_states:',next_states.shape,next_states)
  #print('next_q_values:',next_q_values.shape,next_q_values)
  #print('replay_next_qt_max:',replay_next_qt_max.shape,replay_next_qt_max)
  logsum = logsumexp((next_q_values-replay_next_qt_max)/tau,1).reshape(next_q_values.shape[0],1)
  #print('logsum:',logsum.shape,logsum)
  #B
  tau_log_pi_next = next_q_values-replay_next_qt_max-tau*logsum
  #print('tau_log_pi_next:',tau_log_pi_next.shape,tau_log_pi_next)

  # Equa.3 [Johan]

  pi_target = jax.nn.softmax(next_q_values/tau, axis=-1)
  #print('pi_target:',pi_target.shape,pi_target)

  # Equa.4+3+1 [Johan]
  #next_qt_softmax = cumulative_gamma*(q_ope* jnp.sum((next_q_values-tau_log_pi_next),axis=0))
  #print('(1. - terminals:)',(1. - terminals),(1. - terminals).shape)
  ter = (1. - terminals).reshape((1. - terminals).shape[0],1)
  #print('ter',ter,ter.shape)
  
  Q_target = cumulative_gamma*jnp.sum(pi_target*(next_q_values-tau_log_pi_next)*ter,axis=1)
  Q_target = Q_target.reshape(Q_target.shape[0],1)
  #print('Q_target:',Q_target.shape,Q_target)
  
    # Equa.2 [Johan]
  logsum_q_targe = logsumexp((q_state_values-replay_qt_max)/tau,1).reshape(next_q_values.shape[0],1)
  #print('logsum_q_targe:',logsum_q_targe.shape,logsum_q_targe)

  tau_log_pi = q_state_values-replay_qt_max-tau*logsum_q_targe
  #print('tau_log_pi:',tau_log_pi.shape,tau_log_pi)

  # tau_log_pi_a = jnp.sum(replay_log_policy * replay_action_one_hot, axis=1)

  #munchausen_addon = tau_log_pi * replay_action_one_hot
  munchausen_addon = jax.vmap(lambda x, y: x[y])(tau_log_pi, actions)
  #print('munchausen_addon:',munchausen_addon.shape,munchausen_addon)

  munchausen_reward = rewards + alpha* jnp.clip(munchausen_addon, a_min=clip_value_min,a_max=0)
  #print('munchausen_reward:',munchausen_reward.shape,munchausen_reward)

  #print('rewards:',rewards.shape)
  #print('terminals:',terminals.shape)


  q_target = munchausen_reward+Q_target 
  #print('q_target:',q_target.shape,q_target)
  return jax.lax.stop_gradient(q_target)

@gin.configurable
class JaxDQNAgentNew(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,

               tau,
               alpha=1,
               clip_value_min=-10,
               #interact='greedy',

               net_conf = None,
               env = "CartPole", 
               normalize_obs = True,
               hidden_layer=2, 
               neurons=512,
               prioritized=False,
               noisy = False,
               dueling = False,
               double_dqn=False,
               mse_inf=False,
               network=networks.NatureDQNNetwork,
               optimizer='adam',
               epsilon_fn=dqn_agent.linearly_decaying_epsilon):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    # We need this because some tools convert round floats into ints.


    self._tau = tau
    self._alpha = alpha
    self._clip_value_min = clip_value_min

    self._net_conf = net_conf
    self._env = env 
    self._normalize_obs = normalize_obs
    self._hidden_layer = hidden_layer
    self._neurons=neurons 
    self._noisy = noisy
    self._dueling = dueling
    self._double_dqn = double_dqn
    self._mse_inf = mse_inf
    #print('ACTIONS 222',num_actions)

    super(JaxDQNAgentNew, self).__init__(
        num_actions= num_actions,
        network=network.partial(num_actions=num_actions,
                                net_conf=self._net_conf,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling),
        optimizer=optimizer,
        epsilon_fn=dqn_agent.identity_epsilon if self._noisy == True else epsilon_fn)

    #print('ACTIONS 333',num_actions)
    self._num_actions=num_actions
    #print('ACTIONS 444',self._num_actions)
    self._prioritized=prioritized
    self._rng = jax.random.PRNGKey(0)
    state_shape = self.observation_shape + (self.stack_size,)
    self.state = onp.zeros(state_shape)
    self._replay = self._build_replay_buffer_prioritized() if self._prioritized == True else self._build_replay_buffer()
    self._optimizer_name = optimizer
    self._build_networks_and_optimizer()


  def _build_replay_buffer_prioritized(self):
    """Creates the prioritized replay buffer used by the agent."""
    return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)
    
  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        self.optimizer, loss = train(self.target_network,
                                     self.optimizer,
                                     self.replay_elements['state'],
                                     self.replay_elements['action'],
                                     self.replay_elements['next_state'],
                                     self.replay_elements['reward'],
                                     self.replay_elements['terminal'],
                                     self.cumulative_gamma,
                                     self._double_dqn,
                                     self._mse_inf,
                                     self._tau,
                                     self._alpha,
                                     self._clip_value_min,
                                     self._num_actions)
        if self._prioritized == 'prioritized':
        #It must be: if self._prioritized == True:
          #print('OJOOOOOOO prioritized')


          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)

          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='HuberLoss', simple_value=loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _store_transition(self, last_observation, action, reward, is_terminal):
    """Stores an experienced transition.

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    if self._prioritized==True:
      priority = self._replay.sum_tree.max_recorded_priority
      self._replay.add(last_observation, action, reward, is_terminal, priority)
    else:
      self._replay.add(last_observation, action, reward, is_terminal)