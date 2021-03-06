# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pickle
import sys
import time
import re
from tqdm import tqdm

from absl import logging
from copy import deepcopy

# from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import highway_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import gin.tf

import logging


def process_key(key):
  key = re.sub(r'Online/basic_discrete_domain_network_\d+', 'Online/basic_discrete_domain_network', key)
  key = re.sub(r'Target/basic_discrete_domain_network_\d+', 'Target/basic_discrete_domain_network_1', key)
  return key


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


@gin.configurable
class TestRunner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               model_dir,
               n_runs,
               total_num,
               create_agent_fn,
               create_environment_fn=highway_lib.create_gym_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
              #  evaluation_steps=125000,
               evaluation_steps=1000,
               max_window_size=5,
               max_steps_per_episode=27000,
               clip_rewards=True):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    tf.compat.v1.disable_v2_behavior()

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._max_window_size = max_window_size
    self._clip_rewards = clip_rewards
    self._n_runs = n_runs
    self._total_num = total_num

    self._base_dir = base_dir
    self._model_dir = model_dir
    self._create_directories()

    self._environment = create_environment_fn()

    self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    self.config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.

    self.create_agent_fn = create_agent_fn
    self.checkpoint_file_prefix = checkpoint_file_prefix

  def _create_directories(self):
    """Create necessary sub-directories."""
    if not os.path.exists(self._base_dir):
      os.makedirs(self._base_dir)
    setup_logger('window action cert', os.path.join(self._base_dir, 'test.log'))
    self._logger = logging.getLogger('window action cert')

  def _create_agent(self, config, create_agent_fn):
    sess = tf.compat.v1.Session('', config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    agent = create_agent_fn(sess, self._environment,
                                  summary_writer=None)
    return agent

  def _init_agent_from_ckpt(self, agent, checkpoint_dir, checkpoint_file_prefix):
    self._checkpointer = checkpointer.Checkpointer(checkpoint_dir,
                                                   checkpoint_file_prefix)
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      agent.unbundle(
          checkpoint_dir, latest_checkpoint_version, experiment_data)
    return agent


  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward, terminal=True):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    self._agent.end_episode(reward)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _compute_poison_size(self, action, a, n, c, W):
    # self._logger.info(f'c mat = {c}')
    h = c[action,:] + W - c[a,:]
    # self._logger.info(f'[compute_poison_size] h arr (before sort) = {h}')
    h = np.sort(h)[::-1]
    # self._logger.info(f'[compute_poison_size] h arr (after sort) = {h}')

    delta = n[action] - (n[a] + (a < action))
    # self._logger.info(f'[compute_poison_size] delta = {delta}')

    # self._logger.info(f'delta = {delta}, h arr = {h}')

    prefix_sum_h = np.cumsum(h)
    return np.where(prefix_sum_h > delta)[0][0] + 1

  def _compute_certification_tpa(self, actions, a_star, max_w):
    actions_f = actions.flatten()

    # self._logger.info(f'actions_arr = {actions_arr}')

    n = self._count_freq(actions_f, self.num_actions)
    action = np.argmax(n)

    assert action == a_star, f'action = {action}, a_star = {a_star}. failure!'

    c = np.zeros((self.num_actions, self._total_num), dtype=np.int)
    for a in range(self.num_actions):
      pos = np.where(actions == a)[1]
      c[a] = self._count_freq(pos, self._total_num)

    min_val = 1e10
    for a in range(self.num_actions):
      if a == action: continue
      min_val = min(min_val, self._compute_poison_size(action, a, n, c, W=max_w))

    return min_val-1


  def _best_other_action_at_w(self, cum_actions, w_star, a_3, max_w):
    arr = cum_actions[max_w-w_star]

    ind = np.argsort(arr)
    a_w_star_1 = ind[-1]
    a_w_star_2 = ind[-2]

    self._logger.info(f'arr = {arr}, a_w_star_1 = {a_w_star_1}, a_w_star_2 = {a_w_star_2}')

    return a_w_star_2 if a_w_star_1 == a_3 else a_w_star_1

  def _compute_aux_func(self, w_prime, w_star, a_star, a_2, a_3, a_4, max_w):
    sigma = np.zeros((max_w, self.num_actions), dtype=np.int)

    for w in range(1, max_w+1):
      for a in range(self.num_actions):
        sigma[max_w-w][a] = \
                    w_prime * (a == a_3 and w <= w_star) \
                    - w_prime * (a == a_4 and w <= w_star) \
                    - w_star * (a == a_star and w <= w_prime) \
                    + w_star * (a == a_2 and w <= w_prime)
    return sigma

  def _compute_aux_margin(self, actions, sigma, max_w):
    g = np.zeros(self._total_num, dtype=np.int)

    for w in range(1, max_w+1):
      max_sigma = np.max(sigma[max_w-w])
      for i in range(self._total_num):
        g[i] += max_sigma - sigma[max_w-w][actions[max_w-w][i]]
    return g

  def _compute_certification_dtpa_helper(self, actions, cum_actions, w_prime, w_star, a_star, a_2, a_3, a_4, max_w):
    sigma = self._compute_aux_func(w_prime, w_star, a_star, a_2, a_3, a_4, max_w)

    self._logger.info(f'w_prime = {w_prime}, w_star = {w_star}')
    self._logger.info(f'a_star = {a_star}, a_2 = {a_2}, a_3 = {a_3}, a_4 = {a_4}')
    self._logger.info(f'sigma = {sigma}')

    g = self._compute_aux_margin(actions, sigma, max_w)
    g = np.sort(g)[::-1]

    thre = w_star * (cum_actions[max_w-w_prime][a_star] - cum_actions[max_w-w_prime][a_2]) \
        - w_prime * (cum_actions[max_w-w_star][a_3] - cum_actions[max_w-w_star][a_4]) \
        + (a_3 > a_star)

    self._logger.info(f'g = {g}, thre = {thre}')

    prefix_sum_g = np.cumsum(g)
    return np.where(prefix_sum_g >= thre)[0][0] + 1

  def _compute_certification_dtpa(self, actions, cum_actions, w_prime, a_star, max_w):
    min_poison_size = 1e10

    for w_star in range(1, max_w+1):
      if w_star == w_prime: continue

      for a_3 in range(self.num_actions):
        if a_3 == a_star: continue

        a_4 = self._best_other_action_at_w(cum_actions, w_star, a_3, max_w)

        for a_2 in range(self.num_actions):
          if a_2 == a_star: continue

          poison_size = self._compute_certification_dtpa_helper(actions, cum_actions, w_prime, w_star, a_star, a_2, a_3, a_4, max_w)
          self._logger.info(f'poison size for w_star = {w_star}, a_3 = {a_3}, a_4 = {a_4}, a_2 = {a_2} is {poison_size}')

          min_poison_size = min(min_poison_size, poison_size)
    return min_poison_size - 1

  def _count_freq(self, arr, num):
    freq = np.zeros(num, dtype=np.int)
    unique, count = np.unique(arr, return_counts=True)
    for a, b in zip(unique, count):
      freq[a] = b
    return freq

  def _compute_action_and_certificate(self, actions, max_w):
    actions_arr = np.asarray(actions)
    cum_actions = np.zeros((max_w, self.num_actions), dtype=np.int)

    for w in range(1, max_w+1):
      action_arr = actions_arr[max_w-w]
      action_cnt = self._count_freq(action_arr, self.num_actions)
      cum_actions[max_w-w] = action_cnt

    cum_actions = np.cumsum(cum_actions[::-1], axis=0)[::-1]

    self._logger.info(f'actions = {actions}')
    self._logger.info(f'cum_actions = {cum_actions}')

    delta = np.zeros(max_w)

    for w in range(1, max_w+1):
      arr = cum_actions[max_w-w]
      val_1 = np.max(arr)
      val_2 = np.partition(arr, -2)[-2]
      delta[w-1] = (val_1 - val_2) / w 

    w_prime = np.argmax(delta) + 1
    action = np.argmax(cum_actions[max_w-w_prime])

    term_1 = self._compute_certification_tpa(actions_arr[max_w-w_prime:], action, max_w=w_prime)

    term_2 = self._compute_certification_dtpa(actions_arr, cum_actions, w_prime, action, max_w=max_w)

    self._logger.info(f'term_1 = {term_1}, term_2 = {term_2}')

    return action, min(term_1, term_2), w_prime

  def _save_image(self, img, img_file):
    with open(osp.join(self._base_dir, img_file), 'wb') as f:
        pickle.dump(img, f)

  def _run_one_episode_multi_agent(self, save_fig=False):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    all_obs = []
    step_number = 0
    total_reward = 0.
    all_cert = []
    all_reward = []
    all_action = []
    all_w = []

    action_seqs = []

    initial_observation = self._environment.reset()
    if save_fig:
      all_obs.append(initial_observation)

    actions = []
    for agent in self._agents:
      actions.append(agent.begin_episode(initial_observation))
    action_seqs.append(actions)

    action, cert, w_prime = self._compute_action_and_certificate(action_seqs, max_w=len(action_seqs))
    all_cert.append(cert)
    all_action.append(action)
    all_w.append(w_prime)

    # self._test_deterministic_actions()

    is_terminal = False


    # Keep interacting until we reach a terminal state.
    while True:
      if len(action_seqs) == self._max_window_size:
        action_seqs.pop(0)

      observation, reward, is_terminal = self._run_one_step(action)
      if save_fig:
        all_obs.append(observation)
      all_reward.append(reward)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if step_number % 100 == 0:
        self._logger.info(f'step_number = {step_number}, total reward = {total_reward}')

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        actions = []
        for agent in self._agents:
          actions.append(agent.begin_episode(observation))
        action_seqs.append(actions)

        action, cert, w = self._compute_action_and_certificate(action_seqs, max_w=len(action_seqs))
        all_cert.append(cert)
        all_action.append(action)
        all_w.append(w)
      else:
        actions = []
        for agent in self._agents:
          actions.append(agent.step(reward, observation))
        action_seqs.append(actions)

        # with open('result_2.pkl', 'wb') as f:
        #   pickle.dump(agent.state, f)
        #   exit(-1)
        t = time.time()
        action, cert, w = self._compute_action_and_certificate(action_seqs, max_w=len(action_seqs))
        # self._logger.info(f'voting takes {time.time() - t} seconds!')
        all_cert.append(cert)
        all_action.append(action)
        all_w.append(w)

    return step_number, total_reward, all_cert, all_obs, all_reward, all_action, all_w

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()

    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    return statistics.data_lists


  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    # evaluate for oniteration
    # id_list = list(range(1, self._total_num+1))
    id_list = list(range(self._total_num))

    sess = tf.compat.v1.Session('', config=self.config)
    sess.run(tf.compat.v1.global_variables_initializer())

    t_0 = time.time()
    self._agents = []
    for cur_id in id_list:
      with tf.name_scope(f"net{cur_id}"):
        agent = self.create_agent_fn(sess, self._environment,
                                      summary_writer=None)

      net1_varlist = {process_key(v.op.name.lstrip(f"net{cur_id}/")): v
                    for v in tf1.get_collection(tf1.GraphKeys.VARIABLES, scope=f"net{cur_id}/")}
      print(net1_varlist)
      net1_saver = tf1.train.Saver(var_list=net1_varlist)

      t = time.time()
      # net1_saver.restore(sess, f'{self._model_dir}test{cur_id}/checkpoints/tf_ckpt-49')
      net1_saver.restore(sess, osp.join(self._model_dir, f'hash_{"%02d"%cur_id}/checkpoints/tf_ckpt-2999'))
      self._logger.info(f'loading ckpts {cur_id} taking {time.time() - t} seconds!')

      self._logger.info(f'loading {cur_id} done!')

      self._agents.append(agent)

    self._logger.info(f'loading all models using {time.time() - t_0} seconds!')

    self.num_actions = self._agents[0].num_actions


    for idx in tqdm(range(self._n_runs)):

      t = time.time()
      step_number, total_reward, all_cert, all_obs, all_reward, all_action, all_w \
          = self._run_one_episode_multi_agent()
      self._logger.info(f'running one episode takes {time.time() - t} seconds!')

      self._logger.info(f'step_number = {step_number}')
      self._logger.info(f'total_reward = {total_reward}')
      self._logger.info(f'all_cert = {all_cert}')
      self._logger.info(f'all_reward = {all_reward}')
      self._logger.info(f'all_action = {all_action}')
      self._logger.info(f'all_w = {all_w}')

      result = {
        'step_number': step_number,
        'total_reward': total_reward,
        'all_cert': all_cert,
        'all_obs': all_obs,
        'all_reward': all_reward,
        'all_action': all_action,
        'all_w': all_w,
      }
      save_filename = os.path.join(self._base_dir, f'result-{idx}.pkl')
      with open(save_filename, 'wb') as f:
        pickle.dump(result, f)

      self._logger.info(f'result saved to {save_filename}')

    # for cur_id, agent in zip(id_list, agents):
    #   self._agent = agent

    #   logging.info(f'Beginning evaluation agent {cur_id}...')
    #   statistics = self._run_one_iteration()
    #   self._logger.info(statistics)
