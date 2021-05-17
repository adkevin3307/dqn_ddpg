'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'

import random
import argparse
import itertools
from typing import Any, Generator
from collections import deque, OrderedDict
import gym
from gym.spaces import Discrete
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, *transition: Any) -> None:
        # (state, action, reward, next_state, done)

        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size: int, device: str) -> Generator:
        '''sample a batch of transition tensors'''

        transitions = random.sample(self.buffer, batch_size)

        return (torch.tensor(x, dtype=torch.float, device=device) for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()

        # TODO Net __init__
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO Net forward
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)

        return x


class DQN:
    def __init__(self, args: argparse.Namespace) -> None:
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)

        # initialize target network
        self._target_net.load_state_dict(OrderedDict(self._behavior_net.state_dict()))

        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        # TODO DQN __init__
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=5e-4)
        self._criterion = nn.SmoothL1Loss()

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq
        self.use_ddqn = args.use_ddqn

    def select_action(self, state: np.ndarray, epsilon: float, action_space: Discrete) -> int:
        '''epsilon-greedy based on behavior network'''

        # TODO DQN select_action
        if random.random() < epsilon:
            return action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state]).to(self.device)
                action = self._behavior_net(state_tensor).argmax(dim=1).item()

            return action

    def append(self, state: torch.Tensor, action: int, reward: torch.Tensor, next_state: torch.Tensor, done: bool) -> None:
        self._memory.append(state, [action], [reward / 10], next_state, [int(done)])

    def update(self, total_steps: int) -> None:
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma: float) -> None:
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        # TODO DQN _update_behavior_network
        q_value = self._behavior_net(state).gather(1, action.type(torch.long))
        with torch.no_grad():
            if self.use_ddqn:
                index = self._behavior_net(next_state).argmax(1).view(-1, 1)

                q_next = self._target_net(next_state).gather(1, index)
            else:
                q_next = self._target_net(next_state).max(1)[0].view(-1, 1)

            q_target = reward + (1 - done) * gamma * q_next

        loss = self._criterion(q_value, q_target)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)

        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''

        # TODO DQN _update_target_network
        self._target_net.load_state_dict(OrderedDict(self._behavior_net.state_dict()))

    def save(self, model_path: str, checkpoint: bool = False) -> None:
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path: str, checkpoint: bool = False) -> None:
        model = torch.load(model_path)

        self._behavior_net.load_state_dict(model['behavior_net'])

        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args: argparse.Namespace, env: TimeLimit, agent: DQN, writer: SummaryWriter) -> None:
    print('Start Training')

    action_space = env.action_space
    total_steps, epsilon = 0, 1.0
    ewma_reward = 0

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)

            # execute action
            next_state, reward, done, _ = env.step(action)

            # store transition
            agent.append(state, action, reward, next_state, done)

            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1

            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, total_steps)

                print(f'Step: {total_steps}\tEpisode: {episode}\tLength: {t:3d}\tTotal reward: {total_reward:.2f}\tEwma reward: {ewma_reward:.2f}\tEpsilon: {epsilon:.3f}')

                break
    env.close()


def test(args: argparse.Namespace, env: Any, agent: DQN, writer: SummaryWriter) -> None:
    print('Start Testing')

    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)

        state = env.reset()

        # TODO test
        while True:
            if args.render:
                env.render()

            action = agent.select_action(state, epsilon, action_space)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f'Episode: {n_episode}, Reward: {total_reward:.3f}')
                rewards.append(total_reward)

                break

    print('Average Reward', np.mean(rewards))

    env.close()


def main() -> None:
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    parser.add_argument('--use_ddqn', action='store_true')
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)

    if not args.test_only:
        train(args, env, agent, writer)

        agent.save(args.model)

    agent.load(args.model)

    test(args, env, agent, writer)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main()
