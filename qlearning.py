# encoding: utf8

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# 然后设置相关变量

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions available in theCartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# 下一步，我们创建一个简单的单隐层神经网络模型。

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# 接下来，配置并编译我们的代理端。我们将策略设成ε-贪心算法，并且将存储设置成顺序存储方式因为我们想要存储执行操作的结果和每一操作得到的奖励。

print 'preparing'
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

print 'fitting'
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

# 现在测试强化学习模型

print 'dtesting'
dqn.test(env, nb_episodes=50, visualize=True)

print 'done'

