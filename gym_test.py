# encoding: utf8

import gym
import os
import time
import random
import numpy as np


def test2():

    env = gym.make('Taxi-v2')

    Q = np.random.random([env.observation_space.n, env.action_space.n])
    alpha = 0.5
    gamma = 0.5

    for episode in range(2000):
        state = env.reset()
        G = 0
        done = False
        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, info = env.step(action)
            Q[state, action] = alpha * Q[state, action] + (1 - alpha) * (reward + gamma * np.max(Q[next_state]))
            G += reward
            state = next_state
        print 'rewards:', G

    state = env.reset()
    G = 9
    while True:
        os.system('clear')
        if G < -10:
            action = env.action_space.sample()
            G += 10
        else:
            action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)
        G += reward
        if done:
            env.reset()
            G = 0
            print 'done'
        else:
            print 'reward:', G
        env.render()
        time.sleep(.1)


def test():
    env = gym.make('LunarLander-v2')   # 选择环境
    for i_episode in range(20):
        observation = env.reset()      # 环境初始化，返回当前环境状态
        for t in range(100):
            env.render()               # 环境显示
            print(observation)
            # 从动作空间中随机采样选择动作
            action = env.action_space.sample()
            # 执行动作获得反馈,从左至右分别是状态，奖励，结束标识符，debug信息
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


def test_pacman():
    env = gym.make('MsPacman-v0')
    print env.reset()
    env.render()


test2()

