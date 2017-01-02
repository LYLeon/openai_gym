import gym
from dqn_agent import DQNAgent

env = gym.make('CartPole-v0')
agent = DQNAgent(env.observation_space, env.action_space)

for i_episode in range(1000000):
    observation = env.reset()
    reward_sum = 0
    if i_episode % 100 == 0:
    	print('episode ', i_episode)
    for t in range(500):
        env.render()
        # print(observation)
        action = agent.get_action_for(observation)
        new_observation, reward, done, info = env.step(action)
        agent.observe(observation, action, None if done else new_observation, reward, reward_sum)
        observation = new_observation
        reward_sum += reward
        if done or t == 499:
            print('total reward this turn: %d' % reward_sum, ' action controlled: %0.2f ' % agent.controlled_random_action_ratio(), 'action proportion', agent.proportion_action_taken())
            agent.episode_end()
            break