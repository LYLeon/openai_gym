import tensorflow as tf
from collections import deque
import random
import numpy as np
from gym.spaces import Box

class DQNAgent:

	def __construct_nn(self, observation_dimen):
		observation_dimen += 1 # +1 for action input
		self.observation_action_input = tf.placeholder(tf.float32, shape=(None, observation_dimen), name="input")
		self.real_reward = tf.placeholder(tf.float32,shape=(None, 1))

		hidden_1_size = 5
		hidden_2_size = 5
		
		w1 = tf.Variable(tf.truncated_normal([observation_dimen, hidden_1_size], name="w1"))
		b1 = tf.Variable(tf.zeros([hidden_1_size]), name="b1")
		hidden_layer_1 = tf.nn.relu(tf.matmul(self.observation_action_input, w1) + b1, name="hidden1")

		w2 = tf.Variable(tf.truncated_normal([hidden_2_size, 1]), name="w2")
		b2 = tf.Variable(tf.zeros([1]), name="b2")
		self.reward_prediction = tf.add(tf.matmul(hidden_layer_1, w2), b2, name="out_reward")
		cost = tf.reduce_sum(tf.pow(self.reward_prediction-self.real_reward, 2))/(2*self.batch_size)

		with tf.name_scope('rewards_cost'):
			tf.summary.scalar('rms_cost', cost)
			tf.summary.scalar('real_reward', tf.reduce_mean(self.real_reward))
			tf.summary.scalar('reward_prediction', tf.reduce_mean(self.reward_prediction))

		self.optimizer = tf.train.GradientDescentOptimizer(0.00025).minimize(cost)
		self.summary = tf.merge_all_summaries()
		self.init = tf.initialize_all_variables()
		self.sess = tf.InteractiveSession()
		self.train_summary_writer = tf.train.SummaryWriter('summary/train', self.sess.graph)
		self.sess.run(self.init)
		self.total_trained_step = 0		


	def __combine_action_observation(self, observation, action):
		return np.append(observation, action)

	def __predict_ultimate_reward(self, observation, action):
		reward = self.sess.run(self.reward_prediction, feed_dict={self.observation_action_input : [self.__combine_action_observation(observation, action)]})
		return reward[0][0]			

	def __compute_all_possible_reward(self, observation):
		action_to_rewards = [self.__predict_ultimate_reward(observation, i) for i in xrange(self.ACTION_SPACE.n)]
		return action_to_rewards

	def get_action_for(self, observation):
		if random.random() > self.EPSILON:
			self.controled_action_num += 1
			return np.argmax(self.__compute_all_possible_reward(observation))
		else :
			self.random_action_num += 1
			return self.ACTION_SPACE.sample()

	def __get_batch(self, replay, batch_size, i):
		start_idx = i * batch_size
		end_idx = start_idx + batch_size
		batch_replay = replay[start_idx:end_idx]
		return batch_replay

	def __batch_observation_action(self, replay):
		return [self.__combine_action_observation(x[0], x[1]) for x in replay]

	def __batch_reward(self, replay):
		return [[x[3] + self.REWARD_DISCOUNT*np.max(self.__compute_all_possible_reward(x[2])) if x[2]!=None else x[3]] for x in replay]

	def __train_on_replays(self):
		sample_indices = random.sample(xrange(len(self.replay)), int(self.batch_size))
		replay_batch = [self.replay[i] for i in sample_indices]
		observation_action = self.__batch_observation_action(replay_batch)
		reward = self.__batch_reward(replay_batch)
		summary, _ = self.sess.run([self.summary, self.optimizer], feed_dict={self.observation_action_input : observation_action, self.real_reward : reward})
		self.train_summary_writer.add_summary(summary, self.total_trained_step)
		self.total_trained_step += 1
		if self.total_trained_step % 500 == 0:
			if self.EPSILON > self.MIN_EPISON:
				self.EPSILON *= self.EPSILON_DECAY
			self.train_summary_writer.flush()
			print('step ', self.total_trained_step, 'epsilon', self.EPSILON)
			
			
	def observe(self, observation_before_action, action_taken, new_observation, reward):
		self.replay.appendleft((observation_before_action, action_taken, new_observation, reward))
		self.action_record[action_taken] += 1
		if len(self.replay) >= self.batch_size:
			self.__train_on_replays()

	def controlled_random_action_ratio(self):
		return float(self.controled_action_num)/float(self.controled_action_num + self.random_action_num)

	def proportion_action_taken(self):
		num_of_actions = float(np.sum(self.action_record))
		return [float(self.action_record[i])/num_of_actions for i in range(len(self.action_record))]

	def episode_end(self):
		self.controled_action_num = 0
		self.random_action_num = 0
		for i in range(len(self.action_record)):
			self.action_record[i] = 0


	def __init__(self, observation_space, action_space):
		self.REPLAY_SIZE = 600
		self.OBSERVATION_SPACE = observation_space
		self.ACTION_SPACE = action_space
		self.batch_size = 30
		self.EPSILON = 1
		self.EPSILON_DECAY = 0.991
		self.MIN_EPISON = 0.05
		self.REWARD_DISCOUNT = 0.99
		self.TRAIN_INTERVAL = self.batch_size/5
		self.action_record = [0]*action_space.n
		self.steps_since_last_train = 0
		self.random_action_num = 0
		self.controled_action_num = 0
		self.replay = deque(maxlen=self.REPLAY_SIZE)
		self.__construct_nn(observation_space.shape[0])

