# 策略梯度下降
from argparse import Action
from turtle import shape
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import gym
import matplotlib as plt
import time


# define the policyGradient
class PolicyGradient:
    '''initilaize all variable'''

    def __init__(self, n_x, n_y, learning_rate=0.01, reward_decay=0.95):
        # number of states in the env
        self.n_x = n_x
        # number of actions in the env
        self.n_y = n_y
        # learning rate of the network
        self.lr = learning_rate
        # discount factor
        self.gamma = reward_decay

        # initialize the lists of storing oberservation
        # actions and rewards
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        # define function for build the network
        self.build_network()

        # store the cois i.e loss
        self.cost_history = []
        # initialize tensorflow session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    ''' stores the transitions, that is, state,action, and reward.'''

    def store_transition(self, s, a, r):
        self.episode_observations.append(s)
        self.episode_rewards.append(r)

        # store actions as list of arrays
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)

    '''for choosing the action which given the state'''

    def choose_action(self, observation):
        # reshapr observation to (num_feature,1)
        observation = observation[:, np.newaxis]

        # run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: observation})

        # select action using a biased sample

        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    '''building the neural network'''

    def build_network(self):
        # placeholder for input x and output y
        tf.compat.v1.disable_eager_execution()
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(self.n_x, None), name="X")
        self.Y = tf.compat.v1.placeholder(tf.float32, shape=(self.n_y, None), name="Y")

        # placeholder for reward
        self.discounted_episode_rewards_norm = tf.compat.v1.placeholder(tf.float32, [None, ], name="actions_value")

        # build 3 layers
        units_layers_1 = 10
        units_layers_2 = 10

        # number of neurons in the output layer
        units_output_layers = self.n_y

        # initialize weights and bias value using
        W1 = tf.compat.v1.get_variable("W1", [units_layers_1, self.n_x],
                                       initializer=tf.keras.initializers.glorot_normal(seed=1))
        b1 = tf.compat.v1.get_variable("b1", [units_layers_1, 1],
                                       initializer=tf.keras.initializers.glorot_normal(seed=1))
        W2 = tf.compat.v1.get_variable("W2", [units_layers_2, units_layers_1],
                                       initializer=tf.keras.initializers.glorot_normal(seed=1))
        b2 = tf.compat.v1.get_variable("b2", [units_layers_2, 1],
                                       initializer=tf.keras.initializers.glorot_normal(seed=1))

        W3 = tf.compat.v1.get_variable("W3", [self.n_y, units_layers_2],
                                       initializer=tf.keras.initializers.glorot_normal(seed=1))
        b3 = tf.compat.v1.get_variable("b3", [self.n_y, 1], initializer=tf.keras.initializers.glorot_normal(seed=1))

        # perform forward propagation
        z1 = tf.add(tf.matmul(W1, self.X), b1)
        A1 = tf.nn.relu(z1)
        z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(z2)
        z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.softmax(z3)

        # apply softmax activation function in the output layer
        logits = tf.transpose(z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        # define the loss function as cross entropy loss
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # reward guided loss
        loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)

        # use adam optimizer for minimzing the loss
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

    '''function which result in the discount and normalized reward'''

    def discount_and_norm_rewards(self):
        discounted_episodes_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episodes_rewards[t] = cumulative
        discounted_episodes_rewards -= np.mean(discounted_episodes_rewards)
        discounted_episodes_rewards /= np.std(discounted_episodes_rewards)
        return discounted_episodes_rewards

    '''perform the learning'''

    def learn(self):
        # discount and normalize episodic reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # train the nework
        self.sess.run(self.train_op, feed_dict={
            self.X: np.vstack(self.episode_observations).T,
            self.Y: np.vstack(np.array(self.episode_actions)).T,
            self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # reset the episodic data
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        return discounted_episode_rewards_norm


# initalize the env
env = gym.make('LunarLander-v2')
env = env.unwrapped
# 可视化
RENDER_ENV = True
EPISODES = 5000
rewards = []
RENDER_REWARD_MIN = 5000

PG = PolicyGradient(
    n_x=env.observation_space.shape[0],
    n_y=env.action_space.n,
    learning_rate=0.02,
    reward_decay=0.99,
)

for episode in range(EPISODES):

    # get the state
    observation = env.reset()
    episode_reward = 0

    while True:

        if RENDER_ENV: env.render()

        # choose an action based on the state
        action = PG.choose_action(observation)

        # perform action in the environment and move to next state and receive reward
        observation_, reward, done, info = env.step(action)

        # store the transition information
        PG.store_transition(observation, action, reward)

        # sum the rewards obtained in each episode
        episode_rewards_sum = sum(PG.episode_rewards)

        # if the reward is less than -259 then terminate the episode
        if episode_rewards_sum < -250:
            done = True

        if done:
            episode_rewards_sum = sum(PG.episode_rewards)
            rewards.append(episode_rewards_sum)
            max_reward_so_far = np.amax(rewards)

            print("Episode: ", episode)
            print("Reward: ", episode_rewards_sum)
            print("Max reward so far: ", max_reward_so_far)

            # train the network
            discounted_episode_rewards_norm = PG.learn()

            if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = False

            break

        # update the next state as current state
        observation = observation_

plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
