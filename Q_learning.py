import random
import gymnasium as gym
import numpy as np
#creating an environment for agent
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
#defining the states,actions and q_tabel
state=env.observation_space.n
action=env.action_space.n
q_table=np.zeros((state,action))
#defining the episodes,learning_rate,decay_rate
total_episodes=2000
learning_rate=0.8
discount_rate=0.96
#epsilon values and decy_rate
epsilon=1.0
max_epsilon=1.0
min_epsilon=0.01
decay_rate=0.005
print('training the agent')
env=gym.make("FrozenLake-v1",is_slippery=False)
for episode in range(total_episodes):
    state,info=env.reset()
    done=False
    while not done:
        exp_tradeoff=random.uniform(0,1)
        if exp_tradeoff > epsilon:
            action=np.argmax(q_table[state,:])
        else:
            action=env.action_space.sample()
        new_state,rewards,terminate,truncate,info=env.step(action)
        done=terminate or truncate
        #formula of belman equation
        # Q(s,a) = Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (
            rewards
            + discount_rate * np.max(q_table[new_state, :])
            - q_table[state, action]
        )
        state=new_state
        #the epsilon value will slowdown at finishing of training
        epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_rate+episode)
        print("training finished")
        print("final q_tabel")
        print(q_table)
        # Watch the Trained Agent Play
        print("testing the training agent")
        env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='human')
        state,info=env.reset()
        done=False
        while not done:
            action=np.argmax(q_table[new_state,:])
            new_state,rewards,terminate,truncate,info=env.step(action)
            done=terminate or truncate
        env.close
