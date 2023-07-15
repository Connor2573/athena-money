import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Dropout
from collections import deque
from torch.optim import Adam
import numpy as np
from random import sample


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent_MK1():
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size,
                 load_models=False):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = torch.Tensor(gamma).to(dtype=torch.int32, device=device)
        self.architecture = architecture
        self.l2_reg = l2_reg

        #build models
        if load_models:
            self.online_network = self.load_model('online')
            self.target_network = self.load_model('target')
        else:
            self.online_network = self.build_model()
            self.target_network = self.build_model()

        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = torch.range(0, batch_size - 1).to(dtype=torch.int32, device=device)
        self.train = True
        
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        
    def load_model(self, model_name):
        model = torch.load('./models/savedModels/lr_' + model_name + '_mk1_' + str(30) + '.pth') #20 is the model number that is to be loaded
        return model

    def build_model(self):
        model = Sequential()
        n = len(self.architecture)
        last_out = 0
        for i, units in enumerate(self.architecture, 1):
            model.append(Linear(out_features=units, 
                                 in_features=self.state_dim if i == 1 else last_out))
            last_out = units
            model.append(nn.ReLU())
        model.append(Dropout(.1))
        model.append(Linear(out_features=self.num_actions, in_features=last_out))
        return model.to(device)
    
    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        tensor_state = torch.from_numpy(state).squeeze().to(dtype=torch.float32, device=device)
        #print(tensor_state)
        q = self.online_network(tensor_state)
        out = torch.argmax(q)#.squeeze()
        #print('out:', out)
        return out.cpu().numpy()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            #print('not enough for one batch')
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        if minibatch is None:
            #print('mini batch is none')
            return
        minibatch_list = list(minibatch)
        states, actions, rewards, next_states, not_done = minibatch_list
        states = torch.from_numpy(states).to(dtype=torch.float32, device=device)
        actions = torch.from_numpy(actions).to(dtype=torch.float32, device=device)
        rewards = torch.from_numpy(rewards).to(dtype=torch.float32, device=device)
        next_states = torch.from_numpy(next_states).to(dtype=torch.float32, device=device)
        not_done = torch.from_numpy(not_done).to(dtype=torch.float32, device=device)
        """
        got_mini_batch = False
        while not got_mini_batch:
            try:
                minibatch_list = list(minibatch)
                states, actions, rewards, next_states, not_done = minibatch_list
                states = torch.from_numpy(states)
                actions = torch.from_numpy(actions)
                rewards = torch.from_numpy(rewards)
                next_states = torch.from_numpy(next_states)
                not_done = torch.from_numpy(not_done)
                got_mini_batch = True
            except:
                #print(self.experience)
                print('OOPS SOMETHING WENT WRONG WITH MAKING THE MINIBATCH')
                print('trying again')
        """
        with torch.no_grad():
            #figure out the best_action we can do now that will result in the best next_q_value, for the online network
            next_q_values = []
            for next_state in next_states:
                tensor_state = next_state.squeeze().to(dtype=torch.float32, device=device)
                next_q_values.append(self.online_network(tensor_state))
            next_q_values_tensor = torch.stack(next_q_values)
            best_actions = torch.argmax(next_q_values_tensor, dim=1)

            #predict the target
            next_q_values_target = []
            for next_state in next_states:
                tensor_state = next_state.squeeze().to(dtype=torch.float32, device=device)
                next_q_values_target.append(self.target_network(tensor_state))
            next_q_values_target_tensor = torch.stack(next_q_values_target)
            indices = torch.stack((self.idx, best_actions), dim=1) #, torch.cast(best_actions, torch.int32)
            target_q_values = torch.Tensor(next_q_values_target_tensor[list(indices.T)])
            
            
            targets = torch.mul(torch.add(rewards, not_done), torch.mul(self.gamma, target_q_values))
            
            q_values = []
            for state in states:
                tensor_state = state.squeeze().to(dtype=torch.float32, device=device)
                q_values.append(self.online_network(tensor_state))
            q_values = torch.stack(q_values)
            actions = actions.to(torch.int32)
            q_values[[self.idx, actions]] = targets
            #NOW WE TRAIN USE THE q_values as the target
        
        train_output = []
        for state in states:
            tensor_state = state.squeeze().to(dtype=torch.float32, device=device)
            train_output.append(self.online_network(tensor_state))
        train_tensor = torch.stack(train_output)
            
        loss = self.loss_fn(train_tensor, q_values)
        loss.backward()
        self.losses.append(loss)
        self.optim.step()

        if self.total_steps % self.tau == 0:
            self.update_target()
    