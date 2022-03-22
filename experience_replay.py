import numpy as np
import random
from collections import namedtuple, deque
import torch
# prioritized experience replay
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, experiences_per_sampling):
        '''
        Only stroes the last N experience tuples in the replay memory

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            EXPERIENCES_PER_SAMPLING(int): number of experiences
        '''
        # Initialize replay memory
        self.acion_size = action_size
        self.buffer_size = buffer_size
        ##### Prioritization
#         self.memory = deque(maxlen=buffer_size) # set N memory size
        #####
        self.batch_size = batch_size
        # build named experience tuples
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        ##### Prioritization
        self.experiences_per_sampling = experiences_per_sampling
        
        self.alpha = 0.5
        self.alpha_decay_rate = 0.99
        self.beta = 0.5
        self.beta_growth_rate = 1.001
        self.experience_count = 0
        self.data = namedtuple("Data", field_names=["priority", "probability", "weight","index"])
        
        indexes = []
        datas = []
        for i in range(buffer_size):
            indexes.append(i)
            d = self.data(0,0,0,i)
            datas.append(d)
        
        self.memory = {key: self.experience for key in indexes}
        self.memory_data = {key: data for key,data in zip(indexes, datas)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):
            N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            if self.weights_max > 0:
                updated_weight = ((N * updated_priority)**(-self.beta))/self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight


            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority**self.alpha - old_priority**self.alpha
            updated_probability = td[0]**self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index) 
            self.memory_data[index] = data

    def update_memory_sampling(self):
        """Randomly sample X batches of experiences from memory."""
        # X is the number of steps before updating memory
        self.current_batch = 0
        values = list(self.memory_data.values())
        random_values = random.choices(self.memory_data, 
                                       [data.probability for data in values], 
                                       k=self.experiences_per_sampling)
        self.sampled_batches = [random_values[i:i + self.batch_size] 
                                    for i in range(0, len(random_values), self.batch_size)]

    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority**self.alpha
        sum_prob_after = 0
        for element in self.memory_data.values():
            probability = element.priority**self.alpha / self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            if element.probability > 0.0:
                weight = ((N *  element.probability)**(-self.beta))/self.weights_max
            d = self.data(element.priority, probability, weight, element.index)
            self.memory_data[element.index] = d

        #####
        
#     def add(self, state, action, reward, next_state, done):
#         '''
#         we store the agent's experiences at each time-step, e_t = (s_t,a_t,r_t,s_(t+1))
#         '''
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.experience_count += 1
        index = self.experience_count % self.buffer_size

        if self.experience_count > self.buffer_size:
            temp = self.memory_data[index]
            self.priorities_sum_alpha -= temp.priority**self.alpha
            if temp.priority == self.priorities_max:
                self.memory_data[index].priority = 0
                self.priorities_max = max(self.memory_data.items(), key=operator.itemgetter(1)).priority

            if temp.weight == self.weights_max:
                self.memory_data[index].weight = 0
                self.weights_max = max(self.memory_data.items(), key=operator.itemgetter(2)).weight

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = self.experience(state, action, reward, next_state, done)
        self.memory[index] = e
        d = self.data(priority, probability, weight, index)
        self.memory_data[index] = d
                    
    def sample(self):
        '''
        Samples uniformly at random from D(D_t = {e_1,...,e_t}) when  performing updates
        '''
        # D
#         experiences = random.sample(self.memory, k=self.batch_size)

        #####
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        experiences = []
        weights = []
        indices = []
        
        for data in sampled_batch:
            experiences.append(self.memory.get(data.index))
            weights.append(data.weight)
            indices.append(data.index)
        
        #####
        #store in
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device) # gpu
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # return D
#         return (states, actions, rewards, next_states, dones)
        #####
        return (states, actions, rewards, next_states, dones, weights, indices)
    
    def __len__(self):
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)
        
        
        