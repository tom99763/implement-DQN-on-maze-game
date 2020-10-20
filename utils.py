from collections import namedtuple
import random
import torch

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    def push(self, experience):
        #if not pass capacity,add into memory
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        #if achieve capacity,update knowing experience
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
    def extract_tensors(self,experiences):
        # Convert batch of Experiences to Experience of batches
        batch = self.experience(*zip(*experiences))
        s = torch.cat(batch.s).float()
        a = torch.cat(batch.a).float()
        r = torch.cat(batch.r).float()
        s_ =torch.cat(batch.s_).float()
        return s,a,r,s_