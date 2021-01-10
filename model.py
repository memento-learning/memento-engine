import torch
from torch import nn
import math

class Leitner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p_a = 0.5

    def forward(self, right, wrong):
        rw = right - wrong
        I = 2.0**rw # Suggested interval by algorithim
        h_t = -I/math.log2(self.p_a) # normalize for target probability assumption
        return h_t

class SM2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p_a = 0.5
        self.MAX_SESSIONS = 400
        self.batch_shape = None

        self.I_i = None
        self.n = None
        self.EF = None
        self.EF_floor = None

    def forward(self, p):

        if self.batch_shape!=p.shape:
            self.batch_shape = p.shape
            self.I_i = torch.ones(p.shape[0]).to(p.device)
            self.n = torch.zeros(p.shape[0]).to(p.device)
            self.EF = torch.ones(p.shape[0]).to(p.device) * 2.5
            self.EF_floor = torch.ones(p.shape[0]).to(p.device) * 1.3 

        I_i = self.I_i
        I = self.I_i * 1.0
        n = self.n
        EF = self.EF
        EF_floor = self.EF_floor

        for i in range(self.MAX_SESSIONS):
            p_i = p[:, i]
            q = p_i * 5
            I_n = I * EF

            correct_mask = (q >= 3)
            break_mask = (p_i == -1)

            n_is_0 = (n == 0)
            n_is_1 = (n == 1)
            n_is_2_or_more = (n >= 2)

            I_n = I_n*n_is_2_or_more + I_i*n_is_0 + I_i*6*n_is_1
            # print(torch.isnan(I_n).any())

            I_n = I_n*correct_mask + I_i*torch.logical_not(correct_mask)
            I = I_n*torch.logical_not(break_mask) + I*break_mask
            
            I = torch.clamp(I, 1.0, 274.0)

            EF = EF + (0.1 - (5-q) * (.08 + (5-q) * 0.02))
            EF = torch.max(EF, EF_floor)

            n = (n + 1)*correct_mask + n*torch.logical_not(correct_mask)

        # for q_n in q:
        #     if q == -1:
        #         break    
        #     elif q_n >= 3:
        #         if n == 0:
        #             I = 1
        #         elif n == 1:
        #             I = 6
        #         else:
        #             I *= EF
        #         EF += (0.1 - (5-q_n) * (.08 + (5-q_n) * 0.02))
        #         EF = torch.max(EF, EF_floor)    
        #         n+=1  
        #     else:
        #         n = 0
        #         I = 1
       
        h_t = I/self.p_a # normalize for target probability assumption

        
        return  torch.clamp(h_t, 15.0 / (24 * 60), 274)
    