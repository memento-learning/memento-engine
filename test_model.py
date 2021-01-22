import unittest
import numpy as np
import torch
import math
from model import SM2

class TestSM2(unittest.TestCase):

    def setUp(self):

        self.sample = {
            'hist_p' : torch.tensor([
                [-1] * 400,
                [0] + [-1] * (400-1),
                [1] + [-1] * (400-1),
                [1,.3,.4,.7,.7,1,1] + [-1] * (400-7),
                [.7,.7,.7,.7,.7,.9,.8] + [-1] * (400-7)
            ])
        }
    
    def test_pytorch_vanilla_parity(self):
        hist_p = self.sample['hist_p']
        h_vanilla = []
        for i in range(len(hist_p)):
            I = 1
            EF = 2.5
            n = 0
            for p in hist_p[i]:
                if p.item() == -1:
                    break
                q = p.item()*5.0
                if q >= 3:
                    if n == 0:
                        I = 1
                    elif n == 1:
                        I = 6
                    else:
                        I = I * EF

                    if I > 274.0:
                        I = 274.0
                    if I < 1.0:
                        I = 1.0
                    
                    EF = EF + (0.1 - (5-q) * (.08 + (5-q) * 0.02))
                    if EF < 1.3:
                        EF = 1.3
                    n += 1
                else:
                    n = 0
                    I = 1
            h_vanilla.append(-I/math.log2(0.5))
        
        h_pytorch = SM2()(self.sample)

        for i in range(len(h_vanilla)):
            assert abs(h_vanilla[i] - h_pytorch[i]) < 1e-3, "Sample {} fails parity:\n{}\n{}".format(i, h_vanilla[i], h_pytorch[i])

if __name__ == "__main__":
    unittest.main()