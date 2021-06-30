#!/usr/bin/env python3
from code.testmodels.runtest import test
from code.testmodels.drawer import create_drawer
import torch

def run():
    test("mixlc_acgan", "code/mixlc_acgan/models/generator.pt", "data")
    
if __name__ == "__main__":
    #run()
    #print(torch.device("cuda:0"))
    create_drawer("data/")