#!/usr/bin/env python3
from code.testmodels.runtest import test

import torch

def run():
    test("code/mixlc/models/generator11.pt", "data")

if __name__ == "__main__":
    #run()
    print(torch.device("cuda:0"))