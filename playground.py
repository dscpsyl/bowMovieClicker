import numpy as np
from scipy import sparse

import os

files = os.listdir("./aclImdb/train/neg/")

with open("./data/training_neg_private.txt", "w") as f:
        for file in files:
            with open("./aclImdb/train/neg/" + file) as f2:
                f.write(f2.readline() + "\n")

files = os.listdir("./aclImdb/train/pos/")

with open("./data/training_pos_private.txt", "w") as f:
        for file in files:
            with open("./aclImdb/train/pos/" + file) as f2:
                f.write(f2.readline()+ "\n")
                
files = os.listdir("./aclImdb/test/neg/")

with open("./data/test_neg_private.txt", "w") as f:
        for file in files:
            with open("./aclImdb/test/neg/" + file) as f2:
                f.write(f2.readline()+ "\n")
                
                
files = os.listdir("./aclImdb/test/pos/")

with open("./data/test_pos_private.txt", "w") as f:
        for file in files:
            with open("./aclImdb/test/pos/" + file) as f2:
                f.write(f2.readline()+ "\n")