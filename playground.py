import numpy as np
from scipy import sparse

from classifier import load_data,tokenize, feature_extractor, data_processor
from classifier import classifier_agent


with open('data/vocab.txt', "r") as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        
feat_map = feature_extractor(vocab_list, tokenize) # You many replace this with a different feature extractor

params = np.array([1.0 for _ in range(4)])
classifier1 = classifier_agent(feat_map,params)

# x = sparse.csc_array([[ 0.1 ,12.3 , 3.4 , 7.1 ], [ 2.3 , 4.4 , 8.9 , 0.99], [ 5.3 , 6.5 , 1.3 , 4.2 ]])
x = np.array([[ 0.1 ,12.3 , 3.4 , 7.1 ], [ 2.3 , 4.4 , 8.9 , 0.99], [ 5.3 , 6.5 , 1.3 , 4.2 ]])
print(classifier1.score_function(x))

# x = np.matrix([[ 0.1 ,12.3 , 3.4 , 7.1 ], [ 2.3 , 4.4 , 8.9 , 0.99], [ 5.3 , 6.5 , 1.3 , 4.2 ]])
# (d, m) = x.shape

# p = np.array([1, 1, 1, 1])
# p = np.tile(p, (d, 1))


# print(x, x.shape)
# print(p, p.shape)

# _t = np.multiply(x, p)
# print(_t)
# _s = _t.sum(axis=1)
# print(_s)
# print(np.asarray(_s).flatten())

