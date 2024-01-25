from classifier import load_data,tokenize, feature_extractor, data_processor
from classifier import classifier_agent

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm as t

import sys
import time

plt.style.context("dark_background")
fig = plt.figure()
gd_eve = fig.add_subplot(221)
gd_evt = fig.add_subplot(222)
sgd_eve = fig.add_subplot(223)
sgd_evt = fig.add_subplot(224)

def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    print("Loading data...")
    sentences_pos = load_data("data/training_pos.txt")
    sentences_neg = load_data("data/training_neg.txt")

    print("Creating training and test data...")
    train_sentences = sentences_pos + sentences_neg

    train_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    sentences_pos = load_data("data/test_pos_public.txt")
    sentences_neg = load_data("data/test_neg_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    print("Creeating feature map...")
    feat_map = feature_extractor(vocab_list, tokenize) # You many replace this with a different feature extractor

    print("Processing data and feature extraction...")
    text2feat = data_processor(feat_map)
    
    if(sys.argv[1] == "-l"):
        print("::Loading the saved Xtrain instead of reprocessing the data")
        Xtrain = text2feat.load_data_from_file("train_data.npy")
        ytrain = np.asarray(train_labels)
    else:
        print("::No option selected. Reprocessing the data. (Use -l to load the saved Xtrain)")
        Xtrain, ytrain = text2feat.process_data_and_save_as_file(train_sentences, train_labels, "train_data.npy")

    

    if(sys.argv[2] == "-g"):
        print("-g flag recieved. Training with gradient descent")
        niter = 1000
        print("::Training using GD for", niter, "iterations.")
        d = len(vocab_list)
        params = np.array([0.0 for _ in range(d)])
        classifier1 = classifier_agent(feat_map,params)
        classifier1.train_gd(Xtrain,ytrain,niter,0.01,RAW_TEXT=False)
        
        err1 = classifier1.eval_model(test_sentences,test_labels)
        print("GD: test err =", err1)
        
        #* Save trained params
        classifier1.save_params_to_file("trained_params_gd.npy")
        
    elif(sys.argv[2] == "-s"):
        print("-s flag recieved. Training with stochastic gradient descent")
        nepoch = 10
        print("Training using SGD for ", nepoch, "data passes (epochs).")
        d = len(vocab_list)
        params = np.array([0.0 for i in range(d)])
        classifier2 = classifier_agent(feat_map, params)
        classifier2.train_sgd(Xtrain, ytrain, nepoch, 0.001,RAW_TEXT=False)
        
        err2 = classifier2.eval_model(test_sentences,test_labels)
        print("SGD: test err =", err2)
        
        #* Save trained params
        classifier2.save_params_to_file("trained_params_sgd.npy")
        
    elif(sys.argv[2] == "-b"):
        print("-b flag recieved. Training with both gradient descent and stochastic")
        
        #* Data extraction
        _gd_epoch = np.arange(1, 1001, 1)
        _sgd_epoch = np.arange(1, 11, 1)        
        
        _NITER = 1000
        _NEPOCH = 10
        
        _gd_time = np.empty((_NITER,))
        _sgd_time = np.empty((_NEPOCH,))
        
        _gd_err = np.empty((_NITER,))
        _sgd_err = np.empty((_NEPOCH,))
        
        #* GD
        d = len(vocab_list)
        params = np.array([0.0 for _ in range(d)])
        classifier1 = classifier_agent(feat_map,params)
        _gd_basetime = time.time()
        for i in t(range(_NITER), desc= f"Training using DG for {_NITER} iterations"):
            (_, terr) = classifier1.train_gd(Xtrain,ytrain,1,0.01,RAW_TEXT=False)
            _gd_time[i] = (time.time() - _gd_basetime)
            _gd_err[i] = terr[-1]

        
        
        #* SGD
        classifier2 = classifier_agent(feat_map, params)
        _sgd_basetime = time.time()
        for i in t(range(_NEPOCH), desc= f"Training using SGD for {_NEPOCH} data passes (epochs)."): 
            (_, terr) = classifier2.train_sgd(Xtrain, ytrain, 1, 0.001,RAW_TEXT=False)
            _sgd_time[i] = (time.time() - _sgd_basetime)
            _sgd_err[i] = terr[-1]
            
        #* Plotting
        print("Plotting the results...")
        gd_eve.plot(_gd_epoch, _gd_err, color="red")
        gd_eve.set_title("GD: Epoch (1000) vs Training Error")
        gd_eve.set_xlabel("Epochs")
        gd_eve.set_ylabel("Error (%)")
        
        gd_evt.plot(_gd_time, _gd_err, color="red")
        gd_evt.set_title("GD: Time vs Training Error")
        gd_evt.set_xlabel("Time (s)")
        gd_evt.set_ylabel("Error (%)")
        
        sgd_eve.plot(_sgd_epoch, _sgd_err, color="blue")
        sgd_eve.set_title("SGD: Epoch (10) vs Training Error")
        sgd_eve.set_xlabel("Epochs")
        sgd_eve.set_ylabel("Error (%)")
        
        sgd_evt.plot(_sgd_time, _sgd_err, color="blue")
        sgd_evt.set_title("SGD: Time vs Training Error")
        sgd_evt.set_xlabel("Time (s)")
        sgd_evt.set_ylabel("Error (%)")
        
        fig.savefig("results.png", pad_inches=0.1)
        
        


if __name__ == "__main__":
    main()
