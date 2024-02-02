from classifier import load_data, tokenize, data_processor
from classifier import classifier_agent
from classifier import custom_feature_extractor


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm as t
from time import time
import sys


def main():
    print("Creating a classifier agent:")

    with open('data/vocab_private.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]

    print("Loading and processing data ...")

    sentences_pos = load_data("data/training_pos_private.txt")
    sentences_neg = load_data("data/training_neg_private.txt")

    train_sentences = sentences_pos + sentences_neg

    train_labels = [1 for _ in range(len(sentences_pos))] + [0 for _ in range(len(sentences_neg))]

    sentences_pos = load_data("data/test_pos_private.txt")
    sentences_neg = load_data("data/test_neg_private.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for _ in range(len(sentences_pos))] + [0 for _ in range(len(sentences_neg))]   

    feat_map = custom_feature_extractor(vocab_list, tokenize, train_sentences)

    # Preprocess the training data into features
    text2feat = data_processor(feat_map)
    
    if(sys.argv[1] == "-l"):
        print("::Loading the saved Xtrain instead of reprocessing the data")
        Xtrain = text2feat.load_data_from_file("custom_feat_train.npy")
        ytrain = np.asarray(train_labels)
    else:
        print("::No option selected. Reprocessing the data. (Use -l to load the saved Xtrain)")
        Xtrain, ytrain = text2feat.process_data_and_save_as_file(train_sentences, train_labels,
                                            "custom_feat_train.npy")
        _, _ = text2feat.process_data_and_save_as_file(test_sentences,test_labels,
                                                "custom_feat_test.npy")


    # train with SGD on SSWE
    _NEPOCH = 10
    d = len(vocab_list)
    params = np.array([0.0 for _ in range(d)])
    custom_classifier = classifier_agent(feat_map, params)
    
    _cc_epoch = np.array([i for i in range(1, _NEPOCH+1)])
    
    _cc_err = np.zeros(_NEPOCH, dtype=np.float64)

    _cc_time = np.zeros(_NEPOCH, dtype=np.float64)    
    for i in t(range(_NEPOCH), desc= "Training with custom feature extractor"):
        _step = 1 / (np.log(i+2)*((i+2)*10))
        sTime = time()
        custom_classifier.train_sgd(Xtrain, ytrain, 1, _step, RAW_TEXT = False)
        _cc_time[i] = time() - sTime
        _cc_err[i] = custom_classifier.eval_model(test_sentences,test_labels)
        print(f"Error after epoch {i+1}: {_cc_err[i]}")

    print(f"Error of TF-IDF after training: {custom_classifier.eval_model(test_sentences, test_labels)}")

    print("Saving the model to a file...")
    custom_classifier.save_params_to_file('best_model.npy')
    
    print("Creating analysis graphs...")
    plt.style.use("dark_background")
    fig = plt.figure()
    cc_eve = fig.add_subplot(121)
    cc_evt = fig.add_subplot(122)
    
    cc_eve.plot(_cc_epoch, _cc_err, color="red")
    cc_eve.set_title("CC: Epoch (1000) vs Training Error")
    cc_eve.set_xlabel("Epochs")
    cc_eve.set_ylabel("Error (%)")
    
    cc_evt.plot(_cc_time, _cc_err, color="red")
    cc_evt.set_title("CC: Time vs Training Error")
    cc_evt.set_xlabel("Time (s)")
    cc_evt.set_ylabel("Error (%)")

    fig.tight_layout()
        
    fig.savefig("Images/cc_results_best.png", pad_inches=0.1)


if __name__ == "__main__":
    main()
