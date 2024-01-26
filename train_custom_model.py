from classifier import load_data, tokenize, data_processor
from classifier import classifier_agent
from classifier import custom_feature_extractor


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm as t
from time import time


def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    print("Loading and processing data ...")

    sentences_pos = load_data("data/training_pos.txt")
    sentences_neg = load_data("data/training_neg.txt")

    train_sentences = sentences_pos + sentences_neg

    train_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    sentences_pos = load_data("data/test_pos_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]   

    feat_map = custom_feature_extractor(vocab_list, tokenize)

    # Preprocess the training data into features
    text2feat = data_processor(feat_map)
    
    Xtrain, ytrain = text2feat.process_data_and_save_as_file(train_sentences, train_labels,
                                            "custom_feat_train.npy")
    _, _ = text2feat.process_data_and_save_as_file(test_sentences,test_labels,
                                            "custom_feat_test.npy")


    # train with SGD
    _NEPOCH = 10
    d = len(vocab_list)
    params = np.array([0.0 for _ in range(d)])
    custom_classifier = classifier_agent(feat_map, params)
    
    _cc_epoch = np.array([i for i in range(1, _NEPOCH+1)])
    
    _cc_err = np.zeros(_NEPOCH, dtype=np.float64)

    _cc_time = np.zeros(_NEPOCH, dtype=np.float64)    
    

    
    for i in t(range(_NEPOCH), desc= "Training with custom feature extractor"):
        sTime = time()
        custom_classifier.train_sgd(Xtrain, ytrain, i, 0.01, RAW_TEXT = False)
        _cc_time[i] = time() - sTime
        _cc_err[i] = custom_classifier.eval_model(test_sentences,test_labels)
    

    print("Saving the model to a file...")
    custom_classifier.save_params_to_file('custom_model.npy')
    
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
        
    fig.savefig("Images/cc_results.png", pad_inches=0.1)


if __name__ == "__main__":
    main()
