import scipy
from scipy import sparse
import numpy as np
from collections import Counter
import string

# Provided utility functions

def load_data(file_name: str) -> list:
    '''
    @input:
     file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
     a list of sentences
    '''
    with open(file_name, "r") as file:
        sentences: list = file.readlines()
    return sentences


def tokenize(sentence: str) -> list:
    # Convert a sentence into a list of words
    wordlist: list = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(' ')

    return [word.strip() for word in wordlist]


class data_processor:
    '''
    Please do NOT modify this class.
    This class basically takes any FeatureExtractor class, and provide utility functions
    1. to process data in batches
    2. to save them to npy files
    3. to load data from npy files.
    '''
    # This class
    def __init__(self,feat_map):
        self.feat_map = feat_map

    def batch_feat_map(self, sentences):
        '''
        This function processes data according to your feat_map. Please do not modify.

        :param sentences:  A single text string or a list of text string
        :return: the resulting feature matrix in sparse.csc_array of shape d by m
        '''
        if isinstance(sentences, list):
            X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in sentences])
        else:
            X = self.feat_map(sentences)
        return X

    def load_data_from_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            if data.shape == ():
                X = data[()]
            else:
                X = data
        return X

    def process_data_and_save_as_file(self,sentences,labels, filename):
        # The filename should be *.npy
        X = self.batch_feat_map(sentences)
        y = np.array(labels)
        with open(filename, 'wb') as f:
            np.save(f, X, allow_pickle=True)
        return X, y


# Classes to be implemented

class feature_extractor:
    def __init__(self, vocab, tokenizer):
        self.tokenize = tokenizer
        self.vocab = vocab  # This is a list of words in vocabulary
        self.vocab_dict = {item: i for i, item in enumerate(vocab)}  # This constructs a word 2 index dictionary
        self.d = len(vocab)

    def bag_of_word_feature(self, sentence: str) -> sparse.csc_array:
        '''
        Bag of word feature extactor. Reminder: The 
        `vocab_dict` has the following format:
        {word1: 0, word2: 1, word3: 2, ...} where
        word1, word2, word3, ... are the words in the vocabulary
        0, 1, 2, ... are the indices of the words in the vocabulary (aka the indicies of the feature vector)
        
        :param sentence: A text string representing one "movie review"
        :return: The feature vector in the form of a "sparse.csc_array" with shape = (d,1)
        '''
        
        #* Tokenize and sanitize the sentence
        wordList: list = self.tokenize(sentence)
        
        #* Count the number of times each word appears in the sentence
        wordCount: dict = Counter(wordList)
        
        #* Create the lil vector with the len of the different words in the sentence
        t: sparse.lil_array = sparse.lil_array((self.d, 1), dtype=np.int64) 
        
        #* Populate the sparse vector with the count of each word in the sentence
        for word, count in wordCount.items():
            # Check of the word is in the vocab list that we care about
            if word in self.vocab_dict:
                # Populate the sparse vector with the count of the word at the index of the word defined in the vocab list
                t[self.vocab_dict[word], 0] = count
        
        #* Convert the lil vector to a csc vector
        x: sparse.csc_array = t.tocsc()
        
        #* Sanity check
        assert x.shape == (self.d, 1), "bag_of_word_feature::The shape of the sparse vector is not the same as the vocab list: %s." % str(x.shape)
        
        #* Return the sparse feature vector
        return x

    def __call__(self, sentence):
        # This function makes this any instance of this python class a callable object
        return self.bag_of_word_feature(sentence)

# class custom_feature_extractor(feature_extractor):
#     '''
#     This is a template for implementing more advanced feature extractor
#     '''
#     def __init__(self, vocab, tokenizer, other_inputs=None):
#         super().__init__(vocab, tokenizer)
#         # TODO ======================== YOUR CODE HERE =====================================
#         # Adding external inputs that need to be saved.
#         # TODO =============================================================================

#     def feature_map(self,sentence):
#         # -------- Your implementation of the advanced feature ---------------
#         # TODO ======================== YOUR CODE HERE =====================================
#         x = self.bag_of_word_feature(sentence)
#         # Implementing the advanced feature.
#         # TODO =============================================================================
#         return x

#     def __call__(self, sentence):
#         # If you don't edit anything you will use the standard bag of words feature
#         return self.feature_map(sentence)


class classifier_agent():
    def __init__(self, feat_map, params):
        '''
        This is a constructor of the 'classifier_agent' class. Please do not modify.

         - 'feat_map'  is a function that takes the raw data sentence and convert it
         into a data vector compatible with numpy.array

         Once you implement Bag Of Word and TF-IDF, you can pass an instantiated object
          of these class into this classifier agent

         - 'params' is an numpy array that describes the parameters of the model.
          In a linear classifer, this is the coefficient vector. This can be a zero-initialization
          if the classifier is not trained, but you need to make sure that the dimension is correct.
        '''
        self.feat_map = feat_map
        self.data2feat = data_processor(feat_map)
        self.batch_feat_map = self.data2feat.batch_feat_map

        self.params: np.ndarray = np.array(params)

    def score_function(self, X: sparse.csc_array) -> np.ndarray:
        '''
        This function computes the score function of the classifier.
        Note that the score function is linear in X. 
        :param X: A scipy.sparse.csc_array of size d by m, each column denotes one feature vector.
                    Thus, reach row (0...d) denotes one feature dimension, and each column (0...m) denotes one data point.
                    Thus, each column is a feature vector.
        :return: A numpy.array of length m with the score computed for each data point
        '''

        #* Check the size of the params vector against the feature vector
        (d,m) = X.shape
        d1 = self.params.shape[0]
        if d != d1:
            self.params = np.array([0.0 for _ in range(d)])
        # assert d == d1, f"score_function::The size of the params vector is not the same as the feature vector: {d},{d1}."
        # self.params = np.array([0.0 for i in range(d)]) <-- This was included in the original code, but I don't think it is correct

        #* Initialize the score vector
        s: np.ndarray = np.zeros(shape=m, dtype=float)  # this is the desired type and shape for the output
        
        #* Score each feature vector
        for i in range(m): # Loop through each row of the feature array (aka loop through each feature vector)
            temp: sparse.csc_array = X[:,[i]].T.dot(self.params) # Compute the element product of the params vector and the feature vector
            s[i] = temp.sum()
            
        # # Sanity check
        # assert s.shape == (m, ), "score_function::The score vector is not the same shape as the feature vector: %s." % str(s.shape)
        # assert len(s) == m, "score_function::The score vector is not the same length as the feature vector: %s." % str(s.shape)
            
        #* Return the score vector
        return s
    
    def predict(self, X, RAW_TEXT=False, RETURN_SCORE=False) -> np.ndarray:
        '''
        This function makes a binary prediction or a numerical score
        :param X: d by m sparse (csc_array) matrix
        :param RAW_TEXT: if True, then X is a list of text string
        :param RETURN_SCORE: If True, then return the score directly
        :return: (prediction) 1 for positive and 0 for negative
                (score) a float number for each prediction
        '''
        
        #* Turn the raw text into a feature matrix of the input is raw text
        if RAW_TEXT:
            X = self.batch_feat_map(X)
            
        #* Initialize the prediction vector
        preds: np.ndarray = np.zeros(shape=X.shape[1])
        
        #* Score the feature matrix
        scores: np.ndarray = self.score_function(X)
        
        if RETURN_SCORE:
            return scores # Sanity check not needed as the score function already has a sanity check
        else:
            for i, s in enumerate(scores):
                if s > 0:
                    preds[i] = 1
                else:
                    preds[i] = 0
            
            # # Sanity check
            # assert preds.shape == (X.shape[1],), "predict::The prediction vector is not the same size as the feature vector."
            
            return preds

    def error(self, X, y: np.ndarray, RAW_TEXT=False) -> float:
        '''
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :param RAW_TEXT: if True, then X is a list of text string,
                        and y is a list of true labels
        :return: The average error rate
        '''
        #* The predicted results
        results: np.ndarray = np.zeros(shape=len(y))
        
        #* If the input is raw text, then turn it into the predicted results
        if RAW_TEXT:
            X = self.batch_feat_map(X)
            results = self.predict(X, RAW_TEXT=False)
        else: # If the input is just the results, then use the input as the predicted results
            results = self.predict(X)
        
        #? The error rate is (num_of_wrong_predictions / total_num_of_predictions)
        
        #* Calculate the error rate
        wrongPredictions: float = 0.0
        for pred, actual in zip(results, y):
            if pred != actual:
                wrongPredictions += 1.0
        
        err: float = wrongPredictions / float(len(y))
        
        # # Sanity check
        # assert err >= 0.0 and err <= 1.0, "error::The error rate is not between 0 and 1."
        # assert type(err) == float, "error::The error rate is not a float."

        return err

    def loss_function(self, X: sparse.csc_array, y: np.ndarray) -> np.float64:
        '''
        This function implements the logistic loss at the current self.params

        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return:  a scalar, which denotes the mean of the loss functions on the m data points.

        '''

        #* Calculate the score for each feature vector
        scores: np.ndarray = self.score_function(X)
    
        #* Calculate log(hat(p))    
        m: np.ndarray = np.array(scores, copy=True)
        m[m < 0] = 0
        _sub: np.ndarray = np.add(m, np.logaddexp(np.subtract(np.zeros(shape=m.shape), m), np.subtract(scores, m)))
        log_p: np.ndarray = np.subtract(scores, _sub)
        
        #* Term 1
        term1: np.ndarray = np.negative(np.multiply(log_p, y))
        
        #* Calculate log(1 - hat(p))
        log_nP: np.ndarray = np.negative(_sub)
        
        #* Term 2
        term2: np.ndarray = np.multiply(np.subtract(np.ones(shape=y.shape), y), log_nP)
        
        #* Combine to the loss
        losses: np.ndarray = np.subtract(term1, term2)
        
        #* Calculate the average loss
        avgLoss: np.float64 = np.mean(losses)

        # # Sanity check
        # assert type(avgLoss) == np.float64, "loss_function::The average loss is not a float: %s." % str(type(avgLoss))
        
        return avgLoss
        
    def gradient(self, X: sparse.csc_array, y: np.ndarray) -> np.ndarray:
        '''
        It returns the gradient of the (average) loss function at the current params.
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return: Return an nd.array of size the same as self.params
        '''
        #* Calculate the components of the e score
        scores: np.ndarray = self.score_function(X)
        
        #* Define the constants
        _ones = np.ones(shape=y.shape)
        
        #* Calculate the terms inside to out
        _c =np.subtract(np.divide(np.exp(scores), np.add(np.exp(scores), _ones)), y)
        grad = X.multiply(_c)
        
        #* Calculate the mean gradient
        grad = np.array(grad.mean(axis=1)).flatten()
        
        # # Sanity check
        # assert grad.shape[0] == self.params.shape[0], "gradient::The gradient vector is not the same size as the params vector: %s: %s." % str(grad.shape) % str(self.params.shape)
        return grad

    def train_gd(self, train_sentences, train_labels, niter, lr=0.01, RAW_TEXT=True):
        '''
        The function should updates the parameters of the model for niter iterations using Gradient Descent
        It returns the sequence of loss functions and the sequence of training error for each iteration.

        By default the function takes raw text. But it also takes already pre-processed features,
        if RAW_TEXT is set to False.

        :param train_sentences: Training data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param train_labels: Training data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :param niter: number of iterations to train with Gradient Descent
        :param lr: Choice of learning rate (default to 0.01, but feel free to tweak it)
        :return: A list of loss values, and a list of training errors.
                (Both of them has length niter + 1)
        '''
        if RAW_TEXT:
            # the input is raw text
            Xtrain = self.batch_feat_map(train_sentences)
            ytrain = np.array(train_labels)
        else:
            # the input is the extracted feature vector
            Xtrain = train_sentences
            ytrain = train_labels

        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]

        # Solution:
        for _ in range(niter):
            #* Calculate the gradient and update the params 
            self.params = np.asarray(self.params - (lr * self.gradient(Xtrain, ytrain))).flatten()

            #* Test the new params and record
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))

        return train_losses, train_errors

    def train_sgd(self, train_sentences, train_labels, nepoch, lr=0.001, RAW_TEXT=True):
        '''
        The function should updates the parameters of the model for using Stochastic Gradient Descent.
        (random sample in every iteration, without minibatches,
        pls follow the algorithm from the lecture which picks one data point at random).

        By default the function takes raw text. But it also takes already pre-processed features,
        if RAW_TEXT is set to False.


        :param train_sentences: Training data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param train_labels: Training data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :param nepoch: Number of effective data passes.  One data pass is the same as n iterations
        :param lr: Choice of learning rate (default to 0.001, but feel free to tweak it)
        :return: A list of loss values and a list of training errors.
                (initial loss / error plus  loss / error after every epoch, thus length epoch +1)
        '''

        #* Ensure inputs are feature vectors and labels
        if RAW_TEXT:
            # the input is raw text
            Xtrain = self.batch_feat_map(train_sentences)
            ytrain = np.array(train_labels)
        else:
            # the input is the extracted feature vector
            Xtrain = train_sentences
            ytrain = train_labels

        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]


        sampler = 1/len(ytrain)
        niter = int(nepoch / sampler)

        for _ in range(nepoch): # For each epoch
            for _ in range(niter): # We run through n iterations
                
                #* Choose a random datapoint to train on    
                idx = np.random.choice(len(ytrain), 1)
                xpoint = Xtrain[:,idx]
                ypoint = ytrain[idx]
                
                #* Calculate the gradient and update the params
                self.params = np.asarray(self.params - (lr * self.gradient(xpoint, ypoint))).flatten()
                

            #* For each epoch, test the new params and record
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))


        return train_losses, train_errors

    def eval_model(self, test_sentences, test_labels, RAW_TEXT=True):
        '''
        This function evaluates the classifier agent via new labeled examples.
        Do not edit please.
        :param test_sentences: Test data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param test_labels: Test data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :return: error rate on the input dataset
        '''

        if RAW_TEXT:
            # the input is raw text
            X = self.batch_feat_map(test_sentences)
            y = np.array(test_labels)
        else:
            # the input is the extracted feature vector
            X = test_sentences
            y = test_labels

        return self.error(X, y)

    def save_params_to_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_params_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)
