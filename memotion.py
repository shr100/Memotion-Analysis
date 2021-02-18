# To unzip data file

# from zipfile import ZipFile
# with ZipFile("data_7000_new.csv.zip",'r') as zip:
#   zip.extractall()
#   print("Done")

#Authors: Shruthi Ravishankar, Afia Anjum 

#importing various modules and libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import warnings
import nltk
import csv
import pandas as pd
from pandas import DataFrame
import seaborn as sn
from collections import defaultdict
import string
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import math
import operator
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#method to clean the puctuations from the text
def clean(s):
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator)

#this method handles tokenization(from nltk) by removing or by keeping the nltk stopwords
def tokenize(text, remove_stop_words):
    text = clean(text).lower()
    if remove_stop_words:
        stop_words = set(stopwords.words('english')) 
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if not token in stop_words]
        return tokens
    return word_tokenize(text)

#this method  
def cleaned(pos,neg) :
    pos1 = []
    neg1 = []
    op = open('test.txt','w')
    for item in pos :
        string = clean(item).lower()
        pos1.append(string)
    for item in neg :
        string = clean(item).lower()
        neg1.append(string)
    return pos1,neg1


#this method reads data from the given trial data(data1.csv) and extracts the texts and category of sentiments
#pos and neg are two of the lists that contains all the texts corresponding to a specific label, that is pos is for the positive
#training texts and neg is for the negative training texts.

def extract_test_data():
  pos=[]
  neg=[]
  columns = defaultdict(list)
  with open('data1.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(v) # append the value into the appropriate list
                                     # based on column nam

  for i in range(len(columns['Overall_Sentiment'])) :
      if (columns['Overall_Sentiment'][i] == 'very_positive') or (columns['Overall_Sentiment'][i] == 'positive') :
          if (columns['OCR_extracted_text'][i]!=""):
              pos.append(str(columns['OCR_extracted_text'][i]))
      if (columns['Overall_Sentiment'][i] == 'very_negative') or (columns['Overall_Sentiment'][i] == 'negative') :
          if (columns['OCR_extracted_text'][i]!=""):
              neg.append(str(columns['OCR_extracted_text'][i]))

  return pos,neg

#this method reads data from the given trained file(data_7000_new.csv) and extracts the texts and category of sentiments
#pos and neg are two of the lists that contains all the texts corresponding to a specific label, that is pos is for the positive
#training texts and neg is for the negative training texts.
  
def extract():
    pos = []
    neg = []

    columns = defaultdict(list) # each value in each column is appended to a list
    colnames=['Image_name', 'Image_URL', 'OCR_extracted_text', 'Corrected', 'Humour', 'Sarcasm', 'Offensive', 'Motivation', 'Overall_Sentiment']
    user1 = pd.read_csv('data_7000_new.csv', names=colnames, header=None)
    user1.to_csv('new.csv',index = False)
    
    with open('new.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(v) # append the value into the appropriate list
                                     # based on column nam
    
    for i in range(len(columns['Overall_Sentiment'])) :
        if (columns['Overall_Sentiment'][i] == 'very_positive') or (columns['Overall_Sentiment'][i] == 'positive') :
            if (columns['OCR_extracted_text'][i]!=""):
                pos.append(str(columns['OCR_extracted_text'][i]))
        if (columns['Overall_Sentiment'][i] == 'very_negative') or (columns['Overall_Sentiment'][i] == 'negative') :
            if (columns['OCR_extracted_text'][i]!=""):
                neg.append(str(columns['OCR_extracted_text'][i]))

    return pos,neg

#this method computes the precision and recall scores
def compute_precision_recall(tn,fp,fn,tp):
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  precision1 = tn / (tn + fn)
  recall1 = tn / (tn + fp)
  return precision, recall,precision1,recall1

#this method is used to evaluate accuracy scores manually
def evaluate_accuracy(predictions, actual):
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            accuracy += 1
    return (accuracy * 100) / len(actual)

#this method takes in our training sample, performs a K fold cross validation on the data
# in order to find out the best parameter to train our model.
def cross_validate(Xtrain, Ytrain, K, parameters, category_labels,pos1,neg1):
    all_accs = np.zeros((len(parameters), K))

    subset_size = int(len(Xtrain) / K)
    for k in range(K):
        # Compute slices for validation and train sets
        val_indices = slice(k*subset_size, k*subset_size + subset_size)
        train_indices1 = slice(k*subset_size)
        train_indices2 = slice((k+1)*subset_size, None)

        # Printing data slices
        print("Fold " + str(k) + ": ")
        print("Data slices in the form: slice(start_index, stop_index, step_size)")
        print("Validation set: " + str(val_indices))
        print("Training set: " + str(train_indices1) + " & " + str(train_indices2))
        print()

        X_validation_set = Xtrain[val_indices]
        Y_validation_set = Ytrain[val_indices]
        X_training_set = Xtrain[train_indices1] + Xtrain[train_indices2]
        Y_training_set = Ytrain[train_indices1] + Ytrain[train_indices2]

        # Running the model for each set of params and saving the accuracy scores
        for i, params in enumerate(parameters):
            l_priors, l_likelihoods, vocab, word_counts = trainNB(Y_training_set, X_training_set, category_labels,pos1,neg1,params)
            results = testNB(X_validation_set, l_priors, l_likelihoods, vocab, category_labels, word_counts, params)

            accuracy = evaluate_accuracy(results,Y_validation_set)
            all_accs[i,k] = accuracy

    # Go through all accuracies and pick the best one
    avg_accs = np.mean(all_accs, axis=1)
    best_params = parameters[0]
    best_acc = 0
    for i, params in enumerate(parameters):
        avg_acc = avg_accs[i]
        print('Cross validate parameters:', params)
        print('average accuracy:', avg_acc)

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = params

    return best_params



## Training algorithm of Naive Bayes is implemented as described in Fig. 4.2 of the textbook ##
def trainNB(categories, texts, category_labels,pos1,neg1,params):
    N = len(texts) # No. of training documents

    # Separate lists of texts for each category
    #tech, business, entertainment, politics, sport = split_category_texts(categories, texts)

    category_texts = [pos1, neg1]

    log_class_priors = {}
    word_counts = {}    # Holds dictionary of words and their counts across all texts for each category
    vocab = set()       # Holds the vocabulary of words found in all training texts
    for i in range(len(category_texts)):
        category = category_labels[i]
        texts = category_texts[i]

        # Calculating (log) prior probabilities
        log_class_priors[category] = math.log(len(texts) / N)
        
        # Calculating word counts for the given class and building training vocab
        word_counts[category] = {}
        for j in range(len(texts)):
            text = texts[j]

            word_tokens = tokenize(text, params['remove_stop_words'])
            w_counts = get_word_counts(word_tokens)
            
            for word, count in w_counts.items():
                if word not in vocab:
                    vocab.add(word)
                word_counts[category][word] = word_counts[category].get(word, 0.0) + count

    # Calculating loglikelihood for each word in the vocubulary
    log_likelihoods = {}
    for category in category_labels:
        log_likelihoods[category] = {}
        for word in vocab:
            log_likelihoods[category][word] = math.log((word_counts[category].get(word, 0) + 1) / (sum(word_counts[category].values()) + len(vocab)))
            
    return log_class_priors, log_likelihoods, vocab, word_counts

## Testing algorithm of Naive Bayes is implemented as described in Fig. 4.2 of the textbook ##

def testNB(texts, logpriors, loglikelihoods, vocab, category_labels, word_counts, params):
    results = []
    for i in range(len(texts)):
        logprob_scores = {}
        for category in category_labels:
            logprob_scores[category] = logpriors[category]
            words = tokenize(texts[i], params['remove_stop_words'])
            for word in words:
                if word not in vocab:
                    if params['remove_unknown_words']:
                        logprob_scores[category] += math.log(1 / (sum(word_counts[category].values()) + len(vocab)))
                    else: continue
                else:
                    logprob_scores[category] += loglikelihoods[category][word]        
        chosen_category = max(logprob_scores.items(), key=operator.itemgetter(1))[0]
        results.append(chosen_category)

    return results

#this function returns the word counts 
def get_word_counts(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts


def main():

    pos,neg = extract()
    pos1,neg1= cleaned(pos,neg)
    #two labels are defined, positive & the negative
    category_labels = ('pos1', 'neg1')
    
    Xtrain=[]
    Ytrain=[]
    for i in range(len(pos1)):
        category = "pos1"
        Xtrain.append(pos1[i])
        Ytrain.append(category)

    for i in range(len(neg1)):
        category = "neg1"
        Xtrain.append(neg1[i])
        Ytrain.append(category)
    
    #parameters to tune
    params = [
        { 'remove_stop_words': True, 'remove_unknown_words': True },
        { 'remove_stop_words': False, 'remove_unknown_words': False },
        { 'remove_stop_words': True, 'remove_unknown_words': False },
        { 'remove_stop_words': False, 'remove_unknown_words': True },
    ]
    
    #returns the best parameter that would be used to build our model
    best_params = cross_validate(Xtrain, Ytrain, 3, params, category_labels,pos1,neg1)
    print("\nThe best parameters as a result of cross-validation are:", best_params)

    
    ###TRAINING###

    log_class_priors, log_likelihoods, vocab,word_counts=trainNB(Ytrain, Xtrain, category_labels,pos1,neg1,best_params)

    ###TESTING###
    p_test,n_test=extract_test_data()

    postest,negtest= cleaned(p_test,n_test)

    Xtest=[]
    Ytest=[]
    for i in range(len(postest)):
        category = "pos1"
        Xtest.append(postest[i])
        Ytest.append(category)

    for i in range(len(negtest)):
        category = "neg1"
        Xtest.append(negtest[i])
        Ytest.append(category)
    
    #predicted results on the given trial data
    results=testNB(Xtest, log_class_priors, log_likelihoods, vocab, category_labels,word_counts,best_params)
    
    #calculating TruePositive, FalsePositive, TrueNegative & FalseNegative from our confusion matrix
    tn, fp, fn, tp = confusion_matrix(Ytest, results).ravel()
    print(tn, fp, fn, tp)
    
    
    # Evaluation metrics
    acc = accuracy_score(Ytest, results) * 100
    print("\nAccuracy : " +str(acc) )
    
    # Computing precision and recall scores
    precision,recall,precision1,recall1 = compute_precision_recall(tn,fp,fn,tp)
    print("\nprecision :", precision)
    print("recall :", recall)
    
    #computing F1 score 
    # what should we be most concerned about? recall or precision?
    # a negative meme getting identified as positive is a problem in our scenario, we want
    # to penalize more in this situation so we would favour fp rates, and so the precision.
    
    beta=0.5
    # beta less than 1 to favour precision
    F1=((beta**2+1)*precision*recall)/(((beta**2)*precision)+recall)
    print("\nF1 Score for Naive Bayes is :", F1*100)

    #Preprocessing the predicted category and actual category labels in order to draw the Receiver Operating Characteristics(ROC) curve
    #for Naive Bayes
    ytest=[]

    for i in range(len(Ytest)):
      if Ytest[i]=="pos1":
        ytest.append(1)
      if Ytest[i]=="neg1":
        ytest.append(0)
      
    results1=[]

    for i in range(len(results)):
      if results[i]=="pos1":
        results1.append(1)
      if results[i]=="neg1":
        results1.append(0)
    
    #Drawing the ROC curve for Naive Bayes
    fpr, tpr, _ = roc_curve(ytest, results1)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for Naive Bayes')
    plt.legend(loc="lower left")
    plt.show()
    

    # run block of code and catch warnings
    with warnings.catch_warnings():
      # ignore all caught warnings
      warnings.filterwarnings("ignore")
      
      ###LOGISTIC REGRESSION###
      
      #preprocessing the train and test data to run our Logistic Regression model from sklearn
      ytrain=[]
      for i in range(len(Ytrain)):
        if Ytrain[i]=="pos1":
          ytrain.append(1)
        if Ytrain[i]=="neg1":
          ytrain.append(0)
      
      #transforming our word features into vectors
      vect = CountVectorizer(min_df=5, ngram_range=(1, 1))
      X_train = vect.fit(Xtrain).transform(Xtrain)
      X_test=vect.transform(Xtest)
      
      #defining parameters for our Logistic Regression and performing 3-fold cross validation 
      #with the help of Grid Search to determine the best parameter
      param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
      grid = GridSearchCV(LogisticRegression(), param_grid, cv=3)
      grid.fit(X_train, ytrain)
      best_score = grid.best_score_ * 100
      print("Best cross-validation score for logistic regression : " +str(best_score))
      print("Best parameters: ", grid.best_params_)
      print("Best estimator: ", grid.best_estimator_)


      #Training our model with the best estimator and determining accuracy scores from the predictions 
      clflr=LogisticRegression(C=0.001,random_state=0,multi_class='ovr',fit_intercept=True)
      clflr.fit(X_train, ytrain)
      res=clflr.predict(X_test)
      log_score=accuracy_score(res, ytest)*100
      print("Logistic Regression Accuracy : " +str(log_score)) 

      #calculating the TruePositive, FalsePositive, TrueNegative & FalseNegative from our confusion matrix
      tn, fp, fn, tp = confusion_matrix(ytest, res).ravel()
      print(tn, fp, fn, tp)
      
      #calculating precision and recall for LR
      precision,recall,precision1,recall1 = compute_precision_recall(tn,fp,fn,tp)
      print("\nprecision :", precision)
      print("recall :", recall)

      #F1 score for LR
      beta=0.5
      # beta less than 1 to favour precision
      F1=((beta**2+1)*precision*recall)/(((beta**2)*precision)+recall)
      print("\nF1 Score for Logistic Regression:", F1*100)



main()
