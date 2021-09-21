#-*- codeing=utf-8 -*-
#@time: 2020/7/13 20:21
#@Author: Shang-gang Lee
import numpy as np                # deal with data
import pandas as pd               # deal with data
import re                         # regular expression
from bs4 import BeautifulSoup     # resolver review
from nltk.corpus import stopwords # Import the stop word list
from gensim.models import word2vec# use word2Vec(skip-gram model) making wordfeature vetor
from sklearn.model_selection import train_test_split # use trian data split train and test data
import torch
from torch.utils.data import Dataset,TensorDataset
import torch.nn as nn
from tqdm import tqdm
from run_model.classifier import model
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import metrics

def review_to_wordlist(review,remove_stop_words=False):
    #1.remove HIML
    reivew_text=BeautifulSoup(review,'lxml').get_text()

    #2.Remove non-latters
    latters_only=re.sub("[^a-zA-Z]",' ',reivew_text)

    #3.Convert to lower case,split into individual words
    words=latters_only.lower().split()

    #4.Remove stop words
    if remove_stop_words:
        stop=set(stopwords.words('english'))
        words=[w for w in words if not w in stop]

    #5. reutrn a list of words
    return words



# make features vector by each words
def makeFeatureVec(words,model,num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords=0
    index2word_set = set(model.wv.index2word)    #get name
    for word in words:
        if word in index2word_set:               # if word in index2word and get it's feature
            nwords+=1
            featureVec=np.add(featureVec,model[word])
    if nwords == 0:
        pass
    else:
        featureVec=np.divide(featureVec,nwords)      # average each featureVector
    return featureVec

# make all word's features
def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32") #features size=len(reviews) X num_features
    for review in tqdm(reviews):                                                      # loop each review(word)
        vector = makeFeatureVec(review,model,num_features)
        reviewFeatureVecs[counter]=vector    # get each word's featureVectors
        counter+=1
    return reviewFeatureVecs


def nag_sample_balance_data(dataset):
    # print("********* Negative Sample Balance ***********")
    bugs = dataset.loc[dataset["label"]==0]
    features = dataset.loc[dataset["label"]==1]
    others = dataset.loc[dataset["label"]==2]
    min_len = min(len(bugs), len(features), len(others))

    bugs_b = bugs.sample(n=min_len, random_state=1)
    feature_b = features.sample(n=min_len, random_state=1)
    other_b = others.sample(n=min_len, random_state=1)

    balanced_data = pd.concat([bugs_b, feature_b, other_b])
    balanced_data = balanced_data.sample(frac=1, random_state=1)
    balanced_data = balanced_data.reset_index(drop=True)
    # print("欠采样后，训练集中各类的分布情况：{}".format(Counter(balanced_data.label)))
    return balanced_data

def train_w2v_vector(dataset, text_d):
    # default setting
    # num_features = 300    # Word vector dimensionality
    # min_word_count = 5   # Minimum word count
    # num_workers = 4       # Number of threads to run in parallel
    # context = 10          # Context window size
    # downsampling = 1e-3   # Downsample setting for frequent words

    # num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    word=[]
    for review in tqdm(dataset['doc']):
        if type(review) == float:
            review = ""
        word.append(review_to_wordlist(review,remove_stop_words=True))
    w2vmodel=word2vec.Word2Vec(word,workers=num_workers,size=text_d,
                            min_count=min_word_count,window=context,sample=downsampling)
    vector = getAvgFeatureVecs(reviews=word, model=w2vmodel, num_features=text_d)
    np.save("../data/w2v_vector.npy", vector)


def smote_balance_data(X_train, y_train):
    sos = SMOTE(random_state=0)
    X_train, y_train = sos.fit_resample(X_train.tolist(), y_train.tolist())
    # print("Smote过采样后，训练集中各类的分布情况：{}".format(Counter(y_train)))
    return X_train, y_train

def k_split(X, y, k, total_k):
    fold_size = len(X) // total_k
    X_test = X[k*fold_size:(k+1)*fold_size]
    X_train = np.concatenate((X[:k*fold_size], X[(k+1)*fold_size:]), axis=0)
    y_test = y[k*fold_size:(k+1)*fold_size]
    y_train = np.concatenate((y[:k*fold_size], y[(k+1)*fold_size:]), axis=0)
    return X_train, X_test, y_train, y_test

def main(file_path, repo, epoch, lr, batch, text_d, time_d):
    # 欠采样和过采样二选一
    train_data = pd.read_csv(file_path)
    train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)
    train_data = nag_sample_balance_data(train_data)  # 欠采样
    train_w2v_vector(train_data, text_d)  # 创建一次之后可以注释掉
    vector = np.load("../data/w2v_vector.npy")

    # BATCH_SIZE = 16  # batch size=64
    # Epoch = 20  # use epoch=1 for saving time and calculated amount
    # LR = 1e-5  # learning rate
    # 定义模型
    lstm = model.w2v_LSTM(text_d)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # loss function is CrossEntropyLoss

    total_k = 10
    result = []
    for k in range(total_k):
        X = vector
        y = train_data.label.values
        X_train, X_test, y_train, y_test = k_split(X, y, k, total_k)

        X_test = torch.from_numpy(X_test).view(-1, 1, text_d)
        X_train = torch.from_numpy(np.array(X_train, dtype=np.float32)).view(-1, 1, text_d)
        y_train = torch.from_numpy(np.array(y_train, dtype=np.int64)).view(-1, 1)
        deal_traindata = TensorDataset(X_train, y_train)  # deal with wordVetor and label
        load_train = torch.utils.data.DataLoader(dataset=deal_traindata, batch_size=batch,
                                                 shuffle=True)  # laod data make batch
        for e in range(epoch):
            for step, (x, label) in enumerate(load_train):
                label = label.view(-1)  # loss function need 1 dim! if don't do it, loss function will make error!
                output = lstm(x)
                optimizer.zero_grad()  # clear gradients for this training step
                loss = loss_func(output, label)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
        output_test = lstm(X_test)  # test model and print:loss and accuracy
        pred_y = torch.max(output_test, 1)[1].data.numpy()
        confusematrx = metrics.confusion_matrix(y_test, pred_y)
        precision, recall, f1 = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        for i in range(3):
            precision[i] = float(confusematrx[i][i] / (np.sum(confusematrx, axis=0))[i])
            recall[i] = float(confusematrx[i][i] / np.sum(confusematrx, axis=1)[i])
            try:
                f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
            except Exception as e:
                print(e)
        precision[3] = float(sum(precision) / 3)
        recall[3] = float(sum(recall) / 3)
        f1[3] = (2 * precision[3] * recall[3]) / (precision[3] + recall[3])

        result.append({"repository": repo, "k-cross": k, "data_len": len(y)/3, "precision":precision, "recall": recall, "f1_metrics": f1})

    return result

if __name__ == '__main__':
    main("ClickHouse/ClickHouse")