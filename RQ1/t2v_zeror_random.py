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
from RQ1 import model
from collections import Counter
from sklearn import metrics
from RQ1.utils import file_opt
import jsonlines
from matplotlib import pyplot as plt
import seaborn as sns
import config

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
    print("********* Negative Sample Balance ***********")
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
    print("欠采样后数据集中各类的分布情况：{}".format(Counter(balanced_data["label"])))
    return balanced_data

def binary_nag_sample_balance_data(dataset):
    print("********* Negative Sample Balance ***********")
    bugs = dataset.loc[dataset["label"]==0]
    features = dataset.loc[dataset["label"]==1]
    # others = dataset.loc[train_data["label"]==2]
    min_len = min(len(bugs), len(features))

    bugs_b = bugs.sample(n=min_len, random_state=1)
    feature_b = features.sample(n=min_len, random_state=1)
    # other_b = others.sample(n=min_len, random_state=1)

    balanced_data = pd.concat([bugs_b, feature_b])
    balanced_data = balanced_data.sample(frac=1, random_state=1)
    balanced_data = balanced_data.reset_index(drop=True)
    print("欠采样后数据集中各类的分布情况：{}".format(Counter(balanced_data["label"])))
    return balanced_data

def k_split(X, y, k, total_k):
    fold_size = len(X) // total_k
    X_test = X[k*fold_size:(k+1)*fold_size]
    X_train = np.concatenate((X[:k*fold_size], X[(k+1)*fold_size:]), axis=0)
    y_test = y[k*fold_size:(k+1)*fold_size]
    y_train = np.concatenate((y[:k*fold_size], y[(k+1)*fold_size:]), axis=0)
    return X_train, X_test, y_train, y_test

def t2v(file_path, repo, time_delt, epoch, lr, batch, text_d, time_dim):
    train_data = pd.read_csv(file_path)
    train_data['datetime'] = pd.to_datetime(train_data['datetime'])
    train_data['timedelta'] = train_data['datetime'] - train_data['datetime'][0]
    train_data['timedelta'] = train_data.timedelta.dt.days // time_delt
    train_data = nag_sample_balance_data(train_data)      # 欠采样
    train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)

    lstm = model.t2v_LSTM(time_dim)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # loss function is CrossEntropyLoss

    total_k = 10
    for k in tqdm(range(total_k)):
        X = train_data.timedelta.values
        y = np.array(train_data['label'])
        X_train, X_test, y_train, y_test = k_split(X, y, k, total_k)
        X_test = torch.from_numpy(X_test).view(-1, 1, 1)
        X_train = torch.from_numpy(np.array(X_train, dtype=np.float32)).view(-1, 1,1)
        y_train = torch.from_numpy(np.array(y_train, dtype=np.int64)).view(-1, 1)

        deal_traindata1 = TensorDataset(X_train, y_train)  # deal with wordVetor and label
        load_train1 = torch.utils.data.DataLoader(dataset=deal_traindata1, batch_size=batch,
                                                  shuffle=False)  # laod data make batch
        for e in range(epoch):
            for step, (x_time, label) in enumerate(load_train1):
                label = label.view(-1)  # loss function need 1 dim! if don't do it, loss function will make error!
                x_time = x_time.float()
                output = lstm(x_time)
                optimizer.zero_grad()  # clear gradients for this training step
                loss = loss_func(output, label)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
    X_test = X_test.float()
    output_test = lstm(X_test)  # test model and print:loss and accuracy
    pred_y = torch.max(output_test, 1)[1].data.numpy()
    pred_y_np = output_test.detach().numpy()
    true_y = np.zeros((len(y_test), 3), dtype=np.int)
    for i in range(len(y_test)):
        true_y[i][y_test[i]] = 1
    fpr, tpr, _ = metrics.roc_curve(true_y.ravel(), pred_y_np.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    print(metrics.classification_report(y_test, pred_y))
    precision = metrics.precision_score(y_test, pred_y, average="macro")
    recall = metrics.recall_score(y_test, pred_y, average="macro")
    f1 = metrics.f1_score(y_test, pred_y, average="macro")
    result = {"repository": repo, "precision": precision, "recall": recall, "f1": f1, "auc": roc_auc, "fpr": fpr, "tpr": tpr}
    roc = {"repository": repo, "fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return result, roc


def zeroR(file_path, repo, time_delt, epoch, lr, batch, text_d, time_dim):
    train_data = pd.read_csv(file_path)
    train_data['datetime'] = pd.to_datetime(train_data['datetime'])
    train_data['timedelta'] = train_data['datetime'] - train_data['datetime'][0]
    train_data['timedelta'] = train_data.timedelta.dt.days // time_delt
    train_data = nag_sample_balance_data(train_data)      # 欠采样
    train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)
    X = train_data.timedelta.values
    y = np.array(train_data['label'])
    pred_y = np.zeros((len(y),1), dtype=int)        # ZeroR
    true_y = np.zeros((len(y), 3), dtype=np.int)
    for i in range(len(y)):
        true_y[i][y[i]] = 1
    pred_y_np = np.concatenate( (np.ones([len(y), 1], dtype=int), np.zeros([len(y), 2], dtype=int)), axis=1 )
    fpr, tpr, _ = metrics.roc_curve(true_y.ravel(), pred_y_np.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    print(metrics.classification_report(y, pred_y))
    precision = metrics.precision_score(y, pred_y, average="macro")
    recall = metrics.recall_score(y, pred_y, average="macro")
    f1 = metrics.f1_score(y, pred_y, average="macro")
    result = {"repository": repo, "precision": precision, "recall": recall, "f1": f1, "auc": roc_auc, "fpr": fpr, "tpr": tpr}
    roc = {"repository": repo, "fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return result, roc


def random_guessing(file_path, repo, time_delt, epoch, lr, batch, text_d, time_dim):
    train_data = pd.read_csv(file_path)
    train_data['datetime'] = pd.to_datetime(train_data['datetime'])
    train_data['timedelta'] = train_data['datetime'] - train_data['datetime'][0]
    train_data['timedelta'] = train_data.timedelta.dt.days // time_delt
    train_data = nag_sample_balance_data(train_data)      # 欠采样
    train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)
    X = train_data.timedelta.values
    y = np.array(train_data['label'])
    pred_y = np.random.randint(3,size=((len(y), 1)))      # Randon guessing
    pred_y_np = np.full([len(y), 3], 1/3)
    true_y = np.zeros((len(y), 3), dtype=np.int)
    for i in range(len(y)):
        true_y[i][y[i]] = 1
    fpr, tpr, _ = metrics.roc_curve(true_y.ravel(), pred_y_np.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y, pred_y, average="macro")
    recall = metrics.recall_score(y, pred_y, average="macro")
    f1 = metrics.f1_score(y, pred_y, average="macro")
    result = {"repository": repo, "precision": precision, "recall": recall, "f1": f1, "auc": roc_auc, "fpr": fpr, "tpr": tpr}

    roc = {"repository": repo, "fpr": fpr.tolist(), "tpr": tpr.tolist()}

    return result, roc


def plot_roc():
    with jsonlines.open("./results/t2v_result_roc.jsonl", mode="r") as reader:
        for r in reader:
             roc_t2v = r
    with jsonlines.open("./results/zeroR_result_roc.jsonl", mode="r") as reader:
        for r in reader:
             roc_zeroR = r
    with jsonlines.open("./results/random_result_roc.jsonl", mode="r") as reader:
        for r in reader:
             roc_random = r

    for i in range(len(roc_t2v)):
        plt.plot(roc_t2v[i]["fpr"], roc_t2v[i]["tpr"],  label=roc_t2v[i]["repository"])
    plt.plot(roc_random[0]["fpr"], roc_random[0]["tpr"], label="random guessing & zeroR", linestyle="--")
    plt.show()

def main():
    result = pd.DataFrame(columns=["repository", "precision", "recall", "f1", "auc", "fpr", "tpr"])
    result_zeroR = pd.DataFrame(columns=["repository", "precision", "recall", "f1", "auc", "fpr", "tpr"])
    result_random = pd.DataFrame(columns=["repository", "precision", "recall", "f1", "auc", "fpr", "tpr"])

    roc_t2v, roc_zeroR, roc_random = [], [], []

    repo_list = file_opt.read_txt(config.code_path + "/resource/repo_list.txt")
    for repo in repo_list:
        file_path = config.data_path + "/" + repo + "/" + repo.replace("/", "_") + "_t2v.csv"
        time_delt = 30
        epoch = 20
        lr = 1e-5
        batch = 16
        text_d = 100
        time_d = 40

        result = result.append(t2v(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)[0], ignore_index=True)
        result_zeroR = result_zeroR.append(zeroR(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)[0], ignore_index=True)
        result_random = result_random.append(random_guessing(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)[0], ignore_index=True)

        roc_t2v.append(t2v(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)[1])
        roc_zeroR.append(zeroR(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)[1])
        roc_random.append(random_guessing(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)[1])

    result.to_csv("./results/t2v_result.csv")
    result_zeroR.to_csv("./results/zeroR_result.csv")
    result_random.to_csv("./results/random_result.csv")


    with jsonlines.open("./results/t2v_result_roc.jsonl", mode="w") as writer:
        writer.write(roc_t2v)
    with jsonlines.open("./results/zeroR_result_roc.jsonl", mode="w") as writer:
        writer.write(roc_zeroR)
    with jsonlines.open("./results/random_result_roc.jsonl", mode="w") as writer:
        writer.write(roc_random)

def plot_f1():
    result = pd.read_csv("C:/Users/Administrator/Desktop/issue_classification/plot/rq1_data/t2v_result.csv")
    result_zeroR = pd.read_csv("C:/Users/Administrator/Desktop/issue_classification/plot/rq1_data/zeroR_result.csv")
    result_random = pd.read_csv("C:/Users/Administrator/Desktop/issue_classification/plot/rq1_data/random_result.csv")

    print(result["f1"].describe())
    print(result_zeroR["f1"].describe())
    print(result_random["f1"].describe())

    df = pd.concat([result[["f1"]], result_zeroR[["f1"]], result_random[["f1"]]], axis=1)
    df.columns = ["Time2Vec-LSTM", "ZeroR", "Random Guessing"]
    plt.figure(figsize=[4,3])
    df.boxplot(column=["Time2Vec-LSTM", "ZeroR", "Random Guessing"])
    plt.show()
    df_avg_f1 = pd.DataFrame(columns=["Time2Vec", "ZeroR", "Random Guessing"])
    df_avg_f1 = df_avg_f1.append({"Time2Vec": result["precision"].mean(),
                                  "ZeroR": result_zeroR["precision"].mean(),
                                  "Random Guessing": result_random["precision"].mean()}
                                 ,ignore_index=True)
    df_avg_f1 = df_avg_f1.append({"Time2Vec": result["recall"].mean(),
                                  "ZeroR": result_zeroR["recall"].mean(),
                                  "Random Guessing": result_random["recall"].mean()}
                                 ,ignore_index=True)
    df_avg_f1 = df_avg_f1.append({"Time2Vec": result["f1"].mean(),
                                  "ZeroR": result_zeroR["f1"].mean(),
                                  "Random Guessing": result_random["f1"].mean()}
                                 ,ignore_index=True)
    df_avg_f1 = df_avg_f1.append({"Time2Vec": result["auc"].mean(),
                                  "ZeroR": result_zeroR["auc"].mean(),
                                  "Random Guessing": result_random["auc"].mean()}
                                 ,ignore_index=True)
    df_avg_f1.index = ['Precision', 'Recall', 'F1-measure', 'AUC']

    df_avg_f1.plot(kind='bar', rot=0, fontsize=14)
    plt.show()

def plot_auc_box():
    with jsonlines.open("./results/t2v_result_roc.jsonl", mode="r") as reader:
        for r in reader:
             roc_t2v = r
    with jsonlines.open("./results/zeroR_result_roc.jsonl", mode="r") as reader:
        for r in reader:
             roc_zeroR = r
    with jsonlines.open("./results/random_result_roc.jsonl", mode="r") as reader:
        for r in reader:
             roc_random = r

    result = pd.read_csv("./results/t2v_result.csv")
    result_zeroR = pd.read_csv("./results/zeroR_result.csv")
    result_random = pd.read_csv("./results/random_result.csv")
    auc_pd = pd.DataFrame(columns=["T2V", "ZeroR", "Random Guessing"])

    for i in range(len(result)):
        auc_pd = auc_pd.append({"T2V": result.iloc[i]["auc"],
                                "ZeroR":  result_zeroR.iloc[i]["auc"],
                                "Random Guessing":  result_random.iloc[i]["auc"]},
                               ignore_index=True)
    auc_pd.boxplot(fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()
    plot_roc()
    plot_f1()
    plot_auc_box()