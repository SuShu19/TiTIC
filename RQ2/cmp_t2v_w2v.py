from RQ2 import t2v_w2v_lstm_k_cross, w2v_lstm_k_cross, t2v_lstm_k_cross
from RQ2.utils import file_opt
import jsonlines
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from numpy import *
from collections import Counter
import numpy as np
import pandas as pd
import config

def run():
    # print("balance,t2v: dim=40. timedelta=issue(k) - issue(0), month, w2v: num_feature=300, min_word_count=10, contex=10, sg=0, downsampling=1e-3, lstm: batch=16, epoch=20, lr=1e-5")
    repo_list = file_opt.read_txt(config.code_path+"/resource/repo_list.txt")
    result = []
    for repo in repo_list:
        print(repo)
        # 以月为单位
        time_delt = 30
        epoch = 20
        lr = 1e-5
        batch = 16
        text_d = 100
        time_d = 40

        file_path = config.data_path + "/fasttext_data/" + repo + "/" + repo.replace("/", "_") + "_t2v.csv"
        t2v_w2v_r = t2v_w2v_lstm_k_cross.main(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)
        w2v_r = w2v_lstm_k_cross.main(file_path, repo, epoch, lr, batch, text_d, time_d)
        t2v_r = t2v_lstm_k_cross.main(file_path, repo, time_delt, epoch, lr, batch, text_d, time_d)
        print({"repository": repo, "results": [t2v_w2v_r, w2v_r, t2v_r]})
        result.append({"repository": repo, "results": [t2v_w2v_r, w2v_r, t2v_r]})
    with jsonlines.open("./result/compare_each_26_repo_result_abstime_month_text100d.jsonl", mode="w") as writer:
        writer.write(result)

def plot():
    with jsonlines.open("./result/compare_each_26_repo_result_abstime_month_text100d.jsonl", mode="r") as reader:
        result = []
        for r in reader:
            result = r
    result_pd = pd.DataFrame(columns=["repository", "t2v_w2v_precision", "t2v_w2v_recall", "t2v_w2v_f1",
                                      "w2v_precision", "w2v_recall", "w2v_f1",
                                      "t2v_precision", "t2v_recall", "t2v_f1"])

    for r in result:
        max_metrics = []
        for model_type in r["results"]:
            max_f1 = 0
            max_k = 0
            for k in model_type:
                if k['f1_metrics'][-1] > max_f1:
                    max_k = model_type.index(k)
                    max_f1 = k['f1_metrics'][-1]
            max_metrics.append({"precision": model_type[max_k]["precision"],
                                "recall": model_type[max_k]["recall"],
                                "f1": model_type[max_k]["f1_metrics"]})

        result_pd = result_pd.append({"repository": r["repository"],
                                      "t2v_w2v_precision": max_metrics[0]["precision"][-1],
                                      "t2v_w2v_recall": max_metrics[0]["recall"][-1],
                                      "t2v_w2v_f1": max_metrics[0]["f1"][-1],
                                      "t2v_w2v": max_metrics[0]["f1"][-1],

                                      "w2v_precision": max_metrics[1]["precision"][-1],
                                      "w2v_recall": max_metrics[1]["recall"][-1],
                                      "w2v_f1": max_metrics[1]["f1"][-1],
                                      "w2v": max_metrics[1]["f1"][-1],

                                      "t2v_precision": max_metrics[2]["precision"][-1],
                                      "t2v_recall": max_metrics[2]["recall"][-1],
                                      "t2v_f1": max_metrics[2]["f1"][-1]},
                                     ignore_index=True)

    raise_percent = []
    for i in range(len(result_pd)):
        raise_percent.append((result_pd.loc[i]["t2v_w2v_f1"] - result_pd.loc[i]["w2v_f1"]) / result_pd.loc[i]["w2v_f1"])
    print(pd.DataFrame(raise_percent).describe())

    result_pd.index = result_pd.repository.values.tolist()
    result_pd[["t2v_w2v", "w2v"]].plot.bar()
    plt.legend(bbox_to_anchor=(1, 1.23), borderaxespad=0.)
    # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5, word=100d)")
    plt.show()
    # result_pd[["t2v_w2v_precision", "w2v_precision"]].plot.bar()
    # plt.legend(bbox_to_anchor=(0.8, 1.2), borderaxespad=0.)
    # # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5, word=100d)")
    # plt.show()

def plot2():
    # """带fasttext的结果"""
    # with jsonlines.open("C:/Users/Administrator/Desktop/issue_classification/compare_each_26_repo_result_add_fasttext_abstime_month.jsonl", mode="r") as reader:
    #     result = []
    #     for r in reader:
    #         result = r
    # result_pd = pd.DataFrame(columns=["repository", "t2v_w2v_precision", "t2v_w2v_recall", "t2v_w2v_f1",
    #                                   "w2v_precision", "w2v_recall", "w2v_f1",
    #                                   "t2v_precision", "t2v_recall", "t2v_f1",
    #                                   "fasttext_precision", "fasttext_recall", "fasttext_f1"])
    #
    # for r in result:
    #     max_metrics = []
    #     for model_type in r["results"][:3]:
    #         max_f1 = 0
    #         max_k = 0
    #         for k in model_type:
    #             if k['f1_metrics'][-1] > max_f1:
    #                 max_k = model_type.index(k)
    #                 max_f1 = k['f1_metrics'][-1]
    #         max_metrics.append({"precision": model_type[max_k]["precision"],
    #                             "recall": model_type[max_k]["recall"],
    #                             "f1": model_type[max_k]["f1_metrics"]})
    #
    #     result_pd = result_pd.append({"repository": r["repository"],
    #                                   "t2v_w2v_precision": max_metrics[0]["precision"][-1],
    #                                   "t2v_w2v_recall": max_metrics[0]["recall"][-1],
    #                                   "t2v_w2v_f1": max_metrics[0]["f1"][-1],
    #                                   "t2v_w2v": max_metrics[0]["f1"][-1],
    #
    #                                   "w2v_precision": max_metrics[1]["precision"][-1],
    #                                   "w2v_recall": max_metrics[1]["recall"][-1],
    #                                   "w2v_f1": max_metrics[1]["f1"][-1],
    #                                   "w2v": max_metrics[1]["f1"][-1],
    #
    #                                   "t2v_precision": max_metrics[2]["precision"][-1],
    #                                   "t2v_recall": max_metrics[2]["recall"][-1],
    #                                   "t2v_f1": max_metrics[2]["f1"][-1],
    #
    #                                   "fasttext_precision": r["results"][3]["precison"],
    #                                   "fasttext_recall": r["results"][3]["recall"],
    #                                   "fasttext_f1": r["results"][3]["f1"],
    #                                   "fasttext": r["results"][3]["f1"]},
    #                                  ignore_index=True)
    #
    #     result_pd.to_csv("C:/Users/Administrator/Desktop/issue_classification/plot/rq2/result_pd.csv", index=False)

    result_pd = pd.read_csv("F:/发表论文/apsec 2021/issue_classification/plot/rq2/cmp_fasttext.csv", error_bad_lines=False)

    print(result_pd["t2v_w2v_f1"].describe())
    print(result_pd["w2v_f1"].describe())
    print(result_pd["fasttext_f1"].describe())


    # w2v_raise_percent, fasttext_raise_percent, w2v_fasttext = [], [], []
    # for i in range(len(result_pd)):
    #     w2v_raise_percent.append((result_pd.loc[i]["t2v_w2v_f1"] - result_pd.loc[i]["w2v_f1"]) / result_pd.loc[i]["w2v_f1"])
    #     fasttext_raise_percent.append((result_pd.loc[i]["t2v_w2v_f1"] - result_pd.loc[i]["fasttext_f1"]) / result_pd.loc[i]["fasttext_f1"])
    #     w2v_fasttext.append((result_pd.loc[i]["w2v_f1"] - result_pd.loc[i]["fasttext_f1"]) / result_pd.loc[i]["fasttext_f1"])
    # print("t2v-w2v", pd.DataFrame(w2v_raise_percent).describe())
    # print("t2v-fasttext", pd.DataFrame(fasttext_raise_percent).describe())
    # print("w2v-fasttext", pd.DataFrame(w2v_fasttext).describe())

    # result_pd.index = result_pd.repository.values.tolist()
    # result_pd[["t2v_w2v_f1", "w2v_f1", "fasttext_f1", "t2v_f1",]].plot.bar()
    # plt.legend(bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5)")
    # plt.show()

    # result_pd.index = result_pd.repository.values.tolist()
    # result_pd[["t2v_w2v", "w2v", "fasttext",]].plot.bar()
    # plt.legend(ncol=3,bbox_to_anchor=(1, 1.1), borderaxespad=0.)
    # # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5)")
    # plt.show()


    # result_pd.index = result_pd.repository.values.tolist()
    # box_plot = result_pd[["t2v_w2v_precision", "w2v_precision", "fasttext_precision",]]
    # box_plot.columns = ["TiTIC", "W2v+LSTM", "fasttext"]
    # box_plot.boxplot(fontsize=14)
    # # plt.legend(ncol=3,bbox_to_anchor=(1, 1.1), borderaxespad=0.)
    # # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5)")
    # plt.show()
    #
    #
    # result_pd.index = result_pd.repository.values.tolist()
    # box_plot = result_pd[["t2v_w2v_recall", "w2v_recall", "fasttext_recall",]]
    # box_plot.columns = ["TiTIC", "W2v-LSTM", "fasttext"]
    # box_plot.boxplot(fontsize=14)
    # # plt.legend(ncol=3,bbox_to_anchor=(1, 1.1), borderaxespad=0.)
    # # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5)")
    # plt.show()


    result_pd.index = result_pd.repository.values.tolist()
    box_plot = result_pd[["t2v_w2v_f1", "w2v_f1", "fasttext_f1",]]
    box_plot.columns = ["TiTIC", "W2v-LSTM", "fasttext"]
    box_plot.boxplot(fontsize=14)
    # plt.legend(ncol=3,bbox_to_anchor=(1, 1.1), borderaxespad=0.)
    # plt.title("26 repo f1 compare (time=abs_month, lr=1e-5)")
    plt.show()



    df_avg_f1 = pd.DataFrame(columns=["TiTIC", "Word2Vec", "fasttext"])
    df_avg_f1 = df_avg_f1.append({"TiTIC": result_pd["t2v_w2v_precision"].mean(),
                                  "Word2Vec": result_pd["w2v_precision"].mean(),
                                  "fasttext": result_pd["fasttext_precision"].mean()}
                                 ,ignore_index=True)
    df_avg_f1 = df_avg_f1.append({"TiTIC": result_pd["t2v_w2v_recall"].mean(),
                                  "Word2Vec": result_pd["w2v_recall"].mean(),
                                  "fasttext": result_pd["fasttext_recall"].mean()}
                                 ,ignore_index=True)
    df_avg_f1 = df_avg_f1.append({"TiTIC": result_pd["t2v_w2v_f1"].mean(),
                                  "Word2Vec": result_pd["w2v_f1"].mean(),
                                  "fasttext": result_pd["fasttext_f1"].mean()}
                                 ,ignore_index=True)
    df_avg_f1.index = ['Precision', 'Recall', 'F1-measure']

    df_avg_f1.plot(kind='bar', rot=0)
    plt.show()



if __name__ == '__main__':
    run()
    # plot()
    # plot2()