"""Run experiments"""
import os

# import dga_classifier.bigram as bigram
# import dga_classifier.lstm as lstm
# import dga_classifier.bigram_lstm as bigram_lstm
import dga_classifier.lstm_top_onehot as lstm_top_onehot
# import dga_classifier.doc_model as doc_model
# import dga_classifier.lstm_top as lstm_top

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def run_experiments(method, cata_split = True, multi_class = True, ratio = 0.5):
    # if method == "bigram":
    #     bigram.run(cata_split = cata_split, multi_class = multi_class, ratio = ratio)
    # if method == "lstm":
    #     lstm.run(cata_split = cata_split, multi_class = multi_class, ratio = ratio)
    # if method == "bigram_lstm":
    #     bigram_lstm.run(cata_split = cata_split, multi_class = multi_class, ratio = ratio)
    # if method == "doc":
    #     doc_model.run(cata_split = cata_split, multi_class = multi_class, ratio = ratio)
    if method == "lstm_top_onehot":
        lstm_top_onehot.run(cata_split = cata_split, multi_class = multi_class, ratio = ratio)
    # if method == "lstm_top":
    #     lstm_top.run(cata_split = cata_split, multi_class = multi_class, ratio = ratio)

        
if __name__ == "__main__":
    # method:       Model
    # cata_split:   whether split the dataset as class (set "True" when it used in open set recognition)
    # multi_class:  Multi classification (True) or two classification (False) 
    # ratio:        the ratio of class used in the training data
    run_experiments(method ="lstm_top_onehot") 
