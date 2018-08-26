import numpy as np
import matplotlib.pyplot as plt
import feedparser as fparse

# data:
# https://www.kaggle.com/c/email-spam/data
# https://www.kaggle.com/c/cs189-sp16-hw4-spam/data
# https://www.kaggle.com/c/naive-bayes-spam-filter/data

class NaiveBayes_Struct():
    def __init__(self,inputMat, labelMat):
        self.x = inputMat
        self.y = labelMat
        self.py1 = 0
        self.pj0 = 0
        self.pj1 = 0
        pass


def train(input_mat, labels, laplace = True,k=2.0):

    """

    :param input_mat:
        input matrix: m x n
    :param labels:
        labels matrix: 1x m
    :param laplace:
        flag of using laplace smoothing
    :return:

    calculation:
        p(y=1) = sum(i from 1 to m):[1{y==1}]/m
        p(x[:,j]|y=1) = sum(i from 1 to m)[1{x[j] ==1 ^ y[j]==1}]/ sum(i from 1 to m):[1{y==1}]
        p(x[:,j]|y=0) = sum(i from 1 to m)[1{x[j] ==1 ^ y[j]==0}]/ sum(i from 1 to m):[1{y==0}]
    """
    # # Method 1: partial vectorization method
    # py1=0.0
    # py1 = np.sum(labels) / labels.shape[1]
    # if laplace == True:
    #     pj0 = np.ones([1, input_mat.shape[1]], dtype=float)
    #     pj1 = np.ones([1, input_mat.shape[1]], dtype=float)
    #     py1_cnt = k
    #     py0_cnt = k
    # else:
    #     pj0 = np.zeros([1, input_mat.shape[1]], dtype=float)
    #     pj1 = np.zeros([1, input_mat.shape[1]], dtype=float)
    #     py1_cnt = 0.0
    #     py0_cnt = 0.0
    #
    # for i in range(input_mat.shape[0]):
    #     if labels[0,i] ==1:
    #         pj1 += input_mat[i,:]
    #         # py1_cnt += np.sum(input_mat[i,:])
    #         py1_cnt += 1.0
    #         pass
    #     else:
    #         pj0 += input_mat[i, :]
    #         # py0_cnt += np.sum(input_mat[i, :])
    #         py0_cnt += 1.0
    #         pass
    #
    # pj1 = np.log(pj1 / py1_cnt)
    # pj0 = np.log(pj0 / py0_cnt)

    # Method2: element-wise method
    pj0 = []
    pj1 = []
    # calculate p(y=1) = sum(i from 1 to m):[1{y==1}]/m
    ind_y1 = np.nonzero(labels[0,:].A == 1)[1]
    ind_y0 = np.nonzero(labels[0,:].A == 0)[1]
    m = float(labels.shape[1])
    sum_y1 = float(np.sum(labels[0,ind_y1]))
    sum_y0 = m - sum_y1
    # print("y=1 count: %d y=0 count: %d"%(sum_y1,sum_y0))
    py1 =sum_y1/m
    for j in range(input_mat.shape[1]):
        # p(j|y=1) = sum(i from 1 to m)[1{x[j] ==1 ^ y[j]==1}]/ sum(i from 1 to m):[1{y==1}]
        temp_ls = input_mat[ind_y1, :]
        # x1y1_cnt = len(np.nonzero(temp_ls[:, j].A == 1)[1])
        x1y1_cnt = np.count_nonzero(temp_ls[:, j])
        #  p(j|y=0) = sum(i from 1 to m)[1{x[j] ==1 ^ y[j]==0}]/ sum(i from 1 to m):[1{y==0}]
        temp_ls = input_mat[ind_y0, :]
        x1y0_cnt = np.count_nonzero(temp_ls[:,j])
        if laplace == False:
            pj1.append(np.log(x1y1_cnt / (sum_y1)))
            pj0.append(np.log( x1y0_cnt /(sum_y0)))
        else:
            pj1.append(np.log((x1y1_cnt+1) / (sum_y1 +k)))
            pj0.append(np.log((x1y0_cnt+1) / (sum_y0 +k)))

    print("P(y=1)= ",py1)
    print("P(j|y=1)= ",  pj1)
    print("P(j|y=0)= ",  pj0)
    return np.mat(pj0), np.mat(pj1), py1


def predict(pj0,pj1,py1,input):
    """

    :param pj0:
    p(j|y=0)
    :param pj1:
    p(j|y=1)
    :param py1:
    p(y=1)
    :return: p(y=1|x), probability of y=1

    formula:
    p(y=1|x) = p(x|y=1)*p(y=1)/p(x)
           p(x|y=1) = multiply(i from 1 to n):[p(x[i]|y=1)]
           p(x) = (multiply(i from 1 to n):[p(x[i]|y=1)* p(y=1) + p(x[i]|y=0)*p(y=0)])
    """
    # cal log(P(x|y=1)*P(y=1))
    pj0 = pj0[0,np.nonzero(input ==1)[1]]
    pj1 = pj1[0,np.nonzero(input == 1)[1]]
    px_y0 = 0.0
    px_y1 = 0.0
    for i in range(pj0.shape[1]):
        px_y0 +=pj0[0,i]
    px_y0 +=np.log((1- py1))

    for i in range(pj1.shape[1]):
        px_y1 += pj1[0,i]
    px_y1 += np.log( py1)
    # print("P(x|y=1):",px_y1)
    # print("P(x|y=0):", px_y0)
    # cal prediction
    if px_y1 > px_y0:
        prediction =1
    else:
        prediction =0
    # cal probability of a spam P(x|y=1)*P(y=1) / (P(x|y=1)*P(y=1)+P(x|y=0)*P(y=0))
    P = np.exp(px_y1) /(np.exp(px_y0)+np.exp(px_y1))
    return prediction, P

def read_texttols(dir, num =1):
    """
    :param dir_ham:
        directory name
    :param num:
        num-th file in the directory dir_ham
    :return:
    """
    import re
    # method 1: read a text file as a string
    word_ls =[]
    try:
        fp = open(dir+'{}.txt'.format(num))
    except:
        fp = open(dir+'{}.txt'.format(num), encoding='Windows 1252')

    temp_ls = re.split(r'\W+', fp.read())
    for item in temp_ls:
        if len(item) >2:
            word_ls.append(item)


    # # method 2: iterate each line in text and extend the list
    # lines = fp.readlines()
    # for line in lines:
    #     # line.split(r'\W*') split each line string according to
    #     # the unicode char, where r'\W+' is the regular expression
    #     # to split the 1~any unicode char. if there is no unicode, it won't split string
    #     temp = re.split(r'\W+', line)
    #     if len(temp) >1:
    #         temp = [item for item in temp if len(item)>2]
    #         word_ls.extend(temp)
    return word_ls



def update_dict( word_mat):
    """
    :param word_ls:
     a 1-dimension list of words of a text file
    :return:
    """
    new_set = set()
    # iterate each item in the words list
    # update the union of text files
    for ls in word_mat:
        new_set = new_set | set(ls)
    return new_set

def loadData(dir, num=20):
    """
    :param dir:
        directory of the training set
    :param num:
        the number of files to load
    :return:
        text_ls:
        list of text files
        text_labels:
        list of labels
         text_dict:
         set of dictionary
    """
    text_ls = []
    text_labels = []
    text_dict = set()
    dir_ham = dir+ 'ham/'
    dir_spam = dir + 'spam/'
    # load ham texts
    for i in range(1,num+1):
        # convert text to a word list and add to text list
        text_ls.append(read_texttols(dir_ham,i))
        text_labels.append(0)
        # update the dictionary based on the new word list

    # load spam texts
    for i in range(1,num+1):
        text_ls.append(read_texttols(dir_spam,i))
        # label spam as 1
        text_labels.append(1)

    # update dictionary
    text_dict = update_dict(text_ls)

    return text_ls, text_labels, text_dict



def get_MatData(text_ls,text_dict):
    
    # initialize training input matrix
    # m = length of text list = example amount
    # n = length of dictionary
    # inputMat: m x n zero matrix
    m = len(text_ls)
    n= len(text_dict)
    inputMat = np.zeros([m,n])
    text_dict = list(text_dict)
    # loop through each input vector 
    for i in range(m):
        # check if j-th word in dictionary exist in i-th text file
        for j in range(n):
            # if word exists in the text file, set 1.
            if text_dict[j] in text_ls[i]:
                inputMat[i, j] = 1
            else:
                inputMat[i, j] = 0
        
    return np.mat(inputMat)



def get_trainData(inputMat, text_labels,train_num =40, mode ="rand"):
    """
    :param inputMat:
    :param text_labels:
    :param train_num:
    :return:
     returns the training input, label matrices and test input, label matrices
    """
    # create sparse matrices using zero arrays
    trainMat = np.mat(np.zeros([ train_num,inputMat.shape[1]]))
    train_labels =  np.mat(np.zeros([1,train_num]))
    testMat = np.mat( np.zeros([inputMat.shape[0] - train_num, inputMat.shape[1]]))
    test_labels =  np.mat(np.zeros([1, inputMat.shape[0] - train_num]))
    if mode == 'rand':
        import random
        # obtain test example indices randomly
        test_ind = [int(num) for num in random.sample(range(inputMat.shape[0]), inputMat.shape[0] - train_num)]
        train_ind = list(set(range(inputMat.shape[0])) - set(test_ind))
    else:
        # select the middle 40 examples as training examples
        train_ind = [int(num) for num in range(5, inputMat.shape[0]-5)]
        test_ind = list(set(range(inputMat.shape[0])) - set(train_ind))

    # obtain training examples and test examples
    trainMat[:, :] = inputMat[train_ind, :]
    train_labels[0, :] = text_labels[0, train_ind]
    testMat[:, :] = inputMat[test_ind, :]
    test_labels[0, :] = text_labels[0, test_ind]

    return  trainMat, train_labels, testMat, test_labels

def NaiveBay_test():
    # data directory
    root = "../../MachineLearning-master/MachineLearning-master/input/4.NaiveBayes/email/"
    # load data into list
    text_ls, text_labels, text_dict = loadData(root,25)
    print("dictionary: ",text_dict, len(text_dict))
    print("text ls shape: ", len(text_ls))

    # generate training examples
    inputMat = get_MatData(text_ls,text_dict)
    text_labels = np.mat(text_labels)
    trainMat, train_labels, testMat, test_labels = get_trainData(inputMat, text_labels,40)

    # train Naive Bayesian model
    # p_class0: P(x|y=0).   p_class1: P(x|y=1).  p_spam: P(y=1)
    p_class0, p_class1,p_spam= train(trainMat, train_labels,laplace=True)

    # prediction
    error_cnt =0
    for i in range(testMat.shape[0]):
        p_isSpam, pb = predict(p_class0,p_class1,p_spam,testMat[i,:])
        if p_isSpam != test_labels[0,i]:
            error_cnt+=1
        print("Example: %d Pred: %d Probability of a spam: %f Label: %d"%(i, p_isSpam, pb,test_labels[0,i]))
    print("error rate of test set:", error_cnt/float(testMat.shape[0]))

    error_cnt = 0
    for i in range(inputMat.shape[0]):
        p_isSpam, pb = predict(p_class0, p_class1, p_spam, inputMat[i, :])
        if p_isSpam != text_labels[0, i]:
            error_cnt += 1
        # print("Example: %d Pred: %d Probability of a spam: %f Label: %d"%(i, p_isSpam, pb,test_labels[0,i]))
    print(" error rate of the whole set:", error_cnt / float(inputMat.shape[0]))
    # evaluation
    pass

if __name__ == "__main__":
    NaiveBay_test()
    pass
