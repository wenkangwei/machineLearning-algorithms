import  numpy  as np
import matplotlib.pyplot as plt


def loadData(filename, deliminator= "", labels=False):
    # obtain file
    data_arr =[]
    label_arr= []
    fp = open(filename)
    lines = fp.readlines()
    for line in lines:
        # obtain array
        words = line.split(deliminator)
        if labels == False:
            data_arr.append( list(map(float, words)))
        else:
            data_arr.append( list(map(float,words[0:-2])))
            label_arr.append(float(words[-1]))

    return np.mat(data_arr), np.mat(label_arr)


def elimiNan(data_arr):
    # loop through each column to cal mean
    for i in range(data_arr.shape[1]):
        m = np.mean(data_arr[np.nonzero(~np.isnan(data_arr[:,i].A))[0],i])
        NaN_ind = np.nonzero(np.isnan(data_arr[:,i].A))[0]
        data_arr[NaN_ind,i] = m
    #     check data
    if len(np.nonzero(np.isnan(data_arr.A))[0])>0:
        print("NaN data exists")
    else:
        print("NaN data is cleared")
    return data_arr



def cal_pca(data_arr, features=1000):
    """

    :param data_arr:
    :param features:
    :return:
        red_data： dimension-reduced data set
        eigenVal: eigenvalue
        eigenVec: eigenvecture
        reconMat: new data space


    """
    # calculate the mean of all examples x[i] along the vertical axis
    # data_arr shape: mxn
    # mean shape: 1xn
    # Note: in mean function axis=0: calculate mean of  each column along rows.
    # axis=1: cal mean of each row along columns
    m = np.mean(data_arr,axis=0)
    # calculate the variance respected to each example vector x[i]
    # regarding each column as a variable x and cal cov(x)
    # x[i]: 1x n
    # covariance shape: nxn
    temp = data_arr -m
    covariance =np.cov(temp, rowvar=0)

    print("var shape",covariance.shape)

    # calculate eigenvector and eigenvalue
    eigenVal, eigenVec = np.linalg.eig(covariance)
    # sort and select the top k eigenvectors
    ind = np.argsort(eigenVal)
    sorted_eigvec = eigenVec[:,ind[:-features-1:-1]]
    # calculate new data array
    # data_arr : mxn. sorted_eigvec: 20xn
    # red_data: mx20
    red_data =  data_arr * sorted_eigvec
    # sorted_eigvec: 20xn
    reconMat = red_data* sorted_eigvec.T  + m
    return red_data, eigenVal, eigenVec, reconMat



def eval(eigenVal,eigenVec, features=1000):
    sorted_vec =np.sort(eigenVal)[-1:-features:-1]
    val_sum = np.sum(sorted_vec)
    for val in sorted_vec:
        print("element:%s, percentage:%s %%"%(format(np.real( val),"4.1f"),format( np.real(val/val_sum)*100, "4.2f")))
    pass

def pca_test():
    # load data
    root = "../../MachineLearning-master/MachineLearning-master/input/13.PCA/"
    file = root + "secom.data"
    data_arr,label_arr = loadData(file," ",labels=False)
    print("data:",data_arr.shape)
    #eliminate the NaN data
    data_arr = elimiNan(data_arr)
    # cal PCA eigenvector and eigenvalues
    reduced_data, eigenVal, eigenVec, reconMat= cal_pca(data_arr,20)
    print("New data", reduced_data.shape)
    print("EigenVector", eigenVec.shape)
    print("EigenValue", eigenVal.shape)
    # evaluation
    eval(eigenVal,eigenVec,20)

    show_picture(data_arr,reconMat)
    pass



def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')

    plt.show()

if __name__ == "__main__":
    pca_test()
    pass
