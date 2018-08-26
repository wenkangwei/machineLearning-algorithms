import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from os import  listdir
from operator import  itemgetter

class OptimStruct:
    def __init__(self, input_mat, label_mat, C=10, tol=0.1, k_info="linear"):
        """
        :param input_mat:
                mxn training example input matrix
        :param label_mat:
                mx1 training example label vector
        :param C:
                L1 normalization term parameter, controlling the constraint for the data points that are
                inside the margin, alpha=C -> y(w*x+b)<=1
        :param tol: tolerance

        :param k_info： kernel information
                "linear": linear kernel = g(x)^T*g(x)
                "Gaussian": Gaussian kernel = exp(-||x-z||^2/2sigma)

        members:
        X:  mxn input training matrix
        Y:  mx1 label vector
        m:  the number of examples
        n:  the number of feature in an example
        alpha_vec: mx1 alpha vector
        w_vec:  mx1 weight vector
        b_vec:  bias vector
        k:  kernel matrix
        C:  constant of L1 normalization term in SVM optimization problem
        E_buffer: E_Buffer[0] flag for the first time update.
                    E_buffer[1]: error E value
        SVM problems
        update alphas:
            max [sum(alpha[i], i=1,...,m ) - 1/2 *sum(i=1~m):[sum(j=1~m):[alpha[i]*alpha[j]*Y[i]*Y[j]*<X[i],X[j]>]]]
                respected to alpha
            st: 0<=alpha<= C, sum(i=1~m):(Y[i]*alpha[i])=0
        update w, b:
            w_vec = sum(i=1~m):[alpha[i]*Y[i]*X[i]]
            b_vec = (min(i:Y[i]=1):[w^T* X[i]]+max(i:Y[i]=-1):[w^T*X[i]])/2

        """
        self.X = input_mat
        self.Y = label_mat
        # shape of input data mxn
        self.m = np.shape(input_mat)[0]
        self.n = np.shape(input_mat)[1]
        # alpha vector of SVM
        self.alpha_vec = np.mat(np.zeros([self.m,1]))
        self.w_vec = np.mat(np.zeros([self.n,1]))
        self.b = 0
        self.k= get_kernel(self.X,self.X,k_info)
        self.C = C
        self.tol = tol
        self.E_buffer = np.mat(np.zeros([self.m, 2]))


def predictor(svm_obj,input, kernel= "linear"):
    """

    :param svm_obj:
    :param input:
    :param kernel:
        type of kernel used in prediction formula.
        it could be "Gaussian" or "linear"
    :return: predicted value
    prediction formula:
    y = sum[x=1~m]( y[i]*alpha[i]* <x[i], x[j]>) + b
    where:
    <x[i], x[j]> is the kernel. if kernel is linear, then
    sum[x=1~m]( y[i]*alpha[i]* <x[i], x[j]>) = sum[x=1~m]( y[i]*alpha[i]* x[i]* x[j].T) = w*x.T
    otherwise, it is not w*x.T
    """
    k = get_kernel(svm_obj.X,input,kernel)
    return  k.T* np.multiply(svm_obj.Y,svm_obj.alpha_vec) +svm_obj.b


def get_kernel(x_mat, z_mat, k_info="linear", sigma=10.0):
    # m: rows of matrix. n:cols of matrix
    x_m,x_n = np.shape(x_mat)
    z_m, z_n =np.shape(z_mat)

    kernel= np.mat(np.zeros([x_m, z_m], dtype=float))
    if k_info == "linear":
            # fully-vectorized method for computing kernel
            # kernel= np.matmul(x_mat, np.transpsvm_obje(z_mat)）
            # partially vectorized method for kernel
        for i in range(z_m):
            kernel[:,i]= np.matmul(x_mat, np.transpose(z_mat)[:,i])

    elif k_info == "Gaussian":
        k= np.matrix(np.zeros([1,x_n], dtype=float))
        # element-wise method for Gaussian kernel
        # Gaussian kernel: k[i,j]= exp(-||x[i,:]-z[j,:]||^2/2(sigma)^2)
        # where kernel rows = x rows, kernel cols = z rows.
        for i in range(z_m):
            for j in range(x_m):
                k[0]= x_mat[j,:]-z_mat[i,:]
                sq_delta= k * k.T
                kernel[j,i] = np.exp(-sq_delta/(sigma**2))
    else:
        print("warning Unknown kernel type. ")
    # print("kernel shape: %dx%d" % (kernel.shape[0], kernel.shape[1]))
    return kernel



def readImgVec(filename, size):
    """
    it reads an image matrix into a single vector
    :param filename: file name to read
    :param size: size[0]: image rows size[1]: image cols
    :return: vec : image vector
    """
    file = open(filename)
    # image vector size = row *cols
    vec=list()
    for i in range(size[0]):
        str= file.readline()
    #     convert ASCII to int vector
        for j in range(size[1]):
            if str[j] >='0' and str[j]<='9':
                vec.append(int(str[j]))
    return vec


def loadData(file_name):
    inputData = []
    labels = []
    fp = open(file_name)
    for line in fp.readlines():
        words = line.strip().split('\t')
        example = [float(words[0]), float(words[1])]
        inputData.append(example)
        labels.append(float(words[2]))
        pass
    return  np.mat(inputData), np.mat(labels)


def loadImg(dir_name, ftype='txt', d_size=[32,32]):
    """
    it reads all the files with type and marshal dataset
     then returns input matrix and a label matrix
    :param dir_name: directory name to open
    :param type: type of files it will read
    :param d_size: size of one example
    :return: inputs, labels
    """
    # load file list
    file_ls= listdir(dir_name)
    m= len(file_ls)
    print("Examples:%d"%(m))
    # initialize matrices
    inputs = []
    labels = []
    # loop through all files
    for i in range(m):
        suffix = file_ls[i].split('.')[1]
        file_name= file_ls[i].split('.')[0]
        if suffix == ftype:
            if ftype == 'txt':
                # read a file / an example
                inputs.append(readImgVec( dir_name+'/'+file_ls[i], d_size))
                if file_name.split('_')[0] == '1':
                    labels.append(1)
                else:
                    labels.append(-1)
            elif ftype == 'csv':
                pass
            pass
        pass

    inputs=np.matrix(inputs,copy=False)
    labels=np.matrix(labels,copy=False)
    print("inputs shape: %dx%d"%(inputs.shape[0],inputs.shape[1]))
    print("labels shape: %dx%d"%(labels.shape[0],labels.shape[1]))
    return inputs, labels

def cal_Ei(svm_obj, i):
    """
    it calculates the prediction value by y'=  w^T*X+b
    then find the error value Ei =  || y' - y||
    :param op_struct:
    :param i: i-th example in the matrix
    :return: Ei = || (wx+b) - y||
    """
    # calculate the predication
    # multiply: element-wise operation. matmul: matrix operation
    pred = np.multiply(svm_obj.alpha_vec, svm_obj.Y).T * svm_obj.k[:,i] + svm_obj.b
    Ei = pred - float(svm_obj.Y[i])
    return Ei

def cal_weight(svm_obj):
    """
    :param svm_obj:
    :return:
     w =  sum((a[i]*y[i])*x[i])
     where:
        a : 1xm matrix
        y: 1xm label
        x[i]: nx1 vector
        w: nx1 vector
    """
    labels = svm_obj.Y
    a = svm_obj.alpha_vec
    for i in range(svm_obj.m):
        svm_obj.w_vec += np.multiply( labels[i]*a[i] ,svm_obj.X[i].T)
    return  svm_obj.w_vec


def update_Ei(svm_obj, i):
    Ei = cal_Ei(svm_obj, i)
    svm_obj.E_buffer[i] = [1, Ei]
    return  [1, Ei]


def choose_aj(i, svm_obj, mode="rand"):
    j = i
    # calculate Ei for ai

    Ei = svm_obj.E_buffer[i,1]


    if mode == "rand":
            while j == i:
                j = np.random.randint(0, svm_obj.m - 1)
            Ej = svm_obj.E_buffer[j, 1]
    elif mode == "symE":
            j = svm_obj.m -1 -i
            if j == i:
                j = choose_aj(i, svm_obj, "rand")
            Ej = svm_obj.E_buffer[j, 1]
    elif mode == "maxE":
        j=-1
        # update Ei first before comparing the delta E
        update_Ei(svm_obj,i)
        maxDeltaE = 0.0
        InvalidLs = np.nonzero(svm_obj.E_buffer[:, 0].A)[0]
        if len(InvalidLs) > 1:
            for k in InvalidLs:
                if k != i:
                    # calculate Ej.
                    # Note: don't update Ej here, otherwise
                    # it will only find the maximum |Ei-Ej| once
                    Ek = cal_Ei(svm_obj,k)
                    deltaE = abs(Ei - Ek)
                    if deltaE > maxDeltaE:
                        j = k
                        maxDeltaE = deltaE
                        Ej = Ek
        else:
            # j, aj, Ej = choose_aj(i,svm_obj,"rand")
            deltaE_ls = abs(np.subtract(Ei, svm_obj.E_buffer[:, 1]))
            j = np.argsort(deltaE_ls)[svm_obj.m - 1, 0]
            Ej = svm_obj.E_buffer[j, 1]

    elif mode == "sortE":
        # obtain the list of E that are not initialized and initialize them
        invalid_ls =  np.nonzero(svm_obj.E_buffer[:,0].A)[0]
        if len(invalid_ls) >1:
            j,Ej =selectJ(i,svm_obj,Ei)
        # find the list of |Ei-Ej|
        else:
            deltaE_ls = abs(np.subtract(Ei, svm_obj.E_buffer[:,1]))
            j = np.argsort(deltaE_ls)[svm_obj.m-1,0]
            Ej = svm_obj.E_buffer[j,1]
    else:
            print("Don't recognize selecting mode: ",mode)
    aj = svm_obj.alpha_vec[j].copy()
    return j, aj, Ej



def clip_alpha(a,H,L):
    """
    :param a: alpha of training
    :param H: Higher boundary
    :param L: Lower boundary
    :return:
    """
    a = min(a, H)
    a = max(L, a)
    return  a


def update_alpha(svm_obj, i, j,Ei,Ej):
    """
    :param obj:
    :param ai_old:
    :param aj_old:
    :return: ai_new, aj_new
    update the aj and ai based on the formulas:
    ai_new = ai_old + y[i](Ej- Ei)/eta
    aj_new = aj_old + y[i]y[j](ai_old - ai_new)
    where:
        eta =  Kjj + Kii - 2Kji, Kij is the value of psvm_objition i-row j-col in kernel
    """
    K = svm_obj.k
    ai_old = svm_obj.alpha_vec[i].copy()
    aj_old = svm_obj.alpha_vec[j].copy()

    # check and update new boundary.
    if svm_obj.Y[i] == svm_obj.Y[j]:
        L = max(0, (aj_old + ai_old) - svm_obj.C)
        H = min(svm_obj.C, aj_old + ai_old)
    else:
        L = max(0, aj_old - ai_old)
        H = min(svm_obj.C, svm_obj.C + aj_old - ai_old)
    # if L == H,
    if L == H:
        return 0
    # calculate new aj
    eta= K[j,j] + K[i,i] - 2.0*K[i,j]
    if eta <= 0:
        return 0
    svm_obj.alpha_vec[j] += svm_obj.Y[j]*(Ei - Ej)/eta
    svm_obj.alpha_vec[j] = clip_alpha(svm_obj.alpha_vec[j], H, L)
    update_Ei(svm_obj,j)


    # if aj only changed a little along the line segment,
    # it is not necessay to update ai, return
    if abs(svm_obj.alpha_vec[j] - aj_old) < 0.00001:
        return 0

    # calculate new ai by formula:
    #  y1a1 + y2a2 = y1a1_old +y2a2_old = constant
    svm_obj.alpha_vec[i] += svm_obj.Y[i]*svm_obj.Y[j]*(aj_old - svm_obj.alpha_vec[j])
    # update Ei and Ej for computing new ai, aj
    update_Ei(svm_obj,i)

    return 1

def update_bias(svm_obj, i=0, j=0, ai_old=0, aj_old=0,mode="optimal" ):
    """
    :param svm_obj: structure contains all svm parameters info
    :param Ei: error of ai after updating alphas
    :param Ej: error of aj after updating alphas
    :param i: index of ai
    :param j: index of aj
    :param ai_old: old ai
    :param aj_old: old aj
    :param mode: it is the mode of update the bias
        it could be
        "online": update bias during optimizing alphas
        "optimal": update bias after finding the optimal alphas
    :return: b
    update rule for bia b:
    bi = -Ei - y[i]K[i,i]*(ai_new-ai_old) - y[j]*K[j,i]*(aj_new - aj_old) +b_old
    """

    if mode == "online":
        Ei = svm_obj.E_buffer[i,1]
        Ej = svm_obj.E_buffer[j, 1]
        bi = svm_obj.b - Ei \
             - svm_obj.Y[i]*svm_obj.k[i,i]*(svm_obj.alpha_vec[i] - ai_old) \
             - svm_obj.Y[j]*svm_obj.k[j,i]*(svm_obj.alpha_vec[j] - aj_old)


        bj = svm_obj.b - Ej \
             - svm_obj.Y[j] * svm_obj.k[j, j] * (svm_obj.alpha_vec[j] - aj_old) \
             - svm_obj.Y[i] * svm_obj.k[i, j] * (svm_obj.alpha_vec[i] - ai_old)
        # check KKT condition for alphas, if satisfy KKT, update bias
        if (svm_obj.alpha_vec[i] > 0 and svm_obj.alpha_vec[i] < svm_obj.C):
            b=bi
        elif (svm_obj.alpha_vec[j]>0 and svm_obj.alpha_vec[j]<svm_obj.C):
            b=bj
        else:
            b = (bi+bj)/2
    elif mode == "optimal":
        # update bias vector
        # update b
        # where b= (min(i:y=1):[w*x[i].T] +max(i:y=-1):[w*x[i].T])/2
        # w: nx1 vector, x[i]: 1xn vector.  w*x[i].T = dot product of w, x[i]
        # obtain the indices of Y=-1 and Y=1
        ind_1 = np.nonzero(svm_obj.Y == -1)[0]
        ind_2 = np.nonzero(svm_obj.Y == 1)[0]
        K_1 = get_kernel(svm_obj.X, svm_obj.X[ind_1], "linear")
        K_2 = get_kernel(svm_obj.X, svm_obj.X[ind_2], "linear")
        wx1 = K_1.T * np.multiply(svm_obj.Y, svm_obj.alpha_vec)
        wx2 = K_2.T * np.multiply(svm_obj.Y, svm_obj.alpha_vec)
        m1 = wx1.max()
        m2 = wx2.min()
        b = - (m1 + m2) / 2.0
    else:
        print("Warning: can't recognize mode:",mode)
    return b


def innerLp(svm_obj,i):
    # select ai
    ai = svm_obj.alpha_vec[i]
    label_i = svm_obj.Y[i]
    Ei = cal_Ei(svm_obj, i)
    # check KKT conditions:
    # alpha[i] =0  and y[i](wx[i]+b) >=1 -> Ei=y[i](wx[i]+b-1) >=tol
    # alppha[i]=C and y[i](wx[i]+b) <=1 -> Ei=y[i](wx[i]+b-1) <=tol
    # 0< alpha[i] < C and y[i](wx[i]+b) == 1 -> Ei=y[i](wx[i]+b-1) ==tol
    if ((label_i * Ei < -svm_obj.tol) and (ai < svm_obj.C)) or ((label_i * Ei > svm_obj.tol) and (ai > 0)):
        # select aj for alphas update
        j, aj, Ej = choose_aj(i, svm_obj, "maxE")

        # update alphas
        if update_alpha(svm_obj, i, j,Ei, Ej) !=0:
            # if alpha ai, aj updates successfully, update bias
            # Note: here ai and aj are old values. To get the old values of 
            # ai, aj after updating ai,aj when using online mode, copy() function should be used
            # otherwise, 
            svm_obj.b = update_bias(svm_obj, i, j, ai, aj, mode="online")
            return 1

    return 0

def SMO_Optim(svm_obj, max_itr=10000):
    """
    sequential minimum optimization (SMO)
    it is to solve the problem
    update alphas:
            max [sum(alpha[i], i=1,...,m ) - 1/2 *sum(i=1~m):[sum(j=1~m):[alpha[i]*alpha[j]*Y[i]*Y[j]*<X[i],X[j]>]]]
                respected to alpha
            st: 0<=alpha<= C, sum(i=1~m):(Y[i]*alpha[i])=0
    :param svm_obj:
    :return:
    """
    itr=0
    flag = 1
    alpha_changed =0

    while itr <= max_itr and (alpha_changed>0 or flag==1):
        alpha_changed=0
        if flag == 1:
            for i in range(svm_obj.m):
                alpha_changed += innerLp(svm_obj,i)
        else:
            ai_ls = np.nonzero((svm_obj.alpha_vec.A > 0) * (svm_obj.alpha_vec.A < svm_obj.C))[0]
            for i in ai_ls:
                alpha_changed += innerLp(svm_obj, i)

        if flag ==1:
            flag = 0
        elif alpha_changed == 0:
                flag = 1
        itr += 1
        print("iterate %d times" % (itr))


    svm_obj.w_vec = cal_weight(svm_obj)
    # svm_obj.b = update_bias(svm_obj,mode="optimal")

    return  svm_obj.alpha_vec, svm_obj.b, svm_obj.w_vec

def plotfig_SVM(xArr, yArr, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    # b = np.array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = np.arange(-0.1,10.0,0.1)

    y =((-b-np.multiply(ws[0,0],x))/ws[1, 0])
    y= np.ravel(y)

    ax.plot(x, y)

    for i in range(np.shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')

    plt.show()





def image_test(root):
    # load training dataset
    dir = root + "trainingDigits"
    inputMat, labelMat= loadImg(dir)
    testMat, test_labels = loadImg(root + 'testDigits2')
    # initialize parameters
    params= OptimStruct(input_mat=inputMat,
                        label_mat=labelMat.T,
                        C=200,
                        tol=0.0001,
                        k_info="Gaussian")
    # training svm
    a, b,w= SMO_Optim(params,10000)
    print("alpha:\n",a)
    print("b: %f \n"%(b), "w: ",w)
    # test model
    error_cnt =0.0
    for i in range(inputMat.shape[0]):
        predict = predictor(params,inputMat[i], kernel= "Gaussian")
        if np.sign(predict) != np.sign(labelMat[0,i]):
            error_cnt +=1

    print("training error rate: ", error_cnt/float(inputMat.shape[0]))

    # calculate Acc and error
    error_cnt=0.0
    for i in range(testMat.shape[0]):
        predict = predictor(params,testMat[i],kernel= "linear")
        if predict>0:
            pred =1
        else:
            pred = 9
        print("example %d: %d. label: %d" % (i,pred , test_labels[0,i]))
        if np.sign(predict) != np.sign(test_labels[0,i]):
            error_cnt +=1
    print("test error rate: ", error_cnt/float(testMat.shape[0]), "correct %d, wrong %d"%(testMat.shape[0]-error_cnt,error_cnt))

    # save model


def predTest(files_ls):
    # load training data and test data
    inputMat, labelMat = loadData(files_ls[0])
    # testMat , test_lables=loadData(files_ls[1])
    # initialize svm parameters
    params = OptimStruct(
        input_mat= inputMat,
        label_mat= labelMat.T,
        C=0.6,
        tol= 0.001,
        k_info="linear"
            )
    # training
    alpha, b, w = SMO_Optim(params,500)
    print("a:",alpha,"w:",w,"b:",b)
    # evaluation
    error_cnt = 0.0
    for i in range(inputMat.shape[0]):
        predict = predictor(params, inputMat[i],"linear")
        if np.sign(predict) != np.sign(labelMat[0, i]):
            error_cnt += 1
    print("training error rate: ", error_cnt / float(inputMat.shape[0]))

    # error_cnt = 0.0
    # for i in range(testMat.shape[0]):
    #     predict = predictor(params, testMat[i])
    #     if np.sign(predict) != np.sign(test_lables[0, i]):
    #         error_cnt += 1
    # print("test error rate: ", error_cnt / float(inputMat.shape[0]))

    # plot data
    plotfig_SVM(inputMat, labelMat, w, b, alpha)


    pass

if __name__ == '__main__':
    # select dataset source
    root= "../../MachineLearning-master/MachineLearning-master/input/6.SVM/"

    file = root+"testSet.txt"
    image_test(root)
    # files = []
    # files.append(file)
    # files.append(root+"testSetRBF.txt")
    # predTest(files)
    pass
