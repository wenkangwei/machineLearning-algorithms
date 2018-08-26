#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on Nov 4, 2010
Update on 2017-05-18
Chapter 5 source file for Machine Learing in Action
@author: Peter/geekidentity/片刻
《机器学习实战》更新地址：https://github.com/apachecn/MachineLearning
"""
from numpy import *
import matplotlib.pyplot as plt
from os import  listdir

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
        self.m = shape(input_mat)[0]
        self.n = shape(input_mat)[1]
        # alpha vector of SVM
        self.alpha_vec = mat(zeros([self.m,1]))
        self.w_vec = mat(zeros([self.n,1]))
        self.b = 0
        self.k= get_kernel(self.X,self.X,k_info)
        self.C = C
        self.tol = tol
        self.E_buffer = mat(zeros([self.m, 2]))




def predictor(svm_obj,input):
    # return input*svm_obj.w_vec +svm_obj.b
    k = get_kernel(svm_obj.X, input,"Gaussian")
    return  k.T * multiply(svm_obj.Y, svm_obj.alpha_vec) + svm_obj.b

def get_kernel(x_mat, z_mat, k_info="linear", sigma=10.0):
    # m: rows of matrix. n:cols of matrix
    x_m,x_n = shape(x_mat)
    z_m, z_n =shape(z_mat)

    kernel= mat(zeros([x_m, z_m], dtype=float))
    if k_info == "linear":
            # fully-vectorized method for computing kernel
            # kernel= np.matmul(x_mat, np.transpsvm_obje(z_mat)）
            # partially vectorized method for kernel
        for i in range(z_m):
            kernel[:,i]= matmul(x_mat, transpsvm_obje(z_mat)[:,i])

    elif k_info == "Gaussian":
        k= matrix(zeros([1,x_n], dtype=float))
        # element-wise method for Gaussian kernel
        # Gaussian kernel: k[i,j]= exp(-||x[i,:]-z[j,:]||^2/2(sigma)^2)
        # where kernel rows = x rows, kernel cols = z rows.
        for i in range(z_m):
            for j in range(x_m):
                k[0]= x_mat[j,:]-z_mat[i,:]
                sq_delta= k * k.T
                kernel[j,i] = exp(-sq_delta/(sigma**2))
        pass
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
    return  mat(inputData), mat(labels)


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

    inputs=matrix(inputs,copy=False)
    labels=matrix(labels,copy=False)
    print("inputs shape: %dx%d"%(inputs.shape[0],inputs.shape[1]))
    print("labels shape: %dx%d"%(labels.shape[0],labels.shape[1]))
    return inputs, labels









def calcEk(oS, k):
    """calcEk（求 Ek误差：预测值-真实值的差）

    该过程在完整版的SMO算法中陪出现次数较多，因此将其单独作为一个方法
    Args:
        oS  optStruct对象
        k   具体的某一行

    Returns:
        Ek  预测结果与真实结果比对，计算误差Ek
    """
    fXk = multiply(oS.alpha_vec, oS.Y).T * oS.k[:, k] + oS.b
    Ek = fXk - float(oS.Y[k])
    return Ek


def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    """selectJ（返回最优的j和Ej）

    内循环的启发式方法。
    选择第二个(内循环)alpha的alpha值
    这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。
    该函数的误差与第一个alpha值Ei和下标i有关。
    Args:
        i   具体的第i一行
        oS  optStruct对象
        Ei  预测结果与真实结果比对，计算误差Ei

    Returns:
        j  随机选出的第j一行
        Ej 预测结果与真实结果比对，计算误差Ej
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
    oS.E_buffer[i] = [1, Ei]

    # print('oS.eCache[%s]=%s' % (i, oS.eCache[i]))
    # print('oS.eCache[:, 0].A=%s' % oS.eCache[:, 0].A.T)
    # """
    # # 返回非0的：行列值
    # nonzero(oS.eCache[:, 0].A)= (
    #     行： array([ 0,  2,  4,  5,  8, 10, 17, 18, 20, 21, 23, 25, 26, 29, 30, 39, 46,52, 54, 55, 62, 69, 70, 76, 79, 82, 94, 97]),
    #     列： array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0])
    # )
    # """
    # print('nonzero(oS.eCache[:, 0].A)=', nonzero(oS.eCache[:, 0].A))
    # # 取行的list
    # print('nonzero(oS.eCache[:, 0].A)[0]=', nonzero(oS.eCache[:, 0].A)[0])
    # 非零E值的行的list列表，所对应的alpha值
    validEcacheList = nonzero(oS.E_buffer[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k == i:
                continue  # don't calc for i, waste of time

            # 求 Ek误差：预测值-真实值的差
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                # 选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)

        # 求 Ek误差：预测值-真实值的差
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """updateEk（计算误差值并存入缓存中。）

    在对alpha值进行优化之后会用到这个值。
    Args:
        oS  optStruct对象
        k   某一列的行号
    """

    # 求 误差：预测值-真实值的差
    Ek = calcEk(oS, k)
    oS.E_buffer[k] = [1, Ek]


def clipAlpha(aj, H, L):
    """clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
    Args:
        aj  目标值
        H   最大值
        L   最小值
    Returns:
        aj  目标值
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj


def innerL(i, oS):
    """innerL
    内循环代码
    Args:
        i   具体的某一行
        oS  optStruct对象

    Returns:
        0   找不到最优的值
        1   找到了最优的值，并且oS.Cache到缓存中
    """

    # 求 Ek误差：预测值-真实值的差
    Ei = calcEk(oS, i)

    # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
    # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.Y[i] * Ei < -oS.tol) and (oS.alpha_vec[i] < oS.C)) or ((oS.Y[i] * Ei > oS.tol) and (oS.alpha_vec[i] > 0)):
    # if not( oS.alphas[i]== 0 and oS.labelMat[i] * Ei >= oS.tol ) \
    #         or not( oS.alphas[i]== oS.C and oS.labelMat[i] * Ei <= -oS.tol) \
    #         or not( oS.alphas[i] >0 and oS.alphas[i]<oS.C and oS.labelMat[i] * Ei == oS.tol):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alpha_vec[i].copy()
        alphaJold = oS.alpha_vec[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.Y[i] != oS.Y[j]):
            L = max(0, oS.alpha_vec[j] - oS.alpha_vec[i])
            H = min(oS.C, oS.C + oS.alpha_vec[j] - oS.alpha_vec[i])
        else:
            L = max(0, oS.alpha_vec[j] + oS.alpha_vec[i] - oS.C)
            H = min(oS.C, oS.alpha_vec[j] + oS.alpha_vec[i])
        if L == H:
            # print("L==H")
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * oS.k[i, j] - oS.k[i, i] - oS.k[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alpha_vec[j] -= oS.Y[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alpha_vec[j] = clipAlpha(oS.alpha_vec[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if abs(oS.alpha_vec[j] - alphaJold) < 0.00001:
            # print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alpha_vec[i] += oS.Y[j] * oS.Y[i] * (alphaJold - oS.alpha_vec[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.Y[i] * (oS.alpha_vec[i] - alphaIold) * oS.k[i, i] - oS.Y[j] * (oS.alpha_vec[j] - alphaJold) * oS.k[j, i]
        b2 = oS.b - Ej - oS.Y[i] * (oS.alpha_vec[i] - alphaIold) * oS.k[i, j] - oS.Y[j] * (oS.alpha_vec[j] - alphaJold) * oS.k[j, j]
        if (0 < oS.alpha_vec[i]) and (oS.C > oS.alpha_vec[i]):
            oS.b = b1
        elif (0 < oS.alpha_vec[j]) and (oS.C > oS.alpha_vec[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def SMO_Optim(oS, maxIter):
    """
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
        kTup    包含核函数信息的元组
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """



    # 创建一个 optStruct 对象

    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # ----------- 第一种写法 start -------------------------
        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = nonzero((oS.alpha_vec.A > 0) * (oS.alpha_vec.A < oS.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # ----------- 第一种写法 end -------------------------

        # ----------- 第二种方法 start -------------------------
        # if entireSet:																				#遍历整个数据集
    	# 	alphaPairsChanged += sum(innerL(i, oS) for i in range(oS.m))
		# else: 																						#遍历非边界值
		# 	nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
		# 	alphaPairsChanged += sum(innerL(i, oS) for i in nonBoundIs)
		# iter += 1
        # ----------- 第二种方法 end -------------------------
        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
        oS.w_vec = calcWs(oS.alpha_vec,oS.X,oS.Y)
    return  oS.alpha_vec, oS.w_vec,oS.b


def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集

    Returns:
        wc  回归系数
    """
    X = mat(dataArr)
    labelMat = mat(classLabels)
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i].T)
    return w









def image_test(root):
    # load training dataset
    dir = root + "trainingDigits"
    inputMat, labelMat = loadImg(dir)
    testMat, test_labels = loadImg(root + 'testDigits2')
    # initialize parameters
    params = OptimStruct(input_mat=inputMat,
                         label_mat=labelMat.T,
                         C=200,
                         tol=0.0001,
                         k_info="Gaussian")
    # training svm
    a, w, b = SMO_Optim(params, 10000)
    print("alpha:\n", a)
    print("b:  \n", b, "w: ", w)
    # test model
    error_cnt = 0.0
    for i in range(inputMat.shape[0]):
        predict = predictor(params, inputMat[i])
        if sign(predict) != sign(labelMat[0, i]):
            error_cnt += 1
    print("training error rate: ", error_cnt / float(inputMat.shape[0]))

    error_cnt = 0
    for i in range(testMat.shape[0]):
        predict = testMat[i] * w + b
        if sign(predict) > 0:
            pred = 1
        else:
            pred = 9
        print("example %d: %d. label: %d" % (i, int(pred), test_labels[0, i]))
        if sign(predict) != sign(test_labels[0, i]):
            error_cnt += 1
    print("test error rate: ", error_cnt / float(testMat.shape[0]),
          "count %d out of %d" % (testMat.shape[0] - error_cnt, error_cnt))
    pass

if __name__  == "__main__":
    root = "../../MachineLearning-master/MachineLearning-master/input/6.SVM/"
    file = root + "testSet.txt"
    image_test(root)
    pass