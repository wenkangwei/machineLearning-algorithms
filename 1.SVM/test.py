import  numpy as np

if __name__ == "__main__":
    a= np.mat(np.array(range(20))).T
    b = np.array(range(40,20,-1))
    # c= np.array((a,b)).T
    d = np.argsort(b)
    c = abs(np.subtract(3,a.T ))
    print("a",a,"b",b,"c",c,"d",d)
    pass