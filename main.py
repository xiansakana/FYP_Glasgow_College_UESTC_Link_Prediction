import numpy as np
import random
import time
from numpy.linalg import norm, svd
from load_data import *

def Divide_Network(Networks, ratio):
    train = np.matrix(Networks)
    row, col = np.nonzero(np.tril(Networks))
    probe_size = round(row.shape[0] * (1 - ratio))
    for i in range(probe_size):
        rand_number = int(row.shape[0] * random.random())
        train[row[rand_number], col[rand_number]] = 0
        train[col[rand_number], row[rand_number]] = 0
    test = Networks - train
    return train, test


def CN(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter() 
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    similarity_EndTime = time.perf_counter() 
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for CN: %f s" %(Time_Consumption))
    return Matrix_similarity, Time_Consumption

def PA(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()   
    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (MatrixAdjacency_Train.shape[0],1)
    deg_row_T = deg_row.T 
    Matrix_similarity = np.dot(deg_row,deg_row_T)
    similarity_EndTime = time.perf_counter()
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for PA: %f s" %(Time_Consumption))
    return Matrix_similarity, Time_Consumption

def RA(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()
    RA_Train = sum(MatrixAdjacency_Train)
    RA_Train.shape = (MatrixAdjacency_Train.shape[0],1)
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / RA_Train
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train_Log)
    similarity_EndTime = time.perf_counter()
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for RA: %f s" %(Time_Consumption))
    return Matrix_similarity, Time_Consumption

def AA(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()    
    logTrain = np.log(sum(MatrixAdjacency_Train))
    logTrain = np.nan_to_num(logTrain)
    logTrain.shape = (MatrixAdjacency_Train.shape[0],1)
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / logTrain
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)   
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train_Log)
    similarity_EndTime = time.perf_counter()
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for AA: %f s" %(Time_Consumption))        
    return Matrix_similarity, Time_Consumption

def Katz(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()      
    Parameter = 0.01
    Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])
    Temp = Matrix_EYE - MatrixAdjacency_Train * Parameter    
    Matrix_similarity = np.linalg.inv(Temp)
    Matrix_similarity = Matrix_similarity - Matrix_EYE
    similarity_EndTime = time.perf_counter()
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for Katz: %f s" %(Time_Consumption))       
    return Matrix_similarity, Time_Consumption

def LP(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()      
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)   
    Parameter = 1
    Matrix_LP = np.dot(np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train),MatrixAdjacency_Train) * Parameter  
    Matrix_similarity = np.dot(Matrix_similarity,Matrix_LP)
    similarity_EndTime = time.perf_counter()
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for LP: %f s" %(Time_Consumption))           
    return Matrix_similarity, Time_Consumption

def Rpca(D, lmbda, maxiter=100, tol=1e-3,):
    similarity_StartTime = time.perf_counter()      
    Y = D
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A_hat = np.zeros(Y.shape)
    E_hat = np.zeros(Y.shape)
    d_norm = norm(D, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        tmp = D - A_hat + (1 / mu) * Y
        E_update = np.maximum(tmp - lmbda / mu, 0) + np.minimum(tmp + lmbda / mu, 0)
        U, S, V = svd(D - E_update + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        A_update = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A_hat = A_update
        E_hat = E_update
        Z = D - A_hat - E_hat
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / d_norm) < tol) or (itr >= maxiter):
            break
    similarity_EndTime = time.perf_counter()
    Time_Consumption = similarity_EndTime- similarity_StartTime
    # print(" Time for Rpca: %f s" %(Time_Consumption))   
    return A_hat, Time_Consumption

def Precision_Compute(NetworkPrediction, train, test):
    precision = 0
    train = np.tril(train)
    NetworksP = np.tril(NetworkPrediction - np.multiply(np.eye(NetworkPrediction.shape[0]), NetworkPrediction))
    row, col = np.nonzero(np.tril(test))
    probe_size1 = row.shape[0]
    NetworksP *= np.where(train != 0, 0, 1)
    row, col = np.nonzero(NetworksP)
    probe_size2 = row.shape[0]
    rand_number = probe_size1 if probe_size1 < probe_size2 else probe_size2
    for i in range(rand_number):
        row, col = np.unravel_index(np.argmax(NetworksP), NetworksP.shape)
        NetworksP[row, col] = 0
        if test[row, col] > 0:
            precision += 1
    precision /= probe_size1
    return precision


if __name__ == "__main__":
    dataset = input("Input the network dataset: 0.Celegans 1.Contact 2.Dolphin 3.Email 4.FWM 5.Jazz 6.Karate 7.Macaca 8.Metabolic 9.Political_Blog 10.USAir 11.World_Trade\n")
    Adj = load_data(dataset)

    # The ratio of division of the data to training and probe set 
    # ratio = 0.8
    ratio = float(input("Input the ratio of training to probe set: \n"))

    precision_CN = []
    precision_PA = []
    precision_RA = []
    precision_AA = []
    precision_Katz = []
    precision_LP = []
    precision_LR = []

    Time_CN = []
    Time_PA = []
    Time_RA = []
    Time_AA = []
    Time_Katz = []
    Time_LP = []
    Time_LR = []
    

    # Parameter of LR for each network 
    lmbda = [0.10, 0.10, 0.25, 0.16, 0.13, 0.13, 0.23, 0.17, 0.10, 0.07, 0.10, 0.12]

    loop = 100

    for i in range(loop):
        train, test = Divide_Network(Adj, ratio)

        CN_score, t_CN = CN(train)
        PA_score, t_PA = PA(train)
        RA_score, t_RA = RA(train)
        AA_score, t_AA = AA(train)
        Katz_score, t_Katz = Katz(train)
        LP_score, t_LP = LP(train)
        LR_score, t_LR = Rpca(train, lmbda[int(dataset)])

        CN_score = CN_score + CN_score.T
        PA_score = PA_score + PA_score.T
        RA_score = RA_score + RA_score.T
        AA_score = AA_score + AA_score.T
        Katz_score = Katz_score + Katz_score.T
        LP_score = LP_score + LP_score.T
        LR_score = LR_score + LR_score.T       

        p_CN = Precision_Compute(CN_score, train, test)
        p_PA = Precision_Compute(PA_score, train, test)
        p_RA = Precision_Compute(RA_score, train, test)
        p_AA = Precision_Compute(AA_score, train, test)
        p_Katz = Precision_Compute(Katz_score, train, test)
        p_LP = Precision_Compute(LP_score, train, test)
        p_LR = Precision_Compute(LR_score, train, test)

        precision_CN.append(p_CN)
        precision_PA.append(p_PA)
        precision_RA.append(p_RA)
        precision_AA.append(p_AA)
        precision_Katz.append(p_Katz)
        precision_LP.append(p_LP)
        precision_LR.append(p_LR)   
        
        Time_CN.append(t_CN)
        Time_PA.append(t_PA)
        Time_RA.append(t_RA)
        Time_AA.append(t_AA)
        Time_Katz.append(t_Katz)
        Time_LP.append(t_LP)
        Time_LR.append(t_LR)

    print("Average precision of CN: ", np.mean(precision_CN))
    print("Average precision of PA: ", np.mean(precision_PA))
    print("Average precision of RA: ", np.mean(precision_RA))
    print("Average precision of AA: ", np.mean(precision_AA))
    print("Average precision of Katz: ", np.mean(precision_Katz))
    print("Average precision of LP: ", np.mean(precision_LP))
    print("Average precision of LR: ", np.mean(precision_LR))

    print("Average time consumption of CN: ", np.mean(Time_CN))
    print("Average time consumption of PA: ", np.mean(Time_PA))
    print("Average time consumption of RA: ", np.mean(Time_RA))
    print("Average time consumption of AA: ", np.mean(Time_AA))
    print("Average time consumption of Katz: ", np.mean(Time_Katz))
    print("Average time consumption of LP: ", np.mean(Time_LP))
    print("Average time consumption of LR: ", np.mean(Time_LR))
