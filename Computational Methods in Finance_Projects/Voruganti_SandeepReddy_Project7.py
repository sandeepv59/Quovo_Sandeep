import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
from scipy.interpolate import interp1d

S0 = 10
sigma = 0.2
r = 0.04
K = 10
T = 0.5
min_St = 0
max_St = 20
deltaT = 0.002
deltaS = 0.5

class BlackScholes:    
    def __init__(self,S0,X,r,sigma,t):
        self.S0 = S0
        self.X = X
        self.r = r
        self.sigma = sigma
        self.t = t
        self.d1 = (np.log(self.S0/self.X) + (r + np.power(self.sigma,2)/2)*self.t)/(self.sigma*np.sqrt(self.t))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.t)
    def BS_call(self):
        x = (self.S0*ss.norm.cdf(self.d1) - self.X*np.exp(-self.r*self.t)*ss.norm.cdf(self.d2))
        return x
    def BS_Put(self):
        x = (-self.S0*ss.norm.cdf(-self.d1) + self.X*np.exp(-self.r*self.t)*ss.norm.cdf(-self.d2))
        return x

BS_Put1 =[]
for i in range(4,17,1):
    BS = BlackScholes(i,K,r,sigma,T)
    temp = BS.BS_Put()
    BS_Put1.append(temp)

BS_Put2 =[]
for i in range(2,19,1):
    BS = BlackScholes(i,K,r,sigma,T)
    temp = BS.BS_Put()
    BS_Put2.append(temp)

#Q.1.a

sigma = 0.2
r = 0.04
K = 10
T = 0.5
min_S = np.log(2)
max_S = np.log(18)
deltaT = 0.002
deltaX = sigma*np.sqrt(deltaT)

def prob1(type,deltaT,sigma,deltaX,r):
    if type == 'EFD':
        Pu = deltaT * (pow(sigma,2)/(2*pow(deltaX,2)) + (r - pow(sigma,2)/2)/(2*deltaX))
        Pm = 1 - deltaT * pow(sigma,2)/pow(deltaX,2) - r * deltaT
        Pd = deltaT * (pow(sigma,2)/(2*pow(deltaX,2)) - (r - pow(sigma,2)/2)/(2*deltaX))
    elif type == 'IFD':
        Pu = -1/2 * deltaT * (pow(sigma,2)/(pow(deltaX,2)) + (r - pow(sigma,2)/2)/(deltaX))
        Pm = 1 + deltaT * pow(sigma,2)/pow(deltaX,2) + r * deltaT
        Pd = -1/2 * deltaT * (pow(sigma,2)/(pow(deltaX,2)) - (r - pow(sigma,2)/2)/(deltaX))
    else:
        Pu = -1/4 * deltaT * (pow(sigma,2)/(pow(deltaX,2)) + (r - pow(sigma,2)/2)/(deltaX))
        Pm = 1 + deltaT * pow(sigma,2)/(2*pow(deltaX,2)) + r * (deltaT/2)
        Pd = -1/4 * deltaT * (pow(sigma,2)/(pow(deltaX,2)) - (r - pow(sigma,2)/2)/(deltaX))
    return (Pu,Pm,Pd)

def EFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S):
    N = int(np.ceil((max_S - min_S)/deltaX)/2)
    step = int(T/deltaT)
    [Pu,Pm,Pd] = prob1('EFD',deltaT,sigma,deltaX,r)
    A = np.zeros((2*N + 1, 2*N + 1 ))
    A[0,0] = Pu
    A[0,1] = Pm
    A[0,2] = Pd
    A[2*N,2*N-2] = Pu
    A[2*N,2*N-1] = Pm
    A[2*N,2*N] = Pd
    for i in range(1,2*N):
        A[i,i-1] = Pu
        A[i,i] = Pm
        A[i,i+1] = Pd
    B = np.zeros((2*N + 1,1))
    F_iplus =  np.zeros((2*N + 1, 1 ))
    F_i = np.zeros((2*N + 1, 1 ))
    B[2*N,0] = np.exp(min_S) * (np.exp(deltaX) - 1)
    for i in range(0,2*N+1):
        F_iplus[2*N - i,0] = max(K - np.exp(min_S + (i)*deltaX) , 0)
    for i in range(step,0,-1):
        F_i = np.dot(A,F_iplus) + B
        F_iplus = F_i
    S_price = np.ones((2*N+1,1))    
    for i in range(0,2*N+1):
        S_price[2*N - i,0] = np.exp(min_S + (i)*deltaX)
    Option_Prices = np.ones((2*N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = S_price[:,0]
    return Option_Prices

def IFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S):
    N = int(np.ceil((max_S - min_S)/deltaX)/2)
    step = int(T/deltaT)
    [Pu,Pm,Pd] = prob1('IFD',deltaT,sigma,deltaX,r)
    A = np.zeros((2*N + 1, 2*N + 1 ))
    A[0,0] = 1
    A[0,1] = -1
    A[2*N,2*N-1] = 1
    A[2*N,2*N] = -1
    for i in range(1,2*N):
        A[i,i-1] = Pu
        A[i,i] = Pm
        A[i,i+1] = Pd
    B = np.zeros((2*N + 1,1))
    F_i = np.zeros((2*N + 1, 1 ))
    for i in range(1,2*N):
        B[2*N - i,0] = max(K - np.exp(min_S + (i)*deltaX) , 0)
    B[2*N,0] = np.exp(min_S) * (np.exp(deltaX) - 1)
    for i in range(step,0,-1):
        F_i = np.dot(np.linalg.inv(A),B)
        B = F_i
        B[2*N,0] = np.exp(min_S) * (np.exp(deltaX) - 1)
        B[0,0] = 0
    S_price = np.ones((2*N+1,1))    
    for i in range(0,2*N+1):
        S_price[2*N - i,0] = np.exp(min_S + (i)*deltaX)
    Option_Prices = np.ones((2*N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = S_price[:,0]
    return Option_Prices

def Zmat(Pu,Pm,Pd,Fmat,min_S,deltaX,N):
    Z = np.zeros((2*N+1,1))
    Z[0,0] = 0
    Z[2*N,0] = np.exp(min_S) * (np.exp(deltaX) - 1)
    for i in range(1,2*N):
        Z[i,0] = - Pu * Fmat[i+1,0] - (Pm - 2) * Fmat[i,0] - Pd * Fmat[i-1,0]
    return Z

def CNFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S):
    N = int(np.ceil((max_S - min_S)/deltaX)/2)
    step = int(T/deltaT)
    [Pu,Pm,Pd] = prob1('CNFD',deltaT,sigma,deltaX,r)
    A = np.zeros((2*N + 1, 2*N + 1 ))
    A[0,0] = 1
    A[0,1] = -1
    A[2*N,2*N-1] = 1
    A[2*N,2*N] = -1
    for i in range(1,2*N):
        A[i,i-1] = Pu
        A[i,i] = Pm
        A[i,i+1] = Pd
    F_i = np.zeros((2*N + 1, 1 ))
    term_op = np.zeros((2*N + 1, 1 ))
    for i in range(0,2*N+1):
        term_op[2*N - i,0] = max(K - np.exp(min_S + (i)*deltaX) , 0)
    Z = Zmat(Pu,Pm,Pd,term_op,min_S,deltaX,N)
    for i in range(step,0,-1):
        F_i = np.dot(np.linalg.inv(A),Z)
        Z = Zmat(Pu,Pm,Pd,F_i,min_S,deltaX,N)
    S_price = np.ones((2*N+1,1))    
    for i in range(0,2*N+1):
        S_price[2*N - i,0] = np.exp(min_S + (i)*deltaX)
    Option_Prices = np.ones((2*N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = S_price[:,0]
    return Option_Prices

temppp = CNFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S)

deltaX_total = [sigma*np.sqrt(deltaT),sigma*np.sqrt(3*deltaT),sigma*np.sqrt(4*deltaT)]

for i in [sigma*np.sqrt(deltaT),sigma*np.sqrt(3*deltaT),sigma*np.sqrt(4*deltaT)]:
    Sp_Eur_Put_IFD = IFD(deltaT,sigma,i,r,T,K,max_S,min_S)
    Sp_Eur_Put_EFD = EFD(deltaT,sigma,i,r,T,K,max_S,min_S)
    Sp_Eur_Put_CNFD = CNFD(deltaT,sigma,i,r,T,K,max_S,min_S)
    plt.figure()
    plt.plot(Sp_Eur_Put_EFD[:,0],Sp_Eur_Put_EFD[:,1],label="EFD & deltaX - %s"%i)
    plt.plot(Sp_Eur_Put_IFD[:,0],Sp_Eur_Put_IFD[:,1],label="IFD & deltaX - %s"%i)
    plt.plot(Sp_Eur_Put_CNFD[:,0],Sp_Eur_Put_CNFD[:,1],label="CNFD & deltaX - %s"%i)
    plt.plot(list((range(2,19,1))),BS_Put2,label="Black Scholes")
    legend = plt.legend(loc='upper Right')
    plt.xlabel("Stock Price")
    plt.ylabel("Put Option Price")
    plt.savefig("European Put & deltaX - %s.png"%i)


from scipy.interpolate import interp1d


def table(deltaT,sigma,deltaX,r,T,K,max_S,min_S):
    Eur_Put_Price = np.ones((13,8))
    Sp_Eur_Put_IFD = IFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S)
    Sp_Eur_Put_EFD = EFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S)
    Sp_Eur_Put_CNFD = CNFD(deltaT,sigma,deltaX,r,T,K,max_S,min_S)
    f_EFD = interp1d(Sp_Eur_Put_EFD[:,0],Sp_Eur_Put_EFD[:,1])
    f_IFD = interp1d(Sp_Eur_Put_IFD[:,0],Sp_Eur_Put_IFD[:,1])
    f_CNFD = interp1d(Sp_Eur_Put_CNFD[:,0],Sp_Eur_Put_CNFD[:,1])
    Eur_Put_Price[:,0] = list((range(4,17,1)))
    Eur_Put_Price[:,1] = f_EFD(list(range(4,17,1)))
    Eur_Put_Price[:,2] = f_IFD(list(range(4,17,1)))
    Eur_Put_Price[:,3] = f_CNFD(list(range(4,17,1)))
    Eur_Put_Price[:,4] = BS_Put1
    Eur_Put_Price[:,5] = (Eur_Put_Price[:,1] - Eur_Put_Price[:,4])/Eur_Put_Price[:,4]
    Eur_Put_Price[:,6] = (Eur_Put_Price[:,2] - Eur_Put_Price[:,4])/Eur_Put_Price[:,4]
    Eur_Put_Price[:,7] = (Eur_Put_Price[:,3] - Eur_Put_Price[:,4])/Eur_Put_Price[:,4]
    df = pd.DataFrame(Eur_Put_Price,columns = ['S0', 'EFD', 'IFD','CNFD','BlckSch','Err_IFD','Err_EFD','Err_CNFD' ])
    Output = df.to_string(formatters = {'S0': '{:,.0f}'.format, 'EFD': '{:,.3f}'.format, 'IFD': '{:,.3f}'.format,'CNFD': '{:,.3f}'.format,'BlckSch': '{:,.3f}'.format,'Err_IFD': '{:,.2%}'.format,'Err_EFD': '{:,.2%}'.format,'Err_CNFD': '{:,.2%}'.format})
    return Output
    
    
Eur_Put_Price1 = table(deltaT,sigma,deltaX_total[0],r,T,K,max_S,min_S)
print("table using first deltaX")
print(Eur_Put_Price1)
Eur_Put_Price2 = table(deltaT,sigma,deltaX_total[1],r,T,K,max_S,min_S)
print("table using second deltaX")
print(Eur_Put_Price2)
Eur_Put_Price3 = table(deltaT,sigma,deltaX_total[2],r,T,K,max_S,min_S)
print("table using third deltaX")
print(Eur_Put_Price3)


#Q.2.a

S0 = 10
sigma = 0.2
r = 0.04
K = 10
T = 0.5
min_St = 0
max_St = 20
deltaT = 0.002
deltaS = 0.5

def prob2(type,deltaT,sigma,deltaS,r,N):
    Pu = np.zeros((N-1,1))
    Pd = np.zeros((N-1,1))
    Pm = np.zeros((N-1,1))
    temp = np.matrix(np.transpose(np.matrix(range(1,N))))
    if type == 'EFD':
        Pu = deltaT * (r * (temp/2) + pow(sigma,2) * np.power(temp,2)/2)
        Pm = 1 - deltaT * (pow(sigma,2) * np.power(temp,2) + r)
        Pd = deltaT * ( - r * (temp/2) + pow(sigma,2) * np.power(temp,2)/2)
    elif type == "IFD":
        Pu = -1/2*deltaT * (r * (temp) + pow(sigma,2) * np.power(temp,2))
        Pm = 1 + deltaT * (pow(sigma,2) * np.power(temp,2) + r)
        Pd = -deltaT * ( - r * (temp/2) + pow(sigma,2) * np.power(temp,2)/2)
    else:
        Pu = -1/4*deltaT * (r * (temp) + pow(sigma,2) * np.power(temp,2))
        Pm = 1 + deltaT/2 * (pow(sigma,2) * np.power(temp,2) + r)
        Pd = -deltaT * ( - r * (temp/4) + pow(sigma,2) * np.power(temp,2)/4)
    return (Pu,Pm,Pd)


def EFD_Amer(deltaT,sigma,deltaS,r,T,K,max_St,min_St,option):
    N = int(np.ceil((max_St - min_St)/deltaS))
    step = int(T/deltaT)
    [Pu,Pm,Pd] = prob2('EFD',deltaT,sigma,deltaS,r,N)
    A = np.zeros((N + 1, N + 1 ))
    A[0,0] = Pu[N-2,0]
    A[0,1] = Pm[N-2,0]
    A[0,2] = Pd[N-2,0]
    A[N,N-2] = Pu[0,0]
    A[N,N-1] = Pm[0,0]
    A[N,N] = Pd[0,0]
    for i in range(1,N):
        A[i,i-1] = Pu[N - 1 - i,0]
        A[i,i] = Pm[N - 1 - i,0]
        A[i,i+1] = Pd[N - 1 - i,0]
    B = np.zeros((N + 1,1))
    F_iplus =  np.zeros((N + 1, 1 ))
    F_i = np.zeros((N + 1, 1 ))
    S_price = np.ones((N+1,1))    
    Exer_Val = np.ones((N+1,1)) 
    for i in range(0,N+1):
        S_price[N - i,0] = min_St + (i)*deltaS
    if option == "Put":
        Exer_Val = np.maximum(K - S_price,0)
        B[N,0] = deltaS
        for i in range(0,N+1):
            F_iplus[N - i,0] = max(K - min_St - i*deltaS , 0)
    if option == "Call":
        Exer_Val = np.maximum(S_price - K,0)
        B[0,0] = deltaS
        for i in range(0,N+1):
            F_iplus[N - i,0] = max(min_St + i*deltaS - K, 0)
    for i in range(step,0,-1):
        F_i = np.dot(A,F_iplus) + B
        F_i = np.maximum(F_i,Exer_Val)
        F_iplus = F_i
    Option_Prices = np.ones((N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = S_price[:,0]
    return Option_Prices

def IFD_Amer(deltaT,sigma,deltaS,r,T,K,max_St,min_St,option):
    N = int(np.ceil((max_St - min_St)/deltaS))
    step = int(T/deltaT)
    [Pu,Pm,Pd] = prob2('IFD',deltaT,sigma,deltaS,r,N)
    A = np.zeros((N + 1, N + 1 ))
    A[0,0] = 1
    A[0,1] = -1
    A[N,N-1] = 1
    A[N,N] = -1
    for i in range(1,N):
        A[i,i-1] = Pu[N - 1 - i,0]
        A[i,i] = Pm[N - 1 - i,0]
        A[i,i+1] = Pd[N - 1 - i,0]
    B = np.zeros((N + 1,1))
    F_i = np.zeros((N + 1, 1 ))
    S_price = np.ones((N+1,1))    
    Exer_Val = np.ones((N+1,1)) 
    for i in range(0,N+1):
        S_price[N - i,0] = min_St + (i)*deltaS
    if option == "Put":
        Exer_Val = np.maximum(K - S_price,0)
        B[N,0] = deltaS
        for i in range(0,N+1):
            B[N - i,0] = max(K - min_St - i*deltaS , 0)
    if option == "Call":
        Exer_Val = np.maximum(S_price - K,0)
        B[0,0] = deltaS
        for i in range(0,N+1):
            B[N - i,0] = max(min_St + i*deltaS - K, 0)
    for i in range(step,0,-1):
        F_i = np.dot(np.linalg.inv(A),B)
        F_i = np.maximum(F_i,Exer_Val)
        B = F_i
        if option == "Put":
            B[N,0] = deltaS
            B[0,0] = 0
        else:
            B[N,0] = 0
            B[0,0] = deltaS
    Option_Prices = np.ones((N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = S_price[:,0]
    return Option_Prices


def Zmat_Amer(Pu,Pm,Pd,Fmat,min_St,deltaS,N,option):
    Z = np.zeros((N+1,1))
    if option == "Put":
        Z[N,0] = deltaS
    else:
        Z[0,0] = deltaS
    for i in range(1,N):
        Z[i,0] = - Pu[N-i-1] * Fmat[i+1,0] - (Pm[N-i-1] - 2) * Fmat[i,0] - Pd[N-i-1] * Fmat[i-1,0]
    return Z

def CNFD_Amer(deltaT,sigma,deltaS,r,T,K,max_St,min_St,option):
    N = int(np.ceil((max_St - min_St)/deltaS))
    step = int(T/deltaT)
    [Pu,Pm,Pd] = prob2('CNFD',deltaT,sigma,deltaS,r,N)
    A = np.zeros((N + 1, N + 1 ))
    A[0,0] = 1
    A[0,1] = -1
    A[N,N-1] = 1
    A[N,N] = -1
    for i in range(1,N):
        A[i,i-1] = Pu[N - 1 - i,0]
        A[i,i] = Pm[N - 1 - i,0]
        A[i,i+1] = Pd[N - 1 - i,0]
    F_i = np.zeros((N + 1, 1 ))
    S_price = np.ones((N+1,1))    
    Exer_Val = np.ones((N+1,1))
    for i in range(0,N+1):
        S_price[N - i,0] = min_St + (i)*deltaS
    if option == "Put":
        Exer_Val = np.maximum(K - S_price,0)
    if option == "Call":
        Exer_Val = np.maximum(S_price - K,0)
    Z = Zmat_Amer(Pu,Pm,Pd,Exer_Val,min_St,deltaS,N,option)            
    for i in range(step,0,-1):
        F_i = np.dot(np.linalg.inv(A),Z)
        F_i = np.maximum(F_i,Exer_Val)
        Z = Zmat_Amer(Pu,Pm,Pd,F_i,min_St,deltaS,N,option) 
    Option_Prices = np.ones((N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = S_price[:,0]
    return Option_Prices


def table1(deltaT,sigma,r,T,K,max_St,min_St,option):
    Amer_Put_Price = np.ones((3,5))
    Sp_Amer_Put_IFD = EFD_Amer(deltaT,sigma,0.5,r,T,K,max_St,min_St,option)
    Sp_Amer_Put_EFD = IFD_Amer(deltaT,sigma,0.5,r,T,K,max_St,min_St,option)
    Sp_Amer_Put_CNFD = CNFD_Amer(deltaT,sigma,0.5,r,T,K,max_St,min_St,option)
    f_EFD = interp1d(Sp_Amer_Put_EFD[:,0],Sp_Amer_Put_EFD[:,1])
    f_IFD = interp1d(Sp_Amer_Put_IFD[:,0],Sp_Amer_Put_IFD[:,1])
    f_CNFD = interp1d(Sp_Amer_Put_CNFD[:,0],Sp_Amer_Put_CNFD[:,1])
    Amer_Put_Price[0,0] = 10
    Amer_Put_Price[0,2] = f_EFD(10)
    Amer_Put_Price[0,3] = f_IFD(10)
    Amer_Put_Price[0,4] = f_CNFD(10)
    Amer_Put_Price[0,1] = 0.5
    Sp_Amer_Put_IFD = EFD_Amer(deltaT,sigma,1,r,T,K,max_St,min_St,option)
    Sp_Amer_Put_EFD = IFD_Amer(deltaT,sigma,1,r,T,K,max_St,min_St,option)
    Sp_Amer_Put_CNFD = CNFD_Amer(deltaT,sigma,1,r,T,K,max_St,min_St,option)
    f_EFD = interp1d(Sp_Amer_Put_EFD[:,0],Sp_Amer_Put_EFD[:,1])
    f_IFD = interp1d(Sp_Amer_Put_IFD[:,0],Sp_Amer_Put_IFD[:,1])
    f_CNFD = interp1d(Sp_Amer_Put_CNFD[:,0],Sp_Amer_Put_CNFD[:,1])
    Amer_Put_Price[1,0] = 10
    Amer_Put_Price[1,2] = f_EFD(10)
    Amer_Put_Price[1,3] = f_IFD(10)
    Amer_Put_Price[1,4] = f_CNFD(10)
    Amer_Put_Price[1,1] = 1
    Sp_Amer_Put_IFD = EFD_Amer(deltaT,sigma,1.5,r,T,K,max_St,min_St,option)
    Sp_Amer_Put_EFD = IFD_Amer(deltaT,sigma,1.5,r,T,K,max_St,min_St,option)
    Sp_Amer_Put_CNFD = CNFD_Amer(deltaT,sigma,1.5,r,T,K,max_St,min_St,option)
    f_EFD = interp1d(Sp_Amer_Put_EFD[:,0],Sp_Amer_Put_EFD[:,1])
    f_IFD = interp1d(Sp_Amer_Put_IFD[:,0],Sp_Amer_Put_IFD[:,1])
    f_CNFD = interp1d(Sp_Amer_Put_CNFD[:,0],Sp_Amer_Put_CNFD[:,1])
    Amer_Put_Price[2,0] = 10
    Amer_Put_Price[2,2] = f_EFD(10)
    Amer_Put_Price[2,3] = f_IFD(10)
    Amer_Put_Price[2,4] = f_CNFD(10)
    Amer_Put_Price[2,1] = 1.5
    df = pd.DataFrame(Amer_Put_Price,columns = ['S0', 'deltaS','Put_Price_EFD', 'Put_Price_IFD','Put_Price_CNFD' ])
    Output = df.to_string(formatters = {'S0': '{:,.0f}'.format, 'deltaS': '{:,.1f}'.format,'Put_Price_EFD': '{:,.3f}'.format, 'Put_Price_IFD': '{:,.3f}'.format,'Put_Price_CNFD': '{:,.3f}'.format})
    return Output

Amer_Put_Price1 = table1(deltaT,sigma,r,T,K,max_St,min_St,"Put")
print(Amer_Put_Price1)

Amer_Call_Price1 = table1(deltaT,sigma,r,T,K,max_St,min_St,"Call")
print("table using first deltaS")
print(Amer_Call_Price1)

   
for i in [0.5,1,1.5]:    
    Sp_Amer_Put_EFD = EFD_Amer(deltaT,sigma,i,r,T,K,max_St,min_St,"Put")
    Sp_Amer_Call_EFD = EFD_Amer(deltaT,sigma,i,r,T,K,max_St,min_St,"Call")
    Sp_Amer_Put_IFD = EFD_Amer(deltaT,sigma,i,r,T,K,max_St,min_St,"Put")
    Sp_Amer_Call_IFD = EFD_Amer(deltaT,sigma,i,r,T,K,max_St,min_St,"Call")
    Sp_Amer_Put_CNFD = CNFD_Amer(deltaT,sigma,i,r,T,K,max_St,min_St,"Put")
    Sp_Amer_Call_CNFD = CNFD_Amer(deltaT,sigma,i,r,T,K,max_St,min_St,"Call")
    
    plt.figure()
    plt.plot(Sp_Amer_Call_CNFD[:,0],Sp_Amer_Call_CNFD[:,1],label="Using CNFD & deltaS - %s"%i)
    plt.plot(Sp_Amer_Call_EFD[:,0],Sp_Amer_Call_EFD[:,1],label="Using EFD & deltaS - %s"%i)
    plt.plot(Sp_Amer_Call_IFD[:,0],Sp_Amer_Call_IFD[:,1],label="Using IFD & deltaS - %s"%i)
    legend = plt.legend(loc='upper left')
    plt.xlabel("Stock Price")
    plt.ylabel("American Call Option Price")
    plt.savefig("American Call & deltaX - %s.png"%i)    
    
    plt.figure()
    plt.plot(Sp_Amer_Put_CNFD[:,0],Sp_Amer_Put_CNFD[:,1],label="Using CNFD & deltaS - %s"%i)
    plt.plot(Sp_Amer_Put_EFD[:,0],Sp_Amer_Put_EFD[:,1],label="Using EFD & deltaS - %s"%i)
    plt.plot(Sp_Amer_Put_IFD[:,0],Sp_Amer_Put_IFD[:,1],label="Using IFD & deltaS - %s"%i)
    legend = plt.legend(loc='upper Right')
    plt.xlabel("Stock Price")
    plt.ylabel("American Put Option Price")
    plt.savefig("American Put & deltaX - %s.png"%i)

