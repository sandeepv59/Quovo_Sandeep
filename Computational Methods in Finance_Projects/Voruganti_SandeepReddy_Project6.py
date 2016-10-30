import numpy as np
import matplotlib.pyplot as plt

#1. 

# function to generate stock prices using monte carlo simulation

def Monte_Carlo(N,T,step,S0,sigma,r):
    S = np.ones((N,step+1))
    S[:,0] = S0
    delta = T/step
    for j in range(1,step+1):
        temp = np.random.normal(0,1,int(N/2))
        Z = np.ones((int(N/2),1))
        Z[:,0] = temp
        Sim = np.ones((N/2,1))
        Sim1 = np.ones((N/2,1))
        Sim[:,0] = S[0:N/2,j-1]*np.exp(sigma*np.sqrt(delta)*Z[:,0]+(r-np.power(sigma,2)/2)*delta)
        Sim1[:,0] = S[N/2:N+1,j-1]*np.exp(sigma*np.sqrt(delta)*(-Z[:,0])+(r-np.power(sigma,2)/2)*delta)
        S[0:int(N/2),j] = Sim[:,0]
        S[N/2:N+1,j] = Sim1[:,0]
    return S

# function to calculate Fixed Strike Lookback call and put option prices 

def Fixed_Strike_Lookback(N,T,step,S0,X,sigma,r,method):
    S = Monte_Carlo(N,T,step,S0,sigma,r)
    FEPrice = []
    for i in range(0,N):
        if method == "Call":
            sample = np.max(np.max(S[i,:])- X,0)
        else:
            sample = np.max(X - np.min(S[i,:]),0)
        FEPrice.append(sample)
    return np.exp(-r*T)*np.mean(FEPrice)

FSLookback_Call = []
for sigma in np.arange(0.12,0.52,0.04):
    sample = Fixed_Strike_Lookback(100000,1,100,98,100,sigma,0.03,"Call")
    FSLookback_Call.append(sample)

FSLookback_Put = []
for sigma in np.arange(0.12,0.52,0.04):
    sample = Fixed_Strike_Lookback(100000,1,100,98,100,sigma,0.03,"Put")
    FSLookback_Put.append(sample)

# plot between Fixed Strike Lookback call-option prices and volatility
plt.figure()
plt.plot(np.arange(0.12,0.52,0.04),FSLookback_Call,'k--',color = 'red',label='Fixed Strike Lookback call option price vs. volatility')
legend = plt.legend(loc='upper center', shadow=True)
plt.xlabel("Volatility")
plt.ylabel("Option Prices")
plt.savefig("Proj6_1a.png")

# plot between Fixed Strike Lookback put-option prices and volatility
plt.figure()
plt.plot(np.arange(0.12,0.52,0.04),FSLookback_Put,'k--',color = 'Black',label='Fixed Strike Lookback put option price vs. volatility')
legend = plt.legend(loc='upper center', shadow=True)
plt.xlabel("Volatility")
plt.ylabel("Option Prices")
plt.savefig("Proj6_1b.png")


#2.

# Function to generate matrix qt
def qt_func(n,alpha,beta):
    qt = np.ones((1 , n + 1))
    for i in range (0,n + 1):
        qt[0,i] = alpha + beta*i*(1/12)
    return qt

# Function to generate matrix Lt
def Lt_func(n,a,b,c,L0):
    Lt = np.ones((1,n + 1))
    Lt[0,0] = L0
    for i in range (1,n + 1):
        Lt[0,i] = a - b*pow(c,12*i*(1/12))
    return Lt

# Function to generate matrix Vt
def Vt_func(n,path,mu,sigma,gamma,r,T,lambda1,V0):
    Vt = np.ones((path,n + 1))
    Vt[:,0] = V0
    for i in range(0,path):
# calculating time at which jump occurs in a certain path
        sample1 = 0
        jump_time = []
        while (sample1 < T):
            sample = np.random.exponential(1/lambda1)
            sample1 = sample1 + sample
            if sample1 < T:
                jump_time.append(sample1)
        for j in range(1 , n + 1):
            Vt[i,j] = Vt[i,j-1]*(1 + mu * (1/12) + sigma * np.sqrt(1/12) * np.random.normal(0,1,1))
            sample2 = 1
# adding gamma if there is only one jump, gamma square if there are two jummps and gamma cube if there are three jumps       
            for k in range(1,len(jump_time)):
                if jump_time[k] > (j-1)*(1/12) and jump_time[k] <= (j)*(1/12):
                    sample2 = gamma*sample2
            if sample2 == 1:
                Vt[i,j]
            else:
                Vt[i,j] = Vt[i,j] - Vt[i,j-1]*sample2
    return Vt

# Function to calculate payoff
def payoff_func(Q,S,r0,ephslon,path,Lt,Vt,n):
    min_n = np.ones((path,1))    
    min_n = np.minimum(Q,S)
    payoff = np.ones((path,1))
    for i in range(0,path):
# Using n+1 becuase max could only be n+1  as max of min of q is n+1       
        if min_n[i,0] == n + 1:
            payoff[i,0] = 0
        elif min_n[i,0] == Q[i,0]:
            payoff[i,0] = np.exp(-r0*Q[i,0]*(1/12))*max(Lt[0,Q[i,0]] - ephslon*Vt[i,Q[i,0]],0)
        else:
            payoff[i,0] = np.exp(-r0*S[i,0]*(1/12))*np.absolute(Lt[0,S[i,0]] - ephslon*Vt[i,S[i,0]])
    return np.mean(payoff)

def proj6_2func(lambda1,lambda2,T):
#given, inputs
    V0 = 20000
    L0 = 22000
    mu = -0.1
    sigma = 0.2
# used positive gamma as i changed the postive symbol to negative symbol in the value equation. I did this inorder to take care of symbol squaring or using even powers   
    gamma = 0.4
    r0 = 0.02
    delta = 0.25
    alpha = 0.7
    ephslon = 0.95
    path = 10000
    
    #calculation of r
    n = T*12
    R = r0 + delta*lambda2
    r = R/12
    
    #calculation of a,b,c and beta
    PMT = L0*r/(1 - 1/pow((1+r),n))
    a = PMT/r
    b = PMT/(r*pow((1+r),n))
    c = (1 + r)
    beta = (ephslon - alpha)/T
    
    qt = qt_func(n,alpha,beta)
    Lt = Lt_func(n,a,b,c,L0)
    adj_Lt = qt*Lt 
    Vt = Vt_func(n,path,mu,sigma,gamma,r,T,lambda1,V0)
    Q = np.ones((path,1))
# calculation of Q
    for i in range(0,path):
        sample = np.transpose(np.matrix(np.transpose(Vt[i,:]))) - np.transpose(np.matrix(np.transpose(adj_Lt)))
        if np.min(sample) < 0:
            Q[i,0] = np.asarray(np.where( sample <= 0)).flatten()[0]
        else:
            Q[i,0] = n + 1

    S = np.ones((path,1)) 
# calculation of S    
    for i in range(0,path):
        if lambda2 ==0:
            S[i,0] = n + 1
        else:            
            S[i,0] = np.ceil(np.random.exponential(1/lambda2)*12)
# calculation of payoff using payoff function
    Payoff = payoff_func(Q,S,r0,ephslon,path,Lt,Vt,n)   
    min_n = np.minimum(Q,S)
# calculation of no of defaults
    no_of_defaults = path - list(min_n).count(n+1)
# calculation of expected exercise time of the default option, conditional on exercise time less than T
    exer_time_cond = ((np.mean(min_n)*path - (path - no_of_defaults)*(n + 1))/no_of_defaults)/12
    return (Payoff,exer_time_cond,no_of_defaults/path) 
    

plt.figure()
for lambda2 in np.arange(0,0.9,0.1):
    OP1 = []
    for T in range(3,9,1):
        sample = proj6_2func(0.2,lambda2,T)[0]
        OP1.append(sample)
    plt.plot(list(range(3,9,1)),OP1,label="lambda2 - %s"%lambda2)
legend = plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Option Prices")
plt.savefig("var_lambda2_Optionprice.png")
    
plt.figure()
for lambda1 in np.arange(0.05,0.401,0.05):
    OP = []
    for T in range(3,9,1):
        sample = proj6_2func(lambda1,0.4,T)[0]
        OP.append(sample)
    plt.plot(list(range(3,9,1)),OP,label="lambda1 - %s"%lambda1)
legend = plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Option Prices")
plt.savefig("var_lambda1_Optionprice.png")

plt.figure()
for lambda2 in np.arange(0,0.9,0.1):
    OP1 = []
    for T in range(3,9,1):
        sample = proj6_2func(0.2,lambda2,T)[2]
        OP1.append(sample)
    plt.plot(list(range(3,9,1)),OP1,label="lambda2 - %s"%lambda2)
legend = plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Default Prob")
plt.savefig("var_lambda2_dafault_prob.png")
    
plt.figure()
for lambda1 in np.arange(0.05,0.401,0.05):
    OP = []
    for T in range(3,9,1):
        sample = proj6_2func(lambda1,0.4,T)[2]
        OP.append(sample)
    plt.plot(list(range(3,9,1)),OP,label="lambda1 - %s"%lambda1)
legend = plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Default Prob")
plt.savefig("var_lambda1_dafault_prob.png")

plt.figure()
for lambda2 in np.arange(0,0.9,0.1):
    OP1 = []
    for T in range(3,9,1):
        sample = proj6_2func(0.2,lambda2,T)[1]
        OP1.append(sample)
    plt.plot(list(range(3,9,1)),OP1,label="lambda2 - %s"%lambda2)
legend = plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Exp. conditional Excercise time")
plt.savefig("var_lambda2_Exp_cond_exer_time.png")
    
plt.figure()
for lambda1 in np.arange(0.05,0.401,0.05):
    OP = []
    for T in range(3,9,1):
        sample = proj6_2func(lambda1,0.4,T)[1]
        OP.append(sample)
    plt.plot(list(range(3,9,1)),OP,label="lambda1 - %s"%lambda1)
legend = plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Exp. conditional Excercise time")
plt.savefig("var_lambda1_Exp_cond_exer_time.png")
