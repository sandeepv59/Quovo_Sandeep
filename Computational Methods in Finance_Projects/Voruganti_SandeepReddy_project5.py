
import numpy as np

# defined function for Hermite , Laguerre and monomial
def L(option,K,m,x):
    sample = np.ones((m,K))
    if option == 'Hermite':
        sample[:,0] = 1
        sample[:,1] = 2*x
        if K == 2:
            return sample
        elif K == 3:
            sample[:,2] = (4*pow(x,2) - 2)
            return sample
        else:
            sample[:,3] = (8*pow(x,3) - 12*x)
            return sample
    elif option == 'Laguerre':
        sample[:,0] = np.exp(-x/2)
        sample[:,1] = (np.exp(-x/2)*(1 - x))
        if K == 2:
            return sample
        elif K == 3:
            sample[:,2] = (np.exp(-x/2)*(1 - 2*x + pow(x,2)/2))
            return sample
        else:
            sample[:,3] = (np.exp(-x/2)*(1 - 3*x + 3*pow(x,2)/2 - pow(x,3)/6))
            return sample
    else:
        sample[:,0] = 1
        sample[:,1] = x
        if K == 2:
            return sample
        elif K == 3:
            sample[:,2] = pow(x,2)
            return sample
        else:
            sample[:,3] = pow(x,3)
            return sample 
    

# function to generate stock prices
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

def American_Put(N,T,step,S0,sigma,r,St,option,K):
    S = Monte_Carlo(N,T,step,S0,sigma,r)
    maxim = np.vectorize(max)
    Cash_Flow = np.ones((N,1))
# initializing cash flow
    Cash_Flow[:,0] = maxim(St - S[:,step],0)
# initializing discounting time (step)
    Exc_Time = (step)*np.ones((N,1))
    delta = T/step
    for i in range (step-1,0,-1):
        Exc_value = St - S[:,i]
# selecting index where exercise value is greater than zer0 (in the money)
        Ind1 = np.asarray(np.where(Exc_value > 0)).flatten()
        m = len(Ind1)
# Calling function to generate lfunction values for the selected indices
        Lmat = L(option,K,m,S[Ind1,i])
        A = np.dot(np.transpose(Lmat),Lmat)
# Performing operations only when there are paths with in the money values      
        if np.linalg.det(A) != 0:
            Y = np.ones((m,1))
            Y[:,0] = (np.exp(-r*delta*(Exc_Time[Ind1,0]-i))*Cash_Flow[Ind1,0])
            b = np.transpose(np.dot(np.transpose(Y),Lmat))
# Calculation of coeficients
            Coef = np.dot(np.linalg.inv(A),b)
# Calculating continuation value from coefficients
            Cont_val = np.dot(np.transpose(Coef),np.transpose(Lmat))
            Ind2 = np.asarray(np.where((Exc_value[Ind1] - Cont_val) > 0)).flatten()
            for k in range(len(Ind2)):
                Cash_Flow[Ind1[Ind2[k]],0] = Exc_value[Ind1[Ind2[k]]]
                Exc_Time[Ind1[Ind2[k]],0] = i
    return np.mean(np.exp(-r*delta*Exc_Time[:,0])*Cash_Flow[:,0])

Put_Price1 = []
for i in [36,40,44]:
    for j in [0.5,1,2]:
        for k in [2,3,4]:
            sample = American_Put(100000,j,100,i,0.2,0.06,40,'Hermite',k)
            Put_Price1.append(sample)

Final_price_Hermite = np.ones((9,3))
for i in range(0,9):
    for j in range(0,3):
        Final_price_Hermite[i,j] = Put_Price1[i*3 + j]
        

Put_Price2 = []
for i in [36,40,44]:
    for j in [0.5,1,2]:
        for k in [2,3,4]:
            sample = American_Put(100000,j,100,i,0.2,0.06,40,'Laguerre',k)
            Put_Price2.append(sample)

Final_price_Laguerre = np.ones((9,3))
for i in range(0,9):
    for j in range(0,3):
        Final_price_Laguerre[i,j] = Put_Price2[i*3 + j]

Put_Price3 = []
for i in [36,40,44]:
    for j in [0.5,1,2]:
        for k in [2,3,4]:
            sample = American_Put(100000,j,100,i,0.2,0.06,40,'Monomial',k)
            Put_Price3.append(sample)        

Final_price_Monom = np.ones((9,3))
for i in range(0,9):
    for j in range(0,3):
        Final_price_Monom[i,j] = Put_Price3[i*3 + j]        
        
import pandas as pd

#a)

Final_price_Laguerre = pd.DataFrame(Final_price_Laguerre, index = ['S0 = 36 & T = 0.5','S0 = 36 & T = 1','S0 = 36 & T = 2',\
'S0 = 40 & T = 0.5','S0 = 40 & T = 1','S0 = 40 & T = 2',\
'S0 = 44 & T = 0.5','S0 = 44 & T = 1','S0 = 44 & T = 2'],columns = ['K = 2', 'K = 3', 'K = 4' ])

print("American put option prices using Laguerre function")
print(Final_price_Laguerre)

#b)

Final_price_Hermite = pd.DataFrame(Final_price_Hermite, index = ['S0 = 36 & T = 0.5','S0 = 36 & T = 1','S0 = 36 & T = 2',\
'S0 = 40 & T = 0.5','S0 = 40 & T = 1','S0 = 40 & T = 2',\
'S0 = 44 & T = 0.5','S0 = 44 & T = 1','S0 = 44 & T = 2'],columns = ['K = 2', 'K = 3', 'K = 4' ])

print("American put option prices using Hermite function")
print(Final_price_Hermite)

#c)

Final_price_Monom = pd.DataFrame(Final_price_Monom, index = ['S0 = 36 & T = 0.5','S0 = 36 & T = 1','S0 = 36 & T = 2',\
'S0 = 40 & T = 0.5','S0 = 40 & T = 1','S0 = 40 & T = 2',\
'S0 = 44 & T = 0.5','S0 = 44 & T = 1','S0 = 44 & T = 2'],columns = ['K = 2', 'K = 3', 'K = 4' ])

print("American put option prices using Monomial function")
print(Final_price_Monom)


#2.a

def forward_option_Europeanprice(N1,N2,t,T,step1,step2,S0,sigma,r):
    S = Monte_Carlo(N1,t,step1,S0,sigma,r)
    FEPrice = []
    for i in range(0,N1):
        S1 = Monte_Carlo(N2,T,step2,S[i,step1],sigma,r)
        maxim = np.vectorize(max)
        Cash_Flow = np.ones((N2,1))
        Cash_Flow[:,0] = maxim(S[i,step1] - S1[:,step2],0)
        sample = np.mean(np.exp(-r*(t+T))*Cash_Flow[:,0])
        FEPrice.append(sample)
    return np.mean(FEPrice)

FEPrice = forward_option_Europeanprice(10000,10000,0.2,0.8,100,100,65,0.2,0.06)    
print("The forward-start european put option price is")
print(FEPrice)

#2.b

def forward_option_Americanprice(N1,N2,t,T,step1,step2,S0,sigma,r,option,K):
    S = Monte_Carlo(N1,t,step1,S0,sigma,r)
    FAPrice = []
    for i in range(0,N1):
        sample = American_Put(N2,T,step2,S[i,step1],sigma,r,S[i,step1],option,K)
        FAPrice.append(sample)
    return np.exp(-r*t)*np.mean(FAPrice)

FAPrice = forward_option_Americanprice(1000,1000,0.2,0.8,100,100,65,0.2,0.06,'Monomial',3)
print("The forward-start American put option price is")
print(FAPrice)

