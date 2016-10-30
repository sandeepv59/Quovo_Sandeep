
import numpy as np
import scipy.stats as ss
import scipy.optimize as sci
import matplotlib.pyplot as plt


#Q.1.a)

WAC = 0.08
r = WAC/12
kappa = 0.6
r_avg = 0.08
sigma = 0.12
r0 = 0.078
# defining no of steps per month
step_month = 30
T = 30 
N = 100
PV0 = 100000
nstep = T*12

def r_gen_CIR(step_month,N,T,r0,r_avg,kappa,sigma):
    step = int(step_month*12*T)
    delta = (1/12)*(1/step_month)
    rgen = np.ones((N,step+1))
    rgen[:,0] = r0
    Y0 = rgen[:,0]
    for j in range(1,step+1):
        Z = np.random.normal(0,1,N)
        rgen[:,j] = [ x + kappa * ( r_avg - x ) * delta + sigma *np.sqrt(delta) * np.sqrt(x)*y for (x,y) in zip(Y0,Z)]
        Y0 = rgen[:,j]
    return rgen

def AB(T,t,r0,r_avg,kappa,sigma):
    h1 = np.sqrt(pow(kappa,2) + 2 * pow(sigma,2))
    h2 = (kappa + h1) / 2
    h3 = 2*kappa*r_avg/pow(sigma,2)
    A =  pow( (h1*np.exp(h2*(T-t)))/ \
    (h2 * (np.exp (h1 * (T - t)) - 1) + h1 ) , h3 )
    B =  (np.exp(h1*(T-t)) - 1 ) / \
    (h2 * (np.exp (h1 * (T - t)) - 1) + h1 )  
    return [A,B]



def MBS(step_month,WAC,N,T,r0,r_avg,kappa,sigma,PV0,CPRtype):
    
    PV = np.ones((N,nstep))
    SP = np.ones((N,nstep))
    PP = np.ones((N,nstep))
    IP = np.ones((N,nstep))
    TPP = np.ones((N,nstep))
    RI = np.ones((N,nstep))
    BU = np.ones((N,nstep))
    SY = np.ones((N,nstep))
    TPP = np.ones((N,nstep))
    Ct = np.ones((N,nstep))
    CPR = np.ones((N,nstep))
    
    rmat = r_gen_CIR(step_month,N,T,r0,r_avg,kappa,sigma)
    r = WAC/12
    delta = (1/12)*(1/step_month)
    SY_t = [0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 1.18, 1.22, 1.23, 0.98]*T
    SY = SY_t
    SY = np.matrix(SY)

    if CPRtype == "PSA":
        
        CPR[:,0] = 0.002
        
    else:    
        for k in range(0,N):
            [A,B] = AB(10,0, [k,30],r_avg,kappa,sigma)
            sample =  A * np.exp( - B * rmat[k,30])
            RI[k,0] = 0.28 + 0.14 * np.arctan( - 8.57 + 430 * (WAC - (1/10)*np.log(sample)))
        BU[:,0] = 0.3 + 0.7
        CPR[:,0] = RI[:,0] * BU[:,0] * min(1, 1 / 30) * SY[0,0]
    SP[:,0] = PV0 * r * (1 / (1 - 1 / np.power((1 + r),(nstep))) - 1 )
    PP[:,0] = ( PV0 - SP[:,0] ) * ( 1 - np.power(( 1 - CPR[:,0] ) , 1/12 ))
    IP[:,0] = PV0 * r
    TPP[:,0] = SP[:,0] + PP[:,0]
    PV[:,0] = PV0 - TPP[:,0]
    
    for i in range(1,nstep):
        if CPRtype == "PSA":        
            if i <= 29:
                CPR[:,i] = CPR[:,i-1] + 0.002
            else:
                CPR[:,i] = CPR[:,i-1]
        else:
            for k in range(0,N):
                [A,B] = AB(10,0,rmat[k,(i+1)*30],r_avg,kappa,sigma)
                sample =  A * np.exp( - B * rmat[k,(i+1)*30])
                RI[k,i] = 0.28 + 0.14 * np.arctan(-8.57 + 430 * (WAC - (1/10)*np.log(sample)))
            BU[:,i] = 0.3 + 0.7 * PV[:,i - 1] / PV0
            CPR[:,i] = RI[:,i] * BU[:,i] * min(1, (i+1) / 30) * SY[0,i]
        SP[:,i] = PV[:,i - 1] * r * (1 / (1 - 1 / pow((1 + r),(nstep - (i)))) - 1 )
        PP[:,i] = ( PV[:,i-1] - SP[:,i] ) * ( 1 - pow(( 1 - CPR[:,i] ) , 1/12 ))
        IP[:,i] = PV[:,i-1] * r
        TPP[:,i] = SP[:,i] + PP[:,i]
        PV[:,i] = PV[:,i-1] - TPP[:,i]
    
    Ct = SP + PP + IP
    
    Price = np.ones((N,nstep))
    PO = np.ones((N,nstep))
    
    for i in range(0,nstep):
        
        Price[:,i] = Ct[:,i] * np.exp(-(delta)* np.sum(rmat[:,1:(i+1)*30],axis = 1))
        PO[:,i] = (TPP[:,i]) * np.exp(-(delta)* np.sum(rmat[:,1:(i+1)*30],axis = 1))
    
    Final_Price = np.sum(Price,axis = 1)
    Final_Price = np.mean(Final_Price)
    PO_Price = np.mean(np.sum(PO,axis = 1))
    IO_Price = Final_Price - PO_Price
        
    return [Final_Price, PO_Price, IO_Price]

MBS(step_month,WAC,N,T,r0,r_avg,kappa,sigma,PV0,"Numerix")[0]   
 
Price12 = []
for k  in np.arange(0.3,1,0.1):
    sample = MBS(step_month,WAC,N,T,r0,r_avg,k,sigma,PV0,"Numerix")[0]
    Price12.append(sample)
    
plt.figure()
plt.plot(np.arange(0.3,1,0.1),Price12,label="K vs MBS Price using Numerix")
legend = plt.legend(loc='upper left')
plt.xlabel("K")
plt.ylabel("MBS Price")


#Q.1.c

Price13 = []
for i  in np.arange(0.03,0.1,0.01):
    sample = MBS(step_month,WAC,N,T,r0,i,k,sigma,PV0,"Numerix")[0]
    Price13.append(sample)
    
plt.figure()
plt.plot(np.arange(0.03,0.1,0.01),Price13,label="rbar vs MBS Price using Numerix")
legend = plt.legend(loc='upper left')
plt.xlabel("rbar")
plt.ylabel("MBS Price")


#Q.2.a

Price21 = []
for k  in np.arange(0.3,1,0.1):
    sample = MBS(step_month,WAC,N,T,r0,r_avg,k,sigma,PV0,"PSA")[0]
    Price21.append(sample)
    
plt.figure()
plt.plot(np.arange(0.3,1,0.1),Price21,label="K vs MBS Price using PSA")
legend = plt.legend(loc='upper left')
plt.xlabel("K")
plt.ylabel("MBS Price")


#Q.2.b

Price22 = []
for i  in np.arange(0.03,0.1,0.01):
    sample = MBS(step_month,WAC,N,T,r0,i,kappa,sigma,PV0,"PSA")[0]
    Price22.append(sample)
    
plt.figure()
plt.plot(np.arange(0.03,0.1,0.01),Price22,label="K vs MBS Price using PSA")
legend = plt.legend(loc='upper left')
plt.xlabel("K")
plt.ylabel("MBS Price")

#Q.3

def MBS_Solve(step_month,WAC,N,T,r0,r_avg,kappa,sigma,PV0,CPRtype,x):
    
    PV = np.ones((N,nstep))
    SP = np.ones((N,nstep))
    PP = np.ones((N,nstep))
    IP = np.ones((N,nstep))
    TPP = np.ones((N,nstep))
    RI = np.ones((N,nstep))
    BU = np.ones((N,nstep))
    SY = np.ones((N,nstep))
    TPP = np.ones((N,nstep))
    Ct = np.ones((N,nstep))
    CPR = np.ones((N,nstep))
    
    rmat = r_gen_CIR(step_month,N,T,r0,r_avg,kappa,sigma)
    r = WAC/12
    delta = (1/12)*(1/step_month)
    SY_t = [0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 1.18, 1.22, 1.23, 0.98]*T
    SY = SY_t
    SY = np.matrix(SY)

    if CPRtype == "PSA":
        
        CPR[:,0] = 0.002
        
    else:    
        for k in range(0,N):
            [A,B] = AB(10,0,rmat[k,30],r_avg,kappa,sigma)
            sample =  A * np.exp( - B * rmat[k,30])
            RI[k,0] = 0.28 + 0.14 * np.arctan( - 8.57 + 430 * (WAC - (1/10)*np.log(sample)))
        BU[:,0] = 0.3 + 0.7
        CPR[:,0] = RI[:,0] * BU[:,0] * min(1, 1 / 30) * SY[0,0]
    SP[:,0] = PV0 * r * (1 / (1 - 1 / np.power((1 + r),(nstep))) - 1 )
    PP[:,0] = ( PV0 - SP[:,0] ) * ( 1 - np.power(( 1 - CPR[:,0] ) , 1/12 ))
    IP[:,0] = PV0 * r
    TPP[:,0] = SP[:,0] + PP[:,0]
    PV[:,0] = PV0 - TPP[:,0]
    
    for i in range(1,nstep):
        if CPRtype == "PSA":        
            if i <= 29:
                CPR[:,i] = CPR[:,i-1] + 0.002
            else:
                CPR[:,i] = CPR[:,i-1]
        else:
            for k in range(0,N):
                [A,B] = AB(10,0,rmat[k,(i+1)*30],r_avg,kappa,sigma)
                sample =  A * np.exp( - B * rmat[k,(i+1)*30])
                RI[k,i] = 0.28 + 0.14 * np.arctan( - 8.57 + 430 * (WAC - (1/10) * np.log(sample)))
            BU[:,i] = 0.3 + 0.7 * PV[:,i - 1] / PV0
            CPR[:,i] = RI[:,i] * BU[:,i] * min(1, (i+1) / 30) * SY[0,i]
        SP[:,i] = PV[:,i - 1] * r * (1 / (1 - 1 / pow((1 + r),(nstep - (i)))) - 1 )
        PP[:,i] = ( PV[:,i-1] - SP[:,i] ) * ( 1 - pow(( 1 - CPR[:,i] ) , 1/12 ))
        IP[:,i] = PV[:,i-1] * r
        TPP[:,i] = SP[:,i] + PP[:,i]
        PV[:,i] = PV[:,i-1] - TPP[:,i]
    
    Ct = SP + PP + IP
    
    Price = np.ones((N,nstep))
    rmat = rmat + x

    for i in range(0,nstep):
        
        Price[:,i] = Ct[:,i] * np.exp(-(delta)* np.sum(rmat[:,1:(i+1)*30],axis = 1))
    
    Final_Price = np.sum(Price,axis = 1)
    Final_Price = np.mean(Final_Price) 
        
    return Final_Price



def solve_x(x,*args):
    temp = MBS_Solve(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],x)
    sample = temp - args[10]
    return sample

MBS_x_final = sci.fsolve(solve_x,-0.02,(step_month,WAC,N,T,r0,r_avg,kappa,sigma,PV0,"Numerix",110000))
print(MBS_x_final)


#Q.4

oas_plus = MBS_x_final + 0.0005
oas_minus = MBS_x_final - 0.0005
price_plus = MBS_Solve(step_month,WAC,N,T,r0,r_avg,kappa,sigma,PV0,CPRtype,oas_plus)
price_minus = MBS_Solve(step_month,WAC,N,T,r0,r_avg,kappa,sigma,PV0,CPRtype,oas_minus)
duration = (price_minus - price_plus)/(2 * 110000 * (0.0005))
convexity = (price_plus+ price_minus - 2 * 110000)/(2 * 110000 * pow(0.0005,2))
print(duration)
print(convexity)


#Q.5

Price51 = []
for i  in np.arange(0.03,0.1,0.01):
    sample = MBS(step_month,WAC,N,T,r0,ri,kappa,sigma,PV0,"Numerix")[1]
    Price51.append(sample)
    

Price52 = []
for i  in np.arange(0.03,0.1,0.01):
    sample = MBS(step_month,WAC,N,T,r0,i,kappa,sigma,PV0,"Numerix")[2]
    Price52.append(sample)
    
plt.figure()
plt.plot(np.arange(0.03,0.1,0.01),Price51,label="rbar vs MBS PO Price using Numerix")
plt.plot(np.arange(0.03,0.1,0.01),Price52,label="rbar vs MBS IO Price using Numerix")
legend = plt.legend(loc='upper left')
plt.xlabel("rbar")
plt.ylabel("MBS Price")

