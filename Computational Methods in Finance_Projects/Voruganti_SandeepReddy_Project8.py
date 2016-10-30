
import numpy as np
import scipy.stats as ss


def r_gen_Vasicek(step_month,N,T,r0,r_avg,kappa,sigma):
    step = int(step_month*12*T)
    delta = (1/12)*(1/step_month)
    rgen = np.ones((N,step+1))
    rgen[:,0] = r0
    Y0 = rgen[:,0]
    for j in range(1,step+1):
        Z = np.random.normal(0,1,N)
        rgen[:,j] = [ x + kappa * ( r_avg - x ) * delta + sigma *np.sqrt(delta) * y for (x,y) in zip(Y0,Z)]
        Y0 = rgen[:,j]
    return rgen

def pure_discount_price_Vasicek(step_month,N,T,r0,r_avg,FV,kappa,sigma):
    rmat = r_gen_Vasicek(step_month,N,T,r0,r_avg,kappa,sigma)
    delta = (1/12)*(1/step_month)
    price = np.mean (FV * np.exp(-(delta) * np.sum(rmat[:,1:],axis = 1)))
    return price

def coupon_paying_price(step_month,N,T,r0,r_avg,FV,C,kappa,sigma):
    rmat = r_gen_Vasicek(step_month,N,T,r0,r_avg,kappa,sigma)
    t = int(np.ceil(T*2))
    step = int(step_month*12*T)
    delta = (1/12)*(1/step_month)
    price_term = np.mean ((FV + C) * np.exp(-(delta) * np.sum(rmat[:,1:],axis = 1)))
    price_Final = 0
    for i in range(1,t):
        t_sample = (step + 1 ) - (i*6) * step_month
        price = np.mean( C * np.exp(-(delta)* np.sum(rmat[:,1:t_sample],axis = 1)))
        price_Final = price_Final + price
    return (price_Final + price_term)

def A(T,t,r_avg,kappa,sigma):
   sample =  np.exp( (r_avg - pow(sigma,2)/(2*pow(kappa,2)) ) \
   * ( B(T,t,kappa) - (T - t) ) - pow(sigma,2) *  pow(B(T,t,kappa),2)/(4*kappa))
   return sample

def B(T,t,kappa):
    sample = 1/kappa * (1 - np.exp ( - kappa * (T - t) ) )
    return sample


#Q.1.a

kappa = 0.82
r_avg = 0.05
sigma = 0.10
r0 = 0.05
# defining no of steps per month
step_month = 30
T = 0.5
N = 1000

Bond_price1 = pure_discount_price_Vasicek(step_month,N,T,r0,r_avg,1000,kappa,sigma)

print(Bond_price1)

#Q.1.b

kappa = 0.82
r_avg = 0.05
sigma = 0.10
r0 = 0.05
# defining no of steps per month
step_month = 30
T = 4
N = 1000

Bond_price2 = coupon_paying_price(step_month,N,T,r0,r_avg,1000,30,kappa,sigma)

print(Bond_price2)

#Q.1.c

kappa = 0.82
r_avg = 0.05
sigma = 0.10
r0 = 0.05
# defining no of steps per month
step_month = 30
T = 0.5
t = 90/360
K = 980
N = 10000

def one_c(step_month,N,t,r0,r_avg,kappa,sigma,K):
    rmat =  r_gen_Vasicek(step_month,N,t,r0,r_avg,kappa,sigma)
    a = A(T,t,r_avg,kappa,sigma)
    b = B(T,t,kappa)
    step = int(step_month*12*t)
    price_t = np.ones((N,1))
    delta = (1/12)*(1/step_month) 
    price_t[:,0] = np.maximum(1000*(a * np.exp (-b* rmat[:,step] )) - K, 0)
    sample = np.mean( price_t * np.exp(-(delta)* np.sum(rmat[:,1:],axis = 1)))
    return sample

European_option_price_1c = one_c(step_month,N,t,r0,r_avg,kappa,sigma,K)
print(European_option_price_1c)

#Q.1.d

kappa = 0.82
r_avg = 0.05
sigma = 0.10
r0 = 0.05
# defining no of steps per month
step_month = 30
T = 4
t = 90/360
K = 980
N = 500

def one_d(step_month,N,T,t,r0,r_avg,kappa,sigma,K):
    rmat =  r_gen_Vasicek(step_month,N,t,r0,r_avg,kappa,sigma)
    option_price_t = np.ones((N,1))
    delta = (1/12)*(1/step_month) 
    for i in range(0,N):
        price = coupon_paying_price(step_month,N,(T-t),rmat[i,-1],r_avg,1000,30,kappa,sigma)
        price  = np.max(price - K,0)
        option_price_t[i,:] = price
    sample = np.mean( option_price_t * np.exp(-(delta)* np.sum(rmat[:,1:],axis = 1)))
    return sample

European_option_price_1d = one_d(step_month,N,T,t,r0,r_avg,kappa,sigma,K)
print(European_option_price_1d)

#Q.1.e

kappa = 0.82
r_avg = 0.05
sigma = 0.10
r0 = 0.05
# defining no of steps per month
step_month = 30
T = 4
t0 = 0
t = 90/360
K = 980
FV = 1000
C = 30
N = 10000

def price_solve(T,t,r_avg,kappa,sigma,r_star):
    price = (FV + C) * A(T,t,r_avg,kappa,sigma) * np.exp (- B(T,t,kappa) * r_star)
    for i in range(1,8):
        a = A(T,t,r_avg,kappa,sigma)
        b = B(T,t,kappa)
        sample = C * A( (i/2) ,t,r_avg,kappa,sigma) * np.exp (- B( (i/2) ,t,kappa) * r_star)
        price = price + sample
    return price

def one_e(step_month,N,T,t,r0,r_avg,kappa,sigma,K):
    rmat =  r_gen_Vasicek(step_month,N,t,r0,r_avg,kappa,sigma)
    option_price_t = np.ones((N,1))
    delta = (1/12)*(1/step_month) 
    for i in range(0,N):
        price = price_solve(T,t,r_avg,kappa,sigma,rmat[i,-1])
        price  = np.max(price - K,0)
        option_price_t[i,:] = price
    sample = np.mean( option_price_t * np.exp(-(delta)* np.sum(rmat[:,1:],axis = 1)))
    return sample

#Q.2.

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

def pure_discount_price_CIR(step_month,N,T,r0,r_avg,FV,kappa,sigma):
    rmat = r_gen_CIR(step_month,N,T,r0,r_avg,kappa,sigma)
    delta = (1/12)*(1/step_month)
    price = np.mean (FV * np.exp(-(delta) * np.sum(rmat[:,1:],axis = 1)))
    return price

def two_a(step_month,N1,N2,T,t,r0,r_avg,kappa,sigma,K):
    rmat =  r_gen_CIR(step_month,N1,t,r0,r_avg,kappa,sigma)
    option_price_t = np.ones((N,1))
    delta = (1/12)*(1/step_month) 
    for i in range(0,N):
        price = pure_discount_price_CIR(step_month,N2,(T-t),rmat[i,-1],r_avg,1000,kappa,sigma)
        #[A,B] = AB(T,t,r0,r_avg,kappa,sigma)
        #price = 1000 * A * np.exp ( - B * rmat[i,-1])
        price  = max((price - K),0)
        option_price_t[i,0] = price
    sample = np.mean( option_price_t * np.exp(-(delta)* np.sum(rmat[:,1:],axis = 1)))
    return sample
    
def AB(T,t,r0,r_avg,kappa,sigma):
    h1 = np.sqrt(pow(kappa,2) + 2 * pow(sigma,2))
    h2 = (kappa + h1) / 2
    h3 = 2*kappa*r_avg/pow(sigma,2)
    A =  pow( (h1*np.exp(h2*(T-t)))/ \
    (h2 * (np.exp (h1 * (T - t)) - 1) + h1 ) , h3 )
    B =  (np.exp(h1*(T-t)) - 1 ) / \
    (h2 * (np.exp (h1 * (T - t)) - 1) + h1 )  
    return [A,B]


#Q.2.a

kappa = 0.92
r_avg = 0.055
sigma = 0.12
r0 = 0.05
# defining no of steps per month
step_month = 30
T = 1
t = 0.5
N1 = 10000
N2 = 500
K = 980

option_price_2a = two_a(step_month,N1,N2,T,t,r0,r_avg,kappa,sigma,K)

print(option_price_2a)

#Q.2.b

kappa = 0.92
r_avg = 0.055
sigma = 0.12
r0 = 0.05
# defining no of steps per month
deltaT = 1/360
T = 1
t = 0.5
N = 100
K = 980
min_rt = 0
max_rt = 0.2
deltaR = 0.001

def prob(deltaT,sigma,deltaR,r_avg,kappa,N):
    Pu = np.zeros((N-1,1))
    Pd = np.zeros((N-1,1))
    Pm = np.zeros((N-1,1))
    temp = np.matrix(np.transpose(np.matrix(range(1,N))))
    Pu = -(deltaT/(2*deltaR))*(pow(sigma, 2)*(temp)   +  kappa*(r_avg - temp*deltaR))
    Pm = 1 + (temp*deltaR + pow(sigma,2)*(temp/deltaR))*deltaT
    Pd = -(deltaT/(2*deltaR))*(pow(sigma, 2)*(temp)  -  kappa*(r_avg - temp*deltaR))
    return (Pu,Pm,Pd)
 

def IFD_bond(deltaT,sigma,deltaR,r0, r_avg , N ,T,K,max_rt,min_rt):
    N = int(np.ceil((max_rt - min_rt)/deltaR))
    step = int(t/deltaT)
    [Pu,Pm,Pd] = prob(deltaT,sigma,deltaR,r_avg,kappa,N)
    Amat = np.zeros((N + 1, N + 1 ))
    Amat[0,0] = 1
    Amat[0,1] = -1
    Amat[N,N-1] = 1
    Amat[N,N] = -1
    for i in range(1,N):
        Amat[i,i-1] = Pu[N - 1 - i,0]
        Amat[i,i] = Pm[N - 1 - i,0]
        Amat[i,i+1] = Pd[N - 1 - i,0]
    Bmat = np.zeros((N + 1,1))
    F_i = np.zeros((N + 1, 1 ))
    r_price = np.ones((N+1,1))
    [A,B] = AB(T,t,r0,r_avg,kappa,sigma)
    for i in range(0,N+1):
        r_price[N - i,0] = min_rt + (i)*deltaR
    Bmat[N,0] = FV * A * np.exp( - B * r_price[N,0]) - FV * A * np.exp( - B * r_price[N-1,0])
    for i in range(1,N):
        Bmat[N - i,0] = max(FV * A * np.exp( - B * r_price[N-i,0]) - K, 0)
    for i in range(step):
        F_i = np.dot(np.linalg.inv(Amat),Bmat)
        Bmat = F_i
        Bmat[0,0] = 0
        Bmat[N,0] = FV * A * np.exp( - B * r_price[1,0]) - FV * A * np.exp( - B * r_price[0,0])
    Option_Prices = np.ones((N+1,2))
    Option_Prices[:,1] = F_i[:,0]
    Option_Prices[:,0] = r_price[:,0]
    return Option_Prices   

option_price_2b = IFD_bond(deltaT,sigma,deltaR,r0, r_avg , N ,T,K,max_rt,min_rt)

print(option_price_2b[np.where(option_price_2b[:,0] == 0.05)])

#Q.2.c

kappa = 0.92
r_avg = 0.055
sigma = 0.12
r0 = 0.05
T = 0.5
S = 1
t = 0
K = 980
FV = 1000

def two_c(S,T,t,r_avg,kappa,sigma,r0,FV,K):
    [A,B] = AB(S,T,r0,r_avg,kappa,sigma)
    [A1,B1] = AB(S,t,r0,r_avg,kappa,sigma)
    [A2,B2] = AB(T,t,r0,r_avg,kappa,sigma)
    price_tS = FV * A1 * np.exp( - B1 * r0) 
    price_tT = FV * A2 * np.exp( - B2 * r0)
    r_star = np.log(A/(K/1000))/B
    theta = np.sqrt(pow(kappa,2) + 2 * pow(sigma,2))
    phi = (2 * theta)/(pow(sigma,2) * (np.exp( theta *(T - t) - 1)))
    omega = (kappa + theta)/pow(sigma, 2)
    x1 = 2 * r_star*(phi + omega + B)
    p1 = (4 * kappa * r_avg/pow(sigma, 2))
    q1 = (2 * pow(phi,2) * r0 * np.exp(theta * (T-t)))/(phi + omega + B)
    x2 = 2 * r_star * (phi + omega)
    q2 = (2 * pow(phi,2) * r0 * np.exp(theta * (T-t)))/(phi + omega)
    callprice = price_tS * ss.ncx2.cdf(x1,p1,q1) - (K/1000) * price_tT * ss.ncx2.cdf(x2,p1,q2)
    return callprice

print(two_c(S,T,t,r_avg,kappa,sigma,r0,FV,K))

#Q.3.a

r0 = 0.03
# defining no of steps per month
step_month = 30
T = 1
t = 0.5
N1 = 10000
N12 = 500
K = 950
a = 0.1
b = 0.3
eta = 0.08
phi = 0.03
x0 = 0
y0 = 0
rho = 0.7
sigma = 0.03

def generate_bivariate(N,rho):
    mean = [0,0]
    sigma1 = 1
    sigma2 = 1
    Z1 = np.random.normal(0,1,N)
    Z2 = np.random.normal(0,1,N)
    X1 =  [mean[0]+sigma1*x for x in Z1]
    X2 = [mean[1]+sigma2*(x*rho + y*((1-rho*rho)**(0.5))) for (x,y) in zip(Z1,Z2)]
    return [X1,X2]

def r_gen_G2plus(step_month,N,T,r0,a,b,eta,phi,x0,y0,rho,sigma):
    step = int(step_month*12*T)
    delta = (1/12)*(1/step_month)
    xgen = np.ones((N,step+1))
    xgen[:,0] = x0
    samplex = xgen[:,0]
    ygen = np.ones((N,step+1))
    ygen[:,0] = y0
    sampley = ygen[:,0]    
    for j in range(1,step+1):
        [Z1,Z2] = generate_bivariate(N,rho)
        xgen[:,j] = [ x - a * x * delta + sigma * np.sqrt(delta) * y for (x,y) in zip(samplex,Z1)]
        ygen[:,j] = [ x - b * x * delta + eta * np.sqrt(delta) * y for (x,y) in zip(sampley,Z2)]
        samplex = xgen[:,j]
        sampley = ygen[:,j]
    rgen = xgen + ygen + phi
    return [rgen,xgen,ygen]
    
def pure_discount_price_G2plus(step_month,N,T,r0,a,b,eta,phi,x0,y0,rho,sigma,FV):
    [rmat,xt,yt] = r_gen_G2plus(step_month,N,T,r0,a,b,eta,phi,x0,y0,rho,sigma)
    delta = (1/12)*(1/step_month)
    price = np.mean (FV * np.exp(-(delta) * np.sum(rmat[:,1:],axis = 1)))
    return price    
    
def three(step_month,N1,N2,T,t,r0,a,b,eta,phi,x0,y0,rho,sigma,K):
    [rmat,xt,yt] = r_gen_G2plus(step_month,N,t,r0,a,b,eta,phi,x0,y0,rho,sigma)
    option_price_t = np.ones((N,1))
    delta = (1/12)*(1/step_month)
    for i in range(0,N):
        price = pure_discount_price_G2plus(step_month,N,(T-t),rmat[i,-1],a,b,eta,phi,xt[i,-1],yt[i,-1],rho,sigma,1000)
        price  = max((K - price),0)
        option_price_t[i,:] = price
    sample = np.mean( option_price_t * np.exp(-(delta)* np.sum(rmat[:,1:],axis = 1)))
    return sample 

print(three(step_month,N1,N2,T,t, r0,a,b,eta,phi,x0,y0,rho,sigma,K))

#Q.3.b

r0 = 0.03
T = 0.5
S = 1
t = 0
K = 950
a = 0.1
b = 0.3
eta = 0.08
phi = 0.03
x0 = 0
y0 = 0
rho = 0.7
sigma = 0.03

def V(sigma,eta,a,b,T,t,rho):
    sample = pow((sigma/a),2)*((T-t) + (2/a)*np.exp(-a*(T-t)) - (1/(2*a))*np.exp(-2*a*(T-t)) - (3/(2*a)))\
    + pow((eta/b),2)*((T-t) + (2/b)*np.exp(-b*(T-t)) - (1/(2*b))*np.exp(-2*b*(T-t)) - (3/(2*b)))\
    + (2*rho*((sigma*eta)/(a*b)))*((T-t) + (np.exp(-a*(T-t))-1)/a\
    + (np.exp(-b*(T-t))-1)/b -(np.exp(-(a+b)*(T-t))-1)/(a+b))
    return sample
    
def P(sigma,eta,a,b,T,t,rho,phi,x0,y0):
     sample =  - phi*(T - t) - ((1- np.exp(-a*(T - t)))/a)*x0  + ((1- np.exp(-b* (T - t) ))/b)*y0
     sample = sample + 0.5*V(sigma,eta,a,b,T,t,rho)
     price = np.exp(sample)
     return price

def three_b_explicit(sigma,eta,a,b,T,t,rho,phi,x0,y0,K):
    Price_T = P(sigma,eta,a,b,T,t,rho,phi,x0,y0)
    Price_S = P(sigma,eta,a,b,S,t,rho,phi,x0,y0)
    K = K/1000
    sigma = np.sqrt(0.5*pow((sigma/a),2)*(1/a) * pow((1-np.exp(-a*(S-T))),2) *(1-np.exp(-2*a* (T - t))) \
    + 0.5*pow((eta/b),2)*(1/b) * pow((1-np.exp(-b*(S-T))),2) *(1-np.exp(-2*b*(T - t)))\
    + 2*rho*((sigma*eta)/(a*b*(a+b)))*((1-np.exp(-b*(S - T))))\
    *((1-np.exp(-a*(S - T))))*(1-np.exp(-(a+b)*(T - t))))
    N1 = ss.norm.cdf((np.log(K * Price_T/Price_S)/sigma) - (sigma/2))
    N2 = ss.norm.cdf((np.log(K * Price_T/Price_S)/sigma) + (sigma/2))
    price = FV * (-Price_S * N1 + Price_T * K * N2)
    return price

print(three_b_explicit(sigma,eta,a,b,T,t,rho,phi,x0,y0,K))    
    

