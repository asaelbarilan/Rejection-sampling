'''
this is a basic example for rejection sampling,and importance sampling
'''


import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import random


sns.set()

#create target distribution P(x)
def P(x):
    N=st.norm.pdf(x,loc=0,scale=1)+st.norm.pdf(x,loc=2,scale=3)
    return N

#create gaussian Q(x) proposal distribution
def Q(x):
    return st.norm.pdf(x, loc=0, scale=3.5)

def rejection_sampling_code(iterations=100):
    samples=[]
    i=0
    while i<iterations:
        x_i=np.random.normal(0,2)#point sampled from Q
        u=np.random.uniform(0,1)#point sampled from uniform distribution
        if u<(P(x_i)/c*Q(x_i)):
            samples.append(x_i)
            i=i+1
        else:
            i=i+1
    return np.array(samples)

X=np.arange(-10, 10)
c = max(np.divide(P(X), Q(X)))
samples=rejection_sampling_code(10000)

#importance sampling -choosing the right Q(X)

#choose any functions
def H(y):
    return y**2+8

n=len(P(X))
miu=(1/n)*np.sum(np.dot(P(X)/Q(X),H(X)))

#we want to pick the Q that has smallest variance of H(x)*P(x)/Q(x)

E_q=miu#from central limit theoram it equals...
w=np.dot(P(X)/Q(X),H(X))
VAR_fw=E_q*np.power(H(X),2)*np.power(w,2)
arg=np.abs(H(X))*P(X)

def integratation(A,X):
    '''
    a basic numeric integral
    '''
    h=(X[-1]-X[0])/len(X)
    I=h*(A[0]+np.sum(A[1:-2])+0.5*A[-1])
    return I
Integral=integratation(arg,X)

#from jensen in equallity
#E_q>integral

Q_best=arg/Integral


