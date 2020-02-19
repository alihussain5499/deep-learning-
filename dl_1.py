# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:39:39 2019

@author: ali hussain
"""

f=open("D:/mystuff/patients.txt")
h=f.readline()
lines=f.readlines()
print(lines)

x=[]
y=[]

for line in lines:
    w=line.strip().lower().split(",")
    ins=[float(v) for v in w[1:-1]]
    x.append(ins)
    if w[-1]=="yes":
        y.append(1)
    else:
        y.append(0)
        
print(x)
print(y)

import numpy as np
ones=np.ones(len(x))
X=np.c_[ones,x]
print(X)

Y=np.c_[y]
print(Y)

np.random.seed(101)
r1=X.shape[1]
c1=int(1.5*r1)
W1=2*np.random.random((r1,c1))-1

W2=2*np.random.random((c1,1))-1

print(W2.shape)


def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(y,ycap):
    return ((y-ycap)**2).mean()
def derivative(x):
    return x*(1-x)

ploss=0
flag=0
for i in range(10000):
    l1=sigmoid(X.dot(W1))
    l2=sigmoid(l1.dot(W2))
    e2=Y-l2
    closs=loss(Y,l2)
    diff=abs(ploss-closs)
    if diff<=1e-6:
        print("Training completed ",i+1,"iteration ")
        flag=1
        break
    if i%1000==0:
        print("Current loss",closs)
    d2=e2*derivative(l2)
    e1=d2.dot(W2.T)
    d1=e1*derivative(l1)
    W1+=X.T.dot(d1)
    W2+=l1.T.dot(d2)
    ploss=closs
    
if flag==0:
    print("More iteration needed ")

def predict(x,w):
    r=x
    for v in w:
        r=sigmoid(r.dot(v))
    return r
    
Ycap=predict(X,[W1,W2])

print(Ycap)    
    
Ycap[Ycap<0.5]=0
Ycap[Ycap>0.5]=1
print(Ycap)

def accuracy(y,ycap):
    c=y==ycap
    pcnt=c[c==True].size
    n=y.size
    acc=pcnt/n*100
    return acc
accuracy(Y,Ycap)















