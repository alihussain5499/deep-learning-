# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:30:46 2019

@author: ali hussain
"""
f=open("D:/mystuff/patients2.txt")
lines=f.readlines()[1:]
print(lines)

x=[]
labels=[]
for line in lines:
    w=line.strip().lower().split(",")
    ins=[float(v) for v in w[1:-1]]
    x.append(ins)
    labels.append(w[-1])    
    

print(x)
print(labels)

ulabels=list(set(labels))
print(ulabels)

idx={l:i for i,l in enumerate(ulabels)}
idx

y=[idx[l] for l in labels]
print(y)

import numpy as np
print(np.c_[labels,y])

b=[]
for v in y:
    barr=np.zeros(len(ulabels))
    barr[v]=1
    b.append(barr)
Y=np.array(b)
print(Y)    

Y.shape


o=np.ones(len(lines))
X=np.c_[o,x]

print(X)


from numpy.linalg import inv
Beta=inv(X.T.dot(X)).dot(X.T.dot(Y))
print(Beta)

scap=1/(1+np.exp(X.dot(Beta)))
print(scap)

scap[scap<0.5]=0
scap[scap>0.5]=1
print(scap)

def sd(x):
    return (((x-x.mean())**2).sum()/(x.size-1))**0.5
def scale(x):
    return (x-x.mean())/sd(x)
def scaleMatrix(x):
    for i in range(x.shape[1]):
        col=x[:,i]
        x[:,i]=scale(col)
    o=np.ones(x.shape[0])
    return np.c_[o,x]

#scaleMatrix(X)

ins=np.array(x)
X=scaleMatrix(ins)
print(x)
print(ins)

print(X)

np.random.seed(101)
r=X.shape[1]
h1=int(1.5*r)
h2=int(1.5*h1)
h3=h1
o=Y.shape[1]
W1=2*np.random.random((r,h1))-1
W2=2*np.random.random((h1,h2))-1
W3=2*np.random.random((h2,h3))-1
W4=2*np.random.random((h3,o))-1

print(W1)
print(W2)
print(W3)
print(W4)

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
    l3=sigmoid(l2.dot(W3))
    l4=sigmoid(l3.dot(W4))
    e4=Y-l4
    closs=loss(Y,l4)
    diff=abs(ploss-closs)
    if diff<=1e-6:
        print("Training completed ",i+1,"iteration ")
        flag=1
        break
    if i%1000==0:
        print("Current loss",closs)
    d4=e4*derivative(l4)
    e3=d4.dot(W4.T)
    d3=e3*derivative(l3)
    e2=d3.dot(W3.T)
    d2=e2*derivative(l2)
    e1=d2.dot(W2.T)
    d1=e1*derivative(l1)
    W1+=X.T.dot(d1)
    W2+=l1.T.dot(d2)
    W3+=l2.T.dot(d3)
    W4+=l3.T.dot(d4)
    ploss=closs
    
if flag==0:
    print("More iteration needed ")





def predict(x,w):
    r=x
    for v in w:
        r=sigmoid(r.dot(v))
    return r



dcap=predict(X,[W1,W2,W3,W4])
print(dcap)

dcap[dcap<0.5]=0
dcap[dcap>0.5]=1
print(dcap)

dc=[int(np.where(v==1)[0]) for v in dcap]
print(dc)

print(y)

def accuracy(y,ycap):
    r=y==ycap
    return r[r==True].size/y.size*100
accuracy(np.array(dc),np.array(y))












