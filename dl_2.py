# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:31:30 2019

@author: ali hussain
"""


f=open("D:/mystuff/comm.txt")
h=f.readline()
lines=f.readlines()
print(lines)

x=[]
y=[]

for line in lines:
    w=line.strip().lower().split(",")
    ins=[float(v) for v in w[1:-1]]
    x.append(ins)
    if w[-1]=="high":
        y.append(1)
    else:
        y.append(0)
        
print(x)

import numpy as np
ones=np.ones(len(x))
X=np.c_[ones,x]
print(X)

Y=np.c_[y]
print(Y)

np.random.seed(101)
r1=X.shape[1]
c1=int(1.5*r1)
c2=int(1.5*c1)
W1=2*np.random.random((r1,c1))-1
W2=2*np.random.random((c1,c2))-1
W3=2*np.random.random((c2,c1))-1
W4=2*np.random.random((c1,1))-1

print(W3.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(y,ycap):
    return ((y-ycap)**2).mean()
def derivative(x):
    return x*(1-x)
 
# Trial 1
def predict(x,w):
    r=x
    for v in w:
        r=sigmoid(r.dot(v))
    return r
    
Ycap=predict(X,[W1,W2,W3,W4])

print(Ycap)

loss(Y,Ycap)





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


Ycap=predict(X,[W1,W2,W3,W4])

print(Ycap)    

loss(Y,Ycap)
    
Ycap[Ycap<0.5]=0
Ycap[Ycap>0.5]=1

def accuracy(y,ycap):
    c=y==ycap
    pcnt=c[c==True].size
    n=y.size
    acc=pcnt/n*100
    return acc
accuracy(Y,Ycap)



"""
Scaling required

"""
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
    
ins=np.array(x)
X=scaleMatrix(ins)

print(X)


 # Train network   
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


Ycap=predict(X,[W1,W2,W3,W4])

print(Ycap)    
    
Ycap[Ycap<0.5]=0
Ycap[Ycap>0.5]=1

print(Ycap)

accuracy(Y,Ycap)



new=open("D:/mystuff/newpat.txt")
hd=new.readline()
file=new.readlines()
print(file)
p=[]
for line in file:
    w=line.strip().split(",")
    ins=[float(v) for v in w[2:]]
    p.append(ins)
    
p=np.array(p)    
print(p)   

"""
scaleMatrix(P)           invalid

"""
scaleMatrix(p)
"""
for i in range(p.shape[1]):
    s=sd(x[:,i])
    m=x[:,i].mean()
    p[:,i]=(p[:,i]-m)/s
"""

o=np.ones(p.shape[0])

P=np.c_[o,p]    

print(P)

dstat=predict(P,[W1,W2,W3,W4])
dstat[dstat>0.5]=1
dstat[dstat<0.5]=0

print(dstat)








