#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:15:16 2021

@author: nicklauersdorf
"""
import numpy as np
import matplotlib.pyplot as plt

peA_low=np.arange(0, 550, 50)
peB=np.arange(50, 550, 50)
peRatio_test=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
xA=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])#np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])#np.arange(0.05, 1.05, 0.1)
peNet = np.array([])
peRatio = np.array([])
peRatioWeight = np.array([])
for j in range(0, len(peB)):
    peA_test = peRatio_test*peB[j]
    for k in range(0, len(peA_test)):
        for i in range(0, len(xA)):
            peNet=np.append(peNet, xA[i]*peA_test[k]+(1.0-xA[i])*peB[j])
            #if peA_test[k]<peB[j]:
            peRatioWeight=np.append(peRatioWeight,(xA[i]*peA_test[k])/((1.0-xA[i])*peB[j]))
            #else:
            #    peRatioWeight=np.append(peRatioWeight,((1.0-xA[i])*peB[j])/(xA[i]*peA_test[k]))
            if peB[j]<peA_test[k]:
                peRatio=np.append(peRatio,(peB[j])/(peA_test[k]))
            else:
                peRatio=np.append(peRatio,(peA_test[k])/(peB[j]))
        #plt.scatter(peB[j], peA_test[k], color='black')
            plt.scatter(xA[i], xA[i]*peA_test[k]+(1.0-xA[i])*peB[j], color='black')
plt.ylabel('Net Pe')
plt.xlabel('xA')
plt.show()


print(len(peNet))          

for i in range(0, len(peRatioWeight)):
    if peRatioWeight[i]<=1.0:
        plt.scatter(peNet[i], peRatioWeight[i], color='black')
    else:
        plt.scatter(peNet[i], 1/peRatioWeight[i], color='red')
#plt.scatter(peNet, peRatioWeight, color='black')
plt.ylabel('Pe Ratio Weighted')
plt.xlabel('Net Pe')
#plt.yscale('log')
plt.show()




plt.scatter(peNet, peRatio, color='black')
plt.ylabel('Pe Ratio')
plt.xlabel('Net Pe')
plt.show()
stop   
    
    
    
for i in range(0, len(peA)):
    if i>0:
        for j in range(i, len(peB)):
            for k in range(0, len(xA)):
                peNet=np.append(peNet, xA[k]*peA[i]+(1.0-xA[k])*peB[j])
                peRatio=np.append(peRatio, peA[i]/peB[j])
                peRatioWeight = np.append(peRatioWeight, (peA[i]*xA[k])/(peB[j]*(1.0-xA[k])))
    else:
        for j in range(i+1, len(peB)):
            for k in range(0, len(xA)):
                peNet=np.append(peNet, xA[k]*peA[i]+(1.0-xA[k])*peB[j])
                peRatio=np.append(peRatio, peA[i]/peB[j])
                peRatioWeight = np.append(peRatioWeight, (peA[i]*xA[k])/(peB[j]*(1.0-xA[k])))

print(len(peNet))          
plt.scatter(peNet, peRatio)
plt.ylabel('Pe Ratio')
plt.xlabel('Net Pe')
plt.show()
stop