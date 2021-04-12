#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:31:50 2020

@author: nicklauersdorf
"""

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
x50_n=[0,0,0,0,50,50,50,150,300,300,500,500]
y50_n=[50,150,300,500,0,300,500,0,0,50, 0,50]
x50_c=[50,60,150,150,150,300,300,300,500,500,500]
y50_c=[50,150,150,300,500,150,300,500,150,300,500]
x50_u=[50, 150]
y50_u=[150, 50]
x20=[0,0,60,150]
y20=[150,500,150,500]
x80_n=[0,0]
y80_n=[150,500]
x80_c=[150,150,500,150,500]
y80_c=[500, 0,0,60,150]
x80_u=[60]
y80_u=[150]

plt.plot(x50_n,y50_n,'.',color='red', label='No Clustering',markersize=10)
plt.plot(x50_c,y50_c,'.',color='green', label='Clustering',markersize=10)
plt.plot(x50_u,y50_u,'.',color='orange', label='Unsure/Unstable Clustering',markersize=10)
plt.ylabel('Pe_B')
plt.xlabel('Pe_A')
plt.title('50:50 A:B')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
plt.tight_layout()
plt.show()

plt.plot(x20,y20,'.',color='green', label='Clustering',markersize=10)
#plt.plot(x20_c,y20_c,'.',color='green', label='Clustering',markersize=10)
#plt.plot(x20_u,y20_u,'.',color='orange', label='Unsure/Unstable Clustering',markersize=10)
plt.ylabel('Pe_B')
plt.xlabel('Pe_A')
plt.title('20:80 A:B')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.yticks([0,50,100,150,200,250,300,350,400,450,500])
plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
plt.tight_layout()
plt.show()

plt.plot(x80_n,y80_n,'.',color='red', label='No Clustering',markersize=10)
plt.plot(x80_c,y80_c,'.',color='green', label='Clustering',markersize=10)
plt.plot(x80_u,y80_u,'.',color='orange', label='Unsure/Unstable Clustering',markersize=10)
plt.ylabel('Pe_B')
plt.xlabel('Pe_A')
plt.title('80:20 A:B')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
plt.yticks([0,50,100,150,200,250,300,350,400,450,500])
plt.tight_layout()
plt.show()
stop