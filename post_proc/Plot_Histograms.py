#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:08:49 2020

@author: nicklauersdorf
"""
import matplotlib.pyplot as plt

gsdPath='/Volumes/External/text_files/'
pa=float(0)
pb=float(500)
xa=float(80)
phi=int(60)
ep=1.000
inFile='phase_density_pa'+str(pa)+'_pb'+str(pb)+'_xa'+str(xa)+'_phi'+str(phi)+'_ep1.000.txt'
f= open(gsdPath+inFile, "rb")
contents=f.read()

t150d=float(contents[2290:2296])
t150g=float(contents[2279:2285])

t135d=float(contents[2081:2087])
t135g=float(contents[2070:2076])

t120d=float(contents[1872:1878])
t120g=float(contents[1861:1867])

t105d=float(contents[1663:1669])
t105g=float(contents[1652:1658])

t90d=float(contents[1454:1460])
t90g=float(contents[1443:1449])

t75d=float(contents[1245:1251])
t75g=float(contents[1234:1240])

t60d=float(contents[1036:1042])
t60g=float(contents[1025:1031])

t45d=float(contents[827:833])
t45g=float(contents[816:822])

t30d=float(contents[618:624])
t30g=float(contents[607:613])

t15d=float(contents[409:415])
t15g=float(contents[398:404])

average_dens_d=(t150d+t135d+t120d+t105d+t90d+t75d+t60d+t45d+t30d+t15d)/10
average_dens_g=(t150g+t135g+t120g+t105g+t90g+t75g+t60g+t45g+t30g+t15g)/10
print(average_dens_d)
print(average_dens_g)
print(pa*(xa/100)+pb*(1-(xa/100)))

f.close() 

# pa0_pb50_xa50
# penet=25
# dens = 0.613
# dilute = 0.5795

# pa0_pb150_xa50
# penet=75
# dens = 0.6155
# dilute = 0.554

# pa0_pb300_xa50
# penet=150
# dens = 0.612
# dilute = 0.544

# pa0_pb500_xa50
# penet=250
# dens = 0.612
# dilute = 0.542

# pa50_pb0_xa50
# penet=25
# dens = 0.614
# dilute = 0.567

# pa50_pb50_xa50
# penet=50
# dens = 0.629
# dilute = 0.55

# pa50_pb150_xa50
# penet=100
# dens = 0.614
# dilute = 0.554

# pa50_pb300_xa50
# penet=175
# dens = 0.618
# dilute = 0.546

# pa50_pb500_xa50
# penet=275
# dens = 0.615
# dilute = 0.557

# pa60_pb150_xa50
# penet=105
# dens = 1.0255
# dilute = 0.432

# pa150_pb0_xa50
# penet=75
# dens = 0.617
# dilute = 0.5495

# pa150_pb50_xa50
# penet=100
# dens = 0.6175
# dilute = 0.512

# pa150_pb150_xa50
# penet=150
# dens = 1.1105
# dilute = 0.22

# pa150_pb300_xa50
# penet=225
# dens = 1.212
# dilute = 0.2095

# pa150_pb500_xa50
# penet=325
# dens = 1.2035
# dilute = 0.308

# pa300_pb0_xa50
# penet=150
# dens = 0.614
# dilute = 0.5495

# pa300_pb50_xa50
# penet=175
# dens = 0.614
# dilute = 0.562

# pa300_pb150_xa50
# penet=225
# dens = 1.217
# dilute = 0.2095

# pa300_pb300_xa50
# penet=300
# dens = 1.2735
# dilute = 0.1575

# pa300_pb500_xa50
# penet=400
# dens = 1.3705
# dilute = 0.159

# pa500_pb0_xa50
# penet=250
# dens = 0.615
# dilute = 0.5495

# pa500_pb50_xa50
# penet=275
# dens = 0.614
# dilute = 0.5565

# pa500_pb150_xa50
# penet=325
# dens = 1.2895
# dilute = 0.285

# pa500_pb300_xa50
# penet=400
# dens = 1.371
# dilute = 0.1765

# pa500_pb500_xa50
# penet=500
# dens = 1.42
# dilute = 0.167

no_clust_net_pe = [25, 75, 150, 250, 25, 175, 275, 75, 150, 175, 250, 275, 100]
no_clust_dens_d = [0.613, 0.6155, 0.612, 0.612, 0.614, 0.618, 0.615, 0.617, 0.617, 0.6175, 0.615, 0.614, 0.61 ]
no_clust_dens_g = [0.5795, 0.554, 0.544, 0.542, 0.567, 0.554, 0.546, 0.557, 0.5495, 0.562, 0.5495, 0.5565, 0.544]

clust_net_pe = [105, 150, 225, 325, 225, 300, 400, 325, 400, 500, 120, 400, 132, 430, 220]
clust_dens_d = [1.0255, 1.1105, 1.212, 1.2035, 1.217, 1.2735, 1.3705, 1.2895, 1.371, 1.42, 1.1215, 1.1035, 1.112, 1.4345, 1.246]
clust_dens_g = [0.432, 0.22, 0.2095, 0.308, 0.2095, 0.1575, 0.159, 0.285, 0.1765, 0.167, 0.372, 0.3935, 0.282, 0.2023, 0.286]

not_sure_pe = [50, 100, 100, 78]
not_sure_dens_d = [0.629, 0.614, 0.6175, 0.622]
not_sure_dens_g = [0.55, 0.554, 0.512, 0.533]

plt.plot(no_clust_net_pe, no_clust_dens_g, '.', label = 'No Clustering', color='red')
plt.plot(clust_net_pe, clust_dens_g, '.', label = 'Clustering', color='green')
plt.plot(not_sure_pe, not_sure_dens_g, '.', label = 'Unstable Clustering', color='orange')
plt.plot(no_clust_net_pe, no_clust_dens_d, '.', color='red')
plt.plot(clust_net_pe, clust_dens_d, '.', color='green')
plt.plot(not_sure_pe, not_sure_dens_d, '.', color='orange')
plt.xlabel('Net Pe')
plt.ylabel('Local Area Fraction')
plt.legend()
plt.show()
stop

# pa0_pb150_xa20
# penet=120
# dens = 1.1215
# dilute = 0.372

# pa0_pb500_xa20
# penet=400
# dens = 1.1035
# dilute = 0.3935

# pa60_pb150_xa20
# penet=132
# dens = 1.112
# dilute = 0.282

# pa150_pb500_xa20
# penet=430
# dens = 1.4345
# dilute = 0.2023

# pa0_pb500_xa80
# penet=100
# dens = 0.61
# dilute = 0.544

# pa60_pb150_xa80
# penet=78
# dens = 0.622
# dilute = 0.533

# pa150_pb500_xa80
# penet=220
# dens = 1.246
# dilute = 0.286



[]
stop