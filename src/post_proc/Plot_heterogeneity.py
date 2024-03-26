import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('/Volumes/External/hetero_final_test/output_files/compiled_heterogeneity_dif_Ra.txt', sep='\s+', header=0)

headers = df.columns.tolist()



interpart_peA = df['pa'].values
interpart_peB = df['pb'].values
fa_dens_all = df['fa_dens_all'].values
fa_dens_A = df['fa_dens_A'].values
fa_dens_B = df['fa_dens_B'].values
fa_avg_all = df['fa_avg_all'].values
fa_avg_A = df['fa_avg_A'].values
fa_avg_B = df['fa_avg_B'].values
fa_avg_real_all = df['fa_avg_real_all'].values
num_dens_all = df['num_dens_all'].values
num_dens_A = df['num_dens_A'].values
num_dens_B = df['num_dens_B'].values
align_all = df['align_all'].values
align_A = df['align_A'].values
align_B = df['align_B'].values


fast_act = [200, 300, 350, 400, 450, 500]
#fast_act = [500]
for i in range(0, len(fast_act)):
    test_id = np.where((interpart_peB==fast_act[i]) & (0.5 * interpart_peA + interpart_peB * 0.5>=150))[0]
    if len(test_id)>0:
        mono_id = np.where((interpart_peB==fast_act[i]) & (interpart_peA==fast_act[i]))[0]
        if len(mono_id)==1:

            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], fa_dens_all[test_id]/fa_dens_all[mono_id], c='black', label='all' )
            #plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], fa_dens_A[test_id]/fa_dens_A[mono_id], c='blue', label='slow')
            #plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], fa_dens_B[test_id]/fa_dens_B[mono_id], c='red', label='fast')
#plt.legend()
plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Body Force Density Fluctuations ($\Lambda(n \alpha F^a)$)')
plt.plot([-1, 1], [1,1], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.175, 0.175], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.35, 0.35], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')

plt.xlim([-0.02,1.02])

plt.ylim([0,3])

plt.show()

for i in range(0, len(fast_act)):
    test_id = np.where((interpart_peB==fast_act[i]) & (0.5 * interpart_peA + interpart_peB * 0.5>=150))[0]
    if len(test_id)>0:
        mono_id = np.where((interpart_peB==fast_act[i]) & (interpart_peA==fast_act[i]))[0]
        if len(mono_id)==1:

            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], fa_avg_real_all[test_id]/fa_avg_real_all[mono_id], c='black', label='all' )

#plt.legend()
plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Average Active Force Fluctuations ($\Lambda(F^a)$)')
plt.plot([-1, 1], [1,1], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.175, 0.175], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.35, 0.35], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.ylim([0.9,1.2])
plt.xlim([-0.02,1.02])

plt.show()

for i in range(0, len(fast_act)):
    test_id = np.where((interpart_peB==fast_act[i]) & (0.5 * interpart_peA + interpart_peB * 0.5>=150))[0]
    if len(test_id)>0:
        mono_id = np.where((interpart_peB==fast_act[i]) & (interpart_peA==fast_act[i]))[0]
        if len(mono_id)==1:
            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], num_dens_all[test_id]/num_dens_all[mono_id], c='black', label='all' )
            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], num_dens_A[test_id]/num_dens_A[mono_id], c='blue', label='slow')
            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], num_dens_B[test_id]/num_dens_B[mono_id], c='red', label='fast')

plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Number Density Fluctuations ($\Lambda(n)$)')
plt.plot([-1, 1], [1,1], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.175, 0.175], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.35, 0.35], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.xlim([-0.02,1.02])

plt.ylim([0,6])

#plt.legend()
plt.show()

for i in range(0, len(fast_act)):
    test_id = np.where((interpart_peB==fast_act[i]) & (0.5 * interpart_peA + interpart_peB * 0.5>=150))[0]
    if len(test_id)>0:
        mono_id = np.where((interpart_peB==fast_act[i]) & (interpart_peA==fast_act[i]))[0]
        if len(mono_id)==1:

            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], align_all[test_id]/align_all[mono_id], c='black', label='all' )
            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], align_A[test_id]/align_A[mono_id], c='blue', label='slow')
            plt.scatter(interpart_peA[test_id]/interpart_peB[test_id], align_B[test_id]/align_B[mono_id], c='red', label='fast')

plt.ylim([0,6])

plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Alignment Fluctuations ($\Lambda(\alpha)$)')
plt.plot([-1, 1], [1,1], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.175, 0.175], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.plot([0.35, 0.35], [-10000,20000], linewidth=1.0, linestyle='dashed', color='gray')
plt.xlim([-0.02,1.02])

#plt.legend()
plt.show()