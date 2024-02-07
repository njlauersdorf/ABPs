import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Volumes/EXTERNAL2/final_hetero/output_files/compiled_heterogeneity.txt', sep='\s+', header=0)

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



plt.scatter(interpart_peA/interpart_peB, fa_dens_all, c='black', label='all' )
plt.scatter(interpart_peA/interpart_peB, fa_dens_A, c='blue', label='slow')
plt.scatter(interpart_peA/interpart_peB, fa_dens_B, c='red', label='fast')
plt.legend()
plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Active Force Density ($n \alpha \langle F^a \rangle$)')

plt.show()

plt.scatter(interpart_peA/interpart_peB, fa_avg_all, c='black', label='all' )
plt.scatter(interpart_peA/interpart_peB, fa_avg_A, c='blue', label='slow')
plt.scatter(interpart_peA/interpart_peB, fa_avg_B, c='red', label='fast')
plt.legend()
plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Average Active Force ($\alpha \langle F^a \rangle$)')

plt.show()

plt.scatter(interpart_peA/interpart_peB, fa_avg_real_all, c='black', label='all' )
plt.legend()
plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Average Active Force ($\alpha \langle F^a \rangle$)')

plt.show()

plt.scatter(interpart_peA/interpart_peB, num_dens_all, c='black', label='all')
plt.scatter(interpart_peA/interpart_peB, num_dens_A, c='blue', label='slow')
plt.scatter(interpart_peA/interpart_peB, num_dens_B, c='red', label='fast')
plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Number Density ($n$)')
plt.legend()
plt.show()

plt.scatter(interpart_peA/interpart_peB, align_all, c='black', label='all')
plt.scatter(interpart_peA/interpart_peB, align_A, c='blue', label='all')
plt.scatter(interpart_peA/interpart_peB, align_B, c='red', label='all')

plt.xlabel(r'Activity Ratio ($Pe_R$)')
plt.ylabel(r'Alignment ($\alpha$)')
plt.legend()
plt.show()