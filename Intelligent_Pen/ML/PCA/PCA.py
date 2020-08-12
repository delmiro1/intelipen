import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as sta
import glob
from sklearn import *
import scipy


df = pd.read_excel(open('DataFinal6.xlsx', 'rb'), sheet_name='Sheet1')
df_target = df.loc[:,'prk']
df_values = df.drop(columns=['prk','Unnamed: 0'])


#FILTRO (Apenas Acelerometro, Apenas Giroscopio, Ac+Gy)

Filtro = df_values.filter(like='Ac').columns

for nomes in Filtro:
 df_values.drop(columns = nomes, inplace = True)

pca = decomposition.PCA(n_components = 2)
pca.fit(df_values)
x_pca = pca.transform(df_values)

print(pca.explained_variance_ratio_)
exit()



#Scatter_Plot

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('\nPrincipal Componente 1', fontsize='12')
ax.set_ylabel('Principal Componente 2', fontsize='12')
ax.set_title('Principal Componente Análise', fontsize='16')


df1 = pd.DataFrame(data = x_pca, columns=['1','2'])


fdf = pd.concat([df1, df_target] ,   axis=1)

targets = ['HC', 'PD']
colors = ['r', 'b']
for target, color in zip(targets, colors):
  
  indicesToKeep = fdf['prk'] == target
  ax.scatter(fdf.loc[indicesToKeep, '1'], fdf.loc[indicesToKeep, '2'], c= color, s=40)

ax.legend(targets)
ax.grid()

plt.show()

exit()



