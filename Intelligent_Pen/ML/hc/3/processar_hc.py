import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as sta
import glob
from sklearn import *
import scipy
from librosa import *


class feat(object):
  def __init__ (self, kwargs):
   
    for key, values in kwargs.items():
      setattr(self, key, values)
    

  def features_ (self):
   
    feats = {}

	#MEDIA

    media = []
    media.append(np.mean(self.AcX))
    media.append(np.mean(self.AcY))
    media.append(np.mean(self.AcZ))
    media.append(np.mean(self.GyX))
    media.append(np.mean(self.GyY))
    media.append(np.mean(self.GyZ))

    feats.update({'media_AcX':[media[0]] })
    feats.update({'media_AcY':[media[1]] })
    feats.update({'media_AcZ':[media[2]] })
    feats.update({'media_GyX':[media[3]] })
    feats.update({'media_GyY':[media[4]] })
    feats.update({'media_GyZ':[media[5]] })


	#MIN

    minimo = []
    minimo.append(np.amin(self.AcX))
    minimo.append(np.amin(self.AcY))
    minimo.append(np.amin(self.AcZ))
    minimo.append(np.amin(self.GyX))
    minimo.append(np.amin(self.GyY))
    minimo.append(np.amin(self.GyZ))


    feats.update({'minimo_AcX':[minimo[0]] })
    feats.update({'minimo_AcY':[minimo[1]] })
    feats.update({'minimo_AcZ':[minimo[2]] })
    feats.update({'minimo_GyX':[minimo[3]] })
    feats.update({'minimo_GyY':[minimo[4]] })
    feats.update({'minimo_GyZ':[minimo[5]] })
	
	#MAX

    maximo = []
    maximo.append(np.amax(self.AcX))
    maximo.append(np.amax(self.AcY))
    maximo.append(np.amax(self.AcZ))
    maximo.append(np.amax(self.GyX))
    maximo.append(np.amax(self.GyY))
    maximo.append(np.amax(self.GyZ))


    feats.update({'maximo_AcX':[maximo[0]] })
    feats.update({'maximo_AcY':[maximo[1]] })
    feats.update({'maximo_AcZ':[maximo[2]] })
    feats.update({'maximo_GyX':[maximo[3]] })
    feats.update({'maximo_GyY':[maximo[4]] })
    feats.update({'maximo_GyZ':[maximo[5]] })
	
	#MEDIANA

    mediana = []
    mediana.append(np.median(self.AcX))
    mediana.append(np.median(self.AcY))
    mediana.append(np.median(self.AcZ))
    mediana.append(np.median(self.GyX))
    mediana.append(np.median(self.GyY))
    mediana.append(np.median(self.GyZ))


    feats.update({'mediana_AcX':[mediana[0]] })
    feats.update({'mediana_AcY':[mediana[1]] })
    feats.update({'mediana_AcZ':[mediana[2]] })
    feats.update({'mediana_GyX':[mediana[3]] })
    feats.update({'mediana_GyY':[mediana[4]] })
    feats.update({'mediana_GyZ':[mediana[5]] })

	#DESVIO_PADRAO

    std = []
    std.append(np.std(self.AcX))
    std.append(np.std(self.AcY))
    std.append(np.std(self.AcZ))
    std.append(np.std(self.GyX))
    std.append(np.std(self.GyY))
    std.append(np.std(self.GyZ))


    feats.update({'std_AcX':[std[0]] })
    feats.update({'std_AcY':[std[1]] })
    feats.update({'std_AcZ':[std[2]] })
    feats.update({'std_GyX':[std[3]] })
    feats.update({'std_GyY':[std[4]] })
    feats.update({'std_GyZ':[std[5]] })


	#VARIANCIA

    var = []
    var.append(np.var(self.AcX))
    var.append(np.var(self.AcY))
    var.append(np.var(self.AcZ))
    var.append(np.var(self.GyX))
    var.append(np.var(self.GyY))
    var.append(np.var(self.GyZ))


    feats.update({'var_AcX':[var[0]] })
    feats.update({'var_AcY':[var[1]] })
    feats.update({'var_AcZ':[var[2]] })
    feats.update({'var_GyX':[var[3]] })
    feats.update({'var_GyY':[var[4]] })
    feats.update({'var_GyZ':[var[5]] })

	#RMS

    rms = []
    rms.append(np.sqrt(media[0]**2))
    rms.append(np.sqrt(media[1]**2))
    rms.append(np.sqrt(media[2]**2))
    rms.append(np.sqrt(media[3]**2))
    rms.append(np.sqrt(media[4]**2))
    rms.append(np.sqrt(media[5]**2))


    feats.update({'rms_AcX':[rms[0]] })
    feats.update({'rms_AcY':[rms[1]] })
    feats.update({'rms_AcZ':[rms[2]] })
    feats.update({'rms_GyX':[rms[3]] })
    feats.update({'rms_GyY':[rms[4]] })
    feats.update({'rms_GyZ':[rms[5]] })


	#SKEWNESS

    skw = []
    skw.append(scipy.stats.skew(np.array(self.AcX).T))
    skw.append(scipy.stats.skew(np.array(self.AcY).T))
    skw.append(scipy.stats.skew(np.array(self.AcZ).T))
    skw.append(scipy.stats.skew(np.array(self.GyX).T))
    skw.append(scipy.stats.skew(np.array(self.GyY).T))
    skw.append(scipy.stats.skew(np.array(self.GyZ).T))


    feats.update({'skw_AcX':[skw[0]] })
    feats.update({'skw_AcY':[skw[1]] })
    feats.update({'skw_AcZ':[skw[2]] })
    feats.update({'skw_GyX':[skw[3]] })
    feats.update({'skw_GyY':[skw[4]] })
    feats.update({'skw_GyZ':[skw[5]] })

        #KURTOSIS

    krt = []
    krt.append(scipy.stats.kurtosis(np.array(self.AcX).T))
    krt.append(scipy.stats.kurtosis(np.array(self.AcY).T))
    krt.append(scipy.stats.kurtosis(np.array(self.AcZ).T))
    krt.append(scipy.stats.kurtosis(np.array(self.GyX).T))
    krt.append(scipy.stats.kurtosis(np.array(self.GyY).T))
    krt.append(scipy.stats.kurtosis(np.array(self.GyZ).T))


    feats.update({'krt_AcX':[krt[0]] })
    feats.update({'krt_AcY':[krt[1]] })
    feats.update({'krt_AcZ':[krt[2]] })
    feats.update({'krt_GyX':[krt[3]] })
    feats.update({'krt_GyY':[krt[4]] })
    feats.update({'krt_GyZ':[krt[5]] })
    
	#MODA

    moda = []
    moda.append(scipy.stats.mode(np.array(self.AcX), axis = None))
    moda.append(scipy.stats.mode(np.array(self.AcY), axis = None))
    moda.append(scipy.stats.mode(np.array(self.AcZ), axis = None))
    moda.append(scipy.stats.mode(np.array(self.GyX), axis = None))
    moda.append(scipy.stats.mode(np.array(self.GyY), axis = None))
    moda.append(scipy.stats.mode(np.array(self.GyZ), axis = None))


    feats.update({'moda_AcX':[moda[0][0]] })
    feats.update({'moda_AcY':[moda[1][0]] })
    feats.update({'moda_AcZ':[moda[2][0]] })
    feats.update({'moda_GyX':[moda[3][0]] })
    feats.update({'moda_GyY':[moda[4][0]] })
    feats.update({'moda_GyZ':[moda[5][0]] })

	#TRIM_MEAN

    t_mean = []
    t_mean.append(scipy.stats.trim_mean(np.array(self.AcX), 0.15, axis=None))
    t_mean.append(scipy.stats.trim_mean(np.array(self.AcY), 0.15, axis=None))
    t_mean.append(scipy.stats.trim_mean(np.array(self.AcZ), 0.15, axis=None))
    t_mean.append(scipy.stats.trim_mean(np.array(self.GyX), 0.15, axis=None))
    t_mean.append(scipy.stats.trim_mean(np.array(self.GyY), 0.15, axis=None))
    t_mean.append(scipy.stats.trim_mean(np.array(self.GyZ), 0.15, axis=None))

    feats.update({'t_mean_AcX':[t_mean[0]] })
    feats.update({'t_mean_AcY':[t_mean[1]] })
    feats.update({'t_mean_AcZ':[t_mean[2]] })
    feats.update({'t_mean_GyX':[t_mean[3]] })
    feats.update({'t_mean_GyY':[t_mean[4]] })
    feats.update({'t_mean_GyZ':[t_mean[5]] })

    
	#TEMPO (s)

    tempo = []
    tempo = np.amax(self.miliseg) / 1000

    feats.update({'Tempo':[tempo] })
   
     
    '''

	#ZCR
   
    zero = []
    zero.append(lir.feature.zero_crossing_rate(np.array(self.AcX)))
    print(zero)
    '''

    '''
	#ENTROPIA

   
    entrop = []
    ent1 = np.split(np.array(self.AcX).T,1)
    ent2 = np.array(len(self.AcX))
    print(len(ent1))

    entrop.append(scipy.stats.entropy(ent1, ent2 ) )   
    print("entro", entrop)
    '''
    #entrop = []
    #entrop.append(scipy.stats.entropy(np.array(self.AcX).T))

   # print("Trim", entrop)
    '''
    tur = np.split(self.AcY,1)
    print(tra)
    #tur = list(map(float, np.split(self.AcY)))
    print(tur)
    print(hc_df[0].AcY)
    '''

    #for k, v in feats.items():
    # print("%s"%k, k,v)
    

    return feats
    
    #self.med = np.mean(4)
    


def normalizar(daframe):
 
 
  df_normalizado = {}
  
  Tempo = np.array(daframe['miliseg'])
  AcX_N = np.array(daframe['AcX'])
  AcY_N = np.array(daframe['AcY'])
  AcZ_N = np.array(daframe['AcZ'])
  GyX_N = np.array(daframe['GyX'])
  GyY_N = np.array(daframe['GyY'])
  GyZ_N = np.array(daframe['GyZ'])

  AcX_Normal = preprocessing.normalize([AcX_N])
  AcY_Normal = preprocessing.normalize([AcY_N])
  AcZ_Normal = preprocessing.normalize([AcZ_N])
  GyX_Normal = preprocessing.normalize([GyX_N])
  GyY_Normal = preprocessing.normalize([GyY_N])
  GyZ_Normal = preprocessing.normalize([GyZ_N])

  df_normalizado['miliseg'] = Tempo
  df_normalizado['AcX'] = AcX_Normal 
  df_normalizado['AcY'] = AcY_Normal 
  df_normalizado['AcZ'] = AcZ_Normal 
  df_normalizado['GyX'] = GyX_Normal 
  df_normalizado['GyY'] = GyY_Normal 
  df_normalizado['GyZ'] = GyZ_Normal 
  
  return df_normalizado

def minmax(frame):
 
 
  df_st = {}
  
  Tempo = frame.miliseg.values
  AcX_S = frame.AcX.values
  AcY_S = frame.AcY.values
  AcZ_S = frame.AcZ.values
  GyX_S = frame.GyX.values
  GyY_S = frame.GyY.values
  GyZ_S = frame.GyZ.values

  AcX_S = AcX_S.reshape((len(AcX_S),1))
  xsx = preprocessing.MinMaxScaler(feature_range=(0,1))
  xsx = xsx.fit_transform(AcX_S)
  #print("TNT",xsx)

  AcY_S = AcY_S.reshape((len(AcY_S),1))
  AcZ_S = AcZ_S.reshape((len(AcZ_S),1))
  GyX_S = GyX_S.reshape((len(GyX_S),1))
  GyY_S = GyY_S.reshape((len(GyY_S),1))
  GyZ_S = GyZ_S.reshape((len(GyZ_S),1))

  


  escala = preprocessing.MinMaxScaler(feature_range = (0,1))


  AcX_St = escala.fit_transform([AcX_S])
  #AcY_St = escala.fit_transform([AcY_S])
  #AcZ_St = escala.fit_transform([AcZ_S])
  #GyX_St = escala.fit_transform([GyX_S])
  #GyY_St = escala.fit_transform([GyY_S])
  #GyZ_St = escala.fit_transform([GyZ_S])
  #print("AcX", AcX_St[0][0], AcX_St[0][1])

  #df_st['miliseg'] = Tempo
  df_st['AcX'] = AcX_St
  #df_st['AcY'] = AcY_Normal 
  #df_st['AcZ'] = AcZ_Normal 
  #df_st['GyX'] = GyX_Normal 
  #df_st['GyY'] = GyY_Normal 
  #df_st['GyZ'] = GyZ_Normal
  
  return df_st



#Receber Dados


hc_files = sorted(glob.glob('n*.xlsx'))


hc_df={}
pd_df={}
normal_hc={}


#for x in range (0,19):

 #hc_df[x] = pd.read_excel(open(hc_files[x], 'rb'), sheet_name='Simple Data')
 #pd_df[x] = pd.read_excel(open(pd_files[x], 'rb'), sheet_name='Simple Data')


#for x in range (0,len(hc_df)):
# normal_hc[x] = 0


testen = {}
testet = {}
'''
entrop = []
ent1 = np.array(hc_df[0].AcX)
ent2 = len(hc_df[0].AcX)
print("LEEB",len(ent1), ent2)

print("shape", ent1.shape)
entrop.append(scipy.stats.entropy(ent1, ent2 ) )   
print("entro", entrop)
'''


#tst = minmax(hc_df[0])

#print("TST", tst)


#Criar SuperTabela

stabela = []
tabela = {}
tabela_lista = []

m = 0
n = 0
ind = 0 
k = 0
dicionario = []

for i in range(m):
 dicionario.append([0] * n)

for x in range (0,len(hc_files)):

 hc_df[x] = pd.read_excel(open(hc_files[x], 'rb'), sheet_name='Simple Data')
 normalizando = normalizar(hc_df[x])
 testet = feat(normalizando)
 tabela = testet.features_()
 #dicionario = tabela
 #dicionario.update(dicionario)

# tabela_lista.append = tabela
# tabela_lista = list(tabela.values(AcX))[0][0]
# stabela = pd.DataFrame(data=tabela, index=[x])
 
 for y in range(len(tabela.values()) ):
  valor = list(tabela.values())[y] 
  #print(y, valor)

 #print("Paciente %s"%x, list(tabela)[0], 'valors', list(tabela.values())[0])

 #key = k in range (len(tabela.values()) )
 dicionario.append([ list(tabela.values())[h] for h in range (len(tabela.values() )) ])
 #print(k)

 

#print(dicionario)

stabela = pd.DataFrame(dicionario)

for feature in range (len(stabela.columns)):
 for paciente in range (len(stabela.index)):
   stabela[feature]= stabela[feature].astype(str).str.strip('[array([])]').astype(float)

  
#stabela = stabela.drop(columns ='Unnamed: 0')

stabela.to_excel('SuperTabela.xlsx')


#xtabela.to_excel('SuperTabela.xlsx')

#xske = scipy.stats.skew(hc_df[0].AcX)





#ttt = testet.media('AcX')


















#terere2 = list(terere.values())[1][0]


#print("TESTE T", testet.features_().get('medias'))
#print("TESTE T", terere1)

#print("TESTE N",len(testen[1]))

#print("LOL", mmm)  

#sq = np.squeeze(testen)
#dd = pd.DataFrame(data=sq)
#print("sq, sqlen", sq)


#exc = pd.DataFrame(data=testen)

#exc.to_excel("Teste.xlsx", header=['miliseg','AcX','AcY','AcZ','GyX','GyY','GyZ'])





#Features

#Min_Max

x_min = {}
x_max = {}

#Media_Mediana

x_mean = {}
x_median = {}

#Variancia_DesvioP

y_var = {}
y_std = {}

#RMS

x_rms = {}

#GeoMean_Harmonic Mean [MinMax]

x_gmean = {}
x_hmean = {}

#Kurtosis_Skewness

#Moda

#TrimMean

#Entropia

#Coef.Assimetrico

#Range

#ZCR [Standardize]

#MCR

#

y_mean = {}
y_median = {}



x_average = {}


x_var = {}
x_std = {}

for x in range (0,5):

 x_min[x] = np.amin(hc_df[x].AcX)
 x_max[x] = np.amax(hc_df[x].AcX)
 x_mean[x] = np.mean(hc_df[x].AcX)
 x_median[x] = np.median(hc_df[x].AcX)
 '''
 y_var[x] = np.var(pd_df[x].AcX)
 y_std[x] = np.std(pd_df[x].AcX)
 y_mean[x] = np.mean(pd_df[x].AcX)
 y_median[x] = np.median(pd_df[x].AcX)
 '''

 # Elementos devem ser maior que 0 [MinMax de 0 a 1] x_hmean[x] = scipy.stats.hmean(hc_df[x].AcX)
 
#ANOTACOES
#Conferir X_mean ; X_Median ; 




 x_average[x] = np.average(hc_df[x].AcX)

 x_rms[x] = np.sqrt(np.mean(hc_df[x].AcX**2))

#Mesmo que Hmean x_gmean[x] = scipy.stats.gmean(hc_df[x].AcX)

 x_var[x] = np.var(hc_df[x].AcX)
 x_std[x] = np.std(hc_df[x].AcX)
 
'''
plt.figure("figure 1")
plt.plot(hc_df[0].miliseg, hc_df[0].AcX)

plt.figure("figure 1")
plt.plot(testet[0].miliseg, testet[0].AcX) 


plt.show()
'''
'''
print(x_min)
print(x_max)
print(x_mean)
print(x_median)

print("PD vAR",y_var)
print("PD stsd",y_std)
print("PD mean",y_mean)
print("PD median",y_median)

#print(x_hmean)
print(x_average)

print(x_rms)

#print(x_gmean)

print(x_var)
print(x_std)


#STANDARDIZAR

valor = pd_df[1].AcX.values
valor1 = pd_df[2].AcX.values

valor = valor.reshape((len(valor),1))
valor1 = valor1.reshape((len(valor1),1))

scaler = preprocessing.StandardScaler()
scaler1 = preprocessing.StandardScaler()

scaled = scaler.fit_transform(valor)
scaled1 = scaler1.fit_transform(valor1)

'''
#TESTE NORMALIZAR
'''
#Normalizar AcX

V0 = pd_df[1].AcX
V2 = pd_df[2].AcX

pv0 = preprocessing.MinMaxScaler(feature_range = (0,1))
pv0 = pv0.fit_transform(valor)

pv2 = preprocessing.MinMaxScaler(feature_range = (0,1))
pv2 = pv2.fit_transform(valor1)

V0N = pv0#preprocessing.normalize([V0])
V2N = pv2#preprocessing.normalize([V2])


N0 = preprocessing.normalize([V0])
N2 = preprocessing.normalize([V2])

res1 = N0#.reshape((len(N0),1))
res2 = N2#.reshape((len(N2),1))

rs1 = preprocessing.MinMaxScaler(feature_range = (0,1))
rs2 = preprocessing.MinMaxScaler(feature_range = (0,1))

r1 = rs1.fit_transform(res1)
r2 = rs2.fit_transform(res2)
'''
X0 = hc_df[0].AcX
NN0 = preprocessing.normalize([X0])

#plt.figure("T")
#plt.plot(hc_df[0].miliseg, NN0.T)
'''
print("r1,r2", r1, r2)

plt.figure('Teste')
plt.plot(pd_df[1].miliseg, N0.T)
 
plt.figure('Normal')
plt.plot(pd_df[1].miliseg, V0N)

plt.figure('Normal')
plt.plot(pd_df[2].miliseg, V2N)

plt.figure('Figure')
plt.plot( hc_df[0].miliseg, hc_df[0].AcX)

plt.figure('Figure')
plt.plot( hc_df[2].miliseg, hc_df[2].AcX)

plt.figure('Figure NPD')
plt.plot( pd_df[1].miliseg, r1.T)


plt.figure('Figure NPD')
plt.plot( pd_df[2].miliseg, r2.T)


plt.figure('Figure PD')
plt.plot( pd_df[1].miliseg, scaled)

plt.figure('Figure PD')
plt.plot( pd_df[2].miliseg, scaled1)
Paciente 0 
plt.figure('Figure PD_N')
plt.plot( pd_df[1].miliseg, pd_df[1].AcX)

plt.figure('Figure PD_N')
plt.plot( pd_df[2].miliseg, pd_df[2].AcX)

print("MinMax",np.amin(scaled), np.amin(scaled1), np.amax(scaled), np.amax(scaled1))


print("VarStd",np.var(V0N), np.var(V2N), np.std(V0N), np.std(V2N))


varx = np.var(V0N)
stdx = np.std(V0N)

vary = np.var(V2N)
stdy = np.std(V2N)

print("Var 0:", varx)
print("Std 0:", stdx)
print("Var 3:", vary)
print("Std 3:", stdy)
'''







'''

for y in range (0,5):

 plt.figure('Figure'+str(y))
 plt.plot(hc_df[y].miliseg, hc_df[y].AcX)

plt.show()
'''
