import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as sta
import glob
from sklearn import *
import scipy
import seaborn as sns
import time
from collections import defaultdict
from pylab import savefig

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score





#MERGE DICIONARIOS

def mergeDict(dict1, dict2):
 dict3 = {**dict1, **dict2}
 for key, value in dict3.items():
  if key in dict1 and key in dict2:
    dict3[key] = [value, dict1[key]]
 return dict3

#ROC FUNCTION

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
#ROC TABLE

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


#INICIALIZAR VARIAVEIS

Data = {}
qe = 1
tempo_treino = {}
tempo_teste = {}
datafil = []

#RECEBER DADOS _TODAS AS QUESTOES

for questao in range(6): 
    Data[questao] = pd.read_excel(open('TabelaFinal.xlsx', 'rb'), sheet_name=('q'+str(questao+1)))

###Filtrar por nome em coluna Data[0].filter(like='AcX').columns

'''
#FILTRO (Apenas Acelerometro, Apenas Giroscopio, Ac+Gy)

Filtro = Data[qe].filter(like='Gy').columns

for nomes in Filtro:
 Data[qe].drop(columns = nomes, inplace = True)
'''



##SPLIT_TRAIN + PCA

array_data = Data[qe].values[:,:len(Data[qe].columns)  -1]
pca = decomposition.PCA(2)
array = pca.fit_transform(array_data)

X = array
Y = Data[qe].values[:,len(Data[qe].columns)-1 :len(Data[qe].columns)].ravel()

#print('\n'"Taxa PCA", pca.explained_variance_ratio_)
#print('\n\n\n\n')
#print(pca.n_components_)


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.75, random_state=2)


# Spot Check Algorithms

models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale')))



# evaluate each model in turn

results = []
names = []
for name, model in models:
        start = time.process_time()
        kfold = model_selection.StratifiedKFold(n_splits = 5,  random_state = 17)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        

        
        
        
        end = time.process_time()
        tempo_treino.update({name : (end-start)*100000 })
        results.append(cv_results)
        names.append(name)
        msg = "%s: Mean = %.2f || Std = (%.2f) || Processamento = (%f)s" % (name, cv_results.mean(), cv_results.std(), end-start)

        print(msg)

'''
#Comparar Algoritmos

plt.boxplot(results, labels = names)
plt.title('Algoritmos Comp.')

plt.show()
#Fazer Predicao

for name, model in models:

        print(list(models))

models= [SVC(gamma='auto')]
'''


model1 = LinearDiscriminantAnalysis()
model2 = LogisticRegression(solver='liblinear', multi_class='ovr')
model3 = KNeighborsClassifier()
model4 = DecisionTreeClassifier()
model5 = GaussianNB()
model6 = SVC(gamma='auto', probability=True)


print("")

print("")

print("")

print("")

print("")

print("")


print("						LDA")
model1.fit(X_train, Y_train)


#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model1.predict_proba(X_validation)[::,1]
fpr, tpr, _ = roc_curve(rocy, probs)
auc = roc_auc_score(rocy, probs)
result_table = result_table.append({'classifiers':model1.__class__.__name__,'fpr':fpr,'tpr':tpr,'auc':auc}, ignore_index=True)





'''
#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model1.predict_proba(X_validation)
probs = probs[:,1]
auc = roc_auc_score(rocy, probs)

                  
print(rocy)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(rocy, probs)
print('fpr', fpr)
plot_roc_curve(fpr, tpr)
#END_ROC
'''	

t_start = time.process_time()

predictions = model1.predict(X_validation)

t_end = time.process_time()

tempo_teste.update({'LDA' : (t_end-t_start)*100000 })




print("MATRIX DE CONFUSAO\n",metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))

#Classification Report

report = metrics.classification_report(Y_validation, predictions, output_dict=True)
df = pd.DataFrame(report).transpose()

df = df.append([metrics.accuracy_score(Y_validation, predictions)] )
df.columns.values[len(df.columns)-1] = 'Accuracy'
df.index.values[len(df.index)-1] = 'Accuracy'


#Heatmaps_LDA

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.subplot()
figura = sns.heatmap(matrix, annot = True, ax = ax, fmt='g',annot_kws={"size": 27})

ax.set_xlabel('True', size = 15);ax.set_ylabel('Predict', size = 15);
ax.set_title("LDA", size = 21 );
ax.xaxis.set_ticklabels(['HC','PD'], size = 15); ax.yaxis.set_ticklabels(['HC','PD'], size = 13);

#figure = figura.get_figure()
#figure.savefig('ML.jpg')


'''
#Heatmap

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.subplot()
sns.heatmap(matrix, annot = True, ax = ax, fmt='g')

ax.set_xlabel('True');ax.set_ylabel('Predict');
ax.set_title('LDA');
ax.xaxis.set_ticklabels(['HC','PD']); ax.yaxis.set_ticklabels(['HC','PD']);

#Scatter_Plot

figpca = plt.figure(10)
figpca = plt.figure(figsize = (8,8))
ax = figpca.add_subplot(1,1,1)
ax.set_xlabel('\nPrincipal Componente 1', fontsize='15')
ax.set_ylabel('Principal Componente 2', fontsize='15')
ax.set_title('Principal Componente Análise', fontsize='16')

df = pd.DataFrame(data = X, columns=['1','2'])



fdf = pd.concat([df, Data[qe]['prk']] ,   axis=1)


targets = ['HC', 'PD']
colors = ['r', 'b']
for target, color in zip(targets, colors):
  
  indicesToKeep = fdf['prk'] == target
  ax.scatter(fdf.loc[indicesToKeep, '1'], fdf.loc[indicesToKeep, '2'], c= color, s=40)

ax.legend(targets)
ax.grid()

plt.show()
exit()
'''


print("						LR")

t_start = time.process_time()


model2.fit(X_train, Y_train)

#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model2.predict_proba(X_validation)[::,1]
fpr, tpr, _ = roc_curve(rocy, probs)
auc = roc_auc_score(rocy, probs)
result_table = result_table.append({'classifiers':model2.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)


#END_ROC


predictions = model2.predict(X_validation)

t_end = time.process_time()

tempo_teste.update({'LR' : (t_end-t_start)*100000 })



print("MATRIX DE CONFUSAO\n",metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))


#Classification Report

report = metrics.classification_report(Y_validation, predictions, output_dict=True)
df1 = pd.DataFrame(report).transpose()	
df1 = df1.append([metrics.accuracy_score(Y_validation, predictions)] )
df1.columns.values[len(df1.columns)-1] = 'Accuracy'
df1.index.values[len(df1.index)-1] = 'Accuracy'



#Heatmaps_LR

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.figure(2)
ax = plt.subplot()
sns.heatmap(matrix, annot = True, ax = ax, fmt='g',annot_kws={"size": 27})
ax.set_xlabel('True', size = 15);ax.set_ylabel('Predict', size = 15);
ax.set_title("LR", size = 21 );
ax.xaxis.set_ticklabels(['HC','PD'], size = 15); ax.yaxis.set_ticklabels(['HC','PD'], size = 13);



print("						KNN")

t_start = time.process_time()


model3.fit(X_train, Y_train)
#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model3.predict_proba(X_validation)[::,1]
fpr, tpr, _ = roc_curve(rocy, probs)
auc = roc_auc_score(rocy, probs)
result_table = result_table.append({'classifiers':model3.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
#ROC_END

predictions = model3.predict(X_validation)



t_end = time.process_time()

tempo_teste.update({'KNN' : (t_end-t_start)*100000 })


print("MATRIX DE CONFUSAO\n",metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))


#Classification Report

report = metrics.classification_report(Y_validation, predictions, output_dict=True)
df2 = pd.DataFrame(report).transpose()	
df2 = df2.append([metrics.accuracy_score(Y_validation, predictions)] )
df2.columns.values[len(df2.columns)-1] = 'Accuracy'
df2.index.values[len(df2.index)-1] = 'Accuracy'



#Heatmaps_KNN

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.figure(3)
ax = plt.subplot()
sns.heatmap(matrix, annot = True, ax = ax, fmt='g',annot_kws={"size": 27})
ax.set_xlabel('True', size = 15);ax.set_ylabel('Predict', size = 15);
ax.set_title("KNN", size = 21 );
ax.xaxis.set_ticklabels(['HC','PD'], size = 15); ax.yaxis.set_ticklabels(['HC','PD'], size = 13);




print("						DecisionTree")


t_start = time.process_time()

model4.fit(X_train, Y_train)
#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model4.predict_proba(X_validation)[::,1]
fpr, tpr, _ = roc_curve(rocy, probs)
auc = roc_auc_score(rocy, probs)
result_table = result_table.append({'classifiers':model4.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
#ROC_END

predictions = model4.predict(X_validation)


t_end = time.process_time()

tempo_teste.update({'CART' : (t_end-t_start)*100000 })


print("MATRIX DE CONFUSAO\n",metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))


#Classification Report

report = metrics.classification_report(Y_validation, predictions, output_dict=True)
df3 = pd.DataFrame(report).transpose()	
df3 = df3.append([metrics.accuracy_score(Y_validation, predictions)] )
df3.columns.values[len(df3.columns)-1] = 'Accuracy'
df3.index.values[len(df3.index)-1] = 'Accuracy'

#Heatmaps_DT

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.figure(4)
ax = plt.subplot()
sns.heatmap(matrix, annot = True, ax = ax, fmt='g',annot_kws={"size": 27})
ax.set_xlabel('True', size = 15);ax.set_ylabel('Predict', size = 15);
ax.set_title("CART", size = 21 );
ax.xaxis.set_ticklabels(['HC','PD'], size = 15); ax.yaxis.set_ticklabels(['HC','PD'], size = 13);



print("						GaussianNB")

t_start = time.process_time()


model5.fit(X_train, Y_train)

#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model5.predict_proba(X_validation)[::,1]
fpr, tpr, _ = roc_curve(rocy, probs)
auc = roc_auc_score(rocy, probs)
result_table = result_table.append({'classifiers':model5.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
#ROC_END

predictions = model5.predict(X_validation)


t_end = time.process_time()

tempo_teste.update({'NB' : (t_end-t_start)*100000 })




print("MATRIX DE CONFUSAO\n",metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))


#Classification Report

report = metrics.classification_report(Y_validation, predictions, output_dict=True)
df4 = pd.DataFrame(report).transpose()	
df4 = df4.append([metrics.accuracy_score(Y_validation, predictions)] )
df4.columns.values[len(df4.columns)-1] = 'Accuracy'
df4.index.values[len(df4.index)-1] = 'Accuracy'


#Heatmaps_NB

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.figure(5)
ax = plt.subplot()
sns.heatmap(matrix, annot = True, ax = ax, fmt='g',annot_kws={"size": 27})
ax.set_xlabel('True', size = 15);ax.set_ylabel('Predict', size = 15);
ax.set_title("NB", size = 21 );
ax.xaxis.set_ticklabels(['HC','PD'], size = 15); ax.yaxis.set_ticklabels(['HC','PD'], size = 13);


print("						SVC")


t_start = time.process_time()

model6.fit(X_train, Y_train)

#ROC
rocy = []
for ele in Y_validation:
        if ele == 'PD':
               rocy.append(1)
        else:   
               rocy.append(0)

probs = model6.predict_proba(X_validation)[::,1]
fpr, tpr, _ = roc_curve(rocy, probs)
auc = roc_auc_score(rocy, probs)
result_table = result_table.append({'classifiers':model6.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
#ROC_END
predictions = model6.predict(X_validation)


t_end = time.process_time()

tempo_teste.update({'SVM' : (t_end-t_start)*100000 })







print("MATRIX DE CONFUSAO\n",metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))


#Classification Report

report = metrics.classification_report(Y_validation, predictions, output_dict=True)
df5 = pd.DataFrame(report).transpose()	
df5 = df5.append([metrics.accuracy_score(Y_validation, predictions)] )
df5.columns.values[len(df5.columns)-1] = 'Accuracy'
df5.index.values[len(df5.index)-1] = 'Accuracy'

#Heatmaps_SVC

matrix = metrics.confusion_matrix(Y_validation.ravel(), predictions.ravel())
ax = plt.figure(6)
ax = plt.subplot()
sns.heatmap(matrix, annot = True, ax = ax, fmt='g',annot_kws={"size": 27})
ax.set_xlabel('True', size = 15);ax.set_ylabel('Predict', size = 15);
ax.set_title("SVM", size = 21 );
ax.xaxis.set_ticklabels(['HC','PD'], size = 15); ax.yaxis.set_ticklabels(['HC','PD'], size = 13);




	
with ExcelWriter("ClassificationReport.xlsx") as writer:
 df.to_excel(writer, sheet_name = 'LDA')
 df1.to_excel(writer, sheet_name = 'LR')
 df2.to_excel(writer, sheet_name = 'KNN')
 df3.to_excel(writer, sheet_name = 'DecisionTree')
 df4.to_excel(writer, sheet_name = 'GaussianNB')
 df5.to_excel(writer, sheet_name = 'SVM')

# Tempo de Processamento


tempo_total = mergeDict(tempo_treino, tempo_teste)

Data_Tempo = pd.DataFrame.from_dict(tempo_total)
Data_Tempo = Data_Tempo.transpose()
Data_Tempo.columns = ['Tempo de Treino(us)', 'Tempo de Teste(us)']

Data_Tempo.to_excel('Processamento.xlsx' )


#Salvar modelos

nome = 'LDA.sav'
joblib.dump(model1, nome)
nome = 'LR.sav'
joblib.dump(model2, nome)
nome = 'KNN.sav'
joblib.dump(model3, nome)
nome = 'CART.sav'
joblib.dump(model4, nome)
nome = 'NB.sav'
joblib.dump(model5, nome)
nome = 'SVM.sav'
joblib.dump(model6, nome)

#PLOT ROC
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')




plt.show()



