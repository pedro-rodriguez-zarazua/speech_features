import numpy     as np
import random    as rnd
import datetime
import time
import features
from   scipy.io.wavfile  import read
import soundfile         as sf
from   sklearn                       import svm
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from   sklearn.tree                  import DecisionTreeClassifier
import sklearn.metrics
###################################################################################################################################################
#Calcula los MFCC de todos los archivos de un mismo usuario y los carga en una tabla
def load_class(file_path_list, param_list, feat_label):
	if(feat_label == 'mfb'):
		data_table   = np.zeros([0, param_list[3]])
		for i in range(len(file_path_list)):
			feature_table = features.mfb(file_path_list[i], param_list[0], param_list[1], param_list[2], param_list[3])
			data_table    = np.append(data_table, feature_table, axis=0)
			#return(data_table)
	elif(feat_label == 'mfcc'):
		data_table   = np.zeros([0, param_list[4] + param_list[4]*param_list[6]])
		for i in range(len(file_path_list)):
			feature_table = features.mfcc(file_path_list[i], param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5], param_list[6])
			data_table    = np.append(data_table, feature_table, axis=0)
			#return(data_table)
	elif(feat_label == 'lpc'):
		data_table   = np.zeros([0, param_list[3] + param_list[4]])
		for i in range(len(file_path_list)):
			feature_table = features.lpc(file_path_list[i], param_list[0], param_list[1], param_list[2], param_list[3], param_list[4])
			data_table    = np.append(data_table, feature_table, axis=0)
	return(data_table)

#Carga en una lista todos los MFFC de todos los usuarios/ cada usuario es un registro de la lista
def get_features_list(file_path_list, param_list, feat_label):
	lista_usuarios   = []
	for i in range(len(file_path_list)):
		dt = load_class(file_path_list[i], param_list, feat_label)
		lista_usuarios.append(dt)
	return(lista_usuarios)

#Devuelve el minimo numero de registros. Para posteriormente balancear las clases del conjunto de entrenamiento
def get_feat_min(feat_list):
    minimo = feat_list[0].shape[0]
    for i in range(1, len(feat_list)):
        if(feat_list[i].shape[0] < minimo):
            minimo = feat_list[i].shape[0]
    return(minimo)

#Devuelve una lista con el mismo numero de caracteristicas por cada usuario
def get_balanced_list(feat_list, minimo):
    balanced_list = []
    for i in range(len(feat_list)):
        dt = feat_list[i][:minimo,:]
        balanced_list.append(dt)
    return(balanced_list)

#Devuelve el conjunto de entrenamiento balanceado y la lista de clasificacion
def get_train_data_table(balanced_list):
    data_table   = np.zeros([0, balanced_list[0].shape[1]])
    data_classes = np.zeros(0)
    for i in range(len(balanced_list)):
        data_table    = np.append(data_table, balanced_list[i], axis=0)
        class_list    = np.zeros(balanced_list[i].shape[0]) + i
        data_classes  = np.append(data_classes, class_list)
    return(data_table, data_classes)
###################################################################################################################################################
#Devuelve el conjunto de entrenamiento balanceado y la lista de clases
def get_train_set(train_path_list, param_list, feat_label):
	print('train set . . . ')
	lista_usuarios                 = get_features_list(train_path_list, param_list, feat_label)
	n_minimo                       = get_feat_min(lista_usuarios)
	balanced_list                  = get_balanced_list(lista_usuarios, n_minimo)
	data_table, data_classes       = get_train_data_table(balanced_list)
	return(data_table, data_classes)

#Devuelve el conjunto de prueba y las lista de clases
def get_test_set(test_path_list, param_list, feat_label):
	print('test set . . . ')
	lista_usuarios                 = get_features_list(test_path_list, param_list, feat_label)
	data_table, data_classes       = get_train_data_table(lista_usuarios)
	return(data_table, data_classes)






