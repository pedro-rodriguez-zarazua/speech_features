import numpy     as np
import random    as rnd
import datetime
import time
import features
import data_sets
from   scipy.io.wavfile  import read
import soundfile         as sf
from   sklearn                       import svm
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from   sklearn.tree                  import DecisionTreeClassifier
import sklearn.metrics
###################################################################################################################################################
def create_txt_all_file(file_name):
	header = "date,num_clases,train_set,test_set,feat,clasificador,test_accuracy,test_precision,test_recall,test_f1,train_accuracy,train_precision,train_recall,train_f1,feat_time,train_time,pre_emph,frame_size,frame_hop,nfilt,nceps,lift,delta,order,gain\n"
	f = open(file_name,"w+")
	f.write(header)
	f.close()
	return None

def add_register_txt_all_file(txt_file, num_clases, train_set_size, test_set_size, param_list, training_time, metrics_str, classifier, feature):
    
	object   = datetime.datetime.now()
	date_str = str(object.date())+"-"+str(object.hour)+"-"+str(object.minute)

	if(feature == 'mfb'):
		feat_str  = str(param_list[0])+","+str(param_list[1])+","+str(param_list[2])+","+str(param_list[3])+",.,.,.,.,."
	elif(feature == 'mfcc'):
		feat_str  = str(param_list[0])+","+str(param_list[1])+","+str(param_list[2])+","+str(param_list[3])+","+str(param_list[4])+","+str(param_list[5])+","+str(param_list[6])+",.,."
	elif(feature == 'lpc'):
		feat_str  = str(param_list[0])+","+str(param_list[1])+","+str(param_list[2])+",.,.,.,.,"+str(param_list[3])+","+str(param_list[4])

	register = date_str+","+str(num_clases)+","+train_set_size+","+test_set_size+","+feature+","+classifier+","+metrics_str+","+training_time+","+feat_str+"\n"
	f = open(txt_file,"a")
	f.write(register)
	f.close()
	return None


def metricas(y, y_hat, num_class):
	dif            = y - y_hat
	accuracy       = float(np.sum(dif==0))/float(len(dif))
		
	precision_list = np.zeros(num_class)
	recall_list    = np.zeros(num_class)
	for i in range(num_class):
		index          = np.where(y == i)
		class_dif      = dif[index]
		true_positives = float(np.sum(class_dif == 0))
		precision_den  = float(np.sum(y_hat == i))
		recall_den     = float(np.sum(y == i))        
		if(precision_den == 0):
			precision_list[i] = 0
		else:
			precision_list[i] = true_positives/precision_den
		if(recall_den == 0):
			recall_list[i] = 0
		else:
			recall_list[i]    = true_positives/recall_den
    	    
	precision = float(np.sum(precision_list))/num_class
	recall    = float(np.sum(recall_list))/num_class
	f1        = (2*precision*recall)/(precision+recall)

	return accuracy, precision, recall, f1
###################################################################################################################################################
###################################################################################################################################################
def set_model(model):
	print('Entrenando ' + str(model) + " . . . ")
	# SVM
	start = time.time()
	model = svm.SVC(kernel=kern, C=svm_param_list[0], gamma='auto', tol=svm_param_list[2], degree=svm_param_list[3], coef0=svm_param_list[4], max_iter=50000)
	model.fit(train_set,train_y)
	end                 = time.time()
	training_time       = feat_time + ',' + str(end - start)
	y_train_hat         = model.predict(train_set)
	y_test_hat          = model.predict(test_set)
    
	train_accuracy, train_precision, train_recall, train_f1  = metricas(train_y, y_train_hat, len(train_path_list))
	test_accuracy, test_precision, test_recall, test_f1      = metricas(test_y, y_test_hat, len(train_path_list))

	train_metrics_str       = str(train_accuracy) +","+ str(train_precision)+","+str(train_recall)+","+str(train_f1)
	test_metrics_str        = str(test_accuracy) +","+ str(test_precision)+","+str(test_recall)+","+str(test_f1)
	metrics_str             = test_metrics_str +","+ train_metrics_str    
    
	add_register_txt_all_file(txt_file_path, num_clases, train_set_size_str, test_set_size_str, param_list, training_time, metrics_str, 'svm', feature)
    return None

def run_model(train_path_list, test_path_list, param_list, txt_file_path, feature):
	#SVM parametros
	kern   = 'poly'
	C      = 1
	gamma  = 0.1
	tol    = 0.001
	degree = 3
	coef0  = 0

	#LDA parametros
	reg_param  = 0.25
	tol        = 0.001

	#DECISION TREE parametros
	criteria          = 'gini'
	min_samples_split = 2
	min_samples_leaf  = 1

	svm_param_list   = [C, gamma, tol, degree, coef0]
	lda_param_list   = [reg_param, tol]
	dt_param_list    = [min_samples_split,min_samples_leaf]


	print('Calculando ' + feature + ' . . .')
	start = time.time()
	train_set, train_y = data_sets.get_train_set(train_path_list, param_list, feature)   #Arreglar este y el siguiente en el doc data_sets
	test_set, test_y   = data_sets.get_test_set(test_path_list, param_list, feature)		#Arreglar este y el anterior en el doc data_sets
	end                = time.time()
	feat_time          = str(end - start)
	num_clases         = len(train_path_list)
	train_set_size_str = str(int(train_set.shape[0]/num_clases))
	test_set_size_str  = str(int(test_set.shape[0]))
	
	print('Entrenando svm . . .')
	# SVM
	start = time.time()
	model = svm.SVC(kernel=kern, C=svm_param_list[0], gamma='auto', tol=svm_param_list[2], degree=svm_param_list[3], coef0=svm_param_list[4], max_iter=50000)
	model.fit(train_set,train_y)
	end                 = time.time()
	training_time       = feat_time + ',' + str(end - start)
	y_train_hat         = model.predict(train_set)
	y_test_hat          = model.predict(test_set)
    
	train_accuracy, train_precision, train_recall, train_f1  = metricas(train_y, y_train_hat, len(train_path_list))
	test_accuracy, test_precision, test_recall, test_f1      = metricas(test_y, y_test_hat, len(train_path_list))

	train_metrics_str       = str(train_accuracy) +","+ str(train_precision)+","+str(train_recall)+","+str(train_f1)
	test_metrics_str        = str(test_accuracy) +","+ str(test_precision)+","+str(test_recall)+","+str(test_f1)
	metrics_str             = test_metrics_str +","+ train_metrics_str    
    
	add_register_txt_all_file(txt_file_path, num_clases, train_set_size_str, test_set_size_str, param_list, training_time, metrics_str, 'svm', feature)	

	print('Entrenando lda . . .')
	# LDA
	start = time.time()
	model = QuadraticDiscriminantAnalysis(reg_param=lda_param_list[0],tol=lda_param_list[1])
	model.fit(train_set,train_y)
	end                 = time.time()
	training_time       = feat_time + ',' + str(end - start)
	y_train_hat         = model.predict(train_set)
	y_test_hat          = model.predict(test_set)
    
	train_accuracy, train_precision, train_recall, train_f1  = metricas(train_y, y_train_hat, len(train_path_list))
	test_accuracy, test_precision, test_recall, test_f1  = metricas(test_y, y_test_hat, len(train_path_list))

	train_metrics_str       = str(train_accuracy) +","+ str(train_precision)+","+str(train_recall)+","+str(train_f1)
	test_metrics_str        = str(test_accuracy) +","+ str(test_precision)+","+str(test_recall)+","+str(test_f1)
	metrics_str             = test_metrics_str +","+ train_metrics_str    
    
	add_register_txt_all_file(txt_file_path, num_clases, train_set_size_str, test_set_size_str, param_list, training_time, metrics_str, 'lda', feature)	


	print('Entrenando decision tree . . .')	
	# DECISION TREE
	start = time.time()
	model = DecisionTreeClassifier(criterion = criteria, min_samples_split = dt_param_list[0], min_samples_leaf = dt_param_list[1],random_state=0)
	model.fit(train_set,train_y)
	end                 = time.time()
	training_time       = feat_time + ',' + str(end - start)
	y_train_hat         = model.predict(train_set)
	y_test_hat          = model.predict(test_set)
    
	train_accuracy, train_precision, train_recall, train_f1  = metricas(train_y, y_train_hat, len(train_path_list))
	test_accuracy, test_precision, test_recall, test_f1  = metricas(test_y, y_test_hat, len(train_path_list))

	train_metrics_str       = str(train_accuracy) +","+ str(train_precision)+","+str(train_recall)+","+str(train_f1)
	test_metrics_str        = str(test_accuracy) +","+ str(test_precision)+","+str(test_recall)+","+str(test_f1)
	metrics_str             = test_metrics_str +","+ train_metrics_str    
    
	add_register_txt_all_file(txt_file_path, num_clases, train_set_size_str, test_set_size_str, param_list, training_time, metrics_str, 'decision_tree', feature)    

	return None

def model_search_svm_mfcc(train_path_list, test_path_list, mfcc_param_list, txt_file):
    
	train_set, train_y = data_sets.get_train_set(train_path_list, mfcc_param_list)
	test_set, test_y   = data_sets.get_test_set(test_path_list, mfcc_param_list, feat_label = "mfcc")
    
	for param_change in range(1, 1000, 100):
        #SVM parametros
		kern   = 'poly'
		C      = 1
		gamma  = 1
		tol    = param_change*0.0001
		degree = 3
		coef0  = 0
		svm_param_list   = [C, gamma, tol, degree, coef0]
    
		start = time.time()
		model = svm.SVC(kernel=kern, C=svm_param_list[0], gamma=svm_param_list[1], tol=svm_param_list[2], degree=svm_param_list[3], coef0=svm_param_list[4])
		model.fit(train_set,train_y)
		end           = time.time()
		training_time = str(end - start)
		y_train_hat   = model.predict(train_set)
		y_test_hat    = model.predict(test_set)
				
		
		accuracy, precision, recall, f1  = metricas(train_y, y_train_hat, len(train_path_list))
		train_metrics_str                = str(accuracy) +","+ str(precision)+","+str(recall)+","+str(f1)
		
		accuracy, precision, recall, f1  = metricas(test_y, y_test_hat, len(train_path_list))
		test_metrics_str                 = str(accuracy) +","+ str(precision)+","+str(recall)+","+str(f1)
		metrics_str                      = train_metrics_str +","+ test_metrics_str
		
		
		train_set_size_str = str(int(train_set.shape[0]/len(train_path_list)))
		add_register_txt_file(txt_file, train_set_size_str, kern, svm_param_list, mfcc_param_list, training_time, metrics_str)
		
	return None
###################################################################################################################################################
###################################################################################################################################################

