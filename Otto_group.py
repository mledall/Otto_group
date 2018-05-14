
from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.neighbors import KNeighborsClassifier as KN I played around with the KNN classifier, and there way too many features to really be efficient.
from sklearn import cross_validation, svm
# from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
import csv as csv
import numpy as np
# import pandas as pd
# import cPickle
import time

np.random.seed(17411)

# Loads the training data
def loading_train_data():
	print(' -- Loading training data')
	f = open('trainData.csv', 'rb')
	csv_train_file_object = csv.reader(f)
	train_header = csv_train_file_object.next()
	train_data=[]
	for row in csv_train_file_object:
	    train_data.append(row[0:]) 
	train_data = np.array(train_data)
	f.close()
	return(train_data)


# Loads the testing data
def loading_test_data():
	print(' -- Loading test data')
	f = open('testData.csv', 'rb')
	csv_test_file_object = csv.reader(f) 
	test_header = csv_test_file_object.next() 
	test_data=[]
	for row in csv_test_file_object:
	    test_data.append(row[0:])
	test_data = np.array(test_data)	
	f.close()	
	return(test_data.astype(float))


# Splits the training data into the training and validation sets
def split_data(train_data, train_size=0.98):
	print(' -- Splitting the data')
	X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(train_data[:,1:-1], train_data[:,-1], train_size = train_size, random_state=10)
	return(X_train, X_valid, Y_train, Y_valid)


# Predicts a specific item class
def predict_train(predict_data, clf):
	item_class = clf.predict(predict_data)
	return(item_class[0])


# Defines my evaluation function
def Evaluation(X_valid, Y_valid, clf, metrics):
	print(' -- Start evaluation with %s' % metrics)
	if metrics is 'stdscore':
		score = clf.score(X_valid, Y_valid)
	if metrics is 'logloss':
		Y_predict = clf.predict_proba(X_valid)
		score = log_loss(Y_valid, Y_predict, eps = 1e-15, normalize = 'True')
	return(score)


# Trains the classifier on the training set of the training data, and evaluates it.
def training(classifier, metrics):
	print('- Start training')
	train_data = loading_train_data()
	X_train, X_valid, Y_train, Y_valid = split_data(train_data)
	if classifier is 'RF':
		print(' -- Start training a Random Forest Classifier')
		clf = RF(n_estimators=250, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, min_density=None, compute_importances=None)	# optimal n_estimators = 300
	if classifier is 'SVM':
		print(' -- Start training a Support Vector Machine Classifier')
		clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.001, kernel='rbf', max_iter=-1, probability=True, random_state=np.random.seed(17411),
shrinking=True, tol=0.001, verbose=False)
	clf.fit(X_train, Y_train)
	score = Evaluation(X_valid, Y_valid, clf, metrics)
	print(' -- score = %f' % score)
	print('- Finished training')
	return(clf, score)


# Write a log file of results obtained thus far
def log_file(clf, classifier, score, metrics):
	with open('log_file', 'a') as f:
		f.write('\n\n')
		f.write('- Classifier: %s \n' % classifier)
		f.write(' -- %s Score = %f , \n' % (metrics,score))
		f.write(' -- Parameters: %s .' % clf.get_params(True))


# Write the submission file
def submission(classifier = 'RF', metrics = 'stdscore', n_class = 9):
	path = 'submission_file_%s.csv' % classifier
	clf, score = training(classifier, metrics)
	print('- Start testing')
	test_data = loading_test_data()
	print(' -- Calculate probabilities')
	test_proba = clf.predict_proba(test_data[:,1:])
	print(' -- Write submission file %s' % path)
	classes = []
	for j in range(n_class):
		classes.append('class_%d' % (j+1))
	with open(path, 'w') as f:
		f.write('id,')
		f.write(','.join(classes))
		f.write('\n')
		for item in range(len(test_data[:,0])):
			f.write('%d,' % (item+1))
			f.write(','.join(map(str, test_proba[item,:])))
			f.write('\n')
	print(' -- Write log_file')
	log_file(clf, classifier, score, metrics)
	print('- Finished testing')		



# Runs the show
def main_function():
	submission()

#main_function()


filename = './testData.csv'

print("# sum(1 for line in open(filename)) ")
t0 = time.time()
n = sum(1 for line in open(filename))
print('Elapsed time : ', time.time() - t0)
print('n = ', n)
print('\n')




