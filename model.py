# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:54:38 2019

@author: win10
"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


class Model():
    def __init__(self,X,y):
        self.X=X_train
        self.y=y_train
        
    def svm(self):
        from sklearn.svm import SVC
        clf=SVC(C=1, gamma=0.05)
        scores = cross_val_score(clf,X, y, cv=5)
        return scores.mean()
        
        

"""=============第一个模型 线性SVM=================""" 

#clf = SVC(kernel='linear',C=0.1)
#clf=SVC(kernel='poly',degree=3,gamma=0.5,coef0=0)
clf=SVC(C=1, gamma=0.05)
#scores = cross_val_score(clf, scaled, feature.iloc[:,-1], cv=5)

clf.fit(X_train,y_train)
pred_y = clf.predict(X_test)
print(classification_report(y_test,pred_y))

#confusion_matrix=confusion_matrix(y_test,pred_y)

   
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1,0.05, 0.005]}, cv=4)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


"""=============第二个模型 adaboost================="""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=5),
                     algorithm="SAMME",
                     n_estimators=20, learning_rate=0.5)
bdt.fit(X_train,y_train)
pred_y = bdt.predict(X_test)
print(classification_report(y_test,pred_y))

"""=============第三个模型 逻辑回归================="""
from sklearn.linear_model.logistic import LogisticRegression
classifier=LogisticRegression(solver='liblinear')
#scores = cross_val_score(classifier, scaled, feature.iloc[:,-1], cv=4)
classifier.fit(X_train,y_train)
pred_y = classifier.predict(X_test)
print(classification_report(y_test,pred_y))
#confusion_matrix=confusion_matrix(y_test,pred_y)

"""==============第四个模型  ELM============================"""
#    from hpelm import ELM
#    elm=ELM(10,18)
#    elm.add_neurons(50,'sigm')
#    elm.train(np.mat(X_train), np.mat(y_train), "LOO")
#    y_pred=elm.predict(X_test)

"""=============第五个模型 LDA================="""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(classification_report(y_test,y_pred))