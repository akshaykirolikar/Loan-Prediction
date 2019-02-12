import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scikitplot as skplt

#importing the dataset
dataset = pd.read_excel('Data1.xlsx')
X=dataset.drop(['Loan_ID','Loan_Status'],axis=1)

y=dataset[['Loan_Status']]
y=pd.get_dummies(y,drop_first='True')

Total_Income=X['ApplicantIncome']+X['CoapplicantIncome']
Total_Income=pd.DataFrame(Total_Income)

#visualisations
plt.style.use('ggplot')
X['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender')
X['Married'].value_counts(normalize=True).plot.bar(title= 'Married')
X['Dependents'].value_counts(normalize=True).plot.bar(title= 'Dependents')
X['Education'].value_counts(normalize=True).plot.bar(title= 'Education')
X['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')
X['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')
X['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')
X['Loan_Amount_Term'].value_counts(normalize=True).plot.bar(title= 'Loan_Amount_Term')
sns.distplot(Total_Income)

#Replacing Missing values
X['Dependents'] = X['Dependents'].fillna(X['Dependents'].mode().iloc[0])
X.Dependents[X.Dependents=='3+'] = 3
X['Dependents']=X['Dependents'].astype("int")
X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode().iloc[0])
X['Gender'] = X['Gender'].fillna(X['Gender'].mode().iloc[0])
X['Married'] = X['Married'].fillna(X['Married'].mode().iloc[0])
X['Education'] = X['Education'].fillna(X['Education'].mode().iloc[0])
X['Self_Employed'] = X['Self_Employed'].fillna(X['Self_Employed'].mode().iloc[0])
X["Loan_Amount_Term"] = X["Loan_Amount_Term"].fillna(X["Loan_Amount_Term"].mode().iloc[0])

#subset Education and Loan amount to replace loan amount missing values
subset_edu_LA=X[["Education","LoanAmount"]]
subset_edu_LA["LoanAmount"].loc[subset_edu_LA['Education'] == 'Graduate']\
=subset_edu_LA["LoanAmount"].loc[subset_edu_LA['Education'] == 'Graduate']\
.fillna(subset_edu_LA["LoanAmount"].loc[subset_edu_LA['Education'] == 'Graduate']\
.median())

subset_edu_LA["LoanAmount"].loc[subset_edu_LA['Education'] == 'Not Graduate']\
=subset_edu_LA["LoanAmount"].loc[subset_edu_LA['Education'] == 'Not Graduate']\
.fillna(subset_edu_LA["LoanAmount"].loc[subset_edu_LA['Education'] == 'Not Graduate']\
.median())

X=X.drop(['LoanAmount'],axis=1)
X=pd.concat([X,subset_edu_LA['LoanAmount']],axis=1,ignore_index=False)

sns.distplot(X['LoanAmount'])

#Categorical variables transformed to Dummy Variables
p=X[['Gender','Married','Education','Self_Employed','Property_Area']]

p1=pd.get_dummies(p,drop_first=True)

q=pd.get_dummies(X['Dependents'],drop_first=True)

p1=pd.concat((q,p1), axis=1,ignore_index=False)

#standardizing 
from sklearn import preprocessing
x = X[['LoanAmount', 'Loan_Amount_Term']]
x=pd.concat((x,Total_Income),axis=1,ignore_index=False)

std_scaler = preprocessing.StandardScaler()
x_scaled = std_scaler.fit_transform(x)
x_scaled= pd.DataFrame(x_scaled)

X2=X[['Credit_History']]

X1=pd.concat([p1,X2],axis=1,ignore_index=False)
X1=pd.concat([X1,x_scaled],axis=1,ignore_index=False)

#Dataset split 80:20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.20, random_state = 0)

#Oversampling of training sets using smote
from imblearn.over_sampling import SMOTE, ADASYN
sm = SMOTE(random_state=12, ratio = 1.0)
X_train_s, y_train_s = sm.fit_sample(X_train, y_train)

X_train_s=pd.DataFrame(X_train_s)
y_train_s=pd.DataFrame(y_train_s)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
pca.fit(X_train_s)
var1=pca.explained_variance_

sns.set()
plt.style.use('ggplot')
plt.plot(var1)
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.show()

skplt.decomposition.plot_pca_component_variance(pca,target_explained_variance=0.90)

pca1 = PCA(n_components=8)
X_train_s= pca1.fit_transform(X_train_s)
X_test= pca1.transform(X_test)

#classification Models
#support vector classifier
from sklearn.svm import SVC  
svclassifier = SVC(kernel='sigmoid',probability=True)  
svclassifier.fit(X_train_s, y_train_s)  

#k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier  
classifier_k = KNeighborsClassifier(n_neighbors=5)  
classifier_k.fit(X_train_s, y_train_s)
   
#naive bayes
from sklearn.naive_bayes import GaussianNB
classifier_n=GaussianNB()
classifier_n.fit(X_train_s, y_train_s)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_s, y_train_s)

# Predicting the Test set results
y_pred_log = pd.DataFrame(classifier.predict(X_test))
y_pred_k = pd.DataFrame(classifier_k.predict(X_test)) 
y_pred_n = pd.DataFrame(classifier_n.predict(X_test)) 
y_pred_svc=pd.DataFrame(svclassifier.predict(X_test))

#classification report report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_log))
print(classification_report(y_test, y_pred_k))
print(classification_report(y_test, y_pred_n))
print(classification_report(y_test, y_pred_svc))

#confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred_log, normalize=False)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_k, normalize=False)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_n, normalize=False)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_svc, normalize=False)

#roc curve
skplt.metrics.plot_roc_curve(y_test,classifier.predict_proba(X_test))
skplt.metrics.plot_roc_curve(y_test,classifier_k.predict_proba(X_test))
skplt.metrics.plot_roc_curve(y_test,classifier_n.predict_proba(X_test))
skplt.metrics.plot_roc_curve(y_test,svclassifier.predict_proba(X_test))

