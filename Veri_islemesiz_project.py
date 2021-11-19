import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler #standardizasyon
from sklearn.model_selection import train_test_split,GridSearchCV #Bu gridSearchCv yi KNN ile en iyi paramatreleri seçerken kullancaz
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis,LocalOutlierFactor
from sklearn.decomposition import PCA


data=pd.read_csv("canser.csv")
data.drop(['Unnamed: 32','id'] , inplace=True , axis=1)
data=data.rename(columns={"diagnosis" : "target"})

data["target"]=[1 if i.strip()=="M" else 0 for i in data.target]

y=data.radius_mean  #bağımlı
x=data.drop(["radius_mean"],axis=1) #bağımsız

test_size=0.9
x_train ,x_test ,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)


knn=KNeighborsClassifier(n_neighbors=2,metric="minkowski")
knn.fit(X_train,y_train)

y_pred  =knn.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print("CM: ",cm)
print("Accuracy: ",acc)  


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)


cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print("CM: ",cm)
print("Accuracy: ",acc)  



#logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)
print("Prediction: ",y_pred)

cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print("CM: ",cm)
print("Accuracy: ",acc)  



#veirleri ölcekle
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli=sc1.fit_transform(x)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(y)




from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf")
svr_reg_fit=(x_olcekli,y_olcekli)

print(svr_reg.predict(1))
print(svr_reg.predict(0.5))    














