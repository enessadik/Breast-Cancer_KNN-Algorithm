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

#warningleri kaldırmak için
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("canser.csv") 
data.drop(['Unnamed: 32','id'] , inplace=True , axis=1)
data=data.rename(columns={"diagnosis" : "target"})

sns.countplot(data["target"])
print(data.target.value_counts())  # 357 tane iyi huylu(benign) 212 tane kötü huylu(malignant) 

#Datasetteki "M" ve "B" ler kategorik numerice çevirelim
#strip() yapmamızın amacı işimizi garantiye almak için mesela " M" bunun basında bir bosluk var eğer veri setinde böyle bisey varsa
#strip ile bu "M" olacaktır.
data["target"]=[1 if i.strip()=="M" else 0 for i in data.target]

print(data.shape)
data.info()
describe=data.describe() 


"""
Veriler arasında büyük bir scale farkı var "area mean" feauturesi gibi bu yüzden "STANDARDİZATION" uygulucaz
Missing valuemiz yok
"""

#%% EDA Keşifsel veri analizi

#correlation

corr_matrix=data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Correlation Between Features")
plt.show()

#threshold
threshold=0.75
filtre=np.abs(corr_matrix["target"])>threshold
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot=True,fmt=".2f")
plt.title("Correlation Between Features with Corr threshold 0.75")

# box_plot
data_melted=pd.melt(data,id_vars="target",
                    var_name="features",
                    value_name="value")

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
plt.xticks(rotation=90)
plt.show()        

"""
standardization
"""

#pair plot
sns.pairplot(data[corr_features],diag_kind="kde",markers=".",hue="target")
plt.show()

"""
skewness
"""
#%% Outlier
y=data.target
x=data.drop(["target"],axis=1)
columns=x.columns.tolist()

clf=LocalOutlierFactor()
y_pred=clf.fit_predict(x)

X_score=clf.negative_outlier_factor_
outlier_score=pd.DataFrame()
outlier_score["score"]=X_score

#threshold
threshold=-2.5
filtre=outlier_score["score"]<threshold
outlier_index=outlier_score[filtre].index.tolist()

plt.figure()

plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color="blue",s=50,label="Outliers")
plt.scatter(x.iloc[:,0],x.iloc[:,1],color="k",s=3,label="Data Points")

radius=(X_score.max() - X_score) / (X_score.max() - X_score.min())
outlier_score["radius"]=radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none",label="Outlier Scores")
plt.legend()
plt.show()

#drop outliers
x=x.drop(outlier_index)
y=y.drop(outlier_index).values

#%% Train test split
test_size=0.3
X_train ,X_test ,Y_train,Y_test = train_test_split(x,y,test_size=test_size,random_state=42)

#%% Standardization
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train_df=pd.DataFrame(X_train,columns=columns)
X_train_df_describe=X_train_df.describe()
X_train_df["target"] = Y_train

#boxplot
data_melted=pd.melt(X_train_df,id_vars="target",
                    var_name="features",
                    value_name="value")

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
plt.xticks(rotation=90)
plt.show()        


#pair plot
sns.pairplot(X_train_df[corr_features],diag_kind="kde",markers=".",hue="target")
plt.show()

#%% basic KNN method
knn=KNeighborsClassifier(n_neighbors=2) 
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
acc=accuracy_score(Y_test,y_pred)
score=knn.score(X_test,Y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)

#%% En iyi KNN parametresi bulma...

def KNN_Best_Params(x_train,x_test,y_train,y_test):
    
    k_range=list(range(1,31))
    weight_options=["uniform","distance"]
    print()
    param_grid=dict(n_neighbors=k_range , weights=weight_options)
    
    knn=KNeighborsClassifier()
    grid=GridSearchCV(knn,param_grid,cv=10,scoring="accuracy")
    grid.fit(x_train,y_train)
    
    print("Best Trainig Score: {} with parameters: {}".format(grid.best_score_,grid.best_params_))
    print()
    
    knn=KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)
    
    y_pred_test=knn.predict(x_test)
    y_pred_train=knn.predict(x_train)
    
    cm_test=confusion_matrix(y_test,y_pred_test)
    cm_train=confusion_matrix(y_train,y_pred_train)
    
    acc_test=accuracy_score(y_test,y_pred_test)
    acc_train=accuracy_score(y_train,y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test,acc_train)) #Burda train score test scoreden yüksek çıkmış direk 1.0 yüzde yüz tahmini doğru yapmış yani burda overfitting var
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)
    
    return grid
    

grid=KNN_Best_Params(X_train, X_test, Y_train, Y_test)
    
#%% PCA 

#bunu verimiz featuremiz fazlaysa ya da sample sayısı fazlaysa  PCA ile featuresleri azaltıyoruz
#PCA nın diger kullanım amacıda görselleştirmek mesela elimizde 30 boyutlu veri varsa bunları görselleştiremeyiz
#en fazla 5 6 boyuta cıkartırız bu yüzden PCA yöntemini kullanacaz... mesela 30 boyutlu verimizden 2 boyuta düşürecez

#verimizi scale etmemiz gerekiyor...
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

#30 boyutu 2 boyuta düşürdük ve görselleştirdik
pca=PCA(n_components=2)
pca.fit(x_scaled)
X_reduced_pca=pca.transform(x_scaled)
pca_data=pd.DataFrame(X_reduced_pca,columns=["p1","p2"])
pca_data["target"]=y
sns.scatterplot(x="p1",y="p2",hue="target",data=pca_data)
plt.title("PCA: p1 vs p2")

#2 boyutlu veriyi kullanarak KNN ile sınıflandırma gerçekleştircez bunun için yeni train test split yapmamız gerekir
test_size=0.3
X_train_pca ,X_test_pca ,Y_train_pca,Y_test_pca = train_test_split(X_reduced_pca ,y , test_size=test_size,random_state=42)
grid_pca=KNN_Best_Params(X_train_pca ,X_test_pca ,Y_train_pca,Y_test_pca)



































