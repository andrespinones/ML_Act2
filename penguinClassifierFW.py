#Andrés Piñones Besnier A01570150
#Implementación de un modelo de machine learning utilizando un framework 
#importamos las librerias necesarias
import pandas as pd

#leemos el dataset
df = pd.read_csv('penguins_size.csv')
#se analizó por fuera el dataset y tiene una muy poca cantidad de datos nulos en algunas columnas 
#los podemos llenar con la media de cada columna respectivamemte sin agregar mucho ruido
df['sex'].fillna(df['sex'].mode()[0],inplace=True)
col_with_null = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']
for i in col_with_null:
    df[i].fillna(df[i].mean(),inplace=True)

#hacemos dummies de la variable objetivo el tipo de especie de pinguino 
df['species']=df['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})

#hacemos dummies de las variables categoricas 
dummies = pd.get_dummies(df[['island','sex']],drop_first=True)

#Como el algoritmo de KNN calcula distancias para la clasificacion y las variables tienen escalas muy distintas
#tenemos que escalar las variables, removemos las categoricas pues estas no se escalan.
df_scalable = df.drop(['island','sex'],axis=1)
target = df_scalable.species
df_feat= df_scalable.drop('species',axis=1)

#usaremos el StandarScaler de sklearn para escalar las demás variables y con esto terminamos la limpieza y preprocesamiento.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_feat)
df_scaled = scaler.transform(df_feat)
df_scaled = pd.DataFrame(df_scaled,columns=df_feat.columns[:4])
df_preprocessed = pd.concat([df_scaled,dummies,target],axis=1)
df_preprocessed.head()

#declaramos como X todas nuestras variables predictoras
X = df_preprocessed.drop(['species'], axis = 1)  
# con esto tomamos solo a la variable 'species' como variable dependiente
Y = target

#KNN
#Como el objetivo es clasificar los pinguinos por especie utilizamos KNN como modelo de clasificación de aprendizaje supervisado 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

# hacemos el split de entrenamiento y prueba pues es un algoritmo supervisado y removemos
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

#creamos el modelo de KNN
knn = KNeighborsClassifier(n_neighbors=1)

#entrenamos el modelo con el training set correspondiente
knn.fit(X_train,y_train)

#realizamos una validación del entrenamiento
y_pred = knn.predict(X_train)

print('Training set matrix and score:')
print(confusion_matrix(y_train,y_pred))
print(accuracy_score(y_train,y_pred))

y_pred = knn.predict(X_test)

print('Testing set matrix and score:')
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

