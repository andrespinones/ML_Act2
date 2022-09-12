#Andrés Piñones Besnier A01570150
#Decision tree classifier
#Implementación de un modelo de machine learning utilizando un framework 
#importamos las librerias necesarias
import pandas as pd

#declaramos el nombre de las columnas
columns = ["id","sepal_length","sepal_width","petal_length","petal_width", "species"]
df = pd.read_csv('iris.data',names = columns)

#nos deshacemos de la columna de id pues no es necesaria
df=df.drop(["id"], axis = 1)

# con esto tomamos todas las variables excepto 'class' como variables independiente
X = df.drop(['species'], axis = 1)  
# con esto tomamos solo a la variable 'class' como variable dependiente
Y = df['species']
# llamamos a la biblioteca que divide nuestros datos en entrenamiento y prueba (70, 30)
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=50)

#creamos un modelo de clasificación utilizando un arbol de decisión
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)

predicciones_modelo = clf.predict(X_test) # con esto aplicamos el modelo a los datos de prueba 

#Evaluamos el modelo utilizando una matriz de confusión

from sklearn.metrics import (confusion_matrix, accuracy_score)

# confusion matrix
cm = confusion_matrix(Y_test, predicciones_modelo) 
print ("Confusion Matrix : \n", cm)


# Exactitud de modelo
print('Test accuracy = ', accuracy_score(Y_test,predicciones_modelo))