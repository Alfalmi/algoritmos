import pandas as pd
from sklearn.model_selection import train_test_split  # permite dividir nuestros datos
from sklearn import metrics  # nos permite evaluar nuestro modelo
from sklearn.linear_model import LogisticRegression  # modelo que vamos a aplicar
import matplotlib.pyplot as plt
import seaborn as sns  # nos permite mejorar la presentación de los gráficos

# %matplotlib inline #nos permite insetar gráficos en el notebook

diabetes = pd.read_csv('/rsc/diabetes.csv')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']  # columnas presentes en el dataset
x = diabetes[feature_cols]
y = diabetes[['Outcome']]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)  # Recogemos las predicciones que entrega nuestro modelo para los datos de prueba
