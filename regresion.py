from sklearn.linear_model import LinearRegression
import plotly.express as px
import pandas as pd
import numpy as np


def correlacion_pearson(X:list, Y:list) -> float:
    """
    Funcioan encargada de calcular la correlacion de person entre 2 listas de datos
    """
    # Volvemoa array 
    X_data = np.array(X)
    Y_data = np.array(Y)
    
    # Correlacion entre X y Y 
    correlacion = np.corrcoef(X_data, Y_data)[0, 1]
    
    return correlacion
    
def validar_datos(X:list , Y:list) -> pd.DataFrame:
    """
    Funcioan para volver los datos en dataframe y hacer validaciones
    """
    try:
        # Verificar que las listas no esten vacias y que tengan igual longitud
        if len(X) == 0 or len(Y) == 0:
            return None, None, None
        if len(X) != len(Y):
            raise ValueError("X y Y deben tener igual longitud")
            
        # Volvemoa array 
        X_data = np.array(X)
        Y_data = np.array(Y)
        
        # Volvemos un diccionario
        datos = {
            "Variables_independiente" : X_data,
            "Variables_dependientes" : Y_data,
        }
        
        # Volvemos un dataframe
        df = pd.DataFrame(datos)
        
        return df
    
    except Exception as e:
        print(f"Error al calcular datos: {e}")
        return None


def regresion_lineal(df:pd.DataFrame) -> tuple:
    """
    Funcion encargada de calcular la regresion lineal  
    """
    try:
        x = df[["Variables_independiente"]].values  
        y = df["Variables_dependientes"].values 
        
        modelo = LinearRegression()
        modelo.fit(x, y)  # Ajustar el modelo a los datos
        
        y_pred = modelo.predict(x)  # Predecir los valores de y
        r2 = modelo.score(x,y) # R^2 del modelo
        intercepcion = modelo.intercept_
        pendiente = modelo.coef_[0] # Pendiente de la recta
        
        return y_pred,pendiente,r2,intercepcion,modelo
    
    except Exception as e:
        print(f"Error al calcular la regresion lineal: {e}")
        return None,None,None,None,None

def predecir_valor(modelo, nuevo_x):
    """
    Funcion para realizar prediciones nuevas
    """
    nuevo_x_array = np.array(nuevo_x).reshape(-1, 1)
    return modelo.predict(nuevo_x_array)


def graficar_regresion(df, predicciones,r2):
    """
    Funcion encargada de graficar la regresion lineal
    """
    df["Predicciones"] = predicciones
    fig = px.scatter(df, x="Variables_independiente", y="Variables_dependientes", title=f"Regresión Lineal (R² = {r2:.2f})")
    fig.add_scatter(x=df["Variables_independiente"], y=df["Predicciones"], mode="lines", name="Línea de regresión")
    #fig.show()
    return fig 


if __name__ == "__main__":
        
    X = [15,14,17,16,15,16,15,13,17,16,16] # Edades
    Y = [2,0,3,4,3,4,3,1,4,3,5] # Horas en internet
        
    relacion = round(correlacion_pearson(X,Y),2)
    print(f"Correlacion entre X y Y: {relacion}")
    
    df = validar_datos(X,Y)
    #print(df)
    
    predicion,pendiente,r2,intercepcion,modelo = regresion_lineal(df)
    
    graficar_regresion(df,predicion,r2)
    
    print(predecir_valor(modelo,50))