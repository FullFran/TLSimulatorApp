import numpy as np
import pandas as pd
import os


def guardar_csv(nombre,t,x,nombre_t):
    r=pd.DataFrame(x.T)
    r[str(nombre_t)]=t
    if os.path.exists('resultados') == False:
        os.mkdir('resultados')
    r.to_csv('resultados/'+str(nombre)+'.csv',index=False)
    return print('se ha guardado correctamente')

def neutralidad(X):
    prueba=np.zeros(len(X[0]))
    x=np.zeros(len(X[:,0]))
    for i in range(len(X[0])):
        x=X[:,i]
        e=np.sum(x[0:-3])
        h=-x[-3]-x[-2]-x[-1]
        prueba[i]=e+h
    return prueba