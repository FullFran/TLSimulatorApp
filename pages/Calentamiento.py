import pandas as pd
import numpy as np
import scipy.integrate as integ
import threading

import matplotlib.pyplot as plt
import time

import streamlit as st

# Ejecuta esta celda para introducir los datos

DatosEntrada = pd.read_excel(
    'datos/datostld100.xlsx', sheet_name='Hoja1', header=0, usecols=None, nrows=None)

dfnew = DatosEntrada

E = dfnew.iloc[:, 0]         # Energia de Activación (eV).
s = dfnew.iloc[:, 1]         # Factor de Frecuencia (s-1).
# Concentracion de electrones atrapados en las trampas (cm-3).
n = dfnew.iloc[:, 2]
# Numero de posiciones acesibles por los electrnes de la BC a la Trampa (cm-3).
N = dfnew.iloc[:, 3]
# Coeficiente de Probabilidad de atrapamiento de electrones de a trampa (cm+3·s-1).
A = dfnew.iloc[:, 4]
# Coeficiente de probabilidad de recombinacion e-h Radiativa (cm+3·s-1).
Amn_R = dfnew.iloc[0:1, 5]
# Coeficiente de probabilidad de recombinacion e-h No Radiativa (cm+3·s-1).
Amn_NR = dfnew.iloc[0:1, 6]
# Coeficiente de Probabilidad de atrapamiento de huecos de la BV al Centro de Recombinacion Radiativo (cm+3·s-1).
A_R = dfnew.iloc[0:1, 7]
# Coeficiente de Probabilidad de atrapamiento de huecos de la BV al Centro de Recombinacion No Radiativo (cm+3·s-1).
A_NR = dfnew.iloc[0:1, 8]
# Numero de posiciones acesibles huecos de la BV al Centro de Recombinacion Radiativo (cm-3).
M_R = dfnew.iloc[0:1, 9]
# Numero de posiciones acesibles huecos de la BV al Centro de Recombinacion No Radiativo (cm-3).
M_NR = dfnew.iloc[0:1, 10]
# Concentracion de h atrapados en el centro de recombonacion radiativo (cm-3).
m_R = dfnew.iloc[0:1, 11]
# Concentracion de h atrapados en el centro de recombonacion no radiativo (cm-3).
m_NR = dfnew.iloc[0:1, 12]
f = dfnew.iloc[0:1, 13]     # Factor de generacion de pares e-h (cm-3·s-1).
# Concentración de electrones libres en la banda de conduccion (cm-3).
n_c = dfnew.iloc[0:1, 14]
# Concentración de huecos libres en la banda de Valencia (cm-3).
n_h = dfnew.iloc[0:1, 15]


E = E.to_numpy()
s = s.to_numpy()
n = n.to_numpy()
N = N.to_numpy()
A = A.to_numpy()
Amn_R = Amn_R.to_numpy()
Amn_NR = Amn_NR.to_numpy()
A_R = A_R.to_numpy()
A_NR = A_NR.to_numpy()
M_R = M_R.to_numpy()
M_NR = M_NR.to_numpy()
m_R = m_R.to_numpy()
m_NR = m_NR.to_numpy()
f = f.to_numpy()
n_c = n_c.to_numpy()
n_h = n_h.to_numpy()

# Condiciones iniciales
nn = np.array(n_c[0])
for i in range(len(n)):
    nn = np.append(nn, n[i])
nn = np.append(nn, m_R[0])
nn = np.append(nn, m_NR[0])
nn = np.append(nn, n_h[0])

# Constante de boltzman
kb = 0.00008617333262


# Ejecuta esta celda para definir las funciones del modelo

# aquí definimos el N de saturación y corregimos el numero de trampas en 4 a parte de definir las funciones de N
Nsat=np.ones(len(N))*10**11 #aquí definimos el N de saturación y corregimos el numero de trampas en 4 a parte de definir las funciones de N
# N[3]=N[3]/2
# N[4]=3*N[4]/4

Nsat[0]=Nsat[0]*25
Nsat[1]=Nsat[1]*10
Nsat[2]=Nsat[2]*5
Nsat[3]=Nsat[3]*5
Nsat[4]=Nsat[4]*65


def Ng(t, i):  # En esta función simulamos la creación de trampas durante la irradiación
    Ngg = []
    C = 1
    Ngg = N[i]*np.exp(np.log10(Nsat[i]/N[i])*(1-np.exp(-C*t)))
    return Ngg


def Nee(E, i, T):  # Esta es la distribución de Fermi-Dirac que utilizaremos en la irradiación
    Ngg = []
    Ngg = N[i]/(1+np.exp((E-2.7)/(kb*T)))
    return Ngg


def NFCT(E, i, T, t):  # Aquí simulamos la creación de trampas durante la irradiación, utilizando también la distribución de Fermi-Dirac
    NNN = Nee(E, i, T)*Ng(t, i)/N[i]
    return NNN


NN = N


def Ne(E, i, T):  # Aquí definimos la distribución de Fermi-Dirac para utilizarla utilizando el numero de trampas disponibles tras la irradiación
    Ngg = []
    Ngg = NN[i]/(1+np.exp((E-2.7)/(kb*T)))
    return Ngg


def TLIN(t, u):  # Con esta función simulamos la irradiación añadiendo la creación de trampas, utilizando la función de arrenius para la probabilidad
    Tamb = 297.15
    kb = 0.00008617333262
    dx = np.zeros(len(nn))
    Constante_A = 0.0
    for i in range(1, len(dx)-3):
        dx[i] = -u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb))) + \
            A[i-1]*(Ng(t, i-1)-u[i])*u[0]
        Constante_A += dx[i]
    dx[0] = f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3] = A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2] = A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1] = f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx, dtype=object)


def TLINE(t, u):  # Aquí simulamos la irradiación añadiendo la creación de trampas, utilizando la distribución de F-D
    Tamb = 297.15
    kb = 0.00008617333262
    dx = np.zeros(len(nn))
    Constante_A = 0.0
    for i in range(1, len(dx)-3):
        dx[i] = -u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb))) + \
            A[i-1]*(NFCT(E[i-1], i-1, Tamb, t)-u[i])*u[0]
        Constante_A += dx[i]
    dx[0] = f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3] = A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2] = A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1] = f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx, dtype=object)


Tamb = 273.15


def TLRNE(t, u):  # Simulamos la relajación tras la irradiación (reduciendo la fuente de radiación en un factor 1000) partiendo de las concentraciones y
    # el número de trampas tras la irradiación añadiendo la creación de trampas, utilizando la distribución de F-D  Nota: revisar que esté bien implementado
    Tamb = 273.15
    kb = 0.00008617333262
    dx = np.zeros(len(nn))
    Constante_A = 0.0
    for i in range(1, len(dx)-3):
        dx[i] = -u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb))) + \
            A[i-1]*(NN[i-1]-u[i])*u[0]
        Constante_A += dx[i]
    dx[0] = f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3] = A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2] = A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1] = f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx, dtype=object)


def TLI(t, u):  # Simulación de la irradiación sin tener en cuenta la creación de trampas usando la ley de arrenius
    Tamb = 297.15
    kb = 0.00008617333262
    dx = np.zeros(len(nn))
    Constante_A = 0.0
    for i in range(1, len(dx)-3):
        dx[i] = -u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb))) + \
            A[i-1]*(N[i-1]-u[i])*u[0]
        Constante_A += dx[i]
    dx[0] = f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3] = A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2] = A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1] = f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx, dtype=object)


def TLR(t, u):  # Simulación de la relajación de la función anterior
    Tamb = 297.15
    kb = 0.00008617333262
    dx = np.zeros(len(nn))
    Constante_A = 0.0
    for i in range(1, len(dx)-3):
        dx[i] = -u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb))) + \
            A[i-1]*(N[i-1]-u[i])*u[0]
        Constante_A += dx[i]
    dx[0] = f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3] = A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2] = A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1] = f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx, dtype=object)





def TLE(t,u,beta):      #Simulación del calentamiento
        Tamb=297.15+float(t)*beta
        kb=0.00008617333262
        dx=np.zeros(len(nn))
        Constante_A=0.0
        for i in range(1,len(dx)-3):
                if u[i]>10:
                        dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(NN[i-1]-u[i])*u[0]
                        Constante_A+=dx[i]
        dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
        dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
        dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
        dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
        return np.array(dx,dtype=object)


def TLECT(t, u):  # Simulación del calentamiento utilizando una curva de temperaturas
    # a elección (CT(t))
    Tamb = CT(t)
    kb = 0.00008617333262
    dx = np.zeros(len(nn))
    Constante_A = 0.0
    for i in range(1, len(dx)-3):
        if u[i] > 10:
            dx[i] = -u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb))
                        )+A[i-1]*(NN[i-1]-u[i])*u[0]
            Constante_A += dx[i]
    dx[0] = f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3] = A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2] = A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1] = f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx, dtype=object)


def tl(X, T, beta):  # Calculamos la curva de termoluminiscencia
    Temp = np.zeros(len(T))
    TL = np.zeros(len(T))
    for i in range(len(T)):
        Temp[i] = T[i]*beta+297.15-273.15
        TL[i] = A_R*X[0, i]*X[-3, i]/beta
    return Temp, TL


def tl200(X, T, beta):  # Guardamos 200 canales de la curva de termoluminiscencia
    Temp, TL = tl(X, T, beta)
    t200 = np.linspace(Temp[0], Temp[-1], 200)
    tl200 = np.interp(t200, Temp, TL)
    return t200, tl200


# Iniziamos la variable que albergará el nuevo número de trampas
NN = np.zeros(len(N))
for i in range(len(N)):
        # aquí estamos actualizando el numero de trampas que hay después de crearse para que lo use el resto de funciones
        NN[i] = NFCT(E[i], i, 297.15, 230+273.15)
st.set_page_config(layout='wide')
st.title('Sumulación calentamiento')
st.write('Esta es una prueba para hacer una herramienta web.')



if st.button('Iniciar calentamiento'):
    
    beta = 2 # Este parámetro es la pendiente de la curva de calentamiento
# aquí definimos la temperatura máxima que queremos alcanzar en grados kelvin
    mt = ((230+273.15)-Tamb)/beta  
    p=15

    CC = pd.read_csv('ci.csv')
    CC = CC.to_numpy()
    solcal=integ.solve_ivp(fun=TLE,t_span=[0,mt/p],y0=CC[:,1],args=[beta])       #Resolvemos el modelo para el calentamiento usando 'TLE'
    tt=solcal.t     #Guardamos el tiempo de la solución en 'tt'
    xx=solcal.y     #Guardamos las concentraciones de la solución en 'xx'
    TT=Tamb+beta*tt-273 #Aquí guardamos a qué temperatura corresponde cada instante de tiempo para representar la curva
    for i in range(1,len(nn)-3):        #Representamos las concentraciones de electrones en las trampas durante el calentamiento
            plt.figure(1)
            plt.plot(mt*beta+Tamb-273,0)
            plt.plot(TT,xx[i],label=['x',i])
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')    
            plt.title('CONCENTRACIONES EN CALENTAMIENTO TLD-100')
            plt.xlabel('Temperatura')
            plt.ylabel('concentracion')
            plt.savefig('cal.jpg')
            imagec = 'cal.jpg'
    temp,tlc=tl200(xx,tt,beta)          #Simulamos la curva de termoluminiscencia en función de la simulación del calentamiento

    canal=np.linspace(1,200,200)        #Representamos la curva de termoluminiscencia
    plt.figure(2)
    plt.plot(temp,tlc)
    plt.plot(Tamb+beta*mt-273,0)
    plt.title('Curva tld-100')
    plt.xlabel('canal')
    plt.savefig('curva.jpg')
    imagecu='curva.jpg'
    imac = st.empty()
    imac.empty()
    with imac.container():
        st.image(imagec, caption='Irradiación',
                    )
        st.image(imagecu,caption='Curva de termoluminiscencia')
    for j in range(1,p+1):
        solcal=integ.solve_ivp(fun=TLE,t_span=[mt/p*(j),mt/p*(j+1)],y0=xx[:,-1],args=[beta])
        ttemp=solcal.t     #Guardamos el tiempo de la solución en 'tt'
        xxemp=solcal.y     #Guardamos las concentraciones de la solución en 'xx'
        ttemp=solcal.t
        TTemp=Tamb+beta*ttemp-273 #Aquí guardamos a qué temperatura corresponde cada instante de tiempo para representar la curva

        xx=np.concatenate((xx,xxemp),axis=1)
        TT=np.append(TT,TTemp)
        tt=np.append(tt,ttemp)
        for i in range(1,len(nn)-3):        #Representamos las concentraciones de electrones en las trampas durante el calentamiento
            plt.figure(1)
            plt.plot(mt,0)
            plt.plot(TT,xx[i],label=['x',i])
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')    
            plt.title('CONCENTRACIONES EN CALENTAMIENTO TLD-100')
            plt.xlabel('Temperatura')
            plt.ylabel('concentracion')
            plt.savefig('cal.jpg')
            imagec = 'cal.jpg'


        temp,tlc=tl200(xx,tt,beta)          #Simulamos la curva de termoluminiscencia en función de la simulación del calentamiento

        canal=np.linspace(1,200,200)        #Representamos la curva de termoluminiscencia
        plt.figure(2)
        plt.plot(temp,tlc)
        plt.plot(Tamb+beta*mt-273,0)
        plt.title('Curva tld-100')
        plt.xlabel('Temperatura')
        plt.savefig('curva.jpg')
        imagecu='curva.jpg'

        
        imac.empty()
        with imac.container():
            st.image(imagec, caption='Irradiación',
                    )
            st.image(imagecu,caption='Curva de termoluminiscencia')

