import pandas as pd
import numpy as np
import scipy.integrate as integ
import threading
from io import BytesIO
import matplotlib.pyplot as plt
import time

import streamlit as st


# Ejecuta esta celda para introducir los datos
datostld100='datos/datostld100.xlsx'
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

def leer_datos(dfnew):

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
    return E,s,n,N,A,Amn_R,Amn_NR, A_R,A_NR,M_R,m_R,m_NR,f,n_c,n_h

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





def TLE(t,u):      #Simulación del calentamiento
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

st.set_page_config(layout='wide')
st.title('Sumulación irradiación y relajación')
st.write('Esta es una prueba para hacer una herramienta web.')

tmax = -9999
form = st.form('Entrada')
a=form.write('Introduzca el tiempo de irradiación')
a = form.number_input('Tiempo de irradiación:')
b = form.number_input('Tiempo de relajación:')
form.form_submit_button('Enviar')
tmax= a
tmaxr=b
tmax = float(tmax)
if tmax == -9999:
    st.write('Introduzca el tiepo de irradiación.')
else:
    st.write('Tiempo de irradiación='+str(tmax))
    st.write('Tiempo de relajación= '+str(tmaxr))

# Preguntar al usuario si desea usar un archivo existente o cargar uno nuevo
option = st.selectbox("Seleccione una opción:", ("Usar archivo existente", "Cargar archivo Excel"))

# Descargar el archivo Excel


#hola()





if option == "Cargar archivo Excel":
    # Subir un archivo Excel para crear un DataFrame
    st.header("Cargar archivo Excel para crear DataFrame")
    file = st.file_uploader("Selecciona un archivo Excel para crear un DataFrame", type=["xlsx"])
    if file:
        df2 = pd.read_excel(file)
        E,s,n,N,A,Amn_R,Amn_NR, A_R,A_NR,M_R,m_R,m_NR,f,n_c,n_h=leer_datos(df2)
        st.write(df2)
if st.button('Empezar simulación'):

    warning = st.empty()
    warning.empty()
    if tmax == -9999:
        with warning.container():
            st.write('introduzca tiempo de irradiación')
    pasos = 50
    # tmax=20                 #tiempo de irradiación
    solirad = integ.RK45(fun=TLINE, t0=0, y0=nn, t_bound=tmax, max_step=np.inf,
                         rtol=0.01, atol=0.01, vectorized=False, first_step=None)

    for i in range(len(N)):
        # aquí estamos actualizando el numero de trampas que hay después de crearse para que lo use el resto de funciones
        NN[i] = NFCT(E[i], i, 297.15, tmax)
    ti = []
    xi = []  # Iniziamos la variable que

    imagen = st.empty()
    imagen.empty()

    for i in range(pasos):

        for i in range(1000):
            # get solution step state
            solirad.step()
            ti.append(solirad.t)
            xi.append(solirad.y)
            # break loop after modeling is finished
            if solirad.status == 'finished':

                break
        xtemp = np.array(xi)
        # Representamos la concentración en las trampas en la irradiación
        for i in range(1, len(nn)-3):
            plt.plot(ti, xtemp[:, i])
            plt.plot(tmax, 0)

            plt.title(
                'Concentracion de electrones en las trampas en la irradiación')
            plt.xlabel('t(s)')

            plt.ylabel('Concentración')
            plt.savefig('irrad.jpg')
        image = 'irrad.jpg'
        imagen.empty()
        with imagen.container():
            st.image(image, caption='Concentración de electrones en las trampas', width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        if solirad.status == 'finished':
            print('Solucionado')
            break

    xi = np.array(xi)
    plt.figure(1)
    for i in range(1, len(nn)-3):  # Representamos la concentración en las trampas en la irradiación
        plt.plot(ti, xi[:, i], label=['x', i])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title('Concentracion de electrones en las trampas en la irradiación')
        plt.xlabel('t(s)')
        plt.ylabel('Concentración')
        plt.savefig('irrad.jpg')
        image = 'irrad.jpg'

    imagen.empty()
    with imagen.container():
        st.image(image, caption='Concentración de electrones en las trampas', width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    solrel = integ.RK45(fun=TLRNE, t0=0, y0=xi[-1], t_bound=tmaxr, max_step=np.inf,
                        rtol=0.01, atol=0.01, vectorized=False, first_step=None)
    tr = []
    xr = []  # Iniziamos la variable que
    for i in range(pasos):
        for i in range(1000):
            # get solution step state
            solrel.step()
            tr.append(solrel.t)
            xr.append(solrel.y)
            # break loop after modeling is finished
            if solrel.status == 'finished':
                break
        xtempr = np.array(xr)
        plt.figure(2)
        # Representamos la concentración en las trampas en la relajación
        for i in range(1, len(nn)-3):
            plt.plot(tr, xtempr[:, i])
            plt.plot(tmax, 0)

            plt.title(
                'Concentracion de electrones en las trampas en la relajación')
            plt.xlabel('t(s)')

            plt.ylabel('Concentración')
            plt.savefig('relaj.jpg')
            imager = 'relaj.jpg'

        imagen.empty()
        with imagen.container():
            st.image([image, imager], caption=['Irradiación', 'Relajación'], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        if solrel.status == 'finished':
            print('Solucionao')
            break

    xr = np.array(xr)
    plt.figure(3)
    for i in range(1, len(nn)-3):  # Representamos la concentración en las trampas en la irradiación
        plt.plot(tr, xr[:, i], label=['x', i])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title('Concentracion de electrones en las trampas en la relajación')
        plt.xlabel('t(s)')
        plt.ylabel('Concentración')
        plt.savefig('relaj.jpg')
        imager = 'relaj.jpg'

    imagen.empty()
    with imagen.container():
        st.image([image, imager], caption=['Irradiación', 'Relajación'], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    ci = pd.DataFrame(xr[-1])
    ci.to_csv('ci.csv')

    st.write('simulación completada')

    if st.button('borrar imagen'):
        imagen.empty()

for i in range(len(N)):
        # aquí estamos actualizando el numero de trampas que hay después de crearse para que lo use el resto de funciones
        NN[i] = NFCT(E[i], i, 297.15, 230+273.15)
beta = 2  # Este parámetro es la pendiente de la curva de calentamiento
# aquí definimos la temperatura máxima que queremos alcanzar en grados kelvin
mtc = ((230+273.15)-Tamb)/beta

CC = pd.read_csv('ci.csv')
CC = CC.to_numpy()
if st.button('iniciar calentamiento'):
    for i in range(len(N)):
        # aquí estamos actualizando el numero de trampas que hay después de crearse para que lo use el resto de funciones
        NN[i] = NFCT(E[i], i, 297.15, tmax)
    beta = 10
    # aquí definimos la temperatura máxima que queremos alcanzar en grados kelvin
    mt = ((230+273.15)-Tamb)/beta

    solcal = integ.RK45(fun=TLE, t0=0, y0=CC[:, 1], t_bound=mt, max_step=np.inf,
                        rtol=0.01, atol=0.01, vectorized=False, first_step=None)
    tc = []
    xc = []  # Iniziamos la variable que
    TT = []
    imac = st.empty()
    imac.empty()
    inicio=time.time()
    for j in range(50):
        for i in range(3000):
            # get solution step state
            solcal.step()
            tc.append(solcal.t)
            TT.append(solcal.t*beta+Tamb-273.15)
            xc.append(solcal.y)
            # break loop after modeling is finished
            if solcal.status == 'finished':
                break
        xtemp = np.array(xc)
        ttemp=np.array(tc)
        # Representamos la concentración en las trampas en la relajación
        temp,tlc=tl(xtemp.T,TT,beta)
        for i in range(1, len(nn)-3):
            plt.figure(1)
            plt.plot(TT, xtemp[:, i], label=['x', i])
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.xlim(0, 230)
            plt.title(
                'Concentracion de electrones en las trampas en la relajación')
            plt.xlabel('t(s)')

            plt.ylabel('Concentración')
        plt.savefig('cal.jpg')
        imagec = 'cal.jpg'
        plt.figure(2)
        plt.plot(ttemp/mt*200,tlc)
        plt.plot(200,0)
        plt.title('Curva tld-100')
        plt.xlabel('canal')
        plt.savefig('curva.jpg')
        imagecu='curva.jpg'
        imac.empty()
        with imac.container():
            st.image(imagec, caption='Irradiación',
                    )
            st.image(imagecu,caption='Curva de termoluminiscencia')
        

        if solcal.status == 'finished':
            print('Solucionao')
            break
        else:
            fin=time.time()
            print(j/49, inicio-fin)