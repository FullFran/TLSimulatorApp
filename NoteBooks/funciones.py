import pandas as pd
import numpy as np
import scipy.integrate as integ
import threading




DatosEntrada=pd.read_excel('datos/datostld100.xlsx', sheet_name='Hoja1', header=0, usecols=None, nrows=None)

dfnew=DatosEntrada

E=dfnew.iloc[:,0]         # Energia de Activación (eV).
s=dfnew.iloc[:,1]         # Factor de Frecuencia (s-1).
n=dfnew.iloc[:,2]         # Concentracion de electrones atrapados en las trampas (cm-3).
N=dfnew.iloc[:,3]         # Numero de posiciones acesibles por los electrnes de la BC a la Trampa (cm-3).
A=dfnew.iloc[:,4]         # Coeficiente de Probabilidad de atrapamiento de electrones de a trampa (cm+3·s-1).
Amn_R=dfnew.iloc[0:1,5]   # Coeficiente de probabilidad de recombinacion e-h Radiativa (cm+3·s-1).
Amn_NR=dfnew.iloc[0:1,6]  # Coeficiente de probabilidad de recombinacion e-h No Radiativa (cm+3·s-1).
A_R=dfnew.iloc[0:1,7]     # Coeficiente de Probabilidad de atrapamiento de huecos de la BV al Centro de Recombinacion Radiativo (cm+3·s-1).
A_NR=dfnew.iloc[0:1,8]    # Coeficiente de Probabilidad de atrapamiento de huecos de la BV al Centro de Recombinacion No Radiativo (cm+3·s-1).
M_R=dfnew.iloc[0:1,9]    # Numero de posiciones acesibles huecos de la BV al Centro de Recombinacion Radiativo (cm-3).
M_NR=dfnew.iloc[0:1,10]  # Numero de posiciones acesibles huecos de la BV al Centro de Recombinacion No Radiativo (cm-3).
m_R=dfnew.iloc[0:1,11]   # Concentracion de h atrapados en el centro de recombonacion radiativo (cm-3).
m_NR=dfnew.iloc[0:1,12]  # Concentracion de h atrapados en el centro de recombonacion no radiativo (cm-3).
f=dfnew.iloc[0:1,13]     # Factor de generacion de pares e-h (cm-3·s-1).
n_c=dfnew.iloc[0:1,14]   # Concentración de electrones libres en la banda de conduccion (cm-3).
n_h=dfnew.iloc[0:1,15]   # Concentración de huecos libres en la banda de Valencia (cm-3).


E=E.to_numpy()
s=s.to_numpy()
n=n.to_numpy()
N=N.to_numpy()
A=A.to_numpy()
Amn_R=Amn_R.to_numpy()
Amn_NR=Amn_NR.to_numpy()
A_R=A_R.to_numpy()
A_NR=A_NR.to_numpy()
M_R=M_R.to_numpy()
M_NR=M_NR.to_numpy()
m_R=m_R.to_numpy()
m_NR=m_NR.to_numpy()
f=f.to_numpy()
n_c=n_c.to_numpy()
n_h=n_h.to_numpy()






kb=0.00008617333262


nn=np.array(n_c[0])
for i in range(len(n)):
    nn=np.append(nn,n[i])
nn=np.append(nn,m_R[0])
nn=np.append(nn,m_NR[0])
nn=np.append(nn,n_h[0])



def Pf(E,Temp,i):
    kb=0.00008617333262
    mu=0
    sf=s*1e6
    p=sf[i]/(1+np.exp((E[i]-mu+0.5)/(kb*Temp)))
    return (p)

def Ng(t,i):
        Ngg=[]
        Nsat=np.ones(len(N))*10**11
        C=1
        Ngg=N[i]*np.exp(np.log10(Nsat[i]/N[i])*(1-np.exp(-C*t)))
        return Ngg    

def Ne(E,i,T):
        Ngg=[]
        Ngg=N[i]/(1+np.exp((E-2)/(kb*T)))
        return Ngg   

def NFCT(E,i,T,t):
        NNN=Ne(E,i,T)*Ng(t,i)/N[i]
        return NNN

def TLIN(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(Ng(t,i-1)-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)

def TLINE(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(NFCT(E[i-1],i-1,Tamb,t)-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)

def TLRNE(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(Ne(E[i-1],i-1,Tamb)-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)

def TLI(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(N[i-1]-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)

def TLIQ(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(Pf(E,Tamb,i-1))+A[i-1]*(N[i-1]-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)

def TLR(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(N[i-1]-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)

def TLRQ(t,u):
    Tamb=297.15
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        dx[i]=-u[i]*(Pf(E,Tamb,i-1))+A[i-1]*(N[i-1]-u[i])*u[0]
        Constante_A+=dx[i]
    dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)


def TLE(t,u,beta):
    Tamb=297.15+float(t)*beta
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        if u[i]>10:
            dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(N[i-1]-u[i])*u[0]
            Constante_A+=dx[i]
    dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)


def TLE(t,u,beta):
    Tamb=297.15+float(t)*beta
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        if u[i]>10:
            dx[i]=-u[i]*(s[i-1]*np.exp(-E[i-1]/(kb*Tamb)))+A[i-1]*(Ne(E[i-1],i-1,Tamb)-u[i])*u[0]
            Constante_A+=dx[i]
    dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)


def TLEQ(t,u,beta):
    Tamb=297.15+float(t)*beta
    kb=0.00008617333262
    dx=np.zeros(len(nn))
    Constante_A=0.0
    for i in range(1,len(dx)-3):
        if u[i]>10:
            dx[i]=-u[i]*(Pf(E,Tamb,i-1))+A[i-1]*(N[i-1]-u[i])*u[0]
            Constante_A+=dx[i]
    dx[0]=f/1000-Constante_A-u[0]*(u[-3]*Amn_R[0]+u[-2]*Amn_NR[0])
    dx[-3]=A_R[0]*(M_R[0]-u[-3])*u[-1]-u[0]*(u[-3]*Amn_R[0])
    dx[-2]=A_NR[0]*(M_NR[0]-u[-2])*u[-1]-u[0]*(u[-2]*Amn_NR[0])
    dx[-1]=f/1000-u[-1]*(+A_R[0]*(M_R[0]-u[-3])+A_NR[0]*(M_NR[0]-u[-2]))
    return np.array(dx,dtype=object)


def tl(X,T,beta):
    Temp=np.zeros(len(T))
    TL=np.zeros(len(T))
    for i in range(len(T)):
        Temp[i]=T[i]*beta+297.15-273.15
        TL[i]=A_R*X[0,i]*X[-3,i]/beta
    return Temp,TL
    
def tl200(X,T,beta):
    Temp,TL=tl(X,T,beta)
    t200=np.linspace(Temp[0],Temp[-1],200)
    tl200=np.interp(t200,Temp,TL)           
    return t200,tl200

def Ne(E,i,T):
        Ngg=[]
        Nsat=np.ones(len(N))*10**11
        C=0.5
        c=5e-6
        mu=2
        Ngg=Nsat[i]*(1-np.exp(-np.log10(Nsat[i]/N[i])*np.exp(-(E-2)/(kb*T))))
        Ngg=Nsat[i]/(1+np.exp((E-2)/(kb*T)))
        return Ngg   