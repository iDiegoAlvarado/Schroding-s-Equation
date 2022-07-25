# Solving Schödinger's eq

import numpy as np
import pylab
import matplotlib.pyplot as plt


# Funciones necesarias

def Gaussian(x,t,sigma): 
    "Gaussian wave packet"
    return np.exp(-(x-t)**2/(2*sigma**2))

def free(npts):
    "Partícula libre"
    return np.zeros(npts)

def step(npts,v0):
    "Escalón de potencial"
    v = free(npts)
    sl = int(npts/2)
    v[sl:] = v0
    return v

def barrier(npts,v0,thickness):
    "Barrera de potencial"
    v = free(npts)
    sl = int(npts/2)
    v[sl-thickness:sl+thickness] = v0
    return v

def fillax(x,y,*args,**kw):
    "Llena el espacio entre una matriz de valores y y el eje x."
    xx = np.concatenate((x,np.array([x[-1],x[0]],x.dtype)))
    yy = np.concatenate((y,np.zeros(2,y.dtype)))
    return pylab.fill(xx, yy, *args,**kw)

#=============================================================================

# Constantes de simulación

N    = 1000     #  Número de puntos espaciales.
T    = 10*N     #  Número de pasos de tiempo.  5*N es un buen valor
                #  antes de que los bordes sean alcanzados.
Tp   = 50       #  Número de pasos de tiempo antes de plotear.
dx   = 1.0e0    #  Resolución espacial
m    = 1.0e0    #  Masa de ka partícula
hbar = 1.0e0    #  Constante de Planck
X    = dx*np.linspace(0,N,N)        #  Eje espacial.


# Potential parameters

V0   = 1.0e-2      # Amplitud del potencial (pasos y potenciales)
THCK = 8           # "Thickness" (ancho) de la barrera de potencial
POTENTIAL = "free" # Elija "free", "barrier" o "step"


#  Initial wave function constants

sigma = 40.0               # Desviación estándar (Principio de incertidumbre)
x0 = round(N/2) - 5*sigma  # Desface de tiempo
k0 = np.pi/20              # Número de onda 
E = (hbar**2/2.0/m)*(k0**2 + 0.5/sigma**2)

#=============================================================================

# Inicia el código

if POTENTIAL=="free":
    V = free(N)
elif POTENTIAL=="step":
    V = step(N,V0)
elif POTENTIAL=="barrier":
    V = barrier(N,V0,THCK)
else:
    raise ValueError("Unrecognized potential type: %s" % POTENTIAL)

Vmax = V.max()            #  Potencial máximo en el dominio
dt   = hbar/(2*hbar**2/(m*dx**2)+V0)      #  diferencial de tiempo crítico.
c1   = hbar*dt/(m*dx**2)                    #  Coeficiente constante 1.
c2   = 2*dt/hbar                            #  Coeficiente constante 2.
c2V  = c2*V  

# Información de simulación
print ("Evolución temporal de la ecuación de Schrödinger unidimensional")
print ("Paquete de energía:            ", E)
print ("Tipo de potencial:             ", POTENTIAL)
print ("Amplitud de potencial inicial: ", V0)
print ("Ancho de barrera:              ", THCK)

#  Función de onda. Tres estados que representan pasado, presente y futuro
psi_r = np.zeros((3,N))   # Real
psi_i = np.zeros((3,N))   # Imaginario
psi_p = np.zeros(N,)      # Probabilidad observable

#  índices temporales para acceder a las columnas de las funciones de onda.
PA = 0                 #  Pasado
PR = 1                 #  Presente
FU = 2                 #  Futuro

#  Inicializamos la función de onda.  

No2 = int(round(N/2))
xn = range(1,No2)
x = X[xn]/dx              # Coordenada de posición normalizada

gg = Gaussian(x,x0,sigma) # Gaussian wave packet
cx = np.cos(k0*x)         # Recuerde la forma exp(x**2) exp(ikt)
sx = np.sin(k0*x)         

psi_r[PR,xn] = cx*gg
psi_i[PR,xn] = sx*gg
psi_r[PA,xn] = cx*gg
psi_i[PA,xn] = sx*gg

# Nomalización inicial de las funciones de onda

# Calculamos la probabilidad observable
psi_p = psi_r[PR]**2 + psi_i[PR]**2

P   = dx * psi_p.sum()      #  Total probability.
nrm = np.sqrt(P)
psi_r = psi_r / nrm
psi_i = psi_i / nrm
psi_p = psi_p/P

#  Inicializamos las figuras y los ejes.

# Asignamos directamente índices para evitar ciclos for
IDX1 = range(1,N-1)            # psi [ k ]
IDX2 = range(2,N)              # psi [ k + 1 ]
IDX3 = range(0,N-2)            # psi [ k - 1 ]

xmin = X.min()
xmax = X.max()
ymax = (psi_r[PR]).max()

mm = 0        
for t in range(T + 1):

    psi_rPR = psi_r[PR]
    psi_iPR = psi_i[PR]

    psi_i[FU,IDX1] = psi_i[PA,IDX1] + \
                      c1*(psi_rPR[IDX2] - 2*psi_rPR[IDX1] +
                          psi_rPR[IDX3])
    psi_i[FU] -= c2V*psi_r[PR]

    psi_r[FU,IDX1] = psi_r[PA,IDX1] - \
                      c1*(psi_iPR[IDX2] - 2*psi_iPR[IDX1] +
                          psi_iPR[IDX3])
    psi_r[FU] += c2V*psi_i[PR]

    psi_r[PA] = psi_rPR
    psi_r[PR] = psi_r[FU]
    psi_i[PA] = psi_iPR
    psi_i[PR] = psi_i[FU]

    if t % Tp == 0:

        psi_p = psi_r[PR]**2 + psi_i[PR]**2
        
        plt.plot(X,psi_p*6, color="k", 
                 linewidth=2, label="$|\psi|^2$")
        plt.plot(X,psi_r[PR]/1.5, color="b", 
                 linewidth=1, label="$\psi^{R}$")
        plt.plot(X,psi_i[PR]/1.5, color="r", 
                 linewidth=1, label="$\psi^{I}$")
          
        
        if Vmax != 0 :
            # Factor de escala para las energías
            Efac = abs(ymax/2.0/Vmax)
            V_plot = V*Efac
    
            plt.plot(X,V_plot,":k",zorder=0)   #  Linea del potencial.
            #fillax(X,V_plot, facecolor='y', alpha=0.2,zorder=0)
            # Energía de la función de onda en la misma escala que el potencial
            plt.axhline(E*Efac,color='g',label="$E$",zorder=1,linestyle ="--")
        
        else:
            Efac = abs(ymax/2.0)
            V_plot = V*Efac
    
            plt.plot(X,V_plot,":k",zorder=0)   #  Linea del potencial.
            #fillax(X,V_plot, facecolor='y', alpha=0.2,zorder=0)
            # Energía de la función de onda en la misma escala que el potencial
            plt.axhline(E*Efac,color='g',label="$E$",zorder=1,linestyle ="--")
            
            
        En = str(float(str(E*Efac)[:5]))
        Vn = str(float(str(Vmax*Efac)[:5]))
        plt.legend(loc="lower right")
        plt.title("Escalón de potencial: "+"$E=$"+En+", $V_0 =$"+str(Vn))
        plt.axis([xmin,xmax,-ymax,ymax]) 
        #plt.axis('off')
        #plt.savefig(directory + '/' + file_name.format(mm), dpi=600)
        plt.show()
       
