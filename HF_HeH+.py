# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:12:55 2020

@author: B568302
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import fmin
import matplotlib.pyplot as plt

def overlap(A,B,Ra): 
    "Returns the overlap integral Sij = <Xi/Xj>"
    return (np.pi/(A+B))**(3/2)*np.exp(-A*B*Ra**2/(A+B))

def kin(A,B,R_AB): 
    "Returns the kinetic integral Kij = <Xi/K/Xj>"
    return (A*B/(A+B))*(3-2*A*B*R_AB**2/(A+B))*(np.pi/(A+B))**(3/2)*\
    np.exp(-A*B*R_AB**2/(A+B))
    
def Fo(t):
        "Error function"
        if t < 1.e-6:
            return 1-t/3
        return 0.5*(np.pi/t)**0.5*sp.erf(t**0.5)        

def pot(A,B, Ra, Rb, Rc, Z):
        "Returns the potential integral Vij= <Xi/V/Xj>"
        Rp=(A*Ra+B*Rb)/(A+B)
        Rab=(Ra-Rb)
        return -2*np.pi*Z*np.exp(-A*B*Rab**2/(A+B))*Fo((A+B)*(Rp-Rc)**2)/(A+B)
    
def bielectron(A,B,C,D,Rab,Rcd,Rpq):
    """
    Returns the bielectronic integral, the colulumbic term <ab/ab> and the
    exchange term <ab/ba>
    Rab equals distance between centre A and centre B
    Rcd equals distance between centre C and centre D
    Rpq equals distance between centre p and q
    """
    return 2.0*(np.pi**2.5)/((A+B)*(C+D)*np.sqrt(A+B+C+D))*Fo((A+B)*(C+D)*Rpq**2/(A+B+C+D))\
    *np.exp(-A*B*Rab**2/(A+B)-C*D*Rcd**2/(C+D))
def integral(N,R,Zeta1,Zeta2,Za,Zb):
    """
    Returns all the values of the integrals in each component of the matrix
    """
   
    global S12,T11,T12,T22,V11A,V12A,V22A,V11B,V12B,V22B,V1111,V2111,V2121,V2211,V2221,V2222
    
    S12 = 0.0
    T11 = 0.0
    T12 = 0.0
    T22 = 0.0
    V11A = 0.0
    V12A = 0.0
    V22A = 0.0
    V11B = 0.0
    V12B = 0.0
    V22B = 0.0
    V1111 = 0.0
    V2111 = 0.0
    V2121 = 0.0
    V2211 = 0.0
    V2221 = 0.0
    V2222 = 0.0
    
    A1 = alpha[N-1]*Zeta1**2
    D1 = d[N-1]*(2*A1/np.pi)**0.75
    A2 = alpha[N-1]*Zeta2**2
    D2 = d[N-1]*(2*A2/np.pi)**0.75
    
    # One electron integrals 
    for i in range(N):
        for j in range(N):
            S12 +=overlap(A1[i],A2[j],R)*D1[i]*D2[j]
            T11 +=kin(A1[i],A1[j],0.0)*D1[i]*D1[j]
            T12 +=kin(A1[i],A2[j],R)*D1[i]*D2[j]
            T22 +=kin(A2[i],A2[j],0.0)*D2[i]*D2[j]
            V11A +=pot(A1[i],A1[j],-R/2,-R/2,-R/2,Za)*D1[i]*D1[j]
            V12A +=pot(A1[i],A2[j],-R/2,+R/2,-R/2,Za)*D1[i]*D2[j]
            V22A +=pot(A2[i],A2[j],+R/2,+R/2,-R/2,Za)*D2[i]*D2[j]
            V11B += pot(A1[i],A1[j],-R/2,-R/2,+R/2,Zb)*D1[i]*D1[j]
            V12B +=pot(A1[i],A2[j],-R/2,+R/2,+R/2,Zb)*D1[i]*D2[j]
            V22B +=pot(A2[i],A2[j],+R/2,+R/2,+R/2,Zb)*D2[i]*D2[j]
    
    # Calculate two electron integrals
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    Rap = A2[i]*R/(A2[i]+A1[j])
                    Raq = A2[k]*R/(A2[k]+A1[l])
                    Rbq = R - Raq
                    Rpq = Rap - Raq
                    V1111 +=bielectron(A1[i],A1[j],A1[k],A1[l],0.0,0.0,0.0)*D1[i]*D1[j]*D1[k]*D1[l]
                    V2111 +=bielectron(A2[i],A1[j],A1[k],A1[l],R,0.0,Rap)*D2[i]*D1[j]*D1[k]*D1[l]
                    V2121 +=bielectron(A2[i],A1[j],A2[k],A1[l],R,R,Rpq)*D2[i]*D1[j]*D2[k]*D1[l]
                    V2211 +=bielectron(A2[i],A2[j],A1[k],A1[l],0.0,0.0,R)*D2[i]*D2[j]*D1[k]*D1[l]
                    V2221 +=bielectron(A2[i],A2[j],A2[k],A1[l],0.0,R,Rbq)*D2[i]*D2[j]*D2[k]*D1[l]
                    V2222 +=bielectron(A2[i],A2[j],A2[k],A2[l],0.0,0.0,0.0)*D2[i]*D2[j]*D2[k]*D2[l]
    return 

def matrix(N,R,Zeta1,Zeta2,Za,Zb):
    """
    Put the integral values in each component of the matrix
    """
    #kinetic matrix <xi/T/Xj>
    T[0,0] = T11
    T[0,1] = T12
    T[1,0] = T[0,1]
    T[1,1] = T22
    # Potential matrix <xi/V/Xj>
    V[0,0] = V11A+V11B
    V[0,1] = V12A+V12B
    V[1,0] = V[0,1]
    V[1,1] = V22A+V22B
    # Core Hamiltonian <xi/H/Xj>
    H[0,0] = T[0,0]+V[0,0]
    H[0,1] = T[0,1]+V[0,1]
    H[1,0] = H[0,1]
    H[1,1] = T[1,1]+V[1,1]

    # Form overlap matrix <Xi/Xj
    S[0,0] = 1.0
    S[0,1] = S12
    S[1,0] = S12
    S[1,1] = 1.0
    
    #  S^-1/2 from unitary transformation matrix
    X[0,0] = 1.0/np.sqrt(2.0*(1.0+S12))
    X[1,0] = X[0,0]
    X[0,1] = 1.0/np.sqrt(2.0*(1.0-S12))
    X[1,1] = -X[0,1]
    

    # the colulumbic term <ab/ab> and the exchange term <ab/ba>
    TT[0,0,0,0] = V1111
    TT[1,0,0,0] = V2111
    TT[0,1,0,0] = V2111
    TT[0,0,1,0] = V2111
    TT[0,0,0,1] = V2111
    TT[1,0,1,0] = V2121
    TT[0,1,1,0] = V2121
    TT[1,0,0,1] = V2121
    TT[0,1,0,1] = V2121
    TT[1,1,0,0] = V2211
    TT[0,0,1,1] = V2211
    TT[1,1,1,0] = V2221
    TT[1,1,0,1] = V2221
    TT[1,0,1,1] = V2221
    TT[0,1,1,1] = V2221
    TT[1,1,1,1] = V2222

def SCF(N,R,Zeta1,Zeta2,Za,Zb):
    
    global Energy, Energytot,epsilon,C
    Crit = 1e-4 # Convergence 
    Maxit = 1000 # nº of iterations
    
    #1. Initial density matrix 
    #The null matrix is used to give the initial guess of the density matrix
    P = np.zeros([2,2])
    
    Iter=0
    
    while (Iter<Maxit):
        Iter += 1
     
        #2. Calculate the Fock matrix by calculatin the G matrix (coulumb and exchange terms)
        G = np.zeros([2,2]) # This is the two electron contribution in the equations above
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        G[i,j]+=P[k,l]*(TT[i,j,k,l]-0.5*TT[i,l,k,j])

        # Calculate the Fock matrix with the core Hamiltonain and the G matrix
        F = H+G
        
        #3. Transform the Fcok matrix with the X matrix 
        ewa = np.matmul(F,X)
        Fprime = np.matmul(X.T,ewa)
        
        #4. Obtain the eigenvalues and the coefficients 
        epsilon,Cprime=np.linalg.eigh(Fprime)
                                     
        #5. Molecular orbitals coefficients
        # Transform Cprime to get the coefficients C
        C = np.matmul(X,Cprime)
        
        #6. Calculate the new density matrix from the old P 
        P_0 = np.array(P)
        P= np.zeros([2,2])
        
        # Form new density matrix
        for i in range(2):
            for j in range(2):
                #New density matrix
                for k in range(1):
                    P[i,j] += 2.0*C[i,k]*C[j,k]
        
        #7. Check the convergence 
        Delta = 0.0
       
        Delta = (P-P_0)
        Delta = np.sqrt(np.sum(Delta**2)/4.0)
        
        if (Delta<Crit):
            # Electronic energy
            Energy = np.sum(0.5*P*(H+F))
            # Add nuclear repulsion to get the total energy
            Energytot = Energy+Za*Zb/R
            return Energy,Energytot,P,epsilon,C 

def HFCALC(N,R,Zeta1,Zeta2,Za,Zb):  
    
    # Calculate one and two electron integrals
    integral(N,R,Zeta1,Zeta2,Za,Zb)
    # Put all integrals into matrix form
    matrix(N,R,Zeta1,Zeta2,Za,Zb)
    # SCF calculation
    SCF(N,R,Zeta1,Zeta2,Za,Zb)
    return

global H,S,X,XT,V,T,TT,G,C,P,Oldp,F,Fprime,Cprime,E,Zb,Energy1

H = np.zeros([2,2])
S = np.zeros([2,2])
X = np.zeros([2,2])
T = np.zeros([2,2])
V = np.zeros([2,2])
TT = np.zeros([2,2,2,2])
G = np.zeros([2,2])
C = np.zeros([2,2])
P= np.zeros([2,2])
U= np.zeros([2,2])
s= np.zeros([2,2])

Oldp = np.zeros([2,2])
F = np.zeros([2,2])
Fpre =np.zeros([2,2])
Fprime = np.zeros([2,2])
Cprime = np.zeros([2,2])
E = np.zeros([2,2])

# The coefficients for the contracted Gaussian functions are below
d = np.array([[1.00000,0.0000000,0.000000],
                      [0.678914,0.430129,0.000000],
                      [0.444635,0.535328,0.154329]])
alpha = np.array([[0.270950,0.000000,0.000000],
                      [0.151623,0.851819,0.000000],
                      [0.109818,0.405771,2.227660]])


Delta = 0.0
N = 3
R = 1.4632
Zeta1 = 2.0925
Zeta2 = 1.24
Za = 2.0
Zb = 1.0
HFCALC(N,R,Zeta1,Zeta2,Za,Zb)
print("Converged electronic energy:", Energy)
print("Converged total energy:",Energytot)
print("Energy eigenvalues",epsilon)
print("Matrix coefficients",C)



#------------------------------------------------------------------------------
#Plot making a function that only returns the Energy and vectorize the function
#for introduce an array length to plot the Energy vs the intranuclear distance
def fun(x):
    HFCALC(N,x,Zeta1,Zeta2,Za,Zb)
    return Energy
def fun1(x):
    HFCALC(N,x,Zeta1,Zeta2,Za,Zb)
    return Energytot
fun=np.vectorize(fun)
fun1=np.vectorize(fun1)
RR=np.linspace(0.5,4, 100) 
print("COMPARATION") 
print("Equilibrium distance from input = ",R)
print("Total Energy = ",fun1(R))
R_e=fmin(lambda R: fun1(R), 0.01, disp=False)[0]
print("Equilibrium distance calculated from the graphic =",R_e)
print("Energy calculated from the graphic =",fun1(R_e))
plt.ylim(-3,-2)
arrow_x = R_e
arrow_y = fun1(R_e)
plt.vlines(R_e,-3,-2, color='r',label='R_e')
plt.annotate("(R_e, Min.Energy)", xy=(arrow_x, arrow_y-0.05))
plt.xlabel('R (a.u)')
plt.ylabel('Energy (Hartrees)')
plt.grid()
plt.plot(RR, fun1(RR), color='blue',linestyle='dashed',label='Energy')
plt.legend(loc='best')





