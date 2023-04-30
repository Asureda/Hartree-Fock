"""
@author: Alexandre Sureda Croguennoc
Electronic Structure - Quantum mechanics homework
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import itertools


class HartreeFock:
    def __init__(self, N,R,Zeta1,Zeta2,Za,Zb, Crit = 1e-4 , Maxit = 10000):
        self.N = N
        self.R = R
        self.Zeta1 = Zeta1
        self.Zeta2 = Zeta2
        self.Za = Za
        self.Zb = Zb
        self.Crit = Crit
        self.Maxit = Maxit

        self.integral()
        self.colect()
        self.SCF()

    def overlap(self,A,B,Ra): 
        "Returns the overlap integral Sij = <Xi/Xj>"
        return (np.pi/(A+B))**(3/2)*np.exp(-A*B*Ra**2/(A+B))

    def kin(self,A,B,R_AB): 
        "Returns the kinetic integral Kij = <Xi/K/Xj>"
        return (A*B/(A+B))*(3-2*A*B*R_AB**2/(A+B))*(np.pi/(A+B))**(3/2)*\
        np.exp(-A*B*R_AB**2/(A+B))
        
    def Fo(self,t):
            "Error function"
            if t < 1.e-6:
                return 1-t/3
            return 0.5*(np.pi/t)**0.5*erf(t**0.5)        

    def pot(self,A,B, Ra, Rb, Rc, Z):
            "Returns the potential integral Vij= <Xi/V/Xj>"
            Rp=(A*Ra+B*Rb)/(A+B)
            Rab=(Ra-Rb)
            return -2*np.pi*Z*np.exp(-A*B*Rab**2/(A+B))*self.Fo((A+B)*(Rp-Rc)**2)/(A+B)
        
    def bielectron(self,A,B,C,D,Rab,Rcd,Rpq):
        """
        Returns the bielectronic integral, the colulumbic term <ab/ab> and the
        exchange term <ab/ba>
        Rab equals distance between centre A and centre B
        Rcd equals distance between centre C and centre D
        Rpq equals distance between centre p and q
        """
        return 2.0*(np.pi**2.5)/((A+B)*(C+D)*np.sqrt(A+B+C+D))*self.Fo((A+B)*(C+D)*Rpq**2/(A+B+C+D))*np.exp(-A*B*Rab**2/(A+B)-C*D*Rcd**2/(C+D))


    def integral(self):
    
        """
        Returns all the values of the integrals in each component of the matrix
        """

        # The coefficients for the contracted Gaussian functions are below
        d = np.array([[1.00000,0.0000000,0.000000],
                            [0.678914,0.430129,0.000000],
                            [0.444635,0.535328,0.154329]])
        alpha = np.array([[0.270950,0.000000,0.000000],
                      [0.151623,0.851819,0.000000],
                      [0.109818,0.405771,2.227660]])
        
        A1 = alpha[self.N-1]*self.Zeta1**2
        D1 = d[self.N-1]*(2*A1/np.pi)**0.75
        A2 = alpha[self.N-1]*self.Zeta2**2
        D2 = d[self.N-1]*(2*A2/np.pi)**0.75
        
    
        self.S12 = np.sum([self.overlap(A1[i],A2[j],self.R)*D1[i]*D2[j] for i in range(self.N) for j in range(self.N)])
        self.T11 = np.sum([self.kin(A1[i],A1[j],0.0)*D1[i]*D1[j] for i in range(self.N) for j in range(self.N)])
        self.T12 = np.sum([self.kin(A1[i],A2[j],self.R)*D1[i]*D2[j] for i in range(self.N) for j in range(self.N)])
        self.T22 = np.sum([self.kin(A2[i],A2[j],0.0)*D2[i]*D2[j] for i in range(self.N) for j in range(self.N)])
        self.V11A = np.sum([self.pot(A1[i],A1[j],-self.R/2,-self.R/2,-self.R/2,self.Za)*D1[i]*D1[j] for i in range(self.N) for j in range(self.N)])
        self.V12A = np.sum([self.pot(A1[i],A2[j],-self.R/2,+self.R/2,-self.R/2,self.Za)*D1[i]*D2[j] for i in range(self.N) for j in range(self.N)])
        self.V22A = np.sum([self.pot(A2[i],A2[j],+self.R/2,+self.R/2,-self.R/2,self.Za)*D2[i]*D2[j] for i in range(self.N) for j in range(self.N)])
        self.V11B = np.sum([self.pot(A1[i],A1[j],-self.R/2,-self.R/2,+self.R/2,self.Zb)*D1[i]*D1[j] for i in range(self.N) for j in range(self.N)])
        self.V12B = np.sum([self.pot(A1[i],A2[j],-self.R/2,+self.R/2,+self.R/2,self.Zb)*D1[i]*D2[j] for i in range(self.N) for j in range(self.N)])
        self.V22B = np.sum([self.pot(A2[i],A2[j],+self.R/2,+self.R/2,+self.R/2,self.Zb)*D2[i]*D2[j] for i in range(self.N) for j in range(self.N)])

        # Calculate two electron integrals
        self.V1111 = self.V2111 = self.V2121 = self.V2211 = self.V2221 = self.V2222 = 0.0


        for i, j, k, l in itertools.product(range(self.N), repeat=4):
            Rap = A2[i]*self.R/(A2[i]+A1[j])
            Raq = A2[k]*self.R/(A2[k]+A1[l])
            Rbq = self.R - Raq
            Rpq = Rap - Raq
            self.V1111 += self.bielectron(A1[i],A1[j],A1[k],A1[l],0.0,0.0,0.0)*D1[i]*D1[j]*D1[k]*D1[l]
            self.V2111 += self.bielectron(A2[i],A1[j],A1[k],A1[l],self.R,0.0,Rap)*D2[i]*D1[j]*D1[k]*D1[l]
            self.V2121 += self.bielectron(A2[i],A1[j],A2[k],A1[l],self.R,self.R,Rpq)*D2[i]*D1[j]*D2[k]*D1[l]
            self.V2211 += self.bielectron(A2[i],A2[j],A1[k],A1[l],0.0,0.0,self.R)*D2[i]*D2[j]*D1[k]*D1[l]
            self.V2221 += self.bielectron(A2[i],A2[j],A2[k],A1[l],0.0,self.R,Rbq)*D2[i]*D2[j]*D2[k]*D1[l]
            self.V2222 += self.bielectron(A2[i],A2[j],A2[k],A2[l],0.0,0.0,0.0)*D2[i]*D2[j]*D2[k]*D2[l]
            
        return 

    def colect(self):
        self.T = np.array([[self.T11, self.T12], [self.T12, self.T22]])
        self.V = np.array([[self.V11A+self.V11B, self.V12A+self.V12B], [self.V12A+self.V12B, self.V22A+self.V22B]])
        self.H = self.T + self.V
        self.S = np.array([[1.0, self.S12], [self.S12, 1.0]])
        self.X = np.array([[1.0/np.sqrt(2.0*(1.0+self.S12)), 1.0/np.sqrt(2.0*(1.0-self.S12))], 
                    [1.0/np.sqrt(2.0*(1.0+self.S12)), -1.0/np.sqrt(2.0*(1.0-self.S12))]])
        self.TT = np.array([[[[self.V1111, self.V2111], [self.V2111, self.V2211]], [[self.V2111, self.V2121], [self.V2121, self.V2221]]],
                    [[[self.V2111, self.V2121], [self.V2121, self.V2221]], [[self.V2211, self.V2221], [self.V2221, self.V2222]]]])
        
    def SCF(self):
        """
        SCF iteration
        """
        
        #1. Initial density matrix 
        #The null matrix is used to give the initial guess of the density matrix
        P = np.zeros([2,2])
        
        Iter=0
        TT = self.TT
        X = self.X
        T = self.T
        H = self.H

        while (Iter<self.Maxit):
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
            
            if (Delta<self.Crit):
                # Electronic energy
                self.P = P
                self.epsilon = epsilon
                self.C = C
                self.Energy = np.sum(0.5*self.P*(H+F))
                # Add nuclear repulsion to get the total energy
                self.Energytot = self.Energy+self.Za*self.Zb/self.R
                return self.Energy,self.Energytot,self.P,self.epsilon,self.C 




