import numpy as n3
import copy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
import time
from numpy.linalg import norm, svd, lstsq
import sys
import numpy as np
import time
import scipy.optimize
import scipy.special as spl
import scipy.sparse as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import os
from scipy.io import savemat, loadmat
from os.path import exists

norm = lambda x: np.linalg.norm(x)


def main():
    
    nmax = 3000
    nt = 2500
    table = np.zeros((nmax+1,nt)) 
    table_scipy = np.zeros((nmax+1,nt))

    filename = "jn_zeros_n=" + str(nmax) + "_nt=" + str(nt) + ".mat"

    if exists(filename):
        data = loadmat(filename)
        table = date["roots_table"]
    else:
        t0 = time.time()
        for n in range(nmax+1):
            table[n,:] = my_jn_zeros(n,nt)
        t1 = time.time()
        dt = t1 - t0
        print("Time my basic code (seconds)",dt)
        data = {"roots_table": table}
        savemat(filename,data)
    if nmax > 200:
        quit()

    print("attemping spl.jn_zeros") 
    for n in range(nmax+1):
        table_scipy[n,:] = spl.jn_zeros(n,nt)
    t1 = time.time()
    dt = t1 - t0
    print("Time scipy (seconds)",dt)


    err = np.linalg.norm(table_scipy - table)/np.linalg.norm(table_scipy)
    print("norm(table - table_scipy)/norm(table_scipy)=",err)
    


def my_jn_zeros(n,nt):
    # Estimate for first zero
    x0 = n 

    m = 3*nt//2
    x = x0 + np.arange(m)*3
    y = spl.jv(n,x)

    z = np.zeros(nt)
    jj = 0
    for k in range(1,m):
        if y[k]*y[k-1] <= 0:
            sgn = 1
            if y[k] <= y[k-1]:
                sgn = -1
            a = x[k-1]
            b = x[k]

            for j in range(4):
                c = (a+b)/2
                if sgn*spl.jv(n,c) <=0:
                    a = c
                else:
                    b = c
            z[jj] = c
            jj = jj + 1
            if jj >= nt:
                break

    for i in range(nt):
        x0 = z[i]
        for itr in range(5):
            f = spl.jv(n,x0)
            df = spl.jvp(n,x0)
            x0 = x0 - f/df
        z[i] = x0

    return z



if __name__ == "__main__":
    main()
