import numpy as np
import math
import csv
import random
from collections import defaultdict 
from datetime import datetime

np.random.seed(42)
random.seed(42)

# --- Fitness Functions ------------------------------------------------------
def fitness5(x):        # Rastrigin
    n = len(x)
    ss = 0
    for l in range(n):
        ss += x[l]**2 - 10*math.cos(2 * math.pi * x[l])
    return 10 * n + ss


def fitness6(x):        # Schwefel
    n = len(x)
    sum_part = 0
    for i in range(n):
        sum_part += x[i] * math.sin(math.sqrt(abs(x[i])))
    return 418.98288727243369 * n - sum_part

def fitness7(x):        # Ackley 
    n = len(x)
    ss1 = 0
    ss2 = 0
    for l in range(n):
        ss1 += x[l]**2
        ss2 += math.cos(2 * math.pi * x[l])
    return -20 * math.exp(-0.2 * math.sqrt(ss1/n)) - math.exp(ss2/n) + 20 + math.exp(1)

def fitness8(x):        # Griewank
    n = len(x)
    ss1 = 0
    ss2 = 1
    for l in range(n):
        ss1 += x[l]**2 / 4000
        ss2 *= math.cos(x[l] / math.sqrt(l+1))
    return ss1 - ss2 + 1

def fitness9(x):          #Neurophysiology application
    f1 = x[0]**2 + x[2]**2 - 1
    f2 = x[1]**2 + x[3]**2 - 1
    f3 = x[4] * x[2]**3 + x[5] * x[3]**3
    f4 = x[4] * x[0]**3 + x[5] * x[1]**3
    f5 = x[4] * x[0] * x[2]**2 + x[5] * x[3]**2 * x[1]
    f6 = x[4] * x[0]**2 * x[2] + x[5] * x[1]**2 * x[3]
    return (f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2) / 6

def fitness10(x):         #Robot kinematics application
    f1 = 4.731 * 10 ** (-3) * x[0] * x[2] - 0.3578 * x[1] * x[2] -0.1238 * x[0] + x[6] - 1.637 * 10 ** (-3) * x[1] - 0.9338 * x[3] -0.3571
    f2 = 0.2238 * x[0] * x[2]  + 0.7623 * x[1] * x[2] + 0.2638 * x[0] - 0.07745 * x[3] - 0.3571
    f3 = x[5] * x[7] + 0.3578 * x[0] + 4.731 * 10 ** (-3) * x[1]
    f4 = -0.7623 * x[0] + 0.2238 * x[1] + 0.3461
    f5 = x[0]**2 + x[1]**2 - 1
    f6 = x[2]**2 + x[3]**2 - 1
    f7 = x[4]**2 + x[5]**2 - 1
    f8 = x[6]**2 + x[7]**2 - 1
    return (f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2 + f7**2 + f8**2) / 8

def fitness11(x):
    phi = [1.3954170041747090114, 1.7444828545735749268, 2.0656234369405315689, 2.4600678478912500533]
    psi = [1.7461756494150842271, 2.0364691127919609051, 2.2390977868265978920, 2.4600678409809344550]
    ss = []
    n = len(x)
    for i in range (n) :
        Ei = x[1] * (np.cos(psi[i+1]) - np.cos(psi[0])) - x[1]*x[2]*(np.sin(psi[i+1]) - np.sin(psi[0])) - (x[1]*np.sin(psi[i+1]) - x)*x[0]
        Fi = -x[1]*np.cos(phi[i+1]) - x[1]*x[2]*np.sin(phi[i+1]) + x[1]*np.cos(phi[0]) + x[0]*x[2] + (x[2] - x[0])*x[1]*np.sin(phi[0])
        ss1 = (Ei*(x[1]*np.sin(phi[i+1]) - x[2]) - Fi*(x[1]*np.sin(psi[i+1]) - x[2]))**2
        ss2 = (Fi*(1 + x[1]*np.cos(psi[i+1])) - Ei*(x[1]*np.cos(phi[i+1]) - 1))**2
        ss3 = ((1 + x[1]*np.cos(psi[i+1]))*(x[1]*np.sin(phi[i+1]) - x[2])*x[0] - (x[1]*np.sin(psi[i+1]) - x[2])*(x[1]*np.cos(phi[i+1]) - x[2])*x[0])**2
        ss.append(ss1 + ss2 - ss3)
    return np.sum(np.array(ss)**2)

def fitness12(x):          # Economics modeling
    n = len(x)
    F = np.zeros(n) 
    for i in range (n):
        if i < (n-1):
            ss = 0
            for j in range (n-i):
                ss += x[j] * x[j+i]
            F[i] = (x[i] + ss) * x[-1]
        else :
            F[i] = np.sum(x) - x[-1] + 1
    return np.sum(F**2) / n

def fitness13(x):   # Chemical equilibrium problem
    R1 = 10
    R2 = 0.193
    R3 = 0.002597 / np.sqrt(40)
    R4 = 0.003448 / np.sqrt(40)
    R5 = 0.00001799 / 40
    R6 = 0.0002155 / np.sqrt(40)
    R7 = 0.00003846 / 40
    
    f1 = x[0] * x[1] + x[0] - 3 * x[4]
    f2 = (2 * x[0] * x[1] + x[0] + x[1] * (x[2]**2) + R5 * x[1] - R1 * x[4] 
          + 2 * R7 * (x[1]**2) + R4 * x[1] * x[2] + R6 * x[1] * x[3])
    f3 = 2 * x[1] * (x[2]**2) + 2 * R2 * (x[2]**2) - 8 * x[4] + R3 * x[2] + R4 * x[1] * x[2]
    f4 = R6 * x[1] * x[3] + 2 * (x[3]**2) - 4 * R1 * x[4]
    f5 = (x[0] * (x[1] + 1) + R7 * (x[1]**2) + x[1] * (x[2]**2) + R5 * x[1] 
          + R2 * (x[2]**2) + (x[3]**2) - 1 + R3 * x[2] + R4 * x[1] * x[2] + R6 * x[1] * x[3])
    
    return (f1**2 + f2**2 + f3**2 + f4**2 + f5**2) / 5

def fitness22(x):    #8 Himmelblau's function
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7
    return (term1**2 + term2**2)

def fitness27(x):  # Six-Hump Camel Function
    x1 = x[0]
    x2 = x[1]
    
    term1 = (4 - 2.1 * (x1**2) + (x1**4) / 3) * (x1**2)
    term2 = x1 * x2
    term3 = (-4 + 4 * (x2**2)) * (x2**2)
    
    return np.abs(term1 + term2 + term3 + 1.03162845348)

def fitness28(x):  #Bukin Function N. 6
    x1 = x[0]
    x2 = x[1]
    
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * (x1 ** 2)))
    
    term2 = 0.01 * np.abs(x1 + 10)
    
    return term1 + term2

def fitness29(x):  #Cross-in-Tray Function
    x1 = x[0]
    x2 = x[1]
    
    # 1. sqrt(x1^2 + x2^2) / pi
    radial_part = np.sqrt(x1**2 + x2**2) / np.pi
    
    # 2. exp(| 100 - radial_part |)
    exp_term = np.exp(np.abs(100 - radial_part))
    
    # 3. | sin(x1) * sin(x2) * exp_term |
    inner_abs = np.abs(np.sin(x1) * np.sin(x2) * exp_term)
    
    result = (-0.0001 * ((inner_abs + 1) ** 0.1)) +2.06261
    
    return result

def fitness30(x):  #Drop-Wave Function

    x1 = x[0]
    x2 = x[1]

    frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2

    y = (-frac1 / frac2) + 1.0

    return y

def fitness31(x):  #Eggholder Function
    x1 = x[0]
    x2 = x[1]

    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

    y = (term1 + term2) + 959.6407

    return y

def fitness32(x):  #Holder Table Function
    x1 = x[0]
    x2 = x[1]

    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))

    y = -np.abs(fact1 * fact2) + 19.2085025

    return y

def fitness33(x):  # Levy Function
    d = len(x)
    w = 1 + (x - 1) / 4
    w1 = w[0]
    wd = w[-1]
    
    term1 = (np.sin(np.pi * w1))**2
    term3 = ((wd - 1)**2) * (1 + (np.sin(2 * np.pi * wd))**2)
    
    w_sum_part = w[:-1] 
    sum_terms = ((w_sum_part - 1)**2) * (1.0 + 10.0 * (np.sin(np.pi * w_sum_part + 1))**2)
    total_sum = np.sum(sum_terms)
    
    y = term1 + total_sum + term3
    
    return y

def fitness34(x):    # Levy Function N.13
    x1 = x[0]
    x2 = x[1]


    term1 = (np.sin(3 * np.pi * x1))**2
    term2 = ((x1 - 1)**2) * (1 + (np.sin(3 * np.pi * x2))**2)
    term3 = ((x2 - 1)**2) * (1 + (np.sin(2 * np.pi * x2))**2)

    y = term1 + term2 + term3

    return y

def fitness35(x):  #Schaffer Function N. 2
    x1 = x[0]
    x2 = x[1]

    fact1 = (np.sin(x1**2 - x2**2))**2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2))**2

    y = 0.5 + fact1 / fact2

    return y

def fitness36(x):  #Schaffer Function N. 4
    x1 = x[0]
    x2 = x[1]

    fact1 = (np.cos(np.sin(np.abs(x1**2 - x2**2))))**2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2))**2

    y = 0.5 + fact1 / fact2

    return y

def fitness37(x):  #Branin function
    x1 = x[0]
    x2 = x[1]

    term1 = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5.0 / np.pi) * x1 - 6.0) ** 2
    term2 = 10.0 * (1.0 - 1.0 / (8 * np.pi)) * np.cos(x1) + 10.0

    y = (term1 + term2) - 0.397887

    return y

def fitness38(x):  #Beale Function
    x1 = x[0]
    x2 = x[1]

    term1 = (1.5 - x1 + x1 * x2) ** 2
    term2 = (2.25 - x1 + x1 * (x2 ** 2)) ** 2
    term3 = (2.625 - x1 + x1 * (x2 ** 3)) ** 2

    y = term1 + term2 + term3

    return y

def fitness39(x):  #Colville Function
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    term1 = 100 * (x1**2 - x2)**2
    term2 = (x1 - 1)**2
    term3 = (x3 - 1)**2
    term4 = 90 * (x3**2 - x4)**2
    term5 = 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
    term6 = 19.8 * (x2 - 1) * (x4 - 1)

    y = term1 + term2 + term3 + term4 + term5 + term6

    return y

def fitness40(x): #Easom Function
    x1 = x[0]
    x2 = x[1]

    term1 = -np.cos(x1) * np.cos(x2)
    term2 = np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))

    y = (term1 * term2) + 1.0

    return y

def fitness41(x): # Goldstein & Price Function
    x1 = x[0]
    x2 = x[1]

    term1 = 1 + ((x1 + x2 + 1) ** 2) * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + ((2*x1 - 3*x2) ** 2) * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)

    y = (term1 * term2) - 3

    return y

def fitness42(x):  #Hump Functions
    x1 = x[0]
    x2 = x[1]

    term1 = 4 * (x1**2) - 2.1 * (x1**4) + (1/3) * (x1**6)
    term2 = x1 * x2
    term3 = -4 * (x2**2) + 4 * (x2**4)

    y = term1 + term2 + term3

    return y

def fitness43(x):  #Matyas Function
    x1 = x[0]
    x2 = x[1]

    term1 = 0.26 * (x1**2 + x2**2)
    term2 = 0.48 * x1 * x2

    y = term1 - term2

    return y

def fitness44(x): #Three-Hump Camel Function
    x1 = x[0]
    x2 = x[1]

    term1 = 2 * (x1 ** 2)
    term2 = -1.05 * (x1 ** 4)
    term3 = (x1 ** 6) / 6
    term4 = x1 * x2
    term5 = x2 ** 2

    y = term1 + term2 + term3 + term4 + term5

    return y

def fitness45(x):     #Styblinski-Tang Function
    d = len(x)

    total_sum = 0
    for i in range(d):
        xi = x[i]
        
        term1 = xi ** 4
        term2 = -16 * (xi ** 2)
        term3 = 5 * xi
        
        total_sum += (term1 + term2 + term3)

    y_raw = total_sum / 2
    
    y = y_raw + (39.1661657037714 * d)

    return y

def fitness46(x):
    sum_sq = np.sum(x ** 2)

    term1 = 1 + np.cos(12 * np.sqrt(sum_sq))
    term2 = 0.5 * sum_sq + 2

    y = -(term1 / term2) + 1

    return y   

# --- Initialization ---
def initial_population(NP, D, L, U, fitness):
    P = np.zeros((NP, D+1))
    nf = 0
    xb, fb = None, float('inf')

    for i in range(NP):
        for j in range(D):
            P[i,j] = L[j] + (U[j] - L[j]) * np.random.rand()
        
        P[i,D] = fitness(P[i,:D])
        nf += 1
        
        if P[i,D] < fb:
            xb = P[i,:D].copy()
            fb = P[i,D]

    return P, nf, xb, fb

def mutation1(i, P, NP, D, F, L, U):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 3, replace=False)
        if i not in idxs and len(set(idxs)) == 3:
            valid = True
            
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    xr3 = P[idxs[2], 0:D]
     
    v = xr1 + F * (xr2 - xr3) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
    
    return v

def mutation2(i, P, NP, D, F, idx_xs1, L, U, setQ): 
    xs1 = P[idx_xs1, 0:D]
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 1, replace=False)
        idx_r1 = idxs[0] 
        if i != idx_r1 and idx_xs1 != idx_r1:
            valid = True
    
    xr1 = P[idx_r1, 0:D]

    # xrq = np.array(random.choice(setQ))
    if len(setQ) > 5:
        xrq = np.array(random.choice(setQ))
    else:
        valid = False
        while not valid:
            xrq_idx = np.random.randint(0, NP)
            if xrq_idx != i and xrq_idx != idx_xs1 and xrq_idx != idx_r1:
                valid = True
        xrq = P[xrq_idx, 0:D]
    
    v = xs1 + F * (xrq - xr1) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
            
    return v, setQ

def mutation3(i, P, NP, D, F, L, U, setQ): 
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 2, replace=False)
        if i not in idxs and len(set(idxs)) == 2:
            valid = True
            
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]

    # xrq = np.array(random.choice(setQ))
    if len(setQ) > 5:
        xrq = np.array(random.choice(setQ))
    else:
        valid = False
        xrq = np.zeros(D)
        while not valid:
            for j in range(D):
                xrq[j] = L[j] + (U[j] - L[j]) * np.random.rand()
            # if xrq_idx != i and xrq_idx != idxs and xrq_idx != xr1:
            # if xrq_idx != i and xrq_idx != idxs :
                valid = True
        # xrq = P[xrq_idx, 0:D]
    
    v = xr1 + F * (xr2 - xrq) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
            
    return v, setQ

def mutation4(i, P, NP, D, F1, xs1, L, U):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 2, replace=False)
        if i not in idxs and len(set(idxs)) == 2:
            valid = True
    
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    
    v = xs1 + F1 * (xr1 - xr2) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + np.random.rand() * (U[k] - L[k])
    return v

def mutation5(i, P, NP, D, F, idx_xs1, L, U, setQ): 
    xs1 = P[idx_xs1, 0:D]
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 3, replace=False)
        if i not in idxs and len(set(idxs)) == 3:
            valid = True
            
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    xr3 = P[idxs[2], 0:D]

    if len(setQ) > 5:
        xrq = np.array(random.choice(setQ))
    else:
        valid_q = False
        while not valid_q:
            xrq_idx = np.random.randint(0, NP)
            if (xrq_idx != i and 
                xrq_idx != idx_xs1 and 
                xrq_idx not in idxs):
                valid_q = True
        xrq = P[xrq_idx, 0:D].copy()
# 
    # v = xr1 + F * (xrq - xr2) + F * (xs1 - xr3)
    # v = xi + F * (xrq - xr1) + F * (xr2 - xr3)
    # v = xs1 + F * (xb - xi) + F * (xr1 - xr2)
    v = xi + F * (xb - xi) + F * (xr1 - xr2)  #### Best
    # v = xi + F * (xs1 - xi) + F * (xr1 - xr2)
    # v = xi + F * (xb - xi) + F * (xs1 - xr2)
    # v = xi + F * (xb - xr1) + F * (xs1 - xr2)
    # v = xs1 + F * (xb - xi) + F * (xr1 - xr2)
    # v = xs1 + F * (xrq - xr1) + F * (xr2 - xr3)
    # v = xs1 + F * (xb - xi)
    # v = xi + F * (xb - xi)
    # v = xi + F * (xb - xi) + F * (xr1 - xr2)
    # v = xi + F * (xb - xi) + F * (xb - xr1)/2
    # v = xi + F * (xb - xi) + F * (xi - xrq)/2
    # v = xi + F * (xb - xi) + F * (xrq - xr1)/2

    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
            
    return v, setQ

def mutation6(i, P, NP, D, F, L, U, xs1, xb):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 2, replace=False)
        if i not in idxs and len(set(idxs)) == 2:
            valid = True
    
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    
    # v = xs1 + F * (xb - xi) + F * (xr1 - xr2)
    v = xi + F * (xb - xi) + F * (xr1 - xr2)
    # v = xi + F * (xs1 - xi) + F * (xr1 - xr2)
    # v = xi + F * (xb - xi) + F * (xs1 - xr2)
    # v = xi + F * (xb - xr1) + F * (xs1 - xr2)
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + np.random.rand() * (U[k] - L[k])
    return v

def mutation7(i, P, NP, D, F, L, U, xs1):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 2, replace=False)
        if i not in idxs and len(set(idxs)) == 2:
            valid = True
    
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    
    # v = xs1 + F * (xb - xi) + F * (xr1-xr2)
    # v = xi + F * (xb - xi) + F * (xr1-xr2)
    v = xi + F * (xs1 - xi) + F * (xr1-xr2)
    # v = xi + F * (xb - xi) + F * (xs1 - xr2)
    # v = xi + F * (xb - xr1) + F * (xs1 - xr2)
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + np.random.rand() * (U[k] - L[k])
    return v


def crossoverDEASC(i, P, D, v, pc1):
    ww = np.random.rand()
    if ww < pc1 :
        ms = 1
        CR = 0 + np.random.rand() * 0.1  
    else:
        ms = 2
        CR = 0.7 + np.random.rand() * 0.3

    u = P[i,:D].copy()
    I_r = np.random.randint(0, D) 
    
    for j in range(D):
        if np.random.rand() <= CR or j == I_r:
            u[j] = v[j]
    
    return u, ms

def selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, ms, nc1, nc2, success_nfs, setQ):
    fitness_P = P[i,D]
    fitness_u = fitness(u)
    nf += 1
    
    mc = False
    if fitness_u < fitness_P:
        setQ.append(P[i, :D].copy())
        if len(setQ) > 2*NP:
            setQ.pop(0)
        P[i,:D] = u.copy()
        P[i,D] = fitness_u

        if ms == 1:
            nc1 += 1
        else:
            nc2 += 1

        if fitness_u <= fb:
            xb = P[i,:D].copy()
            fb = fitness_u
            archive.append([float(fb), xb.tolist()])
            if len(archive) > n_archive:
                archive.sort(key=lambda x: x[0])
                archive = archive[:n_archive]

            if fb <= VTR:
                found_success = True 
                success_nfs.append(nf)
                mc = True

    return P, xb, fb, nf, mc, archive, success_nfs, nc1, nc2, found_success, setQ   


def select_elite(P, D, NP):
    F_vals = P[:, D]    
    sorted_idx = np.argsort(F_vals)
    
    best_x = P[sorted_idx[0], 0:D]
    worst_x = P[sorted_idx[-1], 0:D]
    
    dmax = np.linalg.norm(worst_x - best_x)
    rc = dmax / (2 * (NP ** (1/D)))
    r = max(0.005, rc)

    S = [sorted_idx[0]] 
    f_best = F_vals[sorted_idx[0]]
    f_worst = F_vals[sorted_idx[-1]]

    for idx in sorted_idx[1:]:
        Fi = P[idx, 0:D]
        e = (F_vals[idx] - f_best) / (f_worst - f_best + 1e-6)
        
        far = False
        if (F_vals[idx] - f_best) <= e:
            for elite_idx in S:
                P_elite = P[elite_idx, 0:D]
                distance = np.linalg.norm(P_elite - Fi)
                
                if distance > r:
                    far = True
                    break
        
        if far:
            S.append(idx)
            
    return S

def DE_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success, pc1, nc1, nc2, success_nfs, setQ):
    for i in range(NP):
        if mc or nf >= MAX_NFE:
            break 
        
        # F = 0.5
        F = 0.5 + 0.2 * np.random.rand()
        # if np.random.rand() < 0.8:
        #     F = 0.5 + 0.2 * np.random.rand()
        # else:
        #     F = 0.1 + 0.2 * np.random.rand()

        M = np.random.rand()
        if M <= 0.9: 
            # v, setQ = mutation3(i, P, NP, D, F, L, U, setQ)
            v = mutation1(i, P, NP, D, F, L, U)
            # S = select_elite(P, D, NP)
            # xs1= random.choice(S) 
            # v = mutation7(i, P, NP, D, F, L, U, xs1)
        else:
            S = select_elite(P, D, NP)
            xs1= random.choice(S) 
            idx_xs1 = random.choice(S) 
            # v, setQ = mutation2(i, P, NP, D, F, idx_xs1, L, U, setQ) 
            # v, setQ = mutation3(i, P, NP, D, F, L, U, setQ)
            # v = mutation4(i, NP, F, xs1, L, U)        
            v, setQ = mutation5(i, P, NP, D, F, idx_xs1, L, U, setQ)
            # v = mutation6(i, P, NP, D, F, L, U, xs1, xb)

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc, archive, success_nfs, nc1, nc2, found_success, setQ  = selectionDE(
            u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, ms, nc1, nc2, success_nfs, setQ)

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.8 * pc1 + 0.2 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0

    return P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, found_success, setQ

# --- Parameters Setup ---                                
# fitness = fitness5  ;D = 5; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin Function               
# fitness = fitness6  ;D = 5; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel Function                 
# fitness = fitness7  ;D = 5; NP = 40; L = -32* np.ones(D); U = 32* np.ones(D)         # Ackley Function                    
fitness = fitness8  ;D = 10; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank Function    
# fitness = fitness33  ;D = 5; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)       # Levy Function 
# fitness = fitness34  ;D = 5; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)       # Levy Function  N.13
# fitness = fitness45  ;D = 5; NP = 40; L = -5* np.ones(D); U = 5* np.ones(D)       # Styblinski-Tang Function
# fitness = fitness46  ;D = 5; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)       # Drop-Wave Function  N

# ----------------------------------------------------------------------------------------------------------------------                         
# fitness = fitness22 ;D = 2; NP = 40; L = -6* np.ones(D); U = 6* np.ones(D)            # Himmelblau's function G == 4
# fitness = fitness27 ;D = 2; NP = 40; L = np.array([-3,-2]); U = np.array([3,2])         # Six-Hump Camel  
# fitness = fitness28 ;D = 2; NP = 40; L = np.array([-15,-5]); U = np.array([-3,3])         # Bukin Function N. 6
# fitness = fitness29 ;D = 2; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)         # Cross-in-Tray Function   G == 4
# fitness = fitness30  ;D = 2; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)       # Drop-Wave Function   
# fitness = fitness31  ;D = 2; NP = 40; L = -512* np.ones(D); U = 512* np.ones(D)       # Eggholder Function 
# fitness = fitness32  ;D = 2; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)       # Holder Table Function
# fitness = fitness35  ;D = 2; NP = 40; L = -100* np.ones(D); U = 100* np.ones(D)       # Schaffer Function N. 2  f=0
# fitness = fitness36  ;D = 2; NP = 40; L = -100* np.ones(D); U = 100* np.ones(D)       # Schaffer Function N. 4    G == 4 ไม่เอา
# fitness = fitness37  ;D = 2; NP = 40; L = np.array([-5,-0]); U = np.array([10,15])      # Branin function    G == 3
# fitness = fitness38  ;D = 2; NP = 40; L = -4.5* np.ones(D); U = 4.5* np.ones(D)       # Beale Function
# fitness = fitness39  ;D = 4; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)       # Colville Function
# fitness = fitness40  ;D = 2; NP = 40; L = -100* np.ones(D); U = 100* np.ones(D)       # Easom Function
# fitness = fitness41  ;D = 2; NP = 40; L = -2* np.ones(D); U = 2* np.ones(D)        # Goldstein & Price Function
# fitness = fitness42  ;D = 2; NP = 40; L = -5* np.ones(D); U = 5* np.ones(D)        # Hump Functions
# fitness = fitness43  ;D = 2; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)        # Matyas Function
# fitness = fitness44  ;D = 2; NP = 40; L = -5* np.ones(D); U = 5* np.ones(D)        # Three-Hump Camel Function      ไม่เอา

# ----------------------------------------------------------------------------------------------------------------------             
# fitness = fitness11 ;D = 3; NP = 40; L = 0* np.ones(D); U = 1* np.ones(D)            # Automative steering         
# fitness = fitness12 ;D = 5 ; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)        # Economics modeling                     
# fitness = fitness13 ;D = 5; NP = 40; L = -100* np.ones(D); U = 100* np.ones(D)       # Chemical equilibrium              

VTR = 1e-10
Max_runs = 50
MAX_NFE = 40000 * D
Maxgen = round(MAX_NFE / NP)
n_archive = 20
bs = []
SA = []
success_nfs = []

header_list = ['Run', 'NF', 'fb'] + [f'xb{i+1}' for i in range(D)]

print("\t".join(header_list)) 
print("-" * 150)

for w in range(Max_runs):
    P, nf, xb, fb = initial_population(NP, D, L, U, fitness)
    setQ = []
    archive = []
    found_success = False
    mc = False
    pc1, pc2 = 0.5, 0.5
    nc1, nc2 = 0, 0

    for g in range(Maxgen): 
        if mc or nf >= MAX_NFE:
            break
        P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, found_success, setQ = DE_algorithm(
            P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success, pc1, nc1, nc2, success_nfs, setQ)

    SA.extend(archive) 
    current_row = [w+1, nf, fb] + xb.tolist()
    bs.append(current_row)

    # print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" + "\t".join([f"{v:.8f}" for v in current_row[3:]]))
    print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" )

print("-" * 150)
if success_nfs:
    print(f"Mean ± SD NF : {np.mean(success_nfs):.0f} ± {np.std(success_nfs):.0f}")
    print(f"Success Rate: {len(success_nfs)}/{Max_runs}")
else:
    print("No success.")

current_date = datetime.now().strftime("%d_%m_%y")
filename = f"Results_{current_date}.csv"
csv_header = header_list  

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header) 
    
    for row in bs:
        writer.writerow(row)
    
    writer.writerow([]) 
    if len(success_nfs) > 0:
        me_success = np.mean(success_nfs)
        sd_success = np.std(success_nfs)
        per_sd = sd_success/me_success*100
        success_rate_pct = (len(success_nfs) / Max_runs) * 100
        
        writer.writerow(['Success', f"{len(success_nfs)}/{Max_runs}", f"{success_rate_pct:.2f}%"])
        writer.writerow(['Mean NF ', f"{me_success:.2f}"])
        writer.writerow(['Std NF ', f"{sd_success:.2f}"])
        writer.writerow(['per sd ', f"{per_sd:.2f} %"])
    else:
        writer.writerow(['Status', 'Full Max runs'])