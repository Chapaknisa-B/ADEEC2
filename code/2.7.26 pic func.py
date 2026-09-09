import numpy as np
import math
import csv
import random
from collections import defaultdict 
import matplotlib.pyplot as plt
from matplotlib import cm

np.random.seed(42)
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

def fitness45(x):   #Styblinski-Tang Function
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

def fitness46(x):   # Drop-Wave Function  
    sum_sq = np.sum(x ** 2)

    term1 = 1 + np.cos(12 * np.sqrt(sum_sq))
    term2 = 0.5 * sum_sq + 2

    y = -(term1 / term2) + 1

    return y  


# --- Parameters Setup ---                        
# fitness = fitness5  ;D = 2; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin Function               
# fitness = fitness6  ;D = 2; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel Function                 
# fitness = fitness7  ;D = 2; NP = 40; L = -32* np.ones(D); U = 32* np.ones(D)         # Ackley Function                    
# fitness = fitness8  ;D = 2; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank Function    
# fitness = fitness33  ;D = 5; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)       # Levy Function 
# fitness = fitness34  ;D = 5; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)       # Levy Function  N.13
# fitness = fitness45  ;D = 5; NP = 40; L = -5* np.ones(D); U = 5* np.ones(D)       # Styblinski-Tang Function
fitness = fitness46  ;D = 2; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)       # Drop-Wave Function  

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
# fitness = fitness13 ;D = 5; NP = 100; L = -100* np.ones(D); U = 100* np.ones(D)       # Chemical equilibrium   

def plot_active_fitness(fit_func, D, L, U):
    x1_min, x1_max = L[0], U[0]
    
    x2_min, x2_max = L[1] if D > 1 else L[0], U[1] if D > 1 else U[0]

    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_input = np.zeros(D)
            x_input[0] = X1[i, j]
            if D > 1:
                x_input[1] = X2[i, j]
            
            Z[i, j] = fit_func(x_input)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    # ax.set_zlabel('Fitness Value')
    # ax.set_title(f'Surface Plot of {fit_func.__name__}')
    # fig.colorbar(surf, shrink=0.5, aspect=10, label='Fitness')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    plt.show()

plot_active_fitness(fitness, D, L, U)