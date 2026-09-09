import numpy as np
import math
import csv
import random
from collections import defaultdict 

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

def fitness14(x):          #Combustion application
    f1 = x[1] + 2*x[5] + x[8] + 2*x[9] - 1e-5
    f2 = x[2] + x[7] - 3e-5
    f3 = x[0] + x[2] + 2*x[4] + 2*x[7] + x[8] + x[9] - 5e-5
    f4 = x[3] + 2*x[6] - 1e-5
    f5 = 0.5140437e-7*x[4] - x[0]**2
    f6 = 0.1006932e-6*x[5] - 2*x[1]**2
    f7 = 0.7816278e-15*x[6] - x[3]**2
    f8 = 0.1496236e-6*x[7] - x[0]*x[2]
    f9 = 0.6194411e-7*x[8] - x[0]*x[1]
    f10 = 0.2089296e-14*x[9] - x[0]*x[1]**2
    return (f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2 + f7**2 + f8**2 + f9**2 + f10**2) / 10


def fitness16(x):         #SINQUAD function
    n = len(x)
    f1 = (x[0] - 1)**2
    middle = [np.sin(x[i] - x[-1]) - x[0]**2 + x[i]**2 for i in range(1, n-1)]
    last = x[-1]**2 - x[0]**2
    return f1 + sum(f**2 for f in middle) + last**2

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

def mutation1(i, NP, F, L, U):
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
            v[k] = L[k] + np.random.rand() * (U[k] - L[k])
    return v

def mutation2(i, NP, F1, xs1, L, U):
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

def nearest_idx(i, P, D):
    u = P[i, :D]
    dists = np.linalg.norm(P[:, :D] - u, axis=1)
    return np.argmin(dists)

def Crowding(i, P, D):
    target_idx = nearest_idx(i, P, D)
    
    if P[i, D] < P[target_idx, D]:
        P[target_idx, :D] = P[i, :D].copy()
        P[target_idx, D] = P[i, D]
    return P

def crossoverDE(i, P, D, CR, v):
    u = P[i,:D].copy()
    I_r = np.random.randint(0, D) 
    
    for j in range(D):
        if np.random.rand() <= CR or j == I_r:
            u[j] = v[j]
    return u

def selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success):
    fitness_P = P[i,D]
    fitness_u = fitness(u)
    nf += 1
    
    mc = False
    if fitness_u < fitness_P:
        P[i,:D] = u.copy()
        P[i,D] = fitness_u

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

    return P, xb, fb, nf, mc, archive, success_nfs

def crossoverDEASC(i, P, D, v, pc1):
    ww = np.random.rand()
    if ww < pc1 :
        ms = 1
        CR = 0 + np.random.rand() * 0.1  
    else:
        ms = 2
        CR = 0.9 + np.random.rand() * 0.1

    u = P[i,:D].copy()
    I_r = np.random.randint(0, D) 
    
    for j in range(D):
        if np.random.rand() <= CR or j == I_r:
            u[j] = v[j]
    
    return u, ms

def selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, found_success):
    fitness_P = P[i,D]
    fitness_u = fitness(u)
    nf += 1
    
    mc = False
    if fitness_u < fitness_P:
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

    return P, xb, fb, nf, mc, archive, nc1, nc2 , success_nfs

def selectionCDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, found_success):
    fitness_P = P[i,D]
    fitness_u = fitness(u)
    nf += 1
    
    mc = False
    target_idx = nearest_idx(i, P, D)
    if fitness_u < P[target_idx, D]:
        P[target_idx, :D] = u.copy()
        P[target_idx, D] = fitness_u

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

    return P, xb, fb, nf, mc, archive, nc1, nc2 , success_nfs


def DE_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success):

    for i in range(NP):
        if mc or nf >= MAX_NFE:
            break 

        F = 0.5
        v = mutation1(i, NP, F, L, U)

        CR = 0.9
        u = crossoverDE(i, P, D, CR, v)
                    
        P, xb, fb, nf, mc, archive, success_nfs = selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success)

    return P, xb, fb, nf, mc, archive, success_nfs, i

def DEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, success_nfs, EE,mm):
    
    for i in range(NP):
        if mc or nf >= MAX_NFE:
           break

        F = 0.5 + 0.2 * np.random.rand()
        v = mutation1(i, NP, F, L, U)

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc, archive, nc1, nc2 , success_nfs = selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, found_success)

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.9 * pc1 + 0.1 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0
    
    return P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs,mm,i

def CDEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, success_nfs, EE,mm):
    for i in range(NP):
        if mc or nf >= MAX_NFE:
           break
           
        S = select_elite(P, D, NP, EE)

        F = 0.5 + 0.2 * np.random.rand()
        F1 = 0.5 + 0.2 * np.random.rand()

        M = np.random.rand()
        if M <= 0.9 :
            v = mutation1(i, NP, F, L, U) 

        else: 
            xs1 = P[random.choice(S), 0:D]
            v = mutation2(i, NP, F1, xs1, L, U) 

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc, archive, nc1, nc2 , success_nfs = selectionCDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, found_success)

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.8 * pc1 + 0.2 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0
    
    return P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs,mm,i

def select_elite(P, D, NP, EE):
    F = P[:, D]     
    sorted_idx = np.argsort(F)
    
    best_x = P[sorted_idx[0], 0:D]
    worst_x = P[sorted_idx[-1], 0:D]
    
    dmax = np.linalg.norm(worst_x - best_x)
    
    rc = dmax / (2 * (NP ** (1/D)))
    r = max(0.05,rc)

    S = [sorted_idx[0]]
    f_best = F[sorted_idx[0]]
    f_worst = F[sorted_idx[-1]]

    for idx in sorted_idx[1:]:
        Fi = P[idx, 0:D]
        e = (F[idx] - f_best) / (f_worst - f_best + 1e-2)
        
        far = False
        if abs(f_best - F[idx]) <= e:
            for elite_idx in S:
                P_elite = P[elite_idx, 0:D]
                distance = np.linalg.norm(P_elite - Fi)
                
                if distance <= r:
                    far = True
                    break
        
        if far:
            S.append(idx)
            
    return S

# --- Parameters Setup ---                        
fitness = fitness5  ;D = 5; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin                
# fitness = fitness6  ;D = 5; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel                  
# fitness = fitness7  ;D = 5; NP = 40; L = -32* np.ones(D); U = 32* np.ones(D)         # Ackley                     
# fitness = fitness8  ;D = 5; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank     
# ----------------------------------------------------------------------------------------------------------------------                         
# fitness = fitness22 ;D = 2; NP = 40; L = -6* np.ones(D); U = 6* np.ones(D)            #8 Himmelblau's function g=4
# fitness = fitness27 ;D = 2; NP = 40; L = np.array([-3,-2]); U = np.array([3,2])         # Six-Hump Camel  
# ----------------------------------------------------------------------------------------------------------------------             
# fitness = fitness11 ;D = 3; NP = 40; L = 0* np.ones(D); U = 1* np.ones(D)            # Automative steering         
# fitness = fitness12 ;D = 5 ; NP = 40; L = -10* np.ones(D); U = 10* np.ones(D)        # Economics modeling                     
# fitness = fitness13 ;D = 5; NP = 100; L = -100* np.ones(D); U = 100* np.ones(D)       # Chemical equilibrium       

VTR = 1e-10
# VTR = 1e-20
Max_runs = 50
MAX_NFE = 20000 * D
Maxgen = round(MAX_NFE / NP)
n_archive = 20
far = VTR
bs = []
SA = []
success_nfs = []
r_s = 10
min_species_size = 5     
m = 5

header_list = ['Run', 'NF', 'fb'] + [f'xb{i+1}' for i in range(D)]

print("\t".join(header_list)) 
print("-" * 150)

for w in range(Max_runs):
    P, nf, xb, fb = initial_population(NP, D, L, U, fitness)
    archive = []
    EE = [] 
    mm = 0
    found_success = False
    mc = False
    pc1, pc2 = 0.5, 0.5
    nc1, nc2 = 0, 0

    for g in range(Maxgen): 
        if mc or nf >= MAX_NFE:
            break
        P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs,mm,i = CDEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, success_nfs, EE, mm)
        # P, xb, fb, nf, mc, archive, success_nfs,i = DE_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success)
        # P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs,mm,i = DEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, success_nfs, EE,mm)

    SA.extend(archive) 
    current_row = [w+1, nf, fb] + xb.tolist()
    bs.append(current_row)

    print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" + "\t".join([f"{v:.8}" for v in current_row[3:]]))

# print("mm",mm)
# print("Maxgen",Maxgen)
# SA.sort(key=lambda x: x[0])
# for idx, record in enumerate(SA):
#     fitness_val = record[0]
#     solution = record[1]
#     solution_str = "[" + ", ".join(f"{x:18.15}" for x in solution) + "]"
#     print(f"{idx+1:<6}{fitness_val:<20.10e}{solution_str}")

# final_archive = []
# if len(SA) > 0:
#     final_archive.append(SA[0])
#     reference_fitness = SA[0][0]  
    
#     for current_record in SA[1:]:
#         fitness_val = current_record[0]
#         fitness_difference = abs(fitness_val - reference_fitness)
        
#         if fitness_difference <= far:
#             final_archive.append(current_record)


if success_nfs:
    print(f"Mean ± SD NF : {np.mean(success_nfs):.0f} ± {np.std(success_nfs):.0f}")
    print(f"Success Rate: {len(success_nfs)}/{Max_runs}")
else:
    print("No success.")

filename = "optimization_results.csv"
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