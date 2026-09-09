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

def mutation1(i, P, NP, D, L, U):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 5, replace=False)
        if i not in idxs and len(set(idxs)) == 5:
            valid = True
            
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    xr3 = P[idxs[2], 0:D]
    xr4 = P[idxs[3], 0:D]
    xr5 = P[idxs[4], 0:D]

    F = 0.5 + 0.2 * np.random.rand()
     
    v = xr1 + F * (xr2 - xr3) + F * (xr4 - xr5) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
    
    return v

def mutation2(i, NP, xs1, L, U):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 2, replace=False)
        if i not in idxs and len(set(idxs)) == 2:
            valid = True
    
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]

    F = 0.5 + 0.2 * np.random.rand()
    
    v = xi + F * (xb - xi) + F * (xr1 - xr2)
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

def selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, success_nfs):
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
                success_nfs.append(nf)
                mc = True

    return P, xb, fb, nf, mc, archive, nc1, nc2, success_nfs 

def select_elite(P, D, NP):
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

def DEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, success_nfs, EE, mm):
    
    for i in range(NP):
        if mc or nf >= MAX_NFE:
           break

        M = np.random.rand()
        if M <= 0.8 :
            v = mutation1(i, P, NP, D, L, U)
        
        else: 
            S = select_elite(P, D, NP)
            xs1 = P[random.choice(S), 0:D]
            v = mutation2(i, NP, xs1, L, U) 

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc_new, archive, nc1, nc2, success_nfs = selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, success_nfs)
        if mc_new:
            mc = True

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.9 * pc1 + 0.1 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0
    
    return P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, mm, i

# --- Parameters Setup ---                                
# fitness = fitness5  ;D = 30; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin Function               
# fitness = fitness6  ;D = 30; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel Function                 
# fitness = fitness7  ;D = 30; NP = 40; L = -32* np.ones(D); U = 32* np.ones(D)         # Ackley Function                    
fitness = fitness8  ;D = 30; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank Function    
             

VTR = 1e-10
Max_runs = 30
MAX_NFE = 40000 * D
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
    mc = False
    pc1, pc2 = 0.5, 0.5
    nc1, nc2 = 0, 0

    for g in range(Maxgen): 
        if mc or nf >= MAX_NFE:
            break
        
        P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, mm, i = DEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, success_nfs, EE, mm)

    SA.extend(archive) 
    current_row = [w+1, nf, fb] + xb.tolist()
    bs.append(current_row)

    print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" + "\t".join([f"{v:.8f}" for v in current_row[3:]]))

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