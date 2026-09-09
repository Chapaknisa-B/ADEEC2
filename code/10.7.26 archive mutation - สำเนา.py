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

def fitness8(x):        # Griewank
    n = len(x)
    ss1 = 0
    ss2 = 1
    for l in range(n):
        ss1 += x[l]**2 / 4000
        ss2 *= math.cos(x[l] / math.sqrt(l+1))
    return ss1 - ss2 + 1

def fitness22(x):    #8 Himmelblau's function
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7
    return (term1**2 + term2**2)

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
            
    idx_base = idxs[0] 
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    xr3 = P[idxs[2], 0:D]
    
    v = xr1 + F * (xr2 - xr3) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
    
    return v, xr1, idx_base 

def mutation2(i, P, NP, D, F, idx_xs1, L, U, setQ): 
    xs1 = P[idx_xs1, 0:D]
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 1, replace=False)
        idx_r1 = idxs[0] 
        if i != idx_r1 and idx_xs1 != idx_r1:
            valid = True
    
    xr1 = P[idx_r1, 0:D]

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
            
    return v, xs1, idx_xs1, setQ 


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
    
# def selectionDE(u, i, xr1, idx_xr1, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, ms, nc1, nc2, success_nfs, setQ):
#     dist_to_target = np.linalg.norm(u - P[i, :D])
#     dist_to_xr1 = np.linalg.norm(u - xr1)
    
#     if dist_to_target <= dist_to_xr1:
#         DC_idx = i  
#     else:
#         DC_idx = idx_xr1  
        
#     fitness_DC = P[DC_idx, D]
#     fitness_u = fitness(u)
#     nf += 1
    
#     mc = False
#     if fitness_u < fitness_DC:
#         setQ.append(P[DC_idx, :D].copy())
#         if len(setQ) > NP:
#             setQ.pop(0)

#         P[DC_idx, :D] = u.copy()
#         P[DC_idx, D] = fitness_u

#         if ms == 1:
#             nc1 += 1
#         else:
#             nc2 += 1

#         if fitness_u <= fb:
#             xb = P[DC_idx, :D].copy()
#             fb = fitness_u
#             archive.append([float(fb), xb.tolist()])
#             if len(archive) > n_archive:
#                 archive.sort(key=lambda x: x[0])
#                 archive = archive[:n_archive]

#             if fb <= VTR and not found_success:
#                 found_success = True 
#                 success_nfs.append(nf)
#                 mc = True

#     return P, xb, fb, nf, mc, archive, success_nfs, nc1, nc2, found_success, setQ

def selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, ms, nc1, nc2, success_nfs):
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
        if np.random.rand() < 0.8:
            F = 0.5 + 0.2 * np.random.rand()
        else:
            F = 0.1 + 0.2 * np.random.rand()

        M = np.random.rand()
        if M <= 0.9: 
            v, v_base, idx_base = mutation1(i, P, NP, D, F, L, U)
        else:
            S = select_elite(P, D, NP)
            idx_xs1 = random.choice(S) 
            v, v_base, idx_base, setQ = mutation2(i, P, NP, D, F, idx_xs1, L, U, setQ) 

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc, archive, success_nfs, nc1, nc2, found_success, setQ = selectionDE(
            u, i, v_base, idx_base, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, ms, nc1, nc2, success_nfs, setQ)

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.8 * pc1 + 0.2 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0

    return P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, found_success, setQ

# --- Parameters Setup ---                                
# fitness = fitness5 ; D = 10; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)    
# fitness = fitness6  ;D = 10; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel Function   
# fitness = fitness8  ;D = 10; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank Function   
fitness = fitness22 ;D = 2; NP = 40; L = -6* np.ones(D); U = 6* np.ones(D)            # Himmelblau's function G == 4             

VTR = 1e-10
Max_runs = 10
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
    archive = []
    setQ = []
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