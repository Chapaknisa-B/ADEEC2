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

# --- Parameters Setup ---                        
fitness = fitness5  ;D = 5; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin Function   

VTR = 1e-10
Max_runs = 10
MAX_NFE = 40000 * D
Maxgen = round(MAX_NFE / NP)
n_archive = 20
far = VTR
bs = []
SA = []
success_nfs = []

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
        P, xb, fb, nf, mc, archive, success_nfs,i = DE_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success)

    SA.extend(archive) 
    current_row = [w+1, nf, fb] + xb.tolist()
    bs.append(current_row)

    print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" + "\t".join([f"{v:.8}" for v in current_row[3:]]))

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


