import numpy as np
import math
import csv
import random
from collections import defaultdict 
from datetime import datetime

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

# --- New RGS-Mutation (Reverse Gravitational Sink Mutation) ------------------
def mutationRGS(i, P, D, xb, alpha, beta, epsilon, F, L, U):
    # ข้อมูลของประชากรปัจจุบัน
    xi = P[i, :D]
    fitness_xi = P[i, D]

    f_sink = np.zeros(D)
    f_push = np.zeros(D)

    for j in range(len(P)):
        if j == i: continue 

        xj = P[j, :D]
        fitness_xj = P[j, D]
        
        direction_vector = xj - xi
        distance_squared = np.sum(direction_vector**2)
        denominator = distance_squared + epsilon

        # valid = False
        # while not valid:
        #     idxs = np.random.choice(NP, 1, replace=False)
        #     if i not in idxs and len(set(idxs)) == 1:
        #         valid = True
        # xr1 = P[idxs[0], 0:D]

        if fitness_xj < fitness_xi:
            sink = (fitness_xi - fitness_xj) / denominator
            f_sink += sink * direction_vector

        elif fitness_xj > fitness_xi:
            push = np.exp(-(fitness_xj - fitness_xi)) / denominator
            f_push += push * direction_vector

    v = xi + (alpha * f_sink) - (beta * f_push) + (F * (xb - xi))
    
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

def selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, success_nfs):
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

            if fb <= VTR and not found_success:
                found_success = True 
                success_nfs.append(nf)
                mc = True

    return P, xb, fb, nf, mc, archive, success_nfs, found_success

def DE_algorithm_RGS(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success, success_nfs, alpha_rgs, beta_rgs, epsilon_rgs):

    F_mutation = 0
    CR_crossover = 0.5

    for i in range(NP):
        if mc or nf >= MAX_NFE:
            break 

        v = mutationRGS(i, P, D, xb, alpha_rgs, beta_rgs, epsilon_rgs, F_mutation, L, U)

        u = crossoverDE(i, P, D, CR_crossover, v)
                    
        P, xb, fb, nf, mc, archive, success_nfs, found_success = selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, success_nfs)

    return P, xb, fb, nf, mc, archive, success_nfs, i, found_success

# --- Main Parameters Setup --------------------------------------------------                        
fitness = fitness5  ;D = 5; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin Function               
# fitness = fitness6  ;D = 5; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel Function                 
# fitness = fitness7  ;D = 30; NP = 40; L = -32* np.ones(D); U = 32* np.ones(D)         # Ackley Function                    
# fitness = fitness8  ;D = 10; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank Function  


alpha_rgs = 0.5
beta_rgs = 0.5
epsilon_rgs = 1e-2 

VTR = 1e-10
Max_runs = 5
MAX_NFE = 10000 * D
Maxgen = round(MAX_NFE / NP)
n_archive = 20
far = VTR
bs = []
SA = []
all_success_nfs = [] 

header_list = ['Run', 'NF', 'fb'] + [f'xb{i+1}' for i in range(D)]

print("\t".join(header_list)) 
print("-" * 150)

for w in range(Max_runs):
    P, nf, xb, fb = initial_population(NP, D, L, U, fitness)
    archive = []
    found_success = False
    mc = False
    success_nfs_run = [] 

    for g in range(Maxgen): 
        if mc or nf >= MAX_NFE:
            break
        
        P, xb, fb, nf, mc, archive, success_nfs_run, i, found_success = DE_algorithm_RGS(
            P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, 
            found_success, success_nfs_run, alpha_rgs, beta_rgs, epsilon_rgs
        )

    SA.extend(archive) 
    all_success_nfs.extend(success_nfs_run)
    current_row = [w+1, nf, fb] + xb.tolist()
    bs.append(current_row)

    print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" + "\t".join([f"{v:.8}" for v in current_row[3:]]))

if all_success_nfs:
    print(f"Mean ± SD NF : {np.mean(all_success_nfs):.0f} ± {np.std(all_success_nfs):.0f}")
    print(f"Success Rate: {len(all_success_nfs)}/{Max_runs}")
else:
    print("No success.")

current_date = datetime.now().strftime("%d_%m_%y")
filename = f"Results_RGS_{current_date}.csv"
csv_header = header_list  

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header) 
    
    for row in bs:
        writer.writerow(row)
    
    writer.writerow([]) 
    if len(all_success_nfs) > 0:
        me_success = np.mean(all_success_nfs)
        sd_success = np.std(all_success_nfs)
        per_sd = sd_success/me_success*100
        success_rate_pct = (len(all_success_nfs) / Max_runs) * 100
        
        writer.writerow(['Success', f"{len(all_success_nfs)}/{Max_runs}", f"{success_rate_pct:.2f}%"])
        writer.writerow(['Mean NF ', f"{me_success:.2f}"])
        writer.writerow(['Std NF ', f"{sd_success:.2f}"])
        writer.writerow(['per sd ', f"{per_sd:.2f} %"])
        
    else:
        writer.writerow(['Status', 'Full Max runs'])