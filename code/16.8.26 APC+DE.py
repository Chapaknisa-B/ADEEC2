import numpy as np
import math
import csv
from collections import defaultdict 
from datetime import datetime

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

def affinity_propagation_clustering(P, D, max_its=100, lam=0.9):
    NP = P.shape[0]
    X = P[:, :D]
    
    S = -np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)
    
    np.fill_diagonal(S, np.median(S))
    
    R = np.zeros((NP, NP)) # Responsibilities
    A = np.zeros((NP, NP)) # Availabilities
    
    for _ in range(max_its):
        R_old = R.copy()
        A_old = A.copy()
        
        AS = A_old + S
        I = np.arange(NP)
        max_idx = np.argmax(AS, axis=1)
        max_val = AS[I, max_idx]
        
        AS[I, max_idx] = -np.inf
        max_val2 = np.max(AS, axis=1)
        
        R_new = S - max_val[:, np.newaxis]
        R_new[I, max_idx] = S[I, max_idx] - max_val2
        R = (1 - lam) * R_new + lam * R_old
        
        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, np.diag(R))
        
        A_new = np.sum(Rp, axis=0) - Rp
        dA = np.diag(A_new)
        A_new = np.minimum(A_new, 0)
        np.fill_diagonal(A_new, dA)
        
        A = (1 - lam) * A_new + lam * A_old

    # 4. ระบุ Exemplars (ศูนย์กลางคลัสเตอร์)
    E = A + R
    labels = np.argmax(E, axis=1)
    
    # จัดกลุ่ม index ของประชากรตามคลัสเตอร์
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
        
    return list(clusters.values())

def mutation1(i, cluster_indices, P, D, F, L, U):
    if len(cluster_indices) < 4:
        pool = list(range(P.shape[0]))
    else:
        pool = cluster_indices

    valid = False
    while not valid:
        idxs = np.random.choice(pool, 3, replace=False)
        if i not in idxs:
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
                mc = True

    return P, xb, fb, nf, mc, archive, found_success

def DE_algorithm_with_APC(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success, lam=0.9):
    clusters = affinity_propagation_clustering(P, D, max_its=100, lam=lam)
    
    F = 0.5
    CR = 0.9

    for cluster in clusters:
        for i in cluster:
            if mc or nf >= MAX_NFE:
                return P, xb, fb, nf, mc, archive
            
            v = mutation1(i, cluster, P, D, F, L, U)
            
            u = crossoverDE(i, P, D, CR, v)
                        
            P, xb, fb, nf, mc, archive, found_success = selectionDE(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success)

    return P, xb, fb, nf, mc, archive

# --- Parameters Setup ---                        
fitness = fitness5  ;D = 5; NP = 40; L = -5.12* np.ones(D); U = 5.12* np.ones(D)     # Rastrigin Function               
# fitness = fitness6  ;D = 5; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Schwefel Function                 
# fitness = fitness7  ;D = 5; NP = 40; L = -32* np.ones(D); U = 32* np.ones(D)         # Ackley Function                    
# fitness = fitness8  ;D = 5; NP = 40; L = -500* np.ones(D); U = 500* np.ones(D)       # Griewank Function      

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
    found_success = False
    mc = False

    for g in range(Maxgen): 
        if mc or nf >= MAX_NFE:
            break
        # เรียกใช้งาน DE ที่มีกระบวนการ APC
        P, xb, fb, nf, mc, archive = DE_algorithm_with_APC(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success)

    if found_success:
        success_nfs.append(nf)

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

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header_list) 
    
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