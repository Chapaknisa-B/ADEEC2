import numpy as np
import math
import csv
from datetime import datetime
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)

# --- 1. Fitness Function ----------------------------------------------------
def fitness5(x):        # Rastrigin Function
    n = len(x)
    ss = 0
    for l in range(n):
        ss += x[l]**2 - 10 * math.cos(2 * math.pi * x[l])
    return 10 * n + ss

# --- 2. Initialization ------------------------------------------------------
def initial_population(NP, D, L, U, fitness):
    # สร้างประชากรเริ่มต้น ( P มีขนาด NP x (D+1) โดยคอลัมน์สุดท้ายเก็บค่า Fitness )
    P = np.zeros((NP, D + 1))
    nf = 0
    xb, fb = None, float('inf')

    for i in range(NP):
        for j in range(D):
            P[i, j] = L[j] + (U[j] - L[j]) * np.random.rand()
        
        P[i, D] = fitness(P[i, :D])
        nf += 1
        
        if P[i, D] < fb:
            xb = P[i, :D].copy()
            fb = P[i, D]

    return P, nf, xb, fb

# --- 3. Coarse Modal Detection (NBC Clustering) -----------------------------
def nbc_clustering(P, D, phi=1.3):
    """
    จัดกลุ่มเพื่อค้นหาตัวแทนแต่ละยอดพีก (Seeds) ด้วย Nearest-Better Clustering
    """
    NP = len(P)
    X = P[:, :D]
    Y = P[:, D]
    
    dist_matrix = squareform(pdist(X))
    nb_edges = []
    
    # ลากเส้นเชื่อมไปยังจุดที่ใกล้ที่สุดและมีค่า Fitness ดีกว่า
    for i in range(NP):
        better_idxs = np.where(Y < Y[i])[0]
        if len(better_idxs) > 0:
            min_idx = better_idxs[np.argmin(dist_matrix[i, better_idxs])]
            nb_edges.append((i, min_idx, dist_matrix[i, min_idx]))
            
    if not nb_edges:
        return np.array([X[np.argmin(Y)]]), [0]

    mean_dist = np.mean([e[2] for e in nb_edges])
    threshold = phi * mean_dist
    
    # ถ้าเส้นเชื่อมยาวเกินเกณฑ์ ให้ตัดออกและนับเป็น Seed ใหม่
    seeds_x, counts = [], []
    for i in range(NP):
        is_root = True
        for edge in nb_edges:
            if edge[0] == i and edge[2] <= threshold:
                is_root = False
                break
        if is_root:
            seeds_x.append(X[i].copy())
            counts.append(0)
            
    return np.array(seeds_x), counts

# --- 4. Local Surrogate & PSO Search ----------------------------------------
def local_pso_search(seed, P, D, L, U, k_neighbors=6):
    """
    ใช้ Local RBF Model ร่วมกับ PSO ในการค้นหาคำตอบใหม่รอบๆ Seed
    """
    X = P[:, :D]
    Y = P[:, D]
    
    # 1. หาจุดใกล้เคียง K จุด
    dists = np.linalg.norm(X - seed, axis=1)
    k_idx = np.argsort(dists)[:k_neighbors]
    r = dists[k_idx[-1]] if dists[k_idx[-1]] > 1e-8 else 0.1
    d = math.sqrt(D) * r
    
    # 2. สร้าง Local RBF Surrogate Model จากจุดในรัศมี
    train_idx = np.where(dists <= d)[0]
    if len(train_idx) < k_neighbors:
        train_idx = k_idx
        
    try:
        rbf_model = RBFInterpolator(X[train_idx], Y[train_idx], kernel='thin_plate_spline')
    except:
        # หากเกิดปัญหา Matrix Singular ให้คืนค่าสุ่มรอบๆ Seed แทน
        return np.clip(seed + 0.01 * np.random.randn(D), L, U)
        
    # 3. รัน Simple PSO เพื่อค้นหาคำตอบบน RBF Model
    lb_local = np.maximum(seed - r, L)
    ub_local = np.minimum(seed + r, U)
    
    n_particles = 15
    particles = np.random.uniform(lb_local, ub_local, (n_particles, D))
    velocities = np.zeros((n_particles, D))
    
    pbest = particles.copy()
    pbest_fit = rbf_model(pbest)
    gbest = pbest[np.argmin(pbest_fit)].copy()
    
    for _ in range(20):
        r1, r2 = np.random.rand(n_particles, D), np.random.rand(n_particles, D)
        velocities = 0.5 * velocities + 1.5 * r1 * (pbest - particles) + 1.5 * r2 * (gbest - particles)
        particles = np.clip(particles + velocities, lb_local, ub_local)
        
        fits = rbf_model(particles)
        improved = fits < pbest_fit
        pbest[improved] = particles[improved]
        pbest_fit[improved] = fits[improved]
        gbest = pbest[np.argmin(pbest_fit)].copy()
        
    return gbest

# --- 5. Selection & Algorithm Update ----------------------------------------
def selectionSREA(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, success_nfs):
    fitness_u = fitness(u)
    nf += 1
    
    mc = False
    # ถ้านำ candidate (u) มาEvaluateจริงแล้วดีกว่าค่าเดิม ให้บันทึกลง P
    if fitness_u < P[i, D]:
        P[i, :D] = u.copy()
        P[i, D] = fitness_u

        if fitness_u <= fb:
            xb = P[i, :D].copy()
            fb = fitness_u
            archive.append([float(fb), xb.tolist()])
            if len(archive) > n_archive:
                archive.sort(key=lambda x: x[0])
                archive = archive[:n_archive]

            if fb <= VTR and not found_success:
                found_success = True
                success_nfs.append(nf)
                mc = True

    return P, xb, fb, nf, mc, archive, success_nfs

def SREA_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success, success_nfs, max_st=10):
    # 1. จัดกลุ่มเพื่อหา Seeds
    seeds_x, seeds_count = nbc_clustering(P, D)
    
    for idx, seed in enumerate(seeds_x):
        if mc or nf >= MAX_NFE:
            break
            
        # 2. ค้นหา Candidate Solution ด้วย Local RBF + PSO
        u = local_pso_search(seed, P, D, L, U)
        
        # 3. อัปเดตและตรวจจับความสำเร็จ (เหมือน selectionDE)
        # เราสุ่มอัปเดตใส่ P ในช่องประชากรที่ใกล้เคียงเพื่อรักษารูปแบบ Matrix P
        i = np.random.randint(0, NP)
        P, xb, fb, nf, mc, archive, success_nfs = selectionSREA(
            u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, found_success, success_nfs
        )
        
        # 4. กลไกรีสตาร์ตแบบง่ายเมื่อติดขัด (Stagnation Restart)
        if seeds_count[idx] >= max_st:
            seeds_count[idx] = 0
            P[i, :D] = L + (U - L) * np.random.rand(D)
            P[i, D] = fitness(P[i, :D])
            nf += 1

    return P, xb, fb, nf, mc, archive, success_nfs

# --- 6. Parameters Setup ----------------------------------------------------
fitness = fitness5
D = 10
NP = 40
L = -5.12 * np.ones(D)
U = 5.12 * np.ones(D)

VTR = 1e-10
Max_runs = 10
MAX_NFE = 4000 * D     # ลด Max NFE ลงเพราะ Surrogate ช่วยประหยัดเวลา
Maxgen = round(MAX_NFE / NP)
n_archive = 20
bs = []
SA = []
success_nfs = []

header_list = ['Run', 'NF', 'fb'] + [f'xb{i+1}' for i in range(D)]

print("\t".join(header_list))
print("-" * 150)

# --- 7. Main Loop (โครงสร้างเดิมของคุณทั้งหมด) -----------------------------
for w in range(Max_runs):
    P, nf, xb, fb = initial_population(NP, D, L, U, fitness)
    archive = []
    found_success = False
    mc = False

    for g in range(Maxgen):
        if mc or nf >= MAX_NFE:
            break
            
        P, xb, fb, nf, mc, archive, success_nfs = SREA_algorithm(
            P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, archive, found_success, success_nfs
        )

    SA.extend(archive)
    current_row = [w + 1, nf, fb] + xb.tolist()
    bs.append(current_row)

    print(f"{current_row[0]}\t{current_row[1]}\t{current_row[2]:.2e}\t" + "\t".join([f"{v:.8}" for v in current_row[3:]]))

print("-" * 150)
if success_nfs:
    print(f"Mean ± SD NF : {np.mean(success_nfs):.0f} ± {np.std(success_nfs):.0f}")
    print(f"Success Rate: {len(success_nfs)}/{Max_runs}")
else:
    print("No success.")

# --- 8. Save CSV ------------------------------------------------------------
current_date = datetime.now().strftime("%d_%m_%y")
filename = f"Results_S+PSO_{current_date}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header_list)
    
    for row in bs:
        writer.writerow(row)
    
    writer.writerow([])
    if len(success_nfs) > 0:
        me_success = np.mean(success_nfs)
        sd_success = np.std(success_nfs)
        per_sd = sd_success / me_success * 100
        success_rate_pct = (len(success_nfs) / Max_runs) * 100
        
        writer.writerow(['Success', f"{len(success_nfs)}/{Max_runs}", f"{success_rate_pct:.2f}%"])
        writer.writerow(['Mean NF ', f"{me_success:.2f}"])
        writer.writerow(['Std NF ', f"{sd_success:.2f}"])
        writer.writerow(['per sd ', f"{per_sd:.2f} %"])
    else:
        writer.writerow(['Status', 'Full Max runs'])