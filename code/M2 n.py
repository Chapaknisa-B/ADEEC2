import numpy as np
import random
from datetime import datetime
import pandas as pd
import os

# Set seed เพื่อให้ผลลัพธ์คงที่เวลาทดสอบ
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1. Fitness Functions
# =============================================================================
def fitness20(x):  # Six-Hump Camel Function
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * (x1**2) + (x1**4) / 3) * (x1**2)
    term2 = x1 * x2
    term3 = (-4 + 4 * (x2**2)) * (x2**2)
    return np.abs(term1 + term2 + term3 + 1.03162845348)

def fitness21(x):    # Himmelblau's function 
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7
    return (term1**2 + term2**2)

def fitness22(x):    # Himmelblau's function MD1
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7
    term3 = 0.01 * ((x[0] - 3)**2 + (x[1] - 2)**2)
    return (term1**2 + term2**2 + term3)

def fitness23(x):   # Himmelblau's function MD2
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7
    term3 = x[2]**2 + x[3] - 11
    term4 = x[2] + x[3]**2 - 7
    return (term1**2 + term2**2 + term3**2 + term4**2)

def fitness24(x):   # Himmelblau's function MD3
    term1 = x[0]**2 + x[1] - 11
    term2 = x[0] + x[1]**2 - 7
    term3 = x[2]**2 + x[3] - 11
    term4 = x[2] + x[3]**2 - 7
    term5 = 0.01 * ((x[0] - 3)**2 + (x[1] - 2)**2)
    term6 = 0.01 * ((x[2] - 3)**2 + (x[3] - 2)**2)
    return (term1**2 + term2**2 + term3**2 + term4**2 + term5 + term6)

def fitness25(x): # MD4
    term1 = (x[0] - 2) * (x[0] - 3) * (x[1] - 1) * (x[1] + 3)
    return term1**2


# =============================================================================
# 2. Algorithm Operators
# =============================================================================
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
        idxs = np.random.choice(NP, 3, replace=False)
        if i not in idxs and len(set(idxs)) == 3:
            valid = True
            
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    xr3 = P[idxs[2], 0:D]

    F = 0.5 + 0.2 * np.random.rand()
     
    v = xr1 + F * (xr2 - xr3) 
    for k in range(D):
        if v[k] < L[k] or v[k] > U[k]:
            v[k] = L[k] + (U[k] - L[k]) * np.random.rand()
    
    return v

def mutation2(i, P, D, NP, xs1, L, U):
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 2, replace=False)
        if i not in idxs and len(set(idxs)) == 2:
            valid = True
    
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]

    F = 0.5 + 0.2 * np.random.rand()
    
    v = xs1 + F * (xr1 - xr2)
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

def nearest_idx(u, P, D):
    dists = np.linalg.norm(P[:, :D] - u, axis=1)
    closest_index = np.argmin(dists)
    return closest_index

def selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, ms, nc1, nc2):
    fitness_u = fitness(u)
    nf += 1
    mc = False
    
    target_idx = nearest_idx(u, P, D)   
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

            if fb <= VTR:
                mc = True

    return P, xb, fb, nf, mc, nc1, nc2

def select_elite(P, D, NP):
    F = P[:, D]     
    sorted_idx = np.argsort(F)
    
    best_x = P[sorted_idx[0], 0:D]
    worst_x = P[sorted_idx[-1], 0:D]
    
    dmax = np.linalg.norm(worst_x - best_x)
    
    rc = dmax / (2 * (NP ** (1/D)))
    r = max(0.05, rc)

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

def DEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2):
    
    for i in range(NP):
        if mc or nf >= MAX_NFE:
           break

        M = np.random.rand()
        if M <= 0.9 :
            v = mutation1(i, P, NP, D, L, U)
        else: 
            S = select_elite(P, D, NP)
            xs1 = P[random.choice(S), 0:D]
            v = mutation2(i, P, D, NP, xs1, L, U) 

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc_new, nc1, nc2 = selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, ms, nc1, nc2)
        
        if mc_new:
            mc = True

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.8 * pc1 + 0.2 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0
    
    return P, xb, fb, nf, mc, pc1, nc1, nc2


# =============================================================================
# 3. Main Experiment Execution
# =============================================================================

# --- ตั้งค่าพารามิเตอร์การทดลอง ---
algo_name = "M2_new"
VTR = 1e-10
Max_runs = 100

# กำหนดข้อมูลฟังก์ชันทั้งหมดที่จะรัน 
functions_config = [
    {"name": "Six-Hump_Camel", "func": fitness20, "D": 2, "NP": 40, "L": np.array([-3, -2]), "U": np.array([3, 2])},
    {"name": "Himmelblau",     "func": fitness21, "D": 2, "NP": 40, "L": -6 * np.ones(2),    "U": 6 * np.ones(2)},
    {"name": "Himmelblau_MD1", "func": fitness22, "D": 2, "NP": 40, "L": -6 * np.ones(2),    "U": 6 * np.ones(2)},
    {"name": "Himmelblau_MD2", "func": fitness23, "D": 4, "NP": 40, "L": -6 * np.ones(4),    "U": 6 * np.ones(4)},
    {"name": "Himmelblau_MD3", "func": fitness24, "D": 4, "NP": 40, "L": -6 * np.ones(4),    "U": 6 * np.ones(4)},
    {"name": "MD4",            "func": fitness25, "D": 2, "NP": 40, "L": -10 * np.ones(2),   "U": 10 * np.ones(2)}
]

max_D = 4 # มิติที่สูงที่สุดในรอบนี้

current_date = datetime.now().strftime("%d_%m_%y")
filename = f"Results_{algo_name}_{current_date}.xlsx"
summary_csv = f"Summary_{algo_name}_{current_date}.csv"

header_list = ['Algorithm', 'Function', 'Dimension', 'Run', 'NF', 'fb', 'Success'] + [f'xb{i+1}' for i in range(max_D)]
summary_list = [] 

print("=" * 65)
print(f"  เริ่มการรันทดสอบ: {algo_name}")
print("=" * 65)

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    
    # 1. Loop ทีละฟังก์ชัน
    for f_config in functions_config:
        func_name = f_config["name"]
        fitness = f_config["func"]
        D = f_config["D"]
        NP = f_config["NP"]
        L = f_config["L"]
        U = f_config["U"]
        
        print(f"\n>>> กำลังรันฟังก์ชัน: {func_name} (D={D}) <<<")
        
        sheet_data = [] 
        successful_nfs = [] # เก็บเฉพาะ NF ของ run ที่สำเร็จ
        fb_list = []        # เก็บค่า Best fitness
        success_count = 0   # นับจำนวน run ที่สำเร็จ
        
        MAX_NFE = 20000 * D
        Maxgen = round(MAX_NFE / NP)
        
        # 2. Loop ทีละ Run
        for w in range(Max_runs):
            P, nf, xb, fb = initial_population(NP, D, L, U, fitness)
            
            mc = False
            pc1 = 0.5
            nc1, nc2 = 0, 0

            for g in range(Maxgen): 
                if mc or nf >= MAX_NFE:
                    break
                
                P, xb, fb, nf, mc, pc1, nc1, nc2 = DEASC_algorithm(
                    P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2
                )

            # ตรวจสอบความสำเร็จ (ถ้า mc เป็น True แปลว่าลู่เข้าถึง VTR)
            is_success = mc 
            if is_success:
                success_count += 1
                successful_nfs.append(nf)

            if (w + 1) % 10 == 0 or (w + 1) == 1:
                status = "Success" if is_success else "Fail"
                print(f"        Run {w+1:02d}/{Max_runs} | NF: {nf:6d} | Fit: {fb:.2e} | {status}")

            fb_list.append(fb)

            xb_padded = xb.tolist() + [None] * (max_D - D)
            sheet_data.append([algo_name, func_name, D, w+1, nf, fb, is_success] + xb_padded)

        # 3. คำนวณ Mean NF และ %SD (เฉพาะที่สำเร็จ)
        if success_count > 0:
            mean_nf = np.mean(successful_nfs)
            if success_count > 1:
                std_nf = np.std(successful_nfs, ddof=1)
                pct_sd = (std_nf / mean_nf) * 100 if mean_nf != 0 else 0
            else:
                pct_sd = 0.0
            
            mean_nf_str = f"{mean_nf:.2f}"
            pct_sd_str = f"{pct_sd:.2f}%"
        else:
            mean_nf_str = "N/A"
            pct_sd_str = "N/A"
            mean_nf = None
            pct_sd = None
            
        success_rate_str = f"{success_count}/{Max_runs}"
        
        print("-" * 65)
        print(f"สรุปผล {func_name} -> Success: {success_rate_str} | Mean NF: {mean_nf_str} | %SD: {pct_sd_str}")
        print("-" * 65)
        
        summary_list.append({
            "Algorithm": algo_name,
            "Function": func_name,
            "Dimension": D,
            "Success_Rate": success_rate_str,
            "Mean_NF_SuccessOnly": mean_nf,
            "%SD_SuccessOnly": pct_sd,
            "Best_Fit_Overall_Mean": np.mean(fb_list)
        })

        df = pd.DataFrame(sheet_data, columns=header_list)
        df.to_excel(writer, sheet_name=func_name, index=False)
        
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_excel(writer, sheet_name="Summary_Stats", index=False)

print("\n" + "=" * 65)
print(f"เสร็จสิ้นการทดสอบ! ข้อมูลทั้งหมดถูกบันทึกไว้ที่ไฟล์: {filename}")
print("=" * 65)

# --- 4. Append CSV (กรณีเก็บรวบรวมข้อมูลหลายๆ อัลกอริทึม) ---
file_exists = os.path.isfile(summary_csv)
df_summary.to_csv(summary_csv, mode='a', header=not file_exists, index=False)
print(f"อัปเดตข้อมูลสรุปลงในไฟล์ CSV: {summary_csv}")