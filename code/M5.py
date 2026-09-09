import numpy as np
import math
import random
from collections import defaultdict 
from datetime import datetime
import pandas as pd
import os

# Set seed เพื่อให้ผลลัพธ์คงที่เวลาทดสอบ
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1. Fitness Functions
# =============================================================================
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

def mutation2(i, P, D, NP, xs1, L, U): # *แก้ไขเพิ่มรับค่า P, D ป้องกัน Error
    valid = False
    while not valid:
        idxs = np.random.choice(NP, 4, replace=False)
        if i not in idxs and len(set(idxs)) == 4:
            valid = True
    
    xi = P[i, 0:D]
    xr1 = P[idxs[0], 0:D]
    xr2 = P[idxs[1], 0:D]
    xr3 = P[idxs[2], 0:D]
    xr4 = P[idxs[3], 0:D]

    F = 0.5 + 0.2 * np.random.rand()
    
    v = xs1 + F * (xr1 - xr2) + F * (xr3 - xr4)
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
                if nf not in success_nfs: # ป้องกันบวกซ้ำ
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

def DEASC_algorithm(P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, n_archive, success_nfs, EE, mm): # *เพิ่ม n_archive
    
    for i in range(NP):
        if mc or nf >= MAX_NFE:
           break

        M = np.random.rand()
        if M <= 0.9 :
            v = mutation1(i, P, NP, D, L, U)
        
        else: 
            S = select_elite(P, D, NP)
            xs1 = P[random.choice(S), 0:D]
            v = mutation2(i, P, D, NP, xs1, L, U) # *ส่ง P, D เข้าไปตามที่แก้

        u, ms = crossoverDEASC(i, P, D, v, pc1)
                    
        P, xb, fb, nf, mc_new, archive, nc1, nc2, success_nfs = selectionDEASC(u, i, P, D, fb, xb, nf, VTR, fitness, archive, n_archive, ms, nc1, nc2, success_nfs)
        if mc_new:
            mc = True

        if (nc1 + nc2) >= 100:
            nc1 += 10
            nc2 += 10
            pc1 = 0.8 * pc1 + 0.2 * (nc1 / (nc1 + nc2))
            nc1, nc2 = 0, 0
    
    return P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, mm, i


# =============================================================================
# 3. Main Experiment Execution
# =============================================================================

# --- ตั้งค่าพารามิเตอร์การทดลอง ---
algo_name = "M5"
NP = 40
VTR = 1e-10
Max_runs = 50
dimensions = [5, 10, 30, 50]
max_D = max(dimensions) # คอลัมน์คำตอบสูงสุด (50 คอลัมน์)

# กำหนดข้อมูลฟังก์ชันทั้งหมดที่จะรัน (Bound เปลี่ยนตามแต่ละ Function)
functions_config = [
    {"name": "Rastrigin", "func": fitness5, "L": -5.12, "U": 5.12},
    {"name": "Schwefel",  "func": fitness6, "L": -500.0, "U": 500.0},
    {"name": "Ackley",    "func": fitness7, "L": -32.0, "U": 32.0},
    {"name": "Griewank",  "func": fitness8, "L": -500.0, "U": 500.0}
]

# สร้างชื่อไฟล์ Excel (ใส่วันที่)
current_date = datetime.now().strftime("%d_%m_%y")
filename = f"Results_{algo_name}_{current_date}.xlsx"

# สร้าง Header ให้รองรับ D สูงสุด (เช่น xb1 ไปจนถึง xb50)
header_list = ['Algorithm', 'Function', 'Dimension', 'Run', 'NF', 'fb'] + [f'xb{i+1}' for i in range(max_D)]

print("=" * 60)
print(f"  เริ่มการรันทดสอบ: {algo_name}")
print("=" * 60)

# สร้างไฟล์ Excel
with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    
    # 1. Loop ทีละฟังก์ชัน
    for f_config in functions_config:
        func_name = f_config["name"]
        fitness = f_config["func"]
        print(f"\n>>> กำลังรันฟังก์ชัน: {func_name} <<<")
        
        sheet_data = [] # สำหรับเก็บข้อมูลแต่ละแถวของฟังก์ชันนี้
        
        # 2. Loop ทีละ Dimension
        for D in dimensions:
            print(f"    --> Dimension: {D}")
            
            L = f_config["L"] * np.ones(D)
            U = f_config["U"] * np.ones(D)
            MAX_NFE = 20000 * D
            Maxgen = round(MAX_NFE / NP)
            
            # 3. Loop ทีละ Run
            for w in range(Max_runs):
                P, nf, xb, fb = initial_population(NP, D, L, U, fitness)
                
                archive = []
                success_nfs = []
                EE = [] 
                mm = 0
                mc = False
                pc1 = 0.5
                nc1, nc2 = 0, 0
                n_archive = 20

                # Evolution Loop (รันแต่ละ Generation)
                for g in range(Maxgen): 
                    if mc or nf >= MAX_NFE:
                        break
                    
                    P, xb, fb, nf, mc, pc1, nc1, nc2, archive, success_nfs, mm, i = DEASC_algorithm(
                        P, D, NP, L, U, fitness, xb, fb, nf, mc, VTR, MAX_NFE, pc1, nc1, nc2, archive, n_archive, success_nfs, EE, mm
                    )

                # ปริ้นอัพเดทแค่ตอนเริ่ม ตอนกลาง และตอนจบ ของแต่ละมิติ จะได้ไม่รกจอเกินไป
                if (w + 1) % 10 == 0 or (w + 1) == 1:
                     print(f"        Run {w+1:02d}/{Max_runs} | NF: {nf:6d} | Best Fitness: {fb:.2e}")

                # สำคัญ: เติมค่าว่าง (None) ให้ xb มีขนาดเท่ากับ max_D (50) จะได้พล็อตตาราง Excel สวยๆ
                xb_padded = xb.tolist() + [None] * (max_D - D)
                
                # นำข้อมูลเข้าสู่ List สำหรับบันทึก
                current_row = [algo_name, func_name, D, w+1, nf, fb] + xb_padded
                sheet_data.append(current_row)

        # จบ 1 ฟังก์ชัน นำข้อมูลบันทึกลง Sheet ใน Excel
        df = pd.DataFrame(sheet_data, columns=header_list)
        df.to_excel(writer, sheet_name=func_name, index=False)
        print(f"    [บันทึกข้อมูลฟังก์ชัน '{func_name}' ลง Sheet สำเร็จ]")

print("\n" + "=" * 60)
print(f" เสร็จสิ้นการทดสอบ! ข้อมูลทั้งหมดถูกบันทึกไว้ที่ไฟล์: {filename}")
print("=" * 60)