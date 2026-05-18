import scipy.io
import numpy as np
import mysql.connector
import pandas as pd
from scipy.stats import entropy
from db import get_conn

conn = get_conn()

def load_mat(path):
    mat = scipy.io.loadmat(path, simplify_cells=True)
    return mat

def extract_cycles(mat):
    """Extract per-cycle features from NASA .mat structure."""
    cycles = []
    for key in mat:
        if key.startswith('B'):          # battery keys e.g. B0005
            battery = mat[key]
            for i, cycle in enumerate(battery['cycle']):
                if cycle['type'] == 'discharge':
                    V = np.array(cycle['data']['Voltage_measured'])
                    I = np.array(cycle['data']['Current_measured'])
                    T = np.array(cycle['data']['Temperature_measured'])
                    Q = float(cycle['data']['Capacity'])

                    # ── Physics features ──────────────────────────
                    # Shannon entropy of voltage distribution
                    V_hist, _ = np.histogram(V, bins=50, density=True)
                    V_hist += 1e-10
                    H = entropy(V_hist)

                    # Voltage rebound
                    delta_V = float(V[-1] - V[0]) if len(V) > 1 else 0.0

                    cycles.append({
                        'battery_id': key,
                        'cycle_num':  i,
                        'capacity':   Q,
                        'entropy_H':  H,
                        'delta_V':    delta_V,
                        'fault_label': None   # set by physics_labeler.py
                    })
    return pd.DataFrame(cycles)

def push_to_mysql(df, host='localhost', user='root',
                  password='', database='cellsentinel'):
    conn = mysql.connector.connect(
        host=host, user=user,
        password=password, database=database
    )
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO cycles
              (battery_id, cycle_num, capacity, entropy_H, delta_V, fault_label)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, tuple(row))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ {len(df)} cycles inserted into MySQL")

if __name__ == '__main__':
    mat  = load_mat('data/B0005.mat')
    df   = extract_cycles(mat)
    push_to_mysql(df)
    print(df.head())