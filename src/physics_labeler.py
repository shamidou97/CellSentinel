"""
CellSentinel — Physics Labeler
Reads cycles from MySQL, applies entropy-based fault labeling,
and writes labels back to the database.

Run: python src/physics_labeler.py
"""

import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine, text
from db import get_conn, get_engine

engine = get_engine()
conn   = get_conn()

# ── Database connections ──────────────────────────────────────
SOCKET   = '/var/run/mysqld/mysqld.sock'
USER     = 'celluser'
PASSWORD = 'cellsentinel123'
DATABASE = 'cellsentinel'

def get_engine():
    """SQLAlchemy engine — used for pd.read_sql (no warnings)"""
    return create_engine(
        f'mysql+mysqlconnector://{USER}:{PASSWORD}@localhost/{DATABASE}'
        f'?unix_socket={SOCKET}'
    )

d

# ── Fault labeling ────────────────────────────────────────────
def label_fault_states(df):
    """
    Applies physics-informed fault classification per cycle.

    S0 = Normal  : SOH >= 85% AND no entropy spike
    S1 = Warning : SOH 70-85% OR entropy spike > 2 std
    S2 = Fault   : SOH < 70%  OR capacity < 1.4 Ah (EOL threshold)

    Args:
        df: DataFrame with columns [battery_id, capacity, soh,
                                    entropy_h, delta_v]
    Returns:
        df with fault_label (0/1/2) and fault_state (S0/S1/S2) columns
    """
    df = df.copy()

    # ── Per-battery entropy statistics ────────────────────────
    # Compute mean/std per battery so spikes are relative to
    # each battery's own baseline (not global average)
    df['H_mean']  = df.groupby('battery_id')['entropy_h'].transform('mean')
    df['H_std']   = df.groupby('battery_id')['entropy_h'].transform('std')
    df['H_spike'] = df['entropy_h'] > df['H_mean'] + 2 * df['H_std']

    # ── Classification ────────────────────────────────────────
    def classify(row):
        # S2 Fault — hard failure conditions
        if row['soh'] < 70.0 or row['capacity'] < 1.4:
            return 2, 'S2'
        # S1 Warning — degrading or entropy anomaly detected
        elif row['soh'] < 85.0 or row['H_spike']:
            return 1, 'S1'
        # S0 Normal
        else:
            return 0, 'S0'

    results = df.apply(classify, axis=1, result_type='expand')
    df['fault_label'] = results[0].astype(int)
    df['fault_state'] = results[1]

    # ── Drop helper columns ───────────────────────────────────
    df.drop(columns=['H_mean', 'H_std', 'H_spike'], inplace=True)

    return df

# ── Write labels back to MySQL ────────────────────────────────
def update_labels(df, conn):
    """Batch UPDATE fault_label and fault_state for all cycles."""
    cursor = conn.cursor()

    rows = [
        (int(row['fault_label']),
         str(row['fault_state']),
         int(row['id']))
        for _, row in df.iterrows()
    ]

    cursor.executemany("""
        UPDATE cycles
        SET fault_label = %s,
            fault_state = %s
        WHERE id = %s
    """, rows)

    updated = cursor.rowcount
    conn.commit()
    cursor.close()
    return updated

# ── Summary printout ──────────────────────────────────────────
def print_summary(df):
    print("\n── Fault label distribution ─────────────────────")
    counts = df['fault_label'].value_counts().sort_index()
    labels = {0: 'S0 Normal', 1: 'S1 Warning', 2: 'S2 Fault'}
    total  = len(df)
    for label, count in counts.items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {labels[label]:<12}: {count:>4} cycles ({pct:5.1f}%)  {bar}")
    print(f"  {'Total':<12}: {total:>4} cycles")

    print("\n── Per-battery breakdown ────────────────────────")
    summary = df.groupby(
        ['battery_name', 'fault_state']
    ).size().unstack(fill_value=0)
    for col in ['S0', 'S1', 'S2']:
        if col not in summary.columns:
            summary[col] = 0
    print(summary[['S0', 'S1', 'S2']].to_string())

    print("\n── Sample cycles (first 10) ─────────────────────")
    cols = ['battery_name', 'cycle_num', 'soh',
            'entropy_h', 'delta_v', 'fault_label', 'fault_state']
    print(df[cols].head(10).to_string(index=False))

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🔋 CellSentinel Physics Labeler")

    # ── Load via SQLAlchemy (no pandas warning) ───────────────
    print("  Loading cycles from MySQL...")
    engine = get_engine()
    with engine.connect() as con:
        df = pd.read_sql(text("""
            SELECT
                c.id,
                c.battery_id,
                b.name          AS battery_name,
                c.cycle_num,
                c.capacity,
                c.soh,
                c.entropy_h,
                c.delta_v,
                c.fault_label,
                c.fault_state
            FROM cycles c
            JOIN batteries b ON b.id = c.battery_id
            ORDER BY c.battery_id, c.cycle_num
        """), con=con)

    print(f"  Loaded {len(df)} cycles from "
          f"{df['battery_name'].nunique()} batteries")

    # ── Apply physics-informed labeling ───────────────────────
    print("  Applying fault classification...")
    df = label_fault_states(df)

    # ── Print summary ─────────────────────────────────────────
    print_summary(df)

    # ── Write back via raw connector (faster for UPDATE) ──────
    print("\n  Writing labels back to MySQL...")
    conn = get_conn()
    n    = update_labels(df, conn)
    conn.close()
    print(f"  ✅ Updated {n} cycle records")

    print("\n✅ Physics labeling complete\n")
