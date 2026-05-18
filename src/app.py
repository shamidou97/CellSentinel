import mysql.connector
from db import get_engine

engine = get_engine()

def get_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        database='cellsentinel',
        unix_socket='/var/run/mysqld/mysqld.sock'  # WSL uses socket auth
    )

@app.route('/api/battery/<name>')
def battery(name):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT c.cycle_num, c.soh, c.capacity,
               c.entropy_h, c.delta_v, c.fault_label, c.fault_state
        FROM cycles c
        JOIN batteries b ON b.id = c.battery_id
        WHERE b.name = %s
        ORDER BY c.cycle_num
    """, (name,))
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)