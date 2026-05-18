import os
import mysql.connector
from sqlalchemy import create_engine

WSL_IP   = "172.25.8.43"
USER     = "celluser"
PASSWORD = "cellsentinel123"
DATABASE = "cellsentinel"
SOCKET   = "/var/run/mysqld/mysqld.sock"

def _is_wsl():
    if os.name != "posix":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except:
        return False

def get_conn():
    if _is_wsl():
        return mysql.connector.connect(host="localhost", unix_socket=SOCKET, user=USER, password=PASSWORD, database=DATABASE)
    else:
        return mysql.connector.connect(host=WSL_IP, port=3306, user=USER, password=PASSWORD, database=DATABASE)

def get_engine():
    if _is_wsl():
        url = f"mysql+mysqlconnector://{USER}:{PASSWORD}@localhost/{DATABASE}?unix_socket={SOCKET}"
    else:
        url = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{WSL_IP}:3306/{DATABASE}"
    return create_engine(url)

if __name__ == "__main__":
    print("Testing CellSentinel database connection...")
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cycles")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM batteries")
        bats = cursor.fetchone()[0]
        conn.close()
        print(f"  Connected successfully")
        print(f"  Batteries : {bats}")
        print(f"  Cycles    : {count}")
        print(f"  Platform  : {chr(39)}WSL/Linux{chr(39) if _is_wsl() else chr(39)}Windows{chr(39)}")
    except Exception as e:
        print(f"  Connection failed: {e}")
