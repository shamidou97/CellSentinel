CREATE DATABASE IF NOT EXISTS cellsentinel;
USE cellsentinel;

CREATE TABLE cycles (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    battery_id  VARCHAR(10),
    cycle_num   INT,
    capacity    FLOAT,
    entropy_H   FLOAT,
    delta_V     FLOAT,
    SOH         FLOAT,
    fault_label TINYINT,   -- 0=Normal, 1=Warning, 2=Fault
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE raw_signals (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    cycle_id    INT,
    timestep    INT,
    voltage     FLOAT,
    current     FLOAT,
    temperature FLOAT,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);