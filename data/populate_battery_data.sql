-- ============================================================
-- CellSentinel — MySQL Database
-- Files: schema.sql + populate_battery_data.sql combined
-- Run:   mysql -u root -p < populate_battery_data.sql
-- ============================================================

-- ── Create & select database ─────────────────────────────────
CREATE DATABASE IF NOT EXISTS cellsentinel
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE cellsentinel;

-- ============================================================
-- SCHEMA
-- ============================================================

-- ── Batteries table ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS batteries (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(10)  NOT NULL UNIQUE,
    initial_capacity FLOAT       NOT NULL,
    final_capacity  FLOAT        NOT NULL,
    rated_capacity  FLOAT        NOT NULL DEFAULT 2.0,
    eol_threshold   FLOAT        NOT NULL DEFAULT 1.4,
    temperature_c   INT          NOT NULL DEFAULT 24,
    chemistry       VARCHAR(50)  NOT NULL DEFAULT 'Li-ion 18650',
    source          VARCHAR(100) NOT NULL DEFAULT 'NASA PCoE',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ── Cycles table ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cycles (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    battery_id      INT          NOT NULL,
    cycle_num       INT          NOT NULL,
    cycle_type      ENUM('charge','discharge','impedance') NOT NULL,
    capacity        FLOAT,
    soh             FLOAT,
    entropy_h       FLOAT,
    delta_v         FLOAT,
    fault_label     TINYINT      COMMENT '0=Normal 1=Warning 2=Fault',
    fault_state     VARCHAR(10)  COMMENT 'S0 S1 S2',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (battery_id) REFERENCES batteries(id),
    INDEX idx_battery_cycle (battery_id, cycle_num),
    INDEX idx_fault_label (fault_label)
);

-- ── Raw signals table ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_signals (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    cycle_id        INT          NOT NULL,
    timestep        INT          NOT NULL,
    voltage         FLOAT,
    current_a       FLOAT,
    temperature_c   FLOAT,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id),
    INDEX idx_cycle_timestep (cycle_id, timestep)
);

-- ── Model predictions table ──────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    cycle_id        INT          NOT NULL,
    model_version   VARCHAR(20)  NOT NULL DEFAULT 'v1.0',
    pred_label      TINYINT      NOT NULL,
    pred_state      VARCHAR(10)  NOT NULL,
    confidence_s0   FLOAT,
    confidence_s1   FLOAT,
    confidence_s2   FLOAT,
    is_correct      BOOLEAN,
    predicted_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);

-- ============================================================
-- SEED DATA — Battery metadata (from verified .mat inspection)
-- ============================================================

INSERT INTO batteries
    (name, initial_capacity, final_capacity, rated_capacity, eol_threshold, temperature_c)
VALUES
    ('B0005', 1.8565, 1.3251, 2.0, 1.4, 24),
    ('B0006', 2.0353, 1.1857, 2.0, 1.4, 24),
    ('B0007', 1.8911, 1.4325, 2.0, 1.4, 24),
    ('B0018', 1.8550, 1.3411, 2.0, 1.4, 24);

-- ============================================================
-- SEED DATA — Discharge cycles with SOH + fault labels
-- Generated from real capacity fade trajectory per battery
-- Fault logic: SOH>=85 → S0, SOH 70-85 → S1, SOH<70 → S2
-- ============================================================

-- Helper procedure to insert cycles for one battery
DROP PROCEDURE IF EXISTS insert_cycles;

DELIMITER $$
CREATE PROCEDURE insert_cycles(
    IN p_battery_name VARCHAR(10),
    IN p_initial      FLOAT,
    IN p_final        FLOAT,
    IN p_n_cycles     INT
)
BEGIN
    DECLARE i          INT DEFAULT 1;
    DECLARE bat_id     INT;
    DECLARE soh        FLOAT;
    DECLARE capacity   FLOAT;
    DECLARE entropy_h  FLOAT;
    DECLARE delta_v    FLOAT;
    DECLARE flabel     TINYINT;
    DECLARE fstate     VARCHAR(10);
    DECLARE t          FLOAT;

    SELECT id INTO bat_id FROM batteries WHERE name = p_battery_name;

    WHILE i <= p_n_cycles DO
        -- Simulate capacity fade (power curve)
        SET t        = (i - 1) / (p_n_cycles - 1);
        SET capacity = p_initial - (p_initial - p_final) * POW(t, 0.85);
        SET soh      = ROUND(capacity / p_initial * 100, 2);

        -- Simulate Shannon entropy (increases with degradation)
        SET entropy_h = ROUND(2.8 + t * 0.9 + (RAND() - 0.5) * 0.3, 4);

        -- Simulate voltage rebound (becomes more negative with age)
        SET delta_v = ROUND(-0.8 - t * 0.6 + (RAND() - 0.5) * 0.1, 4);

        -- Fault classification
        IF soh >= 85 THEN
            SET flabel = 0; SET fstate = 'S0';
        ELSEIF soh >= 70 THEN
            SET flabel = 1; SET fstate = 'S1';
        ELSE
            SET flabel = 2; SET fstate = 'S2';
        END IF;

        INSERT INTO cycles
            (battery_id, cycle_num, cycle_type, capacity, soh,
             entropy_h, delta_v, fault_label, fault_state)
        VALUES
            (bat_id, i, 'discharge', ROUND(capacity, 4), soh,
             entropy_h, delta_v, flabel, fstate);

        SET i = i + 1;
    END WHILE;
END$$
DELIMITER ;

-- ── Insert discharge cycles for all 4 batteries ─────────────
CALL insert_cycles('B0005', 1.8565, 1.3251, 168);
CALL insert_cycles('B0006', 2.0353, 1.1857, 168);
CALL insert_cycles('B0007', 1.8911, 1.4325, 168);
CALL insert_cycles('B0018', 1.8550, 1.3411, 132);

-- ── Cleanup procedure ────────────────────────────────────────
DROP PROCEDURE IF EXISTS insert_cycles;

-- ============================================================
-- VERIFICATION QUERIES — run after seeding to confirm data
-- ============================================================

-- Total cycles per battery
SELECT
    b.name,
    COUNT(c.id)                          AS total_cycles,
    ROUND(MIN(c.soh), 1)                 AS min_soh,
    ROUND(MAX(c.soh), 1)                 AS max_soh,
    SUM(c.fault_label = 0)               AS s0_normal,
    SUM(c.fault_label = 1)               AS s1_warning,
    SUM(c.fault_label = 2)               AS s2_fault
FROM batteries b
JOIN cycles c ON c.battery_id = b.id
GROUP BY b.name
ORDER BY b.name;

-- Fault distribution across all batteries
SELECT
    fault_state,
    COUNT(*)                             AS total,
    ROUND(COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM cycles), 1) AS pct
FROM cycles
GROUP BY fault_state
ORDER BY fault_state;

-- ============================================================
-- USEFUL VIEWS
-- ============================================================

CREATE OR REPLACE VIEW v_cycle_summary AS
SELECT
    b.name          AS battery,
    c.cycle_num,
    c.soh,
    c.capacity,
    c.entropy_h,
    c.delta_v,
    c.fault_state
FROM cycles c
JOIN batteries b ON b.id = c.battery_id
ORDER BY b.name, c.cycle_num;

CREATE OR REPLACE VIEW v_fault_counts AS
SELECT
    b.name          AS battery,
    SUM(c.fault_label = 0) AS s0_normal,
    SUM(c.fault_label = 1) AS s1_warning,
    SUM(c.fault_label = 2) AS s2_fault,
    COUNT(*)               AS total
FROM batteries b
JOIN cycles c ON c.battery_id = b.id
GROUP BY b.name;

-- ============================================================
-- DONE
-- ============================================================
-- To connect from Python:
--   import mysql.connector
--   conn = mysql.connector.connect(
--       host='localhost', user='root',
--       password='yourpassword', database='cellsentinel'
--   )
-- ============================================================
