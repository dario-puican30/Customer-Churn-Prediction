-- 1. Creación de la tabla principal de clientes con restricciones profesionales
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    senior_citizen INT NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner VARCHAR(5) NOT NULL,
    dependents VARCHAR(5) NOT NULL,
    tenure INT NOT NULL CHECK (tenure >= 0),
    phone_service VARCHAR(5) NOT NULL,
    multiple_lines VARCHAR(25) NOT NULL,
    internet_service VARCHAR(25) NOT NULL,
    online_security VARCHAR(25) NOT NULL,
    online_backup VARCHAR(25) NOT NULL,
    device_protection VARCHAR(25) NOT NULL,
    tech_support VARCHAR(25) NOT NULL,
    streaming_tv VARCHAR(25) NOT NULL,
    streaming_movies VARCHAR(25) NOT NULL,
    contract VARCHAR(20) NOT NULL,
    paperless_billing VARCHAR(5) NOT NULL,
    payment_method VARCHAR(40) NOT NULL,
    monthly_charges NUMERIC(10, 2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges NUMERIC(10, 2) NOT NULL CHECK (total_charges >= 0),
    churn VARCHAR(5) NOT NULL
);

-- 2. Vista Analítica Avanzada para el Equipo de Negocio (BI / Reportes)
-- Filtra automáticamente a los clientes de alto valor económico con contratos de alto riesgo (mes a mes)
CREATE OR REPLACE VIEW view_high_risk_revenue_customers AS
SELECT 
    customer_id,
    contract,
    internet_service,
    tenure,
    monthly_charges,
    total_charges,
    churn
FROM 
    customers
WHERE 
    contract = 'Month-to-month' 
    AND monthly_charges > 70.00
ORDER BY 
    monthly_charges DESC;
