SELECT 
    customer_id,
    gender,
    contract,
    monthly_charges,
    -- Aplicamos lógica condicional directa en el motor de Postgres
    CASE 
        WHEN monthly_charges > 100 THEN 'Cliente Premium'
        WHEN monthly_charges BETWEEN 50 AND 100 THEN 'Cliente Regular'
        ELSE 'Cliente Básico'
    END AS segmentacion_comercial
FROM customers
WHERE churn = 'No'
ORDER BY monthly_charges DESC;
