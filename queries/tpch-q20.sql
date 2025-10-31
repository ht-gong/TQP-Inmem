-- SELECT 
--     s_name, 
--     s_address
-- FROM 
--     supplier, nation
-- WHERE 
--     s_suppkey IN (
--         SELECT 
--             ps_suppkey
--         FROM 
--             partsupp
--         WHERE 
--             ps_partkey IN (
--                 SELECT 
--                     p_partkey
--                 FROM 
--                     part
--                 WHERE 
--                     p_name LIKE 'forest%'
--                     AND ps_availqty > (
--                         SELECT 
--                             0.5 * SUM(l_quantity)
--                         FROM 
--                             lineitem
--                         WHERE 
--                             l_partkey = ps_partkey 
--                             AND l_suppkey = ps_suppkey 
--                             AND l_shipdate >= '1994-01-01'
--                             AND l_shipdate < DATE_ADD('1994-01-01', 365)
--                     )
--             )
--             AND s_nationkey = n_nationkey
--             AND n_name = 'CANADA'
--     )
-- ORDER BY 
--     s_name;

SELECT 
    s_name, 
    s_address
FROM 
    supplier 
JOIN nation ON s_nationkey = n_nationkey
JOIN partsupp ON s_suppkey = ps_suppkey
JOIN part ON ps_partkey = p_partkey
WHERE 
    p_name LIKE 'forest%'
    AND n_name = 'CANADA'
    AND ps_availqty > (
        SELECT 
            0.5 * SUM(l_quantity)
        FROM 
            lineitem
        WHERE 
            l_partkey = ps_partkey 
            AND l_suppkey = ps_suppkey 
            AND l_shipdate >= '1994-01-01'
            AND l_shipdate < DATE_ADD('1994-01-01', 365)
    )
ORDER BY 
    s_name;


-- SELECT s_name, 
--     s_address
-- FROM 
--     supplier 
-- JOIN nation ON s_nationkey = n_nationkey
-- JOIN partsupp ON s_suppkey = ps_suppkey
-- JOIN part ON ps_partkey = p_partkey
-- WHERE 
--     p_name LIKE 'forest%'
--     AND n_name = 'CANADA'
--     AND ps_availqty > (
--         SELECT 
--             0.5 * SUM(l_quantity)
--         FROM 
--             lineitem
--         WHERE 
--             l_partkey = ps_partkey 
--             AND l_suppkey = ps_suppkey 
--             AND l_shipdate >= DATE('1994-01-01')
--             AND l_shipdate < DATE('1994-01-01') + INTERVAL '1' YEAR
--     )
-- ORDER BY 
--     s_name;
