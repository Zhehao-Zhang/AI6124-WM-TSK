-- Step 1 (cheap check): which GDELT event tables cover our period, and how many rows per year.
-- Reads only the SQLDATE column (13.5 GB on 2026-11-09). Result saved as 01_coverage_result.csv.
SELECT 'v1_full_events' AS src, DIV(SQLDATE, 10000) AS year, COUNT(*) AS n_events,
       MIN(SQLDATE) AS first_day, MAX(SQLDATE) AS last_day
FROM `gdelt-bq.full.events`
WHERE SQLDATE BETWEEN 19990101 AND 20261231
GROUP BY year
UNION ALL
SELECT 'v2_events' AS src, DIV(SQLDATE, 10000) AS year, COUNT(*) AS n_events,
       MIN(SQLDATE) AS first_day, MAX(SQLDATE) AS last_day
FROM `gdelt-bq.gdeltv2.events`
WHERE SQLDATE BETWEEN 19990101 AND 20261231
GROUP BY year
ORDER BY src, year;
