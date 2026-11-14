-- Step 2: one row per day (event date SQLDATE) with global and US aggregates from GDELT 1.0 events.
-- Output is small (~10,000 rows x 14 columns, about 1-2 MB): save it as CSV.
-- Counts grow strongly over the years, so the features we build later use shares and averages, not raw counts.
-- QuadClass: 1 verbal cooperation, 2 material cooperation, 3 verbal conflict, 4 material conflict.
-- EventRootCode '14' = protest. ActionGeo_CountryCode is FIPS ('US' = United States).
SELECT
  SQLDATE AS day,
  COUNT(*) AS n_all,
  SUM(NumMentions) AS mentions_all,
  AVG(GoldsteinScale) AS goldstein_all,
  SAFE_DIVIDE(SUM(AvgTone * NumMentions), SUM(NumMentions)) AS tone_all_w,
  AVG(IF(QuadClass = 4, 1, 0)) AS share_matconflict_all,
  AVG(IF(EventRootCode = '14', 1, 0)) AS share_protest_all,
  COUNTIF(ActionGeo_CountryCode = 'US') AS n_us,
  AVG(IF(ActionGeo_CountryCode = 'US', GoldsteinScale, NULL)) AS goldstein_us,
  SAFE_DIVIDE(SUM(IF(ActionGeo_CountryCode = 'US', AvgTone * NumMentions, 0)),
              SUM(IF(ActionGeo_CountryCode = 'US', NumMentions, 0))) AS tone_us_w,
  AVG(IF(ActionGeo_CountryCode = 'US', IF(QuadClass IN (3, 4), 1, 0), NULL)) AS share_conflict_us,
  AVG(IF(ActionGeo_CountryCode = 'US', IF(EventRootCode = '14', 1, 0), NULL)) AS share_protest_us,
  AVG(IF(ActionGeo_CountryCode = 'US', IF(QuadClass = 4, 1, 0), NULL)) AS share_matconflict_us
FROM `gdelt-bq.full.events`
WHERE SQLDATE BETWEEN 19991101 AND 20260930
GROUP BY day
ORDER BY day;
