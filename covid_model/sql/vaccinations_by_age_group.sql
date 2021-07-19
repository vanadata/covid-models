with distribution_by_age_by_day as (
	select distinct
		county_id
		, date_trunc('week', measure_date) as measure_date
		, age
		, 1.0 * sum(first_doses_given) over (partition by county_id, date_trunc('week', measure_date), age)
			/ sum(first_doses_given) over (partition by county_id)
	as share_first_doses_given_in_this_week_to_this_age
	from cdphe.covid19_vaccinations
)
, federal_doses as (
	select
		county_id
		, measure_date
		, age
		, total_count * share_first_doses_given_in_this_week_to_this_age as est_federal_first_doses
	from cdphe.covid19_county_summary fd
	join distribution_by_age_by_day dad on lpad(fd.county_fips_code::text, 5, '0') = dad.county_id
	where count_type = 'cumulative federal 1+' and share_first_doses_given_in_this_week_to_this_age > 0
)
select
    measure_date::date as measure_date
    , case
	    when age <= 19 then '0-19'
	    when age >= 20 and age <= 39 then '20-39'
	    when age >= 40 and age <= 64 then '40-64'
	    when age >= 65 then '65+'
	end as "group"
	, vacc
    , round(case
    	when vacc = 'mrna' then sum(v.first_doses_given + coalesce(est_federal_first_doses, 0) - v.jnj_doses_given)
    	when vacc = 'jnj' then sum(v.jnj_doses_given)
    end) as rate
from cdphe.covid19_vaccinations v
left join federal_doses fd using (county_id, measure_date, age)
	, unnest(array['mrna', 'jnj']) vacc
group by 1, 2, 3
order by 1, 2, 3;