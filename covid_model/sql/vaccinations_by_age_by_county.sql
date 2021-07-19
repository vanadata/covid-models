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
    v.measure_date::date as measure_date
    , v.county_id
    , v.county
    , case
        when v.age <= 19 then '0-19'
        when v.age >= 20 and v.age <= 39 then '20-39'
        when v.age >= 40 and v.age <= 64 then '40-64'
        when v.age >= 65 then '65+'
    end as age_group
    , coalesce(round(sum(v.first_doses_given + coalesce(fd.est_federal_first_doses, 0))), 0) as first_doses_given
    , coalesce(round(sum(v.first_doses_given + coalesce(fd.est_federal_first_doses, 0) - v.jnj_doses_given)), 0) as mrna_first_doses_given
    , coalesce(round(sum(v.final_doses_given + coalesce(fd2.est_federal_first_doses, 0) - v.jnj_doses_given)), 0) as mrna_second_doses_given
    , coalesce(round(sum(v.jnj_doses_given)), 0) as jnj_doses_given
from cdphe.covid19_vaccinations v
left join federal_doses fd using (county_id, measure_date, age)
left join federal_doses fd2 on v.county_id = fd2.county_id and v.measure_date = fd2.measure_date + interval '21 days' and v.age = fd2.age
group by 1, 2, 3, 4
order by 1, 2, 3, 4;