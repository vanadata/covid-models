import pandas as pd
import numpy as np
import datetime as dt
from db import db_engine


# load actual hospitalization data for fitting
def get_hosps(engine, min_date=dt.datetime(2020, 1, 24)):
    actual_hosp_df = pd.read_sql('select * from cdphe.emresource_hospitalizations', engine)
    actual_hosp_df['t'] = ((pd.to_datetime(actual_hosp_df['measure_date']) - min_date) / np.timedelta64(1, 'D')).astype(int)
    actual_hosp_tmin = actual_hosp_df[actual_hosp_df['currently_hospitalized'].notnull()]['t'].min()
    return [0] * actual_hosp_tmin + list(actual_hosp_df['currently_hospitalized'])


def get_hosps_df(engine):
    return pd.read_sql('select * from cdphe.emresource_hospitalizations', engine).set_index('measure_date')['currently_hospitalized']


# load actual death data for plotting
def get_deaths(engine, min_date=dt.datetime(2020, 1, 24)):
    sql = """
        select 
            reporting_date as measure_date
            , sum(total_count) as new_deaths
        from cdphe.covid19_county_summary ccs 
        where count_type = 'deaths'
        group by 1
        order by 1"""
    df = pd.read_sql(sql, engine, parse_dates=['measure_date']).set_index('measure_date')
    df = pd.date_range(min_date, df.index.max()).to_frame().join(df, how='left').drop(columns=[0]).fillna(0)
    df['cumu_deaths'] = df['new_deaths'].cumsum()

    return df


def get_vaccinations(engine, from_date=None, proj_to_date=None, proj_lookback=7, proj_fixed_rates=None, max_cumu=None, max_rate_per_remaining=1.0, realloc_priority=None):
    sql = f"""select 
                reporting_date as measure_date
                , count_type as "group"
                , case date_type when 'vaccine dose 1/2' then 'mrna' when 'vaccine dose 1/1' then 'jnj' end as vacc
                , sum(total_count) as rate
            from cdphe.covid19_county_summary
            where date_type like 'vaccine dose 1/_'
            group by 1, 2, 3
            order by 1, 2, 3
        """
    df = pd.read_sql(sql, engine, index_col=['measure_date', 'group', 'vacc'])

    # left join from an empty dataframe to fill in gaps
    date_range = pd.date_range(from_date if from_date else df.index.get_level_values('measure_date').min(), df.index.get_level_values('measure_date').max())
    groups = df.index.unique('group')
    vaccs = df.index.unique('vacc')
    full_empty_df = pd.DataFrame(index=pd.MultiIndex.from_product([date_range, groups, vaccs], names=['measure_date', 'group', 'vacc']))
    df = full_empty_df.join(df, how='left').fillna(0)
    df['is_projected'] = False

    # add projections
    proj_from_date = df.index.get_level_values('measure_date').max() + dt.timedelta(days=1)
    if proj_to_date >= proj_from_date:
        proj_date_range = pd.date_range(proj_from_date, proj_to_date)
        # project rates based on the last {proj_lookback} days of data
        projected_rates = df.loc[(proj_from_date - dt.timedelta(days=proj_lookback)):].groupby(['group', 'vacc']).sum() / float(proj_lookback)
        # override rates using fixed values from proj_fixed_rates, when present
        projected_rates['rate'] = pd.DataFrame(proj_fixed_rates).stack().combine_first(projected_rates['rate'])
        # add projections to dataframe
        projections = pd.concat({d: projected_rates for d in proj_date_range})
        projections['is_projected'] = True
        df = pd.concat([df, projections])

        # reduce rates to prevent cumulative vaccination from exceeding max_cumu
        if max_cumu:
            cumu_vacc = df.loc[:proj_from_date, 'rate'].groupby('group').sum()
            groups = realloc_priority if realloc_priority else groups
            for d in proj_date_range:
                for i in range(len(groups)):
                    group = groups[i]
                    current_rate = df.loc[(d, group), 'rate'].sum()
                    if current_rate > 0:
                        max_rate = max_rate_per_remaining * max_cumu[group]
                        percent_excess = max((current_rate - max_rate) / current_rate, 0)
                        for vacc in vaccs:
                            excess_rate = df.loc[(d, group, vacc), 'rate'] * percent_excess
                            df.loc[(d, group, vacc), 'rate'] -= excess_rate
                            # if a reallocate_order is provided, reallocate excess rate to other groups
                            if i < len(groups) - 1 and realloc_priority is not None:
                                df.loc[(d, groups[i + 1], vacc), 'rate'] += excess_rate
                    cumu_vacc[group] += df.loc[(d, group), 'rate'].sum()

    return df


if __name__ == '__main__':
    engine = db_engine()
    get_vaccinations(
        engine
        , proj_to_date=dt.datetime(2021, 10, 31)
        , proj_fixed_rates={'jnj': {"0-19": 200, "20-39": 1000, "40-64": 1500, "65+": 300}}
        , max_cumu={"0-19": 0.8*1513005, "20-39": 0.8*1685869, "40-64": 0.8*1902963, "65+": 0.94*738958}
        , realloc_priority=['65+', '40-64', '20-39', '0-19'])
