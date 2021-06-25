import pandas as pd
import numpy as np
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
from db import db_engine


# load actual hospitalization data for fitting
def get_hosps(engine, min_date=dt.datetime(2020, 1, 24)):
    actual_hosp_df = pd.read_sql('select * from cdphe.emresource_hospitalizations', engine)
    actual_hosp_df['t'] = ((pd.to_datetime(actual_hosp_df['measure_date']) - min_date) / np.timedelta64(1, 'D')).astype(int)
    actual_hosp_tmin = actual_hosp_df[actual_hosp_df['currently_hospitalized'].notnull()]['t'].min()
    return [0] * actual_hosp_tmin + list(actual_hosp_df['currently_hospitalized'])


def get_hosps_df(engine):
    return pd.read_sql('select * from cdphe.emresource_hospitalizations', engine).set_index('measure_date')['currently_hospitalized']


def get_hosps_by_age(engine, fname):
    df = pd.read_csv(fname, parse_dates=['dates']).set_index('dates')
    df = df[[col for col in df.columns if col[:17] == 'HospCOVIDPatients']]
    df = df.rename(columns={col: col.replace('HospCOVIDPatients', '').replace('to', '-').replace('plus', '+') for col in df.columns})
    df = df.stack()
    df.index = df.index.set_names(['measure_date', 'group'])
    cophs_total = df.groupby('measure_date').sum()
    emr_total = get_hosps_df(engine)
    return df * emr_total / cophs_total


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


def get_deaths_by_age(fname):
    raw = pd.read_csv(fname, parse_dates=['deathdate']).set_index('deathdate')
    df = pd.DataFrame(index=raw.index)
    for i, g in enumerate(['0-19', '20-39', '40-64', '65+']):
        df[g] = raw[f'agedeaths{i+1}'].str.replace('.', '0').astype(int)
    df = df.stack()
    df.index = df.index.set_names(['measure_date', 'group'])
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
        if proj_fixed_rates:
            projected_rates['rate'] = pd.DataFrame(proj_fixed_rates).stack().combine_first(projected_rates['rate'])
        # add projections to dataframe
        projections = pd.concat({d: projected_rates for d in proj_date_range})
        projections['is_projected'] = True
        df = pd.concat([df, projections]).sort_index()

        # reduce rates to prevent cumulative vaccination from exceeding max_cumu
        if max_cumu:
            cumu_vacc = df.loc[:proj_from_date, 'rate'].groupby('group').sum()
            groups = realloc_priority if realloc_priority else groups
            for d in proj_date_range:
                for i in range(len(groups)):
                    group = groups[i]
                    current_rate = df.loc[(d, group), 'rate'].sum()
                    if current_rate > 0:
                        max_rate = max_rate_per_remaining * (max_cumu[group] - cumu_vacc[group])
                        percent_excess = max((current_rate - max_rate) / current_rate, 0)
                        for vacc in vaccs:
                            excess_rate = df.loc[(d, group, vacc), 'rate'] * percent_excess
                            df.loc[(d, group, vacc), 'rate'] -= excess_rate
                            # if a reallocate_order is provided, reallocate excess rate to other groups
                            if i < len(groups) - 1 and realloc_priority is not None:
                                df.loc[(d, groups[i + 1], vacc), 'rate'] += excess_rate
                    cumu_vacc[group] += df.loc[(d, group), 'rate'].sum()

    return df


def get_vaccinations_by_county(engine):
    sql = """select 
        dd.measure_date::date as measure_date
        , c.county_id
        , c.county_label as county
        , g.age_group
        , coalesce(sum(total_count) filter (where date_type like 'vaccine dose 1/%'), 0)::int as first_doses_given
        , coalesce(sum(total_count) filter (where date_type = 'vaccine dose 1/2'), 0)::int as mrna_first_doses_given
        , coalesce(sum(total_count) filter (where date_type = 'vaccine dose 2/2'), 0)::int as mrna_second_doses_given
        , coalesce(sum(total_count) filter (where date_type = 'vaccine dose 1/1'), 0)::int  as jnj_doses_given
    from generate_series('2020-12-01'::date, (select max(reporting_date) from cdphe.covid19_county_summary where date_type like 'vaccine dose 1/%'), interval '1 day') as dd (measure_date)
    cross join geo_dev.us_counties c
    cross join (select distinct count_type as age_group from cdphe.covid19_county_summary where date_type like 'vaccine dose 1/%') g
    left join cdphe.covid19_county_summary v on lpad(v.county_fips_code::text, 5, '0') = c.county_id and dd.measure_date = v.reporting_date and g.age_group = v.count_type
    where c.state_iso2 = 'CO'
    group by 1, 2, 3, 4
    order by 1, 2;
    """
    df = pd.read_sql(sql, engine)
    return df


def get_variant_prevalence(fname):
    actual_variant_prev = pd.read_csv(fname, parse_dates=['date'])
    actual_variant_prev['t'] = (actual_variant_prev['date'] - actual_variant_prev['date'].min()).astype('timedelta64[D]').astype(int)
    actual_variant_prev = actual_variant_prev.drop(columns=['date']).set_index('t')

    t0s = [0, 0, 2*7, 18*7]
    def solve_ode(params):
        # t0s = params[:4]
        # growth_rates = params[4:]
        growth_rates = params
        def variant_growth(t, y):
            naive_growth_y = []
            # print(list(zip(y, t0s, growth_rates)))
            for prevalence, t0, rate in zip(y, t0s, growth_rates):
                if prevalence <= 0.01 and t >= t0:
                    naive_growth_y.append(0.01 * (1.0+rate))
                else:
                    # if rate == 0.9:
                    #     print('\n', t, rate)
                    #     print(prevalence)
                    #     print(prevalence * (1.0 + rate))
                    naive_growth_y.append(prevalence * (1.0 + rate))
            # print(t)
            # print(y)
            #
            # print(dy)
            dy = np.array(naive_growth_y) / sum(naive_growth_y) - y
            return dy

        solution = spi.solve_ivp(variant_growth,
                          t_span=(actual_variant_prev.index.min(), actual_variant_prev.index.max()),
                          y0=[1, 0, 0, 0],
                          t_eval=actual_variant_prev.index.values, method='LSODA')

        # print(solution.y.flatten())
        return solution.y

    # params, cov = spo.curve_fit(solve_ode,
    #               xdata=actual_variant_prev.index.values,
    #               ydata=actual_variant_prev.values.T,
    #               p0=[actual_variant_prev.index.min()] * 4 + [1.0] * 4,
    #               bounds=([actual_variant_prev.index.min()] * 4 + [1.0] * 4,
    #                       [actual_variant_prev.index.max()] * 4 + [3.0] * 4)
    # )


    # results = spo.least_squares(lambda params: solve_ode(params).flatten() - actual_variant_prev.T.to_numpy().flatten(),
    #                   x0=[0, 0, 2*7, 18*7] + [0.05] * 4,
    #                   bounds=([0, 0, 2*7, 18*7] + [0.0] * 4,
    #                           [0+1, 0+1, 2*7+1, 18*7+1] + [0.1] * 4))

    results = spo.least_squares(lambda params: solve_ode(params).flatten() - actual_variant_prev.T.to_numpy().flatten(),
                      x0=[0.2] * 4,
                      bounds=([0.0] * 4, [1] * 4))

    print(actual_variant_prev.T.to_numpy()[3][-6:])

    # print(results)
    # print(results.cost)
    # print(results.x)
    print([round(x, 7) for x in solve_ode(results.x)[0]][-6:])
    print([round(x, 7) for x in solve_ode(results.x)[1]][-6:])
    print([round(x, 7) for x in solve_ode(results.x)[2]][-6:])
    print([round(x, 7) for x in solve_ode(results.x)[3]][-6:])
    # print([round(x, 7) for x in solve_ode([0, 0.019, 0.005, 0.9])[3]][-6:])

    print('\n', results.x)
    print(results.cost)

    actual_variant_prev.plot()
    pd.DataFrame(solve_ode(results.x).T).plot()
    plt.show()


if __name__ == '__main__':
    get_variant_prevalence('input/variant_prevalence.csv')
