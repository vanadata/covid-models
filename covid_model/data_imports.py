import pandas as pd
import numpy as np
import datetime as dt
import json
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
from db import db_engine
from utils import get_params


class ExternalData:
    def __init__(self, engine=None, t0_date=None, fill_from_date=None, fill_to_date=None):
        self.engine = engine
        self.t0_date = t0_date
        self.fill_from_date = fill_from_date if fill_from_date is not None else t0_date
        self.fill_to_date = fill_to_date

    def fetch(self, fpath=None, rerun=True, **args):
        if rerun:
            df = self.fetch_from_db(**args)
            if fpath is not None:
                df.reset_index().drop(columns='index', errors='ignore').to_csv(fpath, index=False)
        else:
            df = pd.read_csv(fpath)

        if self.t0_date is not None:
            index_names = [idx for idx in df.index.names if idx not in (None, 'measure_date')]
            df = df.reset_index()
            df['t'] = (pd.to_datetime(df['measure_date']).dt.date - self.t0_date.date()).dt.days
            min_t = min(df['t'])
            df = df.reset_index().drop(columns=['index', 'measure_date'], errors='ignore').set_index(['t'] + index_names)

            trange = range((self.fill_from_date - self.t0_date).days, (self.fill_to_date - self.t0_date).days + 1)
            index = pd.MultiIndex.from_product([trange] + [df.index.unique(level=idx) for idx in index_names]) if index_names else range(min_t)
            empty_df = pd.DataFrame(index=index.set_names(['t'] + index_names))
            df = empty_df.join(df, how='left').fillna(0)

        return df

    def fetch_from_db(self, **args) -> pd.DataFrame:
        return pd.read_sql(args['sql'], self.engine)


class ExternalHosps(ExternalData):
    def fetch_from_db(self):
        pd.read_sql('select * from cdphe.emresource_hospitalizations', self.engine)


def vacc_eff_decay(days_since):
    return 1.1 * (1 - np.exp(-days_since/7)) * np.exp(-days_since/365)


class ExternalVacc(ExternalData):
    def fetch_from_db(self, proj_params=None, group_pop=None):
        sql = open('sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()

        proj_params = proj_params if type(proj_params) == dict else json.load(open(proj_params))
        proj_lookback = proj_params['lookback'] if 'lookback' in proj_params.keys() else 7
        proj_fixed_rates = proj_params['fixed_rates'] if 'fixed_rates' in proj_params.keys() else None
        max_cumu = proj_params['max_cumu'] if 'max_cumu' in proj_params.keys() else 0
        max_rate_per_remaining = proj_params['max_rate_per_remaining'] if 'max_rate_per_remaining' in proj_params.keys() else 1.0
        realloc_priority = proj_params['realloc_priority'] if 'realloc_priority' in proj_params.keys() else None

        df = pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
        shots = list(df.columns)

        # add projections
        proj_from_date = df.index.get_level_values('measure_date').max() + dt.timedelta(days=1)
        if self.fill_to_date.date() >= proj_from_date:
            proj_date_range = pd.date_range(proj_from_date, self.fill_to_date)
            # project rates based on the last {proj_lookback} days of data
            projected_rates = df.loc[(proj_from_date - dt.timedelta(days=proj_lookback)):].groupby('age').sum() / float(proj_lookback)
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                for shot in shots:
                    projected_rates[shot] = pd.DataFrame(proj_fixed_rates)[shot]
            # build projections
            projections = pd.concat({d.date(): projected_rates for d in proj_date_range}).rename_axis(index=['measure_date', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                max_cumu_df = pd.DataFrame(max_cumu) * pd.DataFrame(group_pop, index=shots).transpose()

                cumu_vacc = df.groupby('age').sum()
                groups = realloc_priority if realloc_priority else projections.index.unique('age')
                # vaccs = df.index.unique('vacc')
                for d in projections.index.unique('measure_date'):
                    for i in range(len(groups)):
                        group = groups[i]
                        current_rate = projections.loc[(d, group)]
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        excess_rate = (projections.loc[(d, group)] - max_rate).clip(lower=0)
                        projections.loc[(d, group)] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[(d, groups[i + 1])] += excess_rate

                    cumu_vacc += projections.loc[d]

            df = pd.concat([df, projections]).sort_index()

        # get vacc rates from database, and calc rate and cumulative shots, shifting to account for 7-day delay of effect
        # df = pd.read_sql(sql, self.engine, index_col=['measure_date', 'age', 'vacc']).sort_index()
        shots = list(df.columns)
        rate = df.groupby(['age']).shift(7).fillna(0)
        cumu = rate.groupby(['age']).cumsum()

        # calculate the "terminal rate" for each shot, representing the number of people who received a given shot on a given day who will NOT receive another shot
        terminal_cumu = pd.DataFrame(index=cumu.index)
        for shot, next_shot in zip(shots, shots[1:] + [None]):
            terminal_cumu[shot] = (cumu[shot] - cumu[next_shot].groupby(['age']).max()).clip(lower=0).reorder_levels(['measure_date', 'age']) if next_shot is not None else cumu[shot].copy()
        terminal_rate = terminal_cumu.groupby(['age']).diff().fillna(0)

        # calculate the mean efficacy from the terminal rates
        terminal_cumu_eff = pd.DataFrame(index=terminal_rate.index, columns=shots, data=0)
        for shot, initial_shot_eff in {'shot1': 0.76, 'shot2': 0.97, 'shot3': 0.99}.items():
            for days_ago in range(len(terminal_rate)):
                terminal_cumu_eff[shot] += initial_shot_eff * vacc_eff_decay(days_ago) * terminal_rate[shot].groupby(['age']).shift(days_ago).fillna(0)
        total_mean_eff = terminal_cumu_eff.sum(axis=1) / terminal_cumu.sum(axis=1)

        total_mean_eff.to_csv('output/vacc_total_mean_eff.csv')

        return df
        

# class ExternalVacc(ExternalData):
#     def fetch_from_db(self, proj_params=None, group_pop=None):
#         sql = open('sql/vaccinations_by_age_group.sql', 'r').read()
#
#         proj_params = proj_params if type(proj_params) == dict else json.load(open(proj_params))
#         proj_lookback = proj_params['lookback'] if 'lookback' in proj_params.keys() else 7
#         proj_fixed_rates = proj_params['fixed_rates'] if 'fixed_rates' in proj_params.keys() else None
#         max_cumu = proj_params['max_cumu'] if 'max_cumu' in proj_params.keys() else 0
#         max_rate_per_remaining = proj_params['max_rate_per_remaining'] if 'max_rate_per_remaining' in proj_params.keys() else 1.0
#         realloc_priority = proj_params['realloc_priority'] if 'realloc_priority' in proj_params.keys() else None
#
#         df = pd.read_sql(sql, self.engine, index_col=['measure_date', 'age', 'vacc'])
#
#         # add projections
#         proj_from_date = df.index.get_level_values('measure_date').max() + dt.timedelta(days=1)
#         if self.fill_to_date.date() >= proj_from_date:
#             proj_date_range = pd.date_range(proj_from_date, self.fill_to_date)
#             # project rates based on the last {proj_lookback} days of data
#             projected_rates = df.loc[(proj_from_date - dt.timedelta(days=proj_lookback)):].groupby(['age', 'vacc']).sum() / float(proj_lookback)
#             # override rates using fixed values from proj_fixed_rates, when present
#             if proj_fixed_rates:
#                 projected_rates['rate'] = pd.DataFrame(proj_fixed_rates).stack().combine_first(projected_rates['rate'])
#             # add projections to dataframe
#             projections = pd.concat({d: projected_rates for d in proj_date_range})
#             projections['is_projected'] = True
#             df = pd.concat([df, projections]).sort_index()
#
#             # reduce rates to prevent cumulative vaccination from exceeding max_cumu
#             if max_cumu:
#                 cumu_vacc = df.loc[:(proj_from_date - dt.timedelta(days=1)), 'rate'].groupby('age').sum()
#                 groups = realloc_priority if realloc_priority else df.index.unique('age')
#                 vaccs = df.index.unique('vacc')
#                 for d in proj_date_range:
#                     for i in range(len(groups)):
#                         group = groups[i]
#                         current_rate = df.loc[(d, group), 'rate'].sum()
#                         if current_rate > 0:
#                             this_max_cumu = max_cumu.copy()
#                             max_rate = max_rate_per_remaining * (
#                                     this_max_cumu[group] * group_pop[group] - cumu_vacc[group])
#                             percent_excess = max((current_rate - max_rate) / current_rate, 0)
#                             for vacc in vaccs:
#                                 excess_rate = df.loc[(d, group, vacc), 'rate'] * percent_excess
#                                 df.loc[(d, group, vacc), 'rate'] -= excess_rate
#                                 # if a reallocate_order is provided, reallocate excess rate to other groups
#                                 if i < len(groups) - 1 and realloc_priority is not None:
#                                     df.loc[(d, groups[i + 1], vacc), 'rate'] += excess_rate
#                         cumu_vacc[group] += df.loc[(d, group), 'rate'].sum()
#
#         return df


class ExternalContactMatrices(ExternalData):
    pass


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
    df.index = df.index.set_names(['measure_date', 'age'])
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


def get_deaths_by_age(engine):
    sql = """select
        date::date as measure_date
        , case when age_group in ('0-5', '6-11', '12-17', '18-19') then '0-19' else age_group end as "group"
        , sum(count::int) as new_deaths
    from cdphe.temp_covid19_county_summary
    where count_type like 'deaths, %%' and date_type = 'date of death'
    group by 1, 2
    order by 1, 2"""
    df = pd.read_sql(sql, engine, parse_dates=['measure_date']).set_index(['measure_date', 'age'])
    return df


def get_vaccinations_by_county(engine):
    sql = open('sql/vaccinations_by_age_by_county.sql', 'r').read()
    df = pd.read_sql(sql, engine)
    return df


def get_corrected_emresource(fpath):
    raw_hosps = pd.read_excel(fpath, 'COVID hospitalized_confirmed', engine='openpyxl', index_col='Resource facility name').drop(index='Grand Total').rename(columns=pd.to_datetime).stack()
    raw_hosps.index = raw_hosps.index.set_names(['facility', 'date'])

    raw_reports = pd.read_excel(fpath, 'Latest EMR update', engine='openpyxl', index_col='Resource facility name').drop(index='Grand Total').rename(columns=pd.to_datetime).stack()
    raw_reports.index = raw_reports.index.set_names(['facility', 'date'])
    raw_reports = pd.to_datetime(raw_reports).rename('last_report_date').sort_index()

    print(raw_reports)
    print(pd.to_datetime(pd.to_numeric(raw_reports).groupby('facility').rolling(20).agg(np.max)))


if __name__ == '__main__':
    engine = db_engine()
    # , t0_date = dt.datetime(2020, 1, 24)
    df = ExternalVacc2(engine=engine, fill_to_date=dt.datetime(2022, 12, 31)).fetch(
        proj_params=json.load(open('input/vacc_proj_params.json'))['current trajectory'],
        group_pop=json.load(open('input/params.json'))['group_pop'])
    print(df)
