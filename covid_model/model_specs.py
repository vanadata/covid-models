import pandas as pd
import numpy as np
import datetime as dt
import json

from sqlalchemy import MetaData

from covid_model.db import db_engine
from covid_model.data_imports import ExternalVacc


class CovidModelSpecifications:

    def __init__(self, min_date=dt.date(2020, 1, 24), max_date=dt.date(2022, 5, 31)):

        self.start_date = min_date
        self.end_date = max_date

        self.tslices = None
        self.ef_by_slice = None

        self.model_params = None
        self.vacc_immun_params = None
        self.vacc_proj_params = None
        self.timeseries_effects = {}

        self.vacc_rate_df = None

    def write_to_db(self, engine, schema='covid_model', table='specifications', tags=None):
        metadata = MetaData(schema=schema)
        metadata.reflect(engine, only=['fits'])
        fits_table = metadata.tables[f'{schema}.{table}']

        stmt = fits_table.insert().values(
            created_at=dt.datetime.now(),
            tags=tags,
            start_date=self.start_date,
            end_date=self.end_date,
            tslices=self.tslices,
            efs=self.ef_by_slice,
            efs_cov=self.efs_cov,
            model_params=self.model_params,
            vacc_proj_params=self.vacc_proj_params,
            vacc_immun_params=self.vacc_immun_params,
            # vacc_rates=self.vacc_rate_df,
            timeseries_effects=self.timeseries_effects,
        )

        conn = engine.connect()
        result = conn.execute(stmt)

        if self.model is not None:
            self.model.fit_id = result.inserted_primary_key[0]

        return result.inserted_primary_key[0]

    def set_model_params(self, model_params):
        self.model_params = model_params if type(model_params) == dict else json.load(open(model_params))

    def set_vacc_rates(self, engine, vacc_proj_params):
        self.vacc_proj_params = vacc_proj_params if isinstance(vacc_proj_params, dict) else json.load(open(vacc_proj_params))
        self.vacc_rate_df = ExternalVacc(engine, t0_date=self.start_date, fill_to_date=self.end_date).fetch(proj_params=self.vacc_proj_params, group_pop=self.model_params['group_pop'])

    def set_vacc_immun(self, vacc_immun_params):
        self.vacc_immun_params = vacc_immun_params if isinstance(vacc_immun_params, dict) else json.load(open(vacc_immun_params))

    def add_timeseries_effect(self, effect_type_name, prevalence_data, param_multipliers, fill_forward=False):
        # build prevalence and multiplier dataframes from inputs
        prevalence_df = pd.read_csv(prevalence_data, parse_dates=['date'], index_col=0) if isinstance(prevalence_data, str) else prevalence_data.copy()
        prevalence_df = prevalence_df[prevalence_df.max(axis=1) > 0]
        if fill_forward and self.end_date > prevalence_df.index.max().date():
            projections = pd.DataFrame.from_dict({date: prevalence_df.iloc[-1] for date in pd.date_range(prevalence_df.index.max() + dt.timedelta(days=1), self.end_date)}, orient='index')
            prevalence_df = pd.concat([prevalence_df, projections]).sort_index()
        # prevalence_df.index = (prevalence_df.index.to_series() - self.min_date).dt.days

        multiplier_dict = json.load(open(param_multipliers)) if isinstance(param_multipliers, str) else param_multipliers
        # multiplier_df = pd.DataFrame.from_dict(multiplier_dict, orient='index').rename_axis(index='effect').fillna(1)

        self.timeseries_effects[effect_type_name] = []
        for effect_name in prevalence_df.columns:
            d = {'effect_name': effect_name, 'multipliers': multiplier_dict[effect_name], 'start_date': prevalence_df.index.min(), 'prevalence': list(prevalence_df[effect_name].values)}
            self.timeseries_effects[effect_type_name].append(d)

    def get_timeseries_effect_multipliers(self):
        params = set().union(*[effect_specs['multipliers'].keys() for effects in self.timeseries_effects.values() for effect_specs in effects])
        multipliers = pd.DataFrame(
            index=pd.date_range(self.start_date, self.end_date),
            columns=params,
            data=1.0
        )

        multiplier_dict = {}
        for effect_type in self.timeseries_effects.keys():
            prevalence_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date))
            for effect_specs in self.timeseries_effects[effect_type]:
                end_date = effect_specs['start_date'] + dt.timedelta(days=len(effect_specs['prevalence']) - 1)
                prevalence_df[effect_specs['effect_name']] = 0
                prevalence_df.loc[effect_specs['start_date']:end_date, effect_specs['effect_name']] = effect_specs['prevalence']
                multiplier_dict[effect_specs['effect_name']] = {**{param: 1.0 for param in params}, **effect_specs['multipliers']}

            prevalence_df = prevalence_df.sort_index()
            multiplier_df = pd.DataFrame.from_dict(multiplier_dict, orient='index').rename_axis(index='effect').fillna(1)

            prevalence = prevalence_df.stack().rename_axis(index=['t', 'effect'])
            remainder = 1 - prevalence.groupby('t').sum()

            multipliers_for_this_effect_type = multiplier_df.multiply(prevalence, axis=0).groupby('t').sum().add(remainder, axis=0)
            multipliers = multipliers.multiply(multipliers_for_this_effect_type)

        multipliers.index = (multipliers.index.to_series().dt.date - self.start_date).dt.days

        return multipliers

    def get_vacc_rate_per_unvacc(self):
        # calculate the vaccination rate per unvaccinated
        cumu = self.vacc_rate_df.groupby('age').cumsum()
        age_group_pop = self.vacc_rate_df.index.get_level_values('age').to_series(index=self.vacc_rate_df.index).replace(self.model_params['group_pop'])
        unvacc = cumu.groupby('age').shift(1).fillna(0).apply(lambda s: age_group_pop - s)
        return self.vacc_rate_df / unvacc

    def get_vacc_fail_per_vacc(self):
        return {k: v['fail_rate'] for k, v in self.vacc_immun_params.items()}

    def get_vacc_fail_reduction_per_vacc_fail(self, delay=7):
        vacc_fail_per_vacc_df = pd.DataFrame.from_dict(self.get_vacc_fail_per_vacc()).rename_axis(index='age')

        rate = self.vacc_rate_df.groupby(['age']).shift(delay).fillna(0)
        fail_increase = rate['shot1'] * vacc_fail_per_vacc_df['shot1']
        fail_reduction_per_vacc = vacc_fail_per_vacc_df.shift(1, axis=1) - vacc_fail_per_vacc_df
        fail_reduction = (rate * fail_reduction_per_vacc.fillna(0)).sum(axis=1)
        fail_cumu = (fail_increase - fail_reduction).groupby('age').cumsum()
        return (fail_reduction / fail_cumu).fillna(0)

    def get_vacc_mean_efficacy(self, delay=7):
        shots = list(self.vacc_rate_df.columns)
        rate = self.vacc_rate_df.groupby(['age']).shift(delay).fillna(0)
        cumu = rate.groupby(['age']).cumsum()
        vacc_effs = {k: v['eff'] for k, v in self.vacc_immun_params.items()}

        terminal_cumu_eff = pd.DataFrame(index=rate.index, columns=shots, data=0)
        vacc_eff_decay_mult = lambda days_ago: 1.0718 * (1 - np.exp(-(days_ago + delay) / 7)) * np.exp(-days_ago / 540)
        for shot, next_shot in zip(shots, shots[1:] + [None]):
            nonzero_ts = rate[shot][rate[shot] > 0].index.get_level_values('t')
            if len(nonzero_ts) > 0:
                days_ago_range = range(nonzero_ts.max() - nonzero_ts.min() + 1)
                for days_ago in days_ago_range:
                    terminal_rate = np.minimum(rate[shot], (cumu[shot] - cumu.groupby('age').shift(-days_ago)[next_shot]).clip(lower=0)) if next_shot is not None else rate[shot]
                    terminal_cumu_eff[shot] += vacc_effs[shot] * vacc_eff_decay_mult(days_ago) * terminal_rate.groupby(['age']).shift(days_ago).fillna(0)

        return (terminal_cumu_eff.sum(axis=1) / cumu[shots[0]]).fillna(0)


if __name__ == '__main__':
    engine = db_engine()
    cms = CovidModelSpecifications()

    cms.set_model_params('input/params.json')
    cms.set_vacc_rates(engine, json.load(open('input/vacc_proj_params.json'))['current trajectory'])
    cms.set_vacc_immun('input/vacc_immun_params.json')
    cms.add_timeseries_effect('variant', prevalence_data='input/variant_prevalence.csv', param_multipliers='input/param_multipliers.json', fill_forward=True)
    cms.add_timeseries_effect('mab', prevalence_data='input/mab_prevalence.csv', param_multipliers='input/param_multipliers.json', fill_forward=True)
