import json
import math
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import pyswarms as ps
from sqlalchemy import MetaData
from datetime import datetime
import itertools
from data_imports import ExternalHosps, ExternalVacc
from utils import *
from collections import OrderedDict
from ode_builder import *


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel(ODEBuilder):
    attr = OrderedDict({'seir': ['S', 'E', 'I', 'Ih', 'A', 'R', 'RA', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['unvacc', 'vacc', 'vacc_fail']})

    param_attr_names = ('age', 'vacc')

    # the starting date of the model
    datemin = dt.datetime(2020, 1, 24)

    def __init__(self, tslices, efs=None, fit_id=None, engine=None, **ode_builder_args):
        ODEBuilder.__init__(self, trange=range(max(tslices)), attributes=self.attr, param_attr_names=self.param_attr_names)
        self.tslices = list(tslices)

        # build global parameters from params file, creating dict lookups for easy access
        self.engine = engine
        self.raw_params = None

        # vaccines
        self.vacc_rate_df = None
        self.vacc_immun_df = None
        self.vacc_trans_mults_df = None
        self.vacc_prevalence_df = None

        # variants
        self.variant_prevalence_df = None

        # transmission control parameters
        self.efs = efs
        self.ef_by_t = None

        # the var values for the solution; these get populated when self.solve_seir is run
        self.solution = None
        self.solution_y = None
        self.solution_ydf_full = None

        # used to connect up with the matching fit in the database
        self.fit_id = fit_id

    # a model must be prepped before it can be run; if any params EXCEPT the efs (i.e. TC) change, it must be re-prepped
    def prep(self, params='input/params.json', vacc_proj_params='input/vacc_proj_params.json', vacc_immun_params='input/vacc_immun_params.json'):
        # prep general parameters (gparams_lookup)
        if self.params is None or params is not None:
            self.set_params_from_file(params if params is not None else 'input/params.json')
            # prep variants (self.variant_prevalence_df and updates to self.gparams_lookup)

        # prep vacc rates, including projections (vacc_rate_df)
        if self.vacc_rate_df is None or vacc_proj_params is not None:
            self.set_vacc_rates(vacc_proj_params if vacc_proj_params is not None else json.load(open('input/vacc_proj_params.json'))['current trajectory'])
            self.set_vacc_eff(vacc_immun_params if vacc_immun_params is not None else json.load(open('input/vacc_immun_params.json')))
            if 'variants' in self.raw_params and self.raw_params['variants']:
                self.set_variant_params(self.raw_params['variants'])

        # prep efs (ef_by_t)
        if self.ef_by_t is None:
            self.set_ef_by_t(self.efs if self.efs is not None else [0] * (len(self.tslices) - 1))

        # build ODE
        self.build_ode()

    def set_ef_from_db(self, fit_id, extend=True):
        fit = CovidModelFit.from_db(self.engine, fit_id)
        tslices = fit.tslices
        efs = fit.efs
        if extend and self.tmax > tslices[-1]:
            tslices.append(self.tmax)
            efs.append(efs[-1])

        self.tslices = tslices
        self.efs = efs
        self.set_ef_by_t(self.efs)

    # create using on a fit that was run previously or manually inserted into the database
    @staticmethod
    def from_fit(conn, fit_id):
        fit = CovidModelFit.from_db(conn, fit_id)
        model = CovidModel(tslices=fit.tslices, engine=conn)
        model.set_params_from_file(fit.model_params)
        model.set_ef_by_t(fit.efs)

        return model

    # handy properties for the beginning t, end t, and the full range of t values
    @property
    def tmin(self): return self.tslices[0]

    @property
    def tmax(self): return self.tslices[-1]

    @property
    def daterange(self): return pd.date_range(self.datemin, periods=len(self.trange))

    # new exposures by day by group
    @property
    def new_exposures(self):
        return self.solution_sum('seir')['E'] / self.raw_params['alpha']

    # estimated reproduction number (length of infection * new_exposures / current_infections
    @property
    def re_estimates(self):
        infect_duration = 1 / self.raw_params['gamm']
        infected = self.solution_sum('seir')['I'].shift(3)
        return infect_duration * self.new_exposures.groupby('t').sum() / infected

    # calc the observed TC by applying the variant multiplier to the base TC
    @property
    def obs_ef_by_t(self):
        return {t: sum(1 - (1 - self.ef_by_t[t]) * self.params[t][pcmpt]['betta'] / self.raw_params['betta'] for pcmpt in self.param_compartments) / len(self.param_compartments) for t in self.trange}

    @property
    def obs_ef_by_slice(self):
        oef_by_t = self.obs_ef_by_t
        return [np.array([oef_by_t[t] for t in range(self.tslices[i], self.tslices[i+1])]).mean() for i in range(len(self.tslices) - 1)]

    # build dataframe containing vaccine first-dose rates by day by group by vaccine
    def set_vacc_rates(self, proj_params):
        proj_params_dict = proj_params if isinstance(proj_params, dict) else json.load(open(proj_params))

        self.vacc_rate_df = ExternalVacc(self.engine, t0_date=self.datemin, fill_to_date=max(self.daterange)).fetch('input/past_and_projected_vaccinations.csv', proj_params=proj_params_dict, group_pop=self.raw_params['group_pop'])

        cumu = self.vacc_rate_df.groupby('age').cumsum()
        age_group_pop = self.vacc_rate_df.index.get_level_values('age').to_series(index=self.vacc_rate_df.index).replace(self.raw_params['group_pop'])
        unvacc = cumu.groupby('age').shift(1).fillna(0).apply(lambda s: age_group_pop - s)
        vacc_per_unvacc = self.vacc_rate_df / unvacc

        for shot in vacc_per_unvacc.columns:
            for age in self.attr['age']:
                for t in self.trange:
                    for vacc in ['unvacc', 'vacc_fail']:
                        self.set_param(f'{shot}_per_unvacc', vacc_per_unvacc.loc[(t, age), shot], {'age': age, 'vacc': vacc}, trange=[t])

    def set_vacc_eff(self, vacc_eff_params):
        vacc_eff_params = vacc_eff_params if isinstance(vacc_eff_params, dict) else json.load(open(vacc_eff_params))
        shots = list(self.vacc_rate_df.columns)

        # set vacc fail rates
        vacc_fail_per_vacc = {k: v['fail_rate'] for k, v in vacc_eff_params.items()}
        for shot in shots:
            for age in self.attr['age']:
                self.set_param(f'{shot}_fail_rate', vacc_fail_per_vacc[shot][age], {'age': age})
        vacc_fail_per_vacc_df = pd.DataFrame.from_dict(vacc_fail_per_vacc).rename_axis(index='age')

        # set vacc eff
        vacc_effs = {k: v['eff'] for k, v in vacc_eff_params.items()}
        rate = self.vacc_rate_df.groupby(['age']).shift(7).fillna(0)
        cumu = rate.groupby(['age']).cumsum()

        # calculate fail rates
        fail_increase = rate['shot1'] * vacc_fail_per_vacc_df['shot1']
        fail_reduction_per_vacc = vacc_fail_per_vacc_df.shift(1, axis=1) - vacc_fail_per_vacc_df
        fail_reduction = (rate * fail_reduction_per_vacc.fillna(0)).sum(axis=1)
        fail_cumu = (fail_increase - fail_reduction).groupby('age').cumsum()
        fail_reduction_per_fail = (fail_reduction / fail_cumu).fillna(0)

        pd.concat({'rate': rate.sum(axis=1), 'fail_increase': fail_increase, 'fail_reduction': fail_reduction,
                   'fail_cumu': fail_cumu, 'fail_reduction_per_fail': fail_reduction_per_fail},
                  axis=1).to_csv('output/fail_reduction_per_fail.csv')
        # exit()

        # calculate the mean efficacy from the terminal rates
        terminal_cumu_eff = pd.DataFrame(index=rate.index, columns=shots, data=0)
        vacc_eff_decay_mult = lambda days_ago: 1.0718 * (1 - np.exp(-days_ago/7)) * np.exp(-days_ago/540)
        for shot, next_shot in zip(shots, shots[1:] + [None]):
            for days_ago in range(len(rate)):
                terminal_rate = np.minimum(rate[shot], (cumu[shot] - cumu.groupby('age').shift(14-days_ago)[next_shot]).clip(lower=0)) if next_shot is not None else rate[shot]
                terminal_cumu_eff[shot] += vacc_effs[shot] * vacc_eff_decay_mult(days_ago) * terminal_rate.groupby(['age']).shift(days_ago).fillna(0)
        total_mean_eff = (terminal_cumu_eff.sum(axis=1) / cumu[shots[0]]).fillna(0)
        total_mean_eff.to_csv('vacc_total_mean_eff.csv')

        # set vacc eff to the total mean efficacy
        self.set_param('vacc_eff', 0, {'vacc': 'unvacc'})
        self.set_param('vacc_eff', 0, {'vacc': 'vacc_fail'})
        for t in self.trange:
            for age in self.attr['age']:
                self.set_param('vacc_eff', total_mean_eff.loc[(t, age)], {'age': age, 'vacc': 'vacc'}, trange=[t])
                self.set_param('vacc_fail_reduction_per_vacc_fail', fail_reduction_per_fail.loc[(t, age)], {'age': age, 'vacc': 'vacc_fail'}, trange=[t])
                self.params[t][(age, 'vacc')]['hosp'] *= 0
                self.params[t][(age, 'vacc')]['dnh'] *= 0

    def set_variant_params(self, variant_params):
        dfs = {}
        for variant, specs in variant_params.items():
            var_df = pd.read_csv(specs['theta_file_path'])  # file with at least a col "t" and a col containing variant prevalence
            var_df = var_df.rename(columns={specs['theta_column']: variant})[['t', variant]].set_index('t').rename(columns={variant: 'e_prev'}).astype(float)  # get rid of all columns except t (the index) and the prev value
            if 't_min' in specs.keys():
                var_df['e_prev'].loc[:specs['t_min']] = 0
            mult_df = pd.DataFrame(specs['multipliers'], index=self.attr['age']).rename(columns={col: f'{col}' for col in specs['multipliers'].keys()})
            mult_df.index = mult_df.index.rename('age')
            combined = pd.MultiIndex.from_product([var_df.index, mult_df.index], names=['t', 'age']).to_frame().join(var_df).join(mult_df).drop(columns=['t', 'age'])  # cross join
            dfs[variant] = combined
        df = pd.concat(dfs)
        df.index = df.index.set_names(['variant', 't', 'age']).reorder_levels(['t', 'age', 'variant'])
        df = df.sort_index().fillna(1)

        # fill in future variant prevalence by duplicating the last row
        variant_input_tmax = df.index.get_level_values('t').max()
        if variant_input_tmax < self.tmax:
            projections = pd.concat({t: df.loc[variant_input_tmax] for t in range(variant_input_tmax + 1, self.tmax)})
            df = pd.concat([df, projections]).sort_index()

        # apply multipliers
        wildtype_prevalences = 1 - df['e_prev'].groupby(['t', 'age']).sum()
        agg_multipliers = df.apply(lambda s: s*df['e_prev']).groupby(['t', 'age']).sum().apply(lambda s: s + 1 * wildtype_prevalences)

        for col in agg_multipliers.columns:
            if col != 'e_prev':
                for t, age in agg_multipliers.index.values:
                    if t in self.trange:
                        for vacc in self.attr['vacc']:
                            self.params[t][(age, vacc)][col] *= agg_multipliers.loc[(t, age), col]

        # execute delta impact on vacc prevalence
        delta_prevalence_df = df['e_prev'].xs('delta', level='variant')
        for (t, age), delta_prevalence in delta_prevalence_df.items():
            if t < self.tmax:
                non_delta_vacc_eff = self.params[t][(age, 'vacc')]['vacc_eff']
                self.params[t][(age, 'vacc')]['vacc_eff'] -= 0.5 * non_delta_vacc_eff * (1 - non_delta_vacc_eff) * delta_prevalence

        pd.DataFrame.from_dict(self.params).stack().apply(lambda x: x['vacc_eff']).rename_axis(index=['age', 'vacc', 't']).xs('vacc', level='vacc').to_csv('output/vacc_total_mean_eff.csv')

        self.variant_prevalence_df = df

    def set_param_using_age_dict(self, name, val, trange=None):
        if not isinstance(val, dict):
            self.set_param(name, val, trange=trange)
        else:
            for age, v in val.items():
                self.set_param(name, v, attrs={'age': age}, trange=trange)

    # set global parameters and lookup dicts
    def set_params_from_file(self, params):
        self.raw_params = params if type(params) == dict else json.load(open(params))
        for name, val in self.raw_params.items():
            if name != 'variants':
                if not isinstance(val, dict) or 'tslices' not in val.keys():
                    self.set_param_using_age_dict(name, val)
                else:
                    for i, (tmin, tmax) in enumerate(zip([self.tmin] + val['tslices'], val['tslices'] + [self.tmax])):
                        v = {a: av[i] for a, av in val['value'].items()} if isinstance(val['value'], dict) else val['value'][i]
                        self.set_param_using_age_dict(name, v, trange=range(tmin, tmax))

    # set ef by slice and lookup dicts
    def set_ef_by_t(self, ef_by_slice):
        self.efs = ef_by_slice
        self.ef_by_t = {t: get_value_from_slices(self.tslices, list(ef_by_slice), t) for t in self.trange}
        for tmin, tmax, ef in zip(self.tslices[:-1], self.tslices[1:], self.efs):
            self.set_param('ef', ef, trange=range(tmin, tmax))

        # the ODE needs to be rebuilt with the new TC values
        # it would be good to refactor the ODEBuilder to support fittable parameters that are easier to adjust cheaply
        if len(self.terms) > 0:
            self.rebuild_ode_with_new_tc()

    # extend the time range with an additional slice; should maybe change how this works to return a new CovidModel instead
    def add_tslice(self, t, ef=None):
        if t <= self.tslices[-2]:
            raise ValueError(f'New tslice (t={t}) must be greater than the second to last tslices (t={self.tslices[-2]}).')
        if t < self.tmax:
            tmax = self.tmax
            self.tslices[-1] = t
            self.tslices.append(tmax)
        else:
            self.tslices.append(t)
            self.trange = range(self.tslices[0], self.tslices[-1])
            for this_t in range(self.tslices[-2], t):
                self.params[this_t] = self.params[self.tslices[-2] - 1]
        if ef:
            self.efs.append(ef)
            self.set_ef_by_t(self.efs)
        self.fit_id = None

    # build ODE
    def build_ode(self):
        self.reset_ode()
        for age in self.attributes['age']:
            for seir in self.attributes['seir']:
                self.add_flow((seir, age, 'unvacc'), (seir, age, 'vacc_fail'), 'shot1_per_unvacc * shot1_fail_rate')
                self.add_flow((seir, age, 'unvacc'), (seir, age, 'vacc'), 'shot1_per_unvacc * (1 - shot1_fail_rate)')
                self.add_flow((seir, age, 'vacc_fail'), (seir, age, 'vacc'), 'vacc_fail_reduction_per_vacc_fail')
                # self.add_flow((seir, age, 'vacc_fail'), (seir, age, 'vacc'), 'shot2_per_unvacc * (shot1_fail_rate - shot2_fail_rate) / shot1_fail_rate')
                # self.add_flow((seir, age, 'vacc_fail'), (seir, age, 'vacc'), 'shot3_per_unvacc * (shot2_fail_rate - shot3_fail_rate) / shot2_fail_rate')
            for vacc in self.attributes['vacc']:
                self.add_flow(('S', age, vacc), ('E', age, vacc),
                               'betta * (1 - ef) * (1 - vacc_eff) * lamb / total_pop',
                               scale_by_cmpts=[('I', a, v) for a in self.attributes['age'] for v in self.attributes['vacc']])
                self.add_flow(('S', age, vacc), ('E', age, vacc),
                               'betta * (1 - ef) * (1 - vacc_eff) / total_pop',
                               scale_by_cmpts=[('A', a, v) for a in self.attributes['age'] for v in self.attributes['vacc']])
                self.add_flow(('E', age, vacc), ('I', age, vacc), '1 / alpha * pS')
                self.add_flow(('E', age, vacc), ('A', age, vacc), '1 / alpha * (1 - pS)')
                self.add_flow(('I', age, vacc), ('Ih', age, vacc), 'gamm * hosp')
                self.add_flow(('I', age, vacc), ('D', age, vacc), 'gamm * dnh')
                self.add_flow(('I', age, vacc), ('R', age, vacc), 'gamm * (1 - hosp - dnh) * immune_rate_I')
                self.add_flow(('I', age, vacc), ('S', age, vacc), 'gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
                self.add_flow(('A', age, vacc), ('RA', age, vacc), 'gamm * immune_rate_A')
                self.add_flow(('A', age, vacc), ('S', age, vacc), 'gamm * (1 - immune_rate_A)')
                self.add_flow(('Ih', age, vacc), ('D', age, vacc), '1 / hlos * dh')
                self.add_flow(('Ih', age, vacc), ('R', age, vacc), '1 / hlos * (1 - dh) * immune_rate_I')
                self.add_flow(('Ih', age, vacc), ('S', age, vacc), '1 / hlos * (1 - dh) * (1 - immune_rate_I)')
                self.add_flow(('R', age, vacc), ('S', age, vacc), '1 / dimmuneI')
                self.add_flow(('RA', age, vacc), ('S', age, vacc), '1 / dimmuneA')

    # reset terms that depend on TC; this takes about 0.08 sec, while rebuilding the whole ODE takes ~0.90 sec
    def rebuild_ode_with_new_tc(self):
        self.reset_terms({'seir': 'S'}, {'seir': 'E'})
        for age in self.attributes['age']:
            for vacc in self.attributes['vacc']:
                self.add_flow(('S', age, vacc), ('E', age, vacc),
                              'betta * (1 - ef) * (1 - vacc_eff) * lamb / total_pop',
                              scale_by_cmpts=[('I', a, v) for a in self.attributes['age'] for v in self.attributes['vacc']])
                self.add_flow(('S', age, vacc), ('E', age, vacc),
                              'betta * (1 - ef) * (1 - vacc_eff) / total_pop',
                              scale_by_cmpts=[('A', a, v) for a in self.attributes['age'] for v in self.attributes['vacc']])

    # define initial state y0
    @property
    def y0_dict(self):
        y0d = {('S', age, 'unvacc'): n for age, n in self.raw_params['group_pop'].items()}
        y0d[('I', '40-64', 'unvacc')] = 2
        y0d[('S', '40-64', 'unvacc')] -= 2
        return y0d

    # override solve_ode to use default y0_dict
    def solve_seir(self, method='RK45'):
        self.solve_ode(y0_dict=self.y0_dict, method=method)

    # count the total hosps by t as the sum of Ih and Ic
    def total_hosps(self):
        return self.solution_sum('seir')['Ih']

    # count the new exposed individuals by day
    def new_exposed(self):
        sum_df = self.solution_sum('seir')
        return sum_df['E'] - sum_df['E'].shift(1) + sum_df['E'].shift(1) / self.raw_params['alpha']

    # create a new fit and assign to this model
    def gen_fit(self, engine, tags=None):
        fit = CovidModelFit(tslices=self.tslices, fixed_efs=self.efs, tags=tags)
        fit.fit_params = None
        fit.model = self
        self.fit_id = fit.write_to_db(engine)

    # write to covid_model.results in Postgres
    def write_to_db(self, engine=None, tags=None, new_fit=False):
        if engine is None:
            engine = self.engine

        # if there's no existing fit assigned, create a new fit and assign that one
        if new_fit or self.fit_id is None:
            self.gen_fit(engine, tags)

        # join the ef values onto the dataframe
        ef_series = pd.Series(self.ef_by_t, name='ef').rename_axis('t')
        oef_series = pd.Series(self.obs_ef_by_t, name='observed_ef').rename_axis('t')
        df = self.solution_ydf.stack(level=self.param_attr_names).join(ef_series).join(oef_series)

        # add fit_id and created_at date
        df['fit_id'] = self.fit_id
        df['created_at'] = dt.datetime.now()

        # add params
        params_df = pd.DataFrame.from_dict(self.params, orient='index').stack(level=list(range(len(self.param_attr_names)))).map(lambda d: json.dumps(d, ensure_ascii=False))
        df = df.join(params_df.rename_axis(index=('t', ) + tuple(self.param_attr_names)).rename('params'), how='left')

        # write to database
        df.to_sql('results'
                  , con=engine, schema='covid_model'
                  , index=True, if_exists='append', method='multi')

    def write_gparams_lookup_to_csv(self, fname):
        df_by_t = {t: pd.DataFrame.from_dict(df_by_group, orient='index') for t, df_by_group in self.params.items()}
        pd.concat(df_by_t, names=['t', 'age']).to_csv(fname)


# class to find an optimal fit of transmission control (ef) values to produce model results that align with acutal hospitalizations
class CovidModelFit:

    def __init__(self, tslices, fixed_efs, fitted_efs=None, efs_cov=None, fit_params=None, actual_hosp=None, tags=None, model_params=None):
        self.tslices = tslices
        self.fixed_efs = fixed_efs
        self.fitted_efs = fitted_efs
        self.fitted_efs_cov = efs_cov
        self.tags = tags
        self.actual_hosp = actual_hosp
        self.model_params = model_params

        self.model = None

        self.fit_params = {} if fit_params is None else fit_params

        # set fit param efs0
        if 'efs0' not in self.fit_params.keys():
            self.fit_params['efs0'] = [0.75] * self.fit_count
        elif type(self.fit_params['efs0']) in (float, int):
            self.fit_params['efs0'] = [self.fit_params['efs0']] * self.fit_count

        # set fit params ef_min and ef_max
        if 'ef_min' not in self.fit_params.keys():
            self.fit_params['ef_min'] = 0.00
        if 'ef_max' not in self.fit_params.keys():
            self.fit_params['ef_max'] = 0.99

    @staticmethod
    def from_db(conn, fit_id):
        df = pd.read_sql_query(f"select * from covid_model.fits where id = '{fit_id}'", con=conn, coerce_float=True)
        if len(df) == 0:
            raise ValueError(f'{fit_id} is not a valid fit ID.')
        fit_count = len(df['efs_cov'][0]) if df['efs_cov'][0] is not None else df['fit_params'][0]['fit_count']
        return CovidModelFit(
            tslices=df['tslices'][0]
            , fixed_efs=df['efs'][0][:-fit_count]
            , fitted_efs=df['efs'][0][-fit_count:]
            , efs_cov=df['efs_cov'][0]
            , fit_params=df['fit_params'][0]
            , tags=df['tags'][0]
            , model_params=df['model_params'][0])

    # the number of total ef values, including fixed values
    @property
    def ef_count(self):
        return len(self.tslices) - 1

    # the number of variables to be fit
    @property
    def fit_count(self):
        return self.ef_count - len(self.fixed_efs)

    @property
    def efs(self):
        return (list(self.fixed_efs) if self.fixed_efs is not None else []) + (list(self.fitted_efs) if self.fitted_efs is not None else [])

    # add a tag, to make fits easier to query
    def add_tag(self, tag_type, tag_value):
        self.tags[tag_type] = tag_value

    # runs the model using a given set of efs, and returns the modeled hosp values (which will be fit to actual hosp data)
    def run_model_and_get_total_hosps(self, ef_by_slice):
        extended_ef_by_slice = self.fixed_efs + list(ef_by_slice)
        self.model.set_ef_by_t(extended_ef_by_slice)
        self.model.solve_seir()
        return self.model.solution_sum('seir')['Ih']

    # the cost function: runs the model using a given set of efs, and returns the sum of the squared residuals
    def cost(self, ef_by_slice: list):
        modeled = self.run_model_and_get_total_hosps(ef_by_slice)
        res = [m - a if not np.isnan(a) else 0 for m, a in zip(modeled, list(self.actual_hosp))]
        c = sum(e**2 for e in res)
        return c

    # the cost function to be used for the particle swarm
    def ps_cost(self, xs):
        return sum(self.cost(x) for x in xs)

    # run an optimization to minimize the cost function using scipy.optimize.minimize()
    # method = 'curve_fit' or 'minimize'
    def run(self, engine, method='curve_fit', **model_params):

        if self.actual_hosp is None:
            self.actual_hosp = ExternalHosps(engine, self.model.datemin).fetch('emresource_hosps.csv')

        self.model = CovidModel(tslices=self.tslices, efs=self.fixed_efs + self.fit_params['efs0'], engine=engine)
        self.model.prep(**model_params)

        # run fit
        if self.model.efs != self.fixed_efs:
            if method == 'curve_fit':
                def func(trange, *efs):
                    return self.run_model_and_get_total_hosps(efs)
                self.fitted_efs, self.fitted_efs_cov = spo.curve_fit(
                    f=func
                    , xdata=self.model.trange
                    , ydata=self.actual_hosp[:len(self.model.trange)]
                    , p0=self.fit_params['efs0']
                    , bounds=([self.fit_params['ef_min']] * self.fit_count, [self.fit_params['ef_max']] * self.fit_count))
            elif method == 'minimize':
                minimization_results = spo.minimize(
                    lambda x: self.cost(x)
                    , self.fit_params['efs0']
                    , method='L-BFGS-B'
                    , bounds=[(self.fit_params['ef_min'], self.fit_params['ef_max'])] * self.fit_count
                    , options=self.fit_params)
                print(minimization_results)
                self.fitted_efs = minimization_results.x
            elif method == 'pswarm':
                options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
                bounds = ([self.fit_params['ef_min']] * self.fit_count, [self.fit_params['ef_max']] * self.fit_count)
                optimizer = ps.single.GlobalBestPSO(n_particles=self.fit_count * 2, dimensions=self.fit_count, bounds=bounds, options=options)
                cost, self.fitted_efs = optimizer.optimize(self.ps_cost, iters=10)
                print(self.fitted_efs)

            # self.model.set_ef_by_t(self.efs)

    # write the fit as a single row to covid_model.fits, describing the optimized ef values
    def write_to_db(self, engine):
        metadata = MetaData(schema='covid_model')
        metadata.reflect(engine, only=['fits'])
        fits_table = metadata.tables['covid_model.fits']

        if self.fit_params is not None:
            self.fit_params['fit_count'] = self.fit_count

        stmt = fits_table.insert().values(tslices=[int(x) for x in self.tslices],
                                          model_params=self.model.raw_params if self.model is not None else None,
                                          fit_params=self.fit_params,
                                          efs=list(self.efs),
                                          observed_efs=self.model.obs_ef_by_slice if self.model is not None else None,
                                          created_at=datetime.now(),
                                          tags=self.tags,
                                          efs_cov=[list(a) for a in self.fitted_efs_cov] if self.fitted_efs_cov is not None else None)

        conn = engine.connect()
        result = conn.execute(stmt)

        if self.model is not None:
            self.model.fit_id = result.inserted_primary_key[0]

        return result.inserted_primary_key[0]
