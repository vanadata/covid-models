import json
import math
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import MetaData
from datetime import datetime
from time import perf_counter


# tslices and values define a stepwise function; get the value of that function for a given t
def get_value_from_slices(tslices, values, t):
    if len(values) != len(tslices) - 1:
        raise ValueError(f"Length of values ({len(values)}) must equal length of tslices ({len(tslices)}) - 1.")
    for i in range(len(tslices)-1):
        if tslices[i] <= t < tslices[i+1]:
            # if i > len(values):
            #     raise ValueError(f"")
            return values[i]
    raise ValueError(f"Cannot fetch value from slices because t={t} is out of range.")


# recursive function to process parameters that include values for different time slices, and construct params for a specific t
def get_params(input_params, t, tslices=None):
    if type(input_params) == list:
        value = get_value_from_slices([0]+tslices+[99999], input_params, t)
        return get_params(value, t, tslices)
    elif type(input_params) == dict:
        if 'tslices' in input_params.keys():
            return get_params(input_params['value'], t, tslices=input_params['tslices'])
        else:
            return {k: get_params(v, t, tslices) for k, v in input_params.items()}
    else:
        return input_params


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel:
    # the variables in the differential equation
    vars = ['S', 'E', 'I', 'Ih', 'A', 'R', 'RA', 'V', 'D']
    transitions = [
        ('s', 'e', 'rel_inf_prob'),
        ('e', 'i', 'pS'),
        ('i', 'h', 'hosp'),
        ('i', 'd', 'dnh'),
        ('h', 'd', 'dh')]

    # groups representing four different age groups
    groups = ['0-19', '20-39', '40-64', '65+']

    # the starting date of the model
    date_t0 = dt.datetime(2020, 1, 24)

    def __init__(self, params, tslices, ef_by_slice=None, fit_id=None, engine=None):
        self.tslices = list(tslices)

        # build global parameters from params file, creating dict lookups for easy access
        self.engine = engine
        self.gparams = params if type(params) == dict else json.load(open(params)) if params else None
        self.gparams_lookup = None
        self.vacc_params_df = None
        self.variant_params_df = None
        self.variant_prev = None
        self.variant_mults = None

        # build ef from given efs by slice, creating dict lookup for easy access
        self.ef_by_slice = ef_by_slice
        self.ef_by_t = None

        # the var values for the solution; these get populated when self.solve_seir is run
        self.solution = None
        self.solution_y = None
        self.solution_ydf = None

        # used to connect up with the matching fit in the database
        self.fit_id = fit_id

    def prep(self, vacc_rate_method='db'):
        self.set_gparams(self.gparams, vacc_rate_method)
        if self.ef_by_slice is not None:
            self.set_ef_by_t(self.ef_by_slice)

    # create using on a fit that was run previously or manually inserted into the database
    @staticmethod
    def from_fit(conn, fit_id, params=None):
        df = pd.read_sql_query(f"select * from stage.covid_model_fits where id = '{fit_id}'", con=conn, coerce_float=True)
        gparams = params if params is not None else df['model_params'][0]

        tslices = df['tslices'][0]
        efs = df['efs'][0]
        return CovidModel(gparams, tslices, ef_by_slice=efs, fit_id=df['id'][0], engine=conn)

    # coerce a "y", a list of length (num of vars)*(num of groups), into a dataframe with dims (num of groups)x(num of vars)
    @classmethod
    def y_to_df(cls, y):
        return pd.DataFrame(np.array(y).reshape(len(cls.groups), len(cls.vars)), columns=cls.vars, index=cls.groups)

    # handy properties for the beginning t, end t, and the full range of t values
    @property
    def tmin(self): return self.tslices[0]

    @property
    def tmax(self): return self.tslices[-1]

    @property
    def trange(self): return range(self.tmin, self.tmax)

    # sum the solution vars to create a single df with total values across all groups
    @property
    def solution_ydf_summed(self):
        return self.solution_ydf.groupby(level=['t']).sum()

    # calc the observed TC by applying the variant multiplier to the base TC
    @property
    def obs_ef_by_t(self):
        return {t: sum(1 - (1 - self.ef_by_t[t]) * self.gparams_lookup[t][group]['rel_inf_prob'] for group in self.groups) / len(self.groups) for t in self.trange}

    @property
    def obs_ef_by_slice(self):
        oef_by_t = self.obs_ef_by_t
        return [np.array([oef_by_t[t] for t in range(self.tslices[i], self.tslices[i+1])]).mean() for i in range(len(self.tslices) - 1)]

    def get_vacc_rates_by_t_from_vacc_params(self, vacc_params):
        rates = {}
        for t in self.trange:
            for vacc, specs in get_params(vacc_params, t).items():
                for group in self.groups:
                    rates[(t, group, vacc)] = specs['rate'][group]
        mux = pd.MultiIndex.from_tuples(rates.keys()).set_names(['t', 'group', 'vacc'])
        df = pd.DataFrame(list(rates.values()), index=mux, columns=['rate'])
        return df

    def get_vacc_rates_by_t_from_db(self):
        sql = f"""select 
            (extract(epoch from reporting_date - %s) / 60.0 / 60.0 / 24.0)::int as t
            , count_type as "group"
            , case date_type when 'vaccine dose 1/2' then 'mrna' when 'vaccine dose 1/1' then 'jnj' end as vacc
            , sum(total_count) as rate
        from cdphe.covid19_county_summary
        where date_type like 'vaccine dose 1/_'
        group by 1, 2, 3
        order by 1, 2, 3
        """
        df = pd.read_sql(sql, self.engine, params=(self.date_t0, ), index_col=['t', 'group', 'vacc'])

        # add projections
        max_t = df.index.get_level_values('t').max()
        if self.tmax >= max_t:
            # project rates based on the last 14 days of data
            projected_rates = df.loc[max_t - 13:].groupby(['group', 'vacc']).sum() / 14.0
            # used fixed value from params ('projected_rate'), if present
            projected_rates['rate'] = pd.DataFrame({vacc: specs['projected_rate'] for vacc, specs in self.gparams['vaccines'].items() if 'projected_rate' in specs.keys()}).stack().combine_first(projected_rates['rate'])
            # add projections to dataframe
            projections = pd.concat([projected_rates.assign(t=t) for t in range(max_t + 1, self.tmax + 1)]).set_index('t', append=True).reorder_levels(['t', 'group', 'vacc'])
            df = pd.concat([df, projections])

        return df, max_t + 1

    # build vaccine prevalence lookup
    def set_vacc_params(self, vacc_params, method='db'):
        # get base rates from CDPHE data, projecting future based on 2-week trajectory
        if method == 'db':
            rate_df, tproj = self.get_vacc_rates_by_t_from_db()
        elif method == 'rate_params':
            rate_df = self.get_vacc_rates_by_t_from_vacc_params(vacc_params)
            tproj = 430
        else:
            raise ValueError(f"method must be 'db' or 'rate_params'")

        # left join from an empty dataframe to fill in gaps
        full_empty_df = pd.DataFrame(index=pd.MultiIndex.from_product([self.trange, self.groups, vacc_params.keys()], names=['t', 'group', 'vacc']))
        rate_df = full_empty_df.join(rate_df, how='left').fillna(0)

        # when total vaccines given exceeds max_vacc * groupN, set rate to 0, and reallocate vaccines to younger group
        rate_df = rate_df.join(pd.DataFrame.from_dict(self.gparams['groupN'], orient='index', columns=['population']), on='group')
        cumu_vacc = {g: 0 for g in self.groups}
        for t in self.trange:
            for i in reversed(range(len(self.groups))):
                group = self.groups[i]
                current_rate = rate_df.loc[(t, group), 'rate'].sum()
                if t >= tproj and current_rate > 0:
                    max_rate = self.gparams['max_vacc_share_of_remaining'] * (self.gparams['max_vacc'][group] * self.gparams['groupN'][group] - cumu_vacc[group])
                    percent_excess = max((current_rate - max_rate) / current_rate, 0)
                    for vacc in self.gparams['vaccines'].keys():
                        reallocate = rate_df.loc[(t, group, vacc), 'rate'] * percent_excess
                        rate_df.loc[(t, group, vacc), 'rate'] -= reallocate
                        if i > 0:
                            rate_df.loc[(t, self.groups[i-1], vacc), 'rate'] += reallocate
                cumu_vacc[group] += rate_df.loc[(t, group), 'rate'].sum()

        # expand to have one row for each shot, and set the immun_gain based on the delay and efficacy of the vaccines
        shot_dfs = []
        shots = []
        for vacc, vacc_specs in vacc_params.items():
            for shot, shot_params in vacc_specs['shots'].items():
                shots.append((vacc, shot))
                # get the rates for this vacc
                shot_df = rate_df.xs(vacc, level='vacc').rename(columns={'rate': 'first_shot_rate'})
                # adjust t to reflect the day that the shot becomes effecive
                shot_df['rate'] = shot_df['first_shot_rate'].groupby(['group']).shift(shot_params['delay'])
                # add the efficacy, duration, and transition mults for this shot
                shot_df = shot_df.assign(eff=shot_params['eff'], duration=shot_params['duration'])
                for param, mult in shot_params['multipliers'].items():
                    shot_df[f'{param}_mult'] = mult
                # fill out the index and append to list of dataframes to concat
                shot_df = shot_df.assign(shot=shot, vacc=vacc).set_index(['vacc', 'shot'], append=True)
                shot_dfs.append(shot_df)
        # combine shot dfs
        df = pd.concat(shot_dfs).sort_index()

        # set immunity gain
        df['immun_gain'] = df['rate'] * df['eff']
        # set immunity loss as equal to the immunity gain [duration] days ago
        df['immun_loss'] = df.groupby(['group', 'vacc', 'shot'])['immun_gain'].apply(lambda x: x.shift(int(df.loc[x.index, 'duration'].iloc[0]), fill_value=0))
        # set prevalence, subtracting second_shot prevalence from first_shot prevalence
        df['s_prev'] = df.groupby(['group', 'vacc', 'shot'])['rate'].cumsum() / df['population']
        df.loc[(self.trange, self.groups, 'mrna', 'first_shot'), 's_prev'] = df.xs('first_shot', level='shot')['s_prev'] - df.xs('second_shot', level='shot')['s_prev']
        # set relative infection multiplier based on eff
        df['rel_inf_prob_mult'] = 1 - df.groupby(['t', 'group', 'vacc'])['eff'].cumsum()
        # set flows
        df = self.calc_multipliers(df)

        self.vacc_params_df = df
        sums = df[['immun_gain', 'immun_loss'] + [col for col in df.columns if '_flow' in col]].groupby(['t', 'group']).sum()
        sums = rate_df[['rate']].groupby(['t', 'group']).sum().join(sums, how='outer').fillna(0).rename(columns={'rate': 'first_shot_rate'})
        sums['first_shot_rate'] = rate_df['rate'].groupby(['t', 'group']).sum()
        sums['first_shot_cumu'] = sums['first_shot_rate'].groupby('group').cumsum()
        sums['jnj_first_shot_rate'] = rate_df['rate'].xs('jnj', level='vacc')
        sums['mrna_first_shot_rate'] = rate_df['rate'].xs('mrna', level='vacc')
        sums.rename(columns={col: col[:-5] + '_mult' for col in sums.columns if col[-5:] == '_flow'}).to_csv('daily_vaccination_rates.csv')

        # add to gparam lookups
        for t in self.trange:
            for group in self.groups:
                for param in ['immun_gain', 'immun_loss']:
                    self.gparams_lookup[t][group][f'vacc_{param}'] = sums.loc[(t, group), param]
                # for every transition except s -> e, multiply the transition param by the vacc mult
                for fr, to, label in self.transitions[1:]:
                    if param != 'rel_inf_prob':
                        self.gparams_lookup[t][group][label] *= sums.loc[(t, group), f'{label}_flow']

    def set_variant_params(self, variant_params):
        dfs = {}
        for variant, specs in variant_params.items():
            var_df = pd.read_csv(specs['theta_file_path'])  # file with at least a col "t" and a col containing variant prevalence
            var_df = var_df.rename(columns={specs['theta_column']: variant})[['t', variant]].set_index('t').rename(columns={variant: 'e_prev'})  # get rid of all columns except t (the index) and the prev value
            if 't_min' in specs.keys():
                var_df['e_prev'].loc[:specs['t_min']] = 0
            mult_df = pd.DataFrame(specs['multipliers'], index=self.groups).rename(columns={col: f'{col}_mult' for col in specs['multipliers'].keys()})
            mult_df.index = mult_df.index.rename('group')
            combined = pd.MultiIndex.from_product([var_df.index, mult_df.index], names=['t', 'group']).to_frame().join(var_df).join(mult_df).drop(columns=['t', 'group'])  # cross join
            dfs[variant] = combined
        df = pd.concat(dfs)
        df.index = df.index.set_names(['variant', 't', 'group']).reorder_levels(['t', 'group', 'variant'])
        df = df.sort_index()

        # calculate multipliers; run s -> e separately, because we're setting the same variant prevalences in S and E
        df['s_prev'] = df['e_prev']
        df = self.calc_multipliers(df, start_at=0, end_at=0)
        df = self.calc_multipliers(df, start_at=1, add_remaining=False)
        sums = df.groupby(['t', 'group']).sum()
        variant_tmin = sums.index.get_level_values('t').min()
        variant_tmax = sums.index.get_level_values('t').max()
        for t in self.trange:
            if t >= variant_tmin:
                for fr, to, label in self.transitions:
                    for group in self.groups:
                        # if t is greater than variant_tmax, just pull the multiplier at variant_tmax
                        self.gparams_lookup[t][group][label] *= sums.loc[(min(t, variant_tmax), group), f'{label}_flow']
                    if len(set(self.gparams_lookup[t][g][label] for g in self.groups)) == 1:
                        self.gparams_lookup[t][None][label] = self.gparams_lookup[t][self.groups[0]][label]

    # provide a dataframe with [compartment]_prev as the initial prevalence and this function will add the flows necessary to calc the downstream multipliers
    def calc_multipliers(self, df, start_at=0, end_at=10, add_remaining=True):
        if add_remaining:
            remaining = (1.0 - df[[f'{self.transitions[start_at][0]}_prev']].groupby(['t', 'group']).sum())
            remaining['vacc'] = 'none'
            remaining['shot'] = 'none'
            remaining = remaining.set_index(['vacc', 'shot'], append=True)
            df = df.append(remaining)
        df = df.sort_index()
        mult_cols = [col for col in df.columns if col[-5:] == '_mult']
        df[mult_cols] = df[mult_cols].fillna(1.0)
        for fr, to, label in self.transitions[start_at:(end_at+1)]:
            df[f'{label}_flow'] = df[f'{fr}_prev'] * df[f'{label}_mult']
            df[f'{to}_prev'] = df[f'{label}_flow'] / df[f'{label}_flow'].groupby(['t', 'group']).transform(sum)
        return df

    def set_generic_gparams(self, gparams):
        gparams_by_t = {t: get_params(gparams, t) for t in self.trange}
        for t in self.trange:
            for group in self.groups:
                # set "temp" to 1; if there is no "temp_on" parameter or temp_on == False, it will be 1
                self.gparams_lookup[t][group]['temp'] = 1
                self.gparams_lookup[t][group]['rel_inf_prob'] = 1.0
                for k, v in gparams_by_t[t].items():
                    # vaccines and variants are handled separately, so skip
                    if k in ['vaccines', 'variants']:
                        pass
                    # special rules for the "temp" paramater, which is set dynamically based on t
                    elif k == 'temp_on':
                        if v:
                            self.gparams_lookup[t][group]['temp'] = 0.5 * math.cos((t + 45) * 0.017) + 1.5
                    # for all other cases, if it's a dictionary, it should be broken out by group
                    elif type(v) == dict:
                        self.gparams_lookup[t][group][k] = v[group]
                    # if it's not a dict, it should be a single value: just assign that value to all groups
                    else:
                        self.gparams_lookup[t][group][k] = v
            # if all groups have the same value, create a None entry for that param
            self.gparams_lookup[t][None] = dict()
            for k, v in self.gparams_lookup[t][self.groups[0]].items():
                if type(v) != dict and len(set(self.gparams_lookup[t][g][k] for g in self.groups)) == 1:
                    self.gparams_lookup[t][None][k] = v

    # set global parameters and lookup dicts
    def set_gparams(self, params, vacc_rate_method='db'):
        # load gparams
        self.gparams = params if type(params) == dict else json.load(open(params))
        self.gparams_lookup = {t: {g: dict() for g in self.groups} for t in self.trange}
        # build a dictionary of gparams for every (t, group) for convenient access
        self.set_generic_gparams(self.gparams)
        # set vacc and variant params
        self.set_vacc_params(self.gparams['vaccines'], vacc_rate_method)
        if 'variants' in self.gparams and self.gparams['variants']:
            self.set_variant_params(self.gparams['variants'])

    # set ef by slice and lookup dicts
    def set_ef_by_t(self, ef_by_slice):
        self.ef_by_slice = ef_by_slice
        self.ef_by_t = {t: get_value_from_slices(self.tslices, list(ef_by_slice), t) for t in self.trange}

    # extend the time range with an additional slice; should maybe change how this works to return a new CovidModel instead
    def add_tslice(self, t, ef):
        if t <= self.tslices[-2]:
            raise ValueError('New tslices must be greater than the second to last tslices.')
        if t < self.tmax:
            tmax = self.tmax
            self.tslices[-1] = t
            self.tslices.append(tmax)
        else:
            self.tslices.append(t)
        self.ef_by_slice.append(ef)
        self.fit_id = None

    # this is the rate of flow from S -> E, based on beta, current prevalence, TC and a bunch of paramaters that should probably be deprecated
    @staticmethod
    def daily_transmission_per_susc(ef, I_total, A_total, rel_inf_prob, N, beta, temp, mask, lamb, siI, ramp, **excess_args):
        return beta * (1 - ef) * rel_inf_prob * temp * (
                I_total * (1 - (mask * 0.03)) * lamb * (1 - (siI + ramp)) + A_total * (1 - (mask * 0.2667))) / N

    # the diff eq for a single group; will be called four times in the actual diff eq
    @staticmethod
    def single_group_seir(single_group_y, transm_per_susc, vacc_immun_gain, vacc_immun_loss, alpha, gamma, pS, hosp, hlos, dnh, dh, groupN, dimmuneI=999999, dimmuneA=999999, **excess_args):

        S, E, I, Ih, A, R, RA, V, D = single_group_y

        daily_vacc_per_elig = vacc_immun_gain / (groupN - V - Ih - D)

        dS = - S * transm_per_susc + R / dimmuneI + RA / dimmuneA - S * daily_vacc_per_elig + vacc_immun_loss  # susceptible & not vaccine-immune
        dE = - E / alpha + S * transm_per_susc  # exposed
        dI = (E * pS) / alpha - I * gamma  # infectious & symptomatic
        dIh = I * hosp * gamma - Ih / hlos  # hospitalized (not considered infectious)
        dA = E * (1 - pS) / alpha - A * gamma  # infectious asymptomatic
        dR = I * (gamma * (1 - hosp - dnh)) + (1 - dh) * Ih / hlos - R / dimmuneI - R * daily_vacc_per_elig  # recovered from symp-not-hosp & immune & not vaccine-immune
        dRA = A * gamma - RA / dimmuneA - RA * daily_vacc_per_elig  # recovered from asymptomatic & immune & not vaccine-immune
        dV = (S + R + RA) * daily_vacc_per_elig - vacc_immun_loss  # vaccine-immune
        dD = dnh * I * gamma + dh * Ih / hlos  # death

        return dS, dE, dI, dIh, dA, dR, dRA, dV, dD

    # the differential equation, takes y, outputs dy
    def seir(self, t, y):
        ydf = CovidModel.y_to_df(y)

        # get param and ef values from lookup table
        t_int = min(math.floor(t), len(self.trange) - 1)
        params = self.gparams_lookup[t_int]
        ef = self.ef_by_t[t_int]

        # build dy
        dy = []
        transm = CovidModel.daily_transmission_per_susc(ef, I_total=ydf['I'].sum(), A_total=ydf['A'].sum(), **params[None])
        for group in self.groups:
            dy += CovidModel.single_group_seir(
                single_group_y=list(ydf.loc[group, :]),
                transm_per_susc=transm,
                **params[group])

        return dy

    # the initial values for y
    def y0(self):
        y = []
        for group in self.groups:
            # the first initial value for each group (presumably uninfected) is the population, which we get from gparams
            y.append(self.gparams['groupN'][group] - (1 if group == 'group1' else 0))
            # everything else is 0, except...
            y += [0] * (len(self.vars) - 1)
        # ...we start with one infection in the first group
        y[2] = 1
        return y

    # solve the diff eq using scipy.integrate.solve_ivp; put the solution in    self.solution_y (list) and self.solution_ydf (dataframe)
    def solve_seir(self):
        self.solution = spi.solve_ivp(fun=self.seir, t_span=[self.tmin, self.tmax], y0=self.y0(), t_eval=range(self.tmin, self.tmax))
        if not self.solution.success:
            raise RuntimeError(f'ODE solver failed with message: {self.solution.message}')
        self.solution_y = np.transpose(self.solution.y)
        self.solution_ydf = pd.concat([self.y_to_df(self.solution_y[t]) for t in self.trange], keys=self.trange, names=['t', 'group'])

    # count the total hosps by t as the sum of Ih and Ic
    def total_hosps(self):
        return self.solution_ydf_summed['Ih']

    # count the new exposed individuals by day
    def new_exposed(self):
        sum_df = self.solution_ydf_summed
        return sum_df['E'] - sum_df['E'].shift(1) + sum_df['E'].shift(1) / self.gparams['alpha']

    # plot the hospitalizations using matplotlib
    def plot_hosps(self, actual_hosp=None):
        if actual_hosp is not None:
            tmax = min(self.tmax, len(actual_hosp))
            plt.plot(self.trange[:tmax], actual_hosp[:tmax], 'r', label='Actual Hosp')
        plt.plot(self.trange, self.total_hosps(), label='Modeled Hosp')
        plt.legend(loc='best')
        plt.xlabel('Days')
        plt.grid()
        plt.show()

    # create a new fit and assign to this model
    def gen_fit(self, engine, label=None, tags=None):
        fit = CovidModelFit(self, label=label, tags=tags)
        fit.fit_params = None
        fit.best_efs = self.ef_by_slice
        fit.write_to_db(engine)

    # write to stage.covid_model_results in Postgres
    def write_to_db(self, engine, label=None, tags=None):
        # if there's no existing fit assigned, create a new fit and assign that one
        if self.fit_id is None or label is not None:
            self.gen_fit(engine, label, tags)

        # get the summed solution, add null index for the group, and then append to group solutions
        summed = self.solution_ydf_summed
        summed['group'] = None
        summed.set_index('group', append=True, inplace=True)
        df = pd.concat([self.solution_ydf, summed])

        # join the ef values onto the dataframe
        ef_series = pd.Series(self.ef_by_t, name='ef').rename_axis('t')
        oef_series = pd.Series(self.obs_ef_by_t, name='observed_ef').rename_axis('t')
        df = df.join(ef_series).join(oef_series)

        no_vacc_df = self.vacc_params_df.xs('none', level='vacc').xs('none', level='shot')
        for v in ['s', 'e', 'i', 'h', 'd']:
            df[f'vacc_prev_{v}'] = 1 - no_vacc_df[f'{v}_prev']

        # add fit_id and created_at date
        df['fit_id'] = self.fit_id
        df['created_at'] = dt.datetime.now()

        # write to database
        df.to_sql('covid_model_results'
                  , con=engine, schema='stage'
                  , index=True, if_exists='append', method='multi')


# class to find an optimal fit of transmission control (ef) values to produce model results that align with acutal hospitalizations
class CovidModelFit:

    def __init__(self, model: CovidModel, actual_hosp: list = None, fit_params=None, fixed_efs: list = None, label=None, tags=None):
        self.model = model
        self.actual_hosp = actual_hosp
        self.label = label
        self.tags = tags if tags else {}
        self.fixed_efs = list(fixed_efs) if fixed_efs else []
        self.results = None
        self.best_efs = None

        self.fit_params = {} if fit_params is None else fit_params

        # set fit_count
        if 'fit_count' not in self.fit_params.keys():
            self.fit_params['fit_count'] = self.ef_count

        # set fit param efs0
        if 'efs0' not in self.fit_params.keys():
            self.fit_params['efs0'] = [0.75] * self.fit_count
        elif type(self.fit_params['efs0']) in (float, int):
            self.fit_params['efs0'] = [self.fit_params['efs0']] * self.fit_count

        # set fit params ef_min and ef_max
        if 'ef_min' not in self.fit_params.keys():
            self.fit_params['ef_min'] = 0.3
        if 'ef_max' not in self.fit_params.keys():
            self.fit_params['ef_max'] = 0.99

    # the number of total ef values, including fixed values
    @property
    def ef_count(self):
        return len(self.model.tslices) - 1

    # the number of variables to be fit
    @property
    def fit_count(self):
        return self.ef_count - len(self.fixed_efs)

    # add a tag, to make fits easier to query
    def add_tag(self, tag_type, tag_value):
        self.tags[tag_type] = tag_value

    # the cost function: runs the model using a given set of efs, and returns the sum of the squared residuals
    def cost(self, ef_by_slice: list):
        extended_ef_by_slice = self.fixed_efs + list(ef_by_slice)
        self.model.set_ef_by_t(extended_ef_by_slice)
        t0 = perf_counter()
        self.model.solve_seir()
        t1 = perf_counter()
        # print(f'Cost function ran in {t1 - t0} secs.')
        modeled = self.model.total_hosps()
        res = [m - a if not np.isnan(a) else 0 for m, a in zip(modeled, list(self.actual_hosp))]
        c = sum(e**2 for e in res)
        return c

    # run an optimization to minimize the cost function using scipy.optimize.minimize()
    def run(self):
        if self.model.gparams_lookup is None:
            self.model.prep()
        # run minimization
        self.results = spo.minimize(
            lambda x: self.cost(x)
            , self.fit_params['efs0']
            , method='L-BFGS-B'
            , bounds=[(self.fit_params['ef_min'], self.fit_params['ef_max'])] * self.fit_count
            , options=self.fit_params)

        self.best_efs = list(self.fixed_efs) + list(self.results.x)
        self.model.set_ef_by_t(self.best_efs)

    # write the fit as a single row to stage.covid_model_fits, describing the optimized ef values
    def write_to_db(self, engine):
        metadata = MetaData(schema='stage')
        metadata.reflect(engine, only=['covid_model_fits'])
        fits_table = metadata.tables['stage.covid_model_fits']

        stmt = fits_table.insert().values(tslices=[int(x) for x in self.model.tslices],
                                    model_params=self.model.gparams,
                                    fit_params=self.fit_params,
                                    efs=list(self.best_efs),
                                    observed_efs=self.model.obs_ef_by_slice,
                                    created_at=datetime.now(),
                                    fit_label=self.label,
                                    tags=self.tags
                                          )

        conn = engine.connect()
        result = conn.execute(stmt)
        self.model.fit_id = result.inserted_primary_key[0]

