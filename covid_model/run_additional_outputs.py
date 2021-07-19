import pandas as pd
import numpy as np
import datetime as dt
import json
from db import db_engine
from data_imports import get_hosps_df, get_vaccinations_by_county, get_vaccinations

if __name__ == '__main__':
    engine = db_engine()

    # export actual hospitalizations
    print('Exporting hospitalizations...')
    hosps = pd.DataFrame(get_hosps_df(engine)).reset_index().rename(columns={'currently_hospitalized': 'Iht', 'measure_date': 'date'})
    hosps['time'] = ((pd.to_datetime(hosps['date']) - dt.datetime(2020, 1, 24)) / np.timedelta64(1, 'D')).astype(int)
    hosps[['time', 'date', 'Iht']].to_csv('output/CO_EMR_Hosp.csv', index=False)

    # export vaccinations by county, with projections
    gparams = json.load(open('input/params.json', 'r'))
    proj_param_dict = json.load(open('input/vacc_proj_params.json', 'r'))
    vacc_df_dict = {}
    for label, proj_params in proj_param_dict.items():
        print(f'Exporting vaccination by age for "{label}" scenario...')
        df = get_vaccinations(engine
                             , from_date=dt.datetime(2020, 1, 24)
                             , proj_to_date=dt.datetime(2021, 12, 31)
                             , proj_lookback=proj_params['lookback']
                             , proj_fixed_rates=proj_params['fixed_rates'] if 'fixed_rates' in proj_params.keys() else None
                             , max_cumu={g: gparams['groupN'][g] * proj_params['max_cumu'][g] for g in gparams['groupN'].keys()}
                             , max_rate_per_remaining=proj_params['max_rate_per_remaining']
                             , realloc_priority=proj_params['realloc_priority'])
        vacc_df_dict[label] = df.groupby(['measure_date', 'group']).sum().rename(columns={'rate': 'first_shot_rate'})
        vacc_df_dict[label]['is_projected'] = vacc_df_dict[label]['is_projected'] > 0

    vacc_df = pd.concat(vacc_df_dict).rename_axis(index=['vacc_scen', 'measure_date', 'group']).sort_index()
    vacc_df['first_shot_cumu'] = vacc_df['first_shot_rate'].groupby(['vacc_scen', 'group']).cumsum()
    vacc_df = vacc_df.join(pd.Series(gparams['groupN']).rename('population').rename_axis('group'))
    vacc_df['cumu_share_of_population'] = vacc_df['first_shot_cumu'] / vacc_df['population']
    vacc_df = vacc_df.join(vacc_df[~vacc_df['is_projected']]['first_shot_cumu'].groupby(['vacc_scen', 'group']).max().rename('current_first_shot_cumu'))
    vacc_df.to_csv('output/daily_vaccination_by_age.csv')

    print('Exporting vaccination by county...')
    vacc_by_county = get_vaccinations_by_county(engine)
    vacc_by_county.to_csv('output/daily_vaccination_by_age_by_county.csv', index=False)
