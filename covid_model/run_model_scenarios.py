from model import CovidModel
from db import db_engine
import datetime as dt
import pandas as pd
import numpy as np
import json
import argparse


def build_legacy_output_df(model: CovidModel):
    ydf = model.solution_sum(['seir', 'age']).stack(level='age')
    dfs_by_group = []
    for i, group in enumerate(model.attr['age']):
        dfs_by_group.append(ydf.xs(group, level='age').rename(columns={var: var + str(i+1) for var in model.attr['seir']}))
    df = pd.concat(dfs_by_group, axis=1)

    params_df = model.params_as_df

    totals = model.solution_sum('seir')
    by_vacc_status = model.solution_sum(['seir', 'vacc'])
    df['Iht'] = totals['Ih']
    df['Dt'] = totals['D']
    df['Rt'] = totals['R'] + totals['RA']
    df['Itotal'] = totals['I'] + totals['A']
    df['Etotal'] = totals['E']
    df['Einc'] = df['Etotal'] / model.raw_params['alpha']
    # df['Vt'] = by_vacc['mrna'] * model.params[0][('65+', 'mrna')]['vacc_eff'] + by_vacc['jnj'] * model.params[0][('65+', 'jnj')]['vacc_eff']
    # df['Vt'] = by_vacc_status['shot1'] * model.params[0][('65+', 'mrna')]['vacc_eff'] + by_vacc_status['jnj'] * model.params[0][('65+', 'jnj')]['vacc_eff']
    # df['immune'] = totals['R'] + totals['RA'] + df['Vt']
    df['date'] = model.daterange
    df['Ilag'] = totals['I'].shift(3)
    df['Re'] = model.re_estimates
    df['prev'] = 100000.0 * df['Itotal'] / model.raw_params['total_pop']
    df['oneinX'] = model.raw_params['total_pop'] / df['Itotal']
    df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df


def build_tc_df(model: CovidModel):
    return pd.DataFrame.from_dict({'time': model.tslices[:-1]
                                , 'tc_pb': model.efs
                                , 'tc': model.obs_ef_by_slice})


def tags_to_scen_label(tags):
    if tags['run_type'] == 'Current':
        return 'Current Fit'
    elif tags['run_type'] == 'Prior':
        return 'Prior Fit'
    elif tags['run_type'] == 'Vaccination Scenario':
        return f'Vaccine Scenario: {tags["vacc_cap"]}'
    elif tags['run_type'] == 'TC Shift Projection':
        return f'TC Shift Scenario: {tags["tc_shift"]} on {tags["tc_shift_date"]} ({tags["vacc_cap"]})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--current_fit_id", type=int, help="The current fit ID (run today)")
    parser.add_argument("-pf", "--prior_fit_id", type=int, help="The prior fit ID (run last week)")
    parser.add_argument("-d", "--days", type=int, help="Number of days to include in these scenarios, starting from Jan 24, 2020")
    parser.add_argument("-tcs", "--tc_shifts", nargs='+', type=float, help="Upcoming shifts in TC to simulate (-0.05 represents a 5% reduction in TC)")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    # parser.add_argument("-pvs", "--primary_vaccine_scen", choices=['high', 'low'], type=float, help="The name of the vaccine scenario to be used for the default model scenario.")
    run_params = parser.parse_args()

    engine = db_engine()

    # set various parameters
    tmax = run_params.days if run_params.days is not None else 700
    primary_vacc_scen = 'current trajectory'
    params_fname = run_params.params if run_params.params is not None else 'input/params.json'
    current_fit_id = run_params.current_fit_id if run_params.current_fit_id is not None else 1824
    prior_fit_id = run_params.prior_fit_id if run_params.prior_fit_id is not None else 1516
    tc_shifts = run_params.tc_shifts if run_params.tc_shifts is not None else [-0.05, -0.10]

    tc_shift_dates = [dt.date.today() + dt.timedelta(days=-3)]
    tc_shift_dates = [dt.datetime.combine(d, dt.datetime.min.time()) for d in tc_shift_dates]
    tc_shift_length = None
    tc_shift_dayss = [14, 56]
    batch = 'standard_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    # create models for low- and high-vaccine-uptake scenarios
    vacc_proj_dict = json.load(open('input/vacc_proj_params.json'))
    models_by_vacc_scen = {}
    for vacc_scen, proj_params in vacc_proj_dict.items():
        vacc_proj_params = vacc_proj_dict[vacc_scen]
        print(f'Building {vacc_scen} projection...')
        models_by_vacc_scen[vacc_scen] = CovidModel(tslices=[0, tmax], engine=engine)
        models_by_vacc_scen[vacc_scen].set_ef_from_db(current_fit_id)
        models_by_vacc_scen[vacc_scen].prep(params=params_fname, vacc_proj_params=vacc_proj_params)

    # run model scenarios
    print('Running scenarios...')
    legacy_outputs = {}

    def run_model(model, fit_id, fit_tags=None, tc_shift=None, tc_shift_date=None, tc_shift_length=None, tc_shift_days=None):
        print('Scenario tags: ', fit_tags)
        model.set_ef_from_db(fit_id)
        if tc_shift_days is not None and tc_shift_date is not None:
            model.tslices[-2] = min(model.tslices[-2], (tc_shift_date - model.datemin).days - 1)
        current_ef = model.efs[-1]
        if tc_shift is not None:
            if tc_shift_days is None:
                model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days, current_ef + tc_shift)
            else:
                for i, tc_shift_for_this_day in enumerate(np.linspace(0, tc_shift, tc_shift_days)):
                    model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days + i, current_ef + tc_shift_for_this_day)

            if tc_shift_length is not None:
                model.add_tslice(((tc_shift_date + dt.timedelta(days=tc_shift_length)) - dt.datetime(2020, 1, 24)).days, current_ef)

        model.solve_seir()
        model.write_to_db(tags=fit_tags, new_fit=True)
        legacy_outputs[tags_to_scen_label(fit_tags)] = build_legacy_output_df(model)
        return model

    # current fit
    tags = {'run_type': 'Current', 'batch': batch}
    run_model(models_by_vacc_scen[primary_vacc_scen], current_fit_id, fit_tags=tags)
    # output this one to it's own file
    build_legacy_output_df(models_by_vacc_scen[primary_vacc_scen]).to_csv('output/out2.csv')
    # and output the TCs to their own file
    build_tc_df(models_by_vacc_scen[primary_vacc_scen]).to_csv('output/tc_over_time.csv', index=False)

    # prior fit
    # tags = {'run_type': 'Prior', 'batch': batch}
    # prior_fit_model = CovidModel(tslices=[0, tmax], engine=engine)

    # prior_fit_model.prep(params=params_fname, vacc_proj_params=vacc_proj_dict[primary_vacc_scen])
    # run_model(prior_fit_model, prior_fit_id, fit_tags=tags)

    # vacc cap scenarios
    for vacc_scen in models_by_vacc_scen.keys():
        tags = {'run_type': 'Vaccination Scenario', 'batch': batch, 'vacc_cap': vacc_scen}
        run_model(models_by_vacc_scen[vacc_scen], current_fit_id, fit_tags=tags)
        # run_model(models_with_increased_under20_inf_prob[vacc_scen], current_fit_id, fit_tags={**tags, **{'tc_shift': 'increased under-18 transm.'}})

    # tc shift scenarios
    for tcs in tc_shifts:
        for tcsd in tc_shift_dates:
            for vacc_scen in models_by_vacc_scen.keys():
                for tc_shift_days in tc_shift_dayss:
                    tcsd_label = tcsd.strftime("%b %#d")
                    if tc_shift_days is not None:
                        tcsd_label += f' - {(tcsd + dt.timedelta(days=tc_shift_days)).strftime("%b %#d")}'
                    tags = {'run_type': 'TC Shift Projection', 'batch': batch, 'tc_shift': f'{int(100 * tcs)}%', 'tc_shift_date': tcsd_label, 'vacc_cap': vacc_scen}
                    run_model(models_by_vacc_scen[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, fit_tags=tags, tc_shift_length=tc_shift_length, tc_shift_days=tc_shift_days)
                    # run_model(models_with_increased_under20_inf_prob[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, tc_shift_length=tc_shift_length, fit_tags={**tags, **{'tc_shift': tags['tc_shift'] + '; increased under-18 transm.'}})

    # for vacc_scen in models_by_vacc_scen.keys():
    #     model = CovidModel(params='input/params.json', tslices=[0, tmax], engine=engine)
    #     model.gparams['rel_inf_prob'] = {'tslices': [577, 647], 'value': {'0-19': [1.0, 2.0, 2.0], '20-39': [1.0, 1.0, 1.0], '40-64': [1.0, 1.0, 1.0], '65+': [1.0, 1.0, 1.0]}}
    #     model.prep(vacc_proj_scen=vacc_scen)
    #     run_model(model, current_fit_id, tc_shift=0.05, tc_shift_date=dt.date(2021, 8, 27), tc_shift_length=70, fit_tags={}),
    # custom scenarios
    # for vacc_scen in models_by_vacc_scen.keys():,
    # for scen in models_for_custom_scenarios.keys():
    #     for vacc_scen in models_for_custom_scenarios[scen].keys():
    #         tags = {'run_type': 'Custom Projection', 'batch': batch, 'tc_shift': scen, 'vacc_cap': vacc_scen}
    #         run_model(models_for_custom_scenarios[scen][vacc_scen], current_fit_id, tc_shift, )

    df = pd.concat(legacy_outputs)
    df.index.names = ['scenario', 'time']
    df.to_csv('output/allscenarios.csv')


if __name__ == '__main__':
    main()





