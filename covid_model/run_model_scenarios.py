from model import CovidModel
from db import db_engine
import datetime as dt
import pandas as pd
import numpy as np
import json
import argparse


def build_legacy_output_df(model: CovidModel):
    ydf = model.solution_ydf
    dfs_by_group = []
    for i, group in enumerate(model.groups):
        dfs_by_group.append(ydf.xs(group, level='group').rename(columns={var: var + str(i+1) for var in model.vars}))
    df = pd.concat(dfs_by_group, axis=1)

    totals = model.solution_ydf_summed
    df['Iht'] = totals['Ih']
    df['Dt'] = totals['D']
    df['Rt'] = totals['R'] + totals['RA']
    df['Itotal'] = totals['I'] + totals['A']
    df['Etotal'] = totals['E']
    df['Einc'] = df['Etotal'] / model.gparams['alpha']
    df['Vt'] = totals['V'] + totals['Vxd']
    # for i, group in enumerate(model.groups):
    #     group_df = ydf.xs(group, level='group')
    #     df[f'vacel{i}'] = (group_df['S'] + group_df['R'] + group_df['A']) / (model.gparams['groupN'][group] - (group_df['V'] + group_df['Ih'] + group_df['D']))
    df['immune'] = totals['R'] + totals['RA'] + totals['V'] + totals['Vxd']
    df['immune4'] = ydf.xs('65+', level='group')['R'] + ydf.xs('65+', level='group')['RA'] + ydf.xs('65+', level='group')['V'] + ydf.xs('65+', level='group')['Vxd']
    df['date'] = model.daterange
    df['Ilag'] = totals['I'].shift(3)
    df['Re'] = model.re_estimates
    df['prev'] = 100000.0 * df['Itotal'] / model.gparams['N']
    df['oneinX'] = model.gparams['N'] / df['Itotal']
    df['pimmune'] = 100.0 * df['Einc'].cumsum() / model.gparams['N']
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
    # parser.add_argument("-pvs", "--primary_vaccine_scen", choices=['high', 'low'], type=float, help="The name of the vaccine scenario to be used for the default model scenario.")
    run_params = parser.parse_args()

    engine = db_engine()

    # set various parameters
    tmax = run_params.days if run_params.days is not None else 700
    primary_vacc_scen = 'current trajectory'
    current_fit_id = run_params.current_fit_id if run_params.current_fit_id is not None else 1824
    prior_fit_id = run_params.prior_fit_id if run_params.prior_fit_id is not None else 1516
    tc_shifts = run_params.tc_shifts if run_params.tc_shifts is not None else [-0.07, -0.14]
    next_friday = dt.date.today() + dt.timedelta((4-dt.date.today().weekday()) % 7)
    # tc_shift_dates = [next_friday, next_friday + dt.timedelta(days=14), next_friday + dt.timedelta(days=28), dt.datetime(2021, 8, 15)]
    tc_shift_dates = [next_friday]
    tc_shift_dates = [dt.datetime.combine(d, dt.datetime.min.time()) for d in tc_shift_dates]
    tc_shift_days = 70
    batch = 'standard_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    # create models for low- and high-vaccine-uptake scenarios
    vacc_projection_params = json.load(open('input/vacc_proj_params.json'))
    models_by_vacc_scen = {}
    for vacc_scen, proj_params in vacc_projection_params.items():
        print(f'Building {vacc_scen} projection...')
        models_by_vacc_scen[vacc_scen] = CovidModel(params='input/params.json', tslices=[0, tmax], engine=engine)
        models_by_vacc_scen[vacc_scen].prep(vacc_proj_scen=vacc_scen)
        # models_by_vacc_scen[vacc_scen].write_vacc_to_csv(f'output/daily_vaccination_rates{"_with_lower_vacc_cap" if vacc_scen == "low vacc. uptake" else ""}.csv')

    # run model scenarios
    print('Running scenarios...')
    legacy_outputs = {}

    def run_model(model, fit_id, fit_tags=None, tc_shift=None, tc_shift_date=None):
        print('Scenario tags: ', fit_tags)
        model.set_ef_from_db(fit_id)
        current_ef = model.efs[-1]
        if tc_shift is not None:
            if tc_shift_days is None:
                model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days, current_ef + tc_shift)
            else:
                for i, tc_shift_for_this_day in enumerate(np.linspace(0, tc_shift, tc_shift_days)):
                    model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days + 1, current_ef + tc_shift_for_this_day)

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
    tags = {'run_type': 'Prior', 'batch': batch}
    prior_fit_model = CovidModel(params='input/params.json', tslices=[0, tmax], engine=engine)
    # prior_fit_model.gparams.update({'delta_vacc_escape': 0.0})
    # prior_fit_model.gparams['variants']['delta']['multipliers']['hosp'].update({"0-19": 2.52, "20-39": 2.52, "40-64": 2.52, "65+": 2.52})
    # print(prior_fit_model.gparams['delta_vacc_escape'], models_by_vacc_scen['high vacc. uptake'].gparams['delta_vacc_escape'])
    # print(prior_fit_model.gparams['variants']['delta']['multipliers']['hosp'], models_by_vacc_scen['high vacc. uptake'].gparams['variants']['delta']['multipliers']['hosp'])

    prior_fit_model.prep(vacc_proj_scen=primary_vacc_scen)
    run_model(prior_fit_model, prior_fit_id, fit_tags=tags)

    # vacc cap scenarios
    for vacc_scen in models_by_vacc_scen.keys():
        tags = {'run_type': 'Vaccination Scenario', 'batch': batch, 'vacc_cap': vacc_scen}
        run_model(models_by_vacc_scen[vacc_scen], current_fit_id, fit_tags=tags)

    # tc shift scenarios
    for tcs in tc_shifts:
        for tcsd in tc_shift_dates:
            for vacc_scen in models_by_vacc_scen.keys():
                tags = {'run_type': 'TC Shift Projection', 'batch': batch, 'tc_shift': f'{int(100 * tcs)}%',
                        'tc_shift_date': tcsd.strftime('%b %#d'), 'vacc_cap': vacc_scen}
                run_model(models_by_vacc_scen[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, fit_tags=tags)

    df = pd.concat(legacy_outputs)
    df.index.names = ['scenario', 'time']
    df.to_csv('output/allscenarios.csv')


if __name__ == '__main__':
    main()





