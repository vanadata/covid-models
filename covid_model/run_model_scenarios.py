from model import CovidModel, CovidModelFit
from db import db_engine
import datetime as dt
import pandas as pd


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
    df['Vt'] = totals['V']
    # for i, group in enumerate(model.groups):
    #     group_df = ydf.xs(group, level='group')
    #     df[f'vacel{i}'] = (group_df['S'] + group_df['R'] + group_df['A']) / (model.gparams['groupN'][group] - (group_df['V'] + group_df['Ih'] + group_df['D']))
    df['immune'] = totals['R'] + totals['RA'] + totals['V']
    df['immune4'] = ydf.xs('65+', level='group')['V'] + ydf.xs('65+', level='group')['R']
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
    engine = db_engine()

    # get prior and current fits
    tmax = 600
    current_fit_id = 1318
    prior_fit_id = 1225
    engine = db_engine()
    tc_shifts = [-0.07, -0.14]
    tc_shift_dates = [dt.datetime(2021, 6, 4), dt.datetime(2021, 6, 18), dt.datetime(2021, 7, 2)]
    batch = 'standard_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    # create models for low- and high-vaccine-uptake scenarios
    print('Building high-uptake vaccine projection...')
    hvu_model = CovidModel(params='input/params.json', tslices=[0, tmax], engine=engine)
    hvu_model.prep(vacc_proj_scen='high-uptake')
    hvu_model.write_vacc_to_csv('output/daily_vaccination_rates.csv')
    print('Building low-uptake vaccine projection...')
    lvu_model = CovidModel(params='input/params.json', tslices=[0, tmax], engine=engine)
    lvu_model.prep(vacc_proj_scen='low-uptake')
    lvu_model.write_vacc_to_csv('output/daily_vaccination_rates_with_lower_vacc_cap.csv')
    vacc_scens = {'high-uptake': hvu_model, 'low-uptake': lvu_model}

    # run model scenarios
    print('Running scenarios...')
    legacy_outputs = {}

    def run_model(model, fit_id, fit_tags=None, tc_shift=None, tc_shift_date=None):
        print('Scenario tags: ', fit_tags)
        model.set_ef_from_db(fit_id)
        if tc_shift is not None:
            model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days, model.efs[-1] + tc_shift)
        model.solve_seir()
        model.write_to_db(tags=fit_tags, new_fit=True)
        legacy_outputs[tags_to_scen_label(fit_tags)] = build_legacy_output_df(model)
        return model

    # current fit
    tags = {'run_type': 'Current', 'batch': batch}
    run_model(hvu_model, current_fit_id, fit_tags=tags)
    # output this one to it's own file
    build_legacy_output_df(hvu_model).to_csv('output/out2.csv')
    # and output the TCs to their own file
    build_tc_df(hvu_model).to_csv('output/tc_over_time.csv', index=False)

    # prior fit
    tags = {'run_type': 'Prior', 'batch': batch}
    run_model(hvu_model, prior_fit_id, fit_tags=tags)

    # vacc cap scenarios
    for vacc_scen, model_w_vacc_scen in vacc_scens.items():
        tags = {'run_type': 'Vaccination Scenario', 'batch': batch, 'vacc_cap': vacc_scen}
        run_model(model_w_vacc_scen, current_fit_id, fit_tags=tags)

    # tc shift scenarios
    for tcs in tc_shifts:
        for tcsd in tc_shift_dates:
            for vacc_scen, model_w_vacc_scen in vacc_scens.items():
                tags = {'run_type': 'TC Shift Projection', 'batch': batch, 'tc_shift': f'{int(100 * tcs)}%',
                        'tc_shift_date': tcsd.strftime('%b %#d'), 'vacc_cap': vacc_scen}
                run_model(model_w_vacc_scen, current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, fit_tags=tags)

    df = pd.concat(legacy_outputs)
    df.index.names = ['scenario', 'time']
    df.to_csv('output/allscenarios.csv')


if __name__ == '__main__':
    main()





