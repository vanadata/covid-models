import urllib.request as request
from covid_model.model import CovidModel
import pandas as pd
import numpy as np
import re
import copy
from db_utils.conn import db_engine
import datetime as dt
import matplotlib.pyplot as plt
import json


def run_model(engine, fit_id, tmax=600, tags=None, tc_shift=None, tc_shift_date=None, update_params=None):
    print(tags)
    model = CovidModel.from_fit(engine, fit_id, params='params.json')
    last_tc = model.ef_by_slice[-1]
    if tc_shift is not None:
        model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days, last_tc)
        model.add_tslice(tmax, last_tc + tc_shift)
    else:
        model.add_tslice(tmax, last_tc)
    if update_params:
        model.gparams.update(update_params)
    model.prep()
    model.solve_seir()
    model.write_to_db(engine, tags=tags)


def main():
    engine = db_engine()
    tmax = 600
    prior_fit_id = 905
    current_fit_id = 1026
    tc_shifts = [-0.07, -0.14]
    tc_shift_dates = [dt.datetime(2021, 4, 30), dt.datetime(2021, 5, 14), dt.datetime(2021, 5, 28)]
    vacc_caps = {
        'high vaccine uptake': {"0-19": 0.2046, "20-39": 0.80, "40-64": 0.80, "65+": 0.94},
        'low vaccine uptake': {"0-19": 0.1279, "20-39": 0.5, "40-64": 0.62, "65+": 0.94}}
    batch = 'standard_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    # prior fit
    run_model(engine, prior_fit_id, tmax=tmax, tags={'run_type': 'Prior', 'batch': batch})

    # current fit
    run_model(engine, current_fit_id, tmax=tmax, tags={'run_type': 'Current', 'batch': batch})

    # vacc cap scenarios
    for vacc_cap_label, vacc_cap in vacc_caps.items():
        change_params = {"max_vacc": vacc_cap}
        run_model(engine, current_fit_id, tmax=tmax, update_params=change_params, tags={'run_type': 'Vaccination Scenario', 'batch': batch, 'vacc_cap': vacc_cap_label})

    # tc shift scenarios
    for tcs in tc_shifts:
        for tcsd in tc_shift_dates:
            for vacc_cap_label, vacc_cap in vacc_caps.items():
                change_params = {"max_vacc": vacc_cap}
                run_model(engine, current_fit_id, tmax=tmax, tc_shift=tcs, tc_shift_date=tcsd, update_params=change_params,
                          tags={'run_type': 'TC Shift Projection', 'batch': batch, 'tc_shift': f'{int(100*tcs)}%', 'tc_shift_date': tcsd.strftime('%b %#d'), 'vacc_cap': vacc_cap_label})


if __name__ == '__main__':
    main()





