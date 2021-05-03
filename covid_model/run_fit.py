from db import db_engine
from model import CovidModel, CovidModelFit
import pandas as pd
import datetime as dt
import numpy as np
import re
from time import perf_counter
import urllib.request as request
import json as json
import argparse


def run():
    # get fit params
    parser = argparse.ArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (14-day) windows to look back and refit; default to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    fit_params = parser.parse_args()
    fitted_tc_count = fit_params.look_back if fit_params.look_back is not None else 3
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else fitted_tc_count
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865

    # load actual hospitalization data for fitting
    engine = db_engine()
    actual_hosp_df = pd.read_sql('select * from cdphe.emresource_hospitalizations', engine)
    actual_hosp_df['t'] = ((pd.to_datetime(actual_hosp_df['measure_date']) - dt.datetime(2020, 1, 24)) / np.timedelta64(1, 'D')).astype(int)
    actual_hosp_tmin = actual_hosp_df[actual_hosp_df['currently_hospitalized'].notnull()]['t'].min()
    actual_hosp_tmax = actual_hosp_df[actual_hosp_df['currently_hospitalized'].notnull()]['t'].max()
    hosp_data = [0]*actual_hosp_tmin + list(actual_hosp_df['currently_hospitalized'])

    # fetch external parameters to use for tslices and fixed efs
    model = CovidModel.from_fit(engine, fit_id)
    tslices = [int(x) for x in model.tslices[:-3]]
    tslices += list(range(tslices[-1] + 14, actual_hosp_tmax - 19, 14)) + [actual_hosp_tmax + 1]

    fit_count = fitted_tc_count
    fixed_efs = [float(x) for x in model.ef_by_slice[:(len(tslices) - 1 - fit_count)]]

    # define model
    for i in range(len(tslices) - fit_count, len(tslices) - batch_size + 1):
        model = CovidModel(
            params='params.json',
            tslices=tslices[:(i+batch_size)])
        model.engine = engine
        model.prep()
        fit = CovidModelFit(model, hosp_data, fixed_efs=fixed_efs, fit_params=vars(fit_params))

        # run fit (or set hard-coded best_efs)
        t1_start = perf_counter()
        fit.run()
        t1_stop = perf_counter()
        print(f'Transmission control fitting completed in {t1_stop - t1_start} seconds.')
        print(fit.results)
        fixed_efs.append(fit.best_efs[i-1])
        fit.write_to_db(engine)

    # run model with best_efs, and plot total hosps
    model.solve_seir()
    print('t-slices: ', model.tslices)
    print('TC by t-slice:', fit.best_efs)
    model.write_to_db(db_engine())
    model.plot_hosps(hosp_data)


if __name__ == '__main__':
    run()
