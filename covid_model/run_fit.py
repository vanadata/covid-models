from db import db_engine
from model import CovidModelFit
from data_imports import get_hosps
from charts import actual_hosps, total_hosps
import matplotlib.pyplot as plt
import datetime as dt
from time import perf_counter
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
    hosp_data = get_hosps(engine, dt.datetime(2020, 1, 24))

    # fetch external parameters to use for tslices and fixed efs
    fit = CovidModelFit.from_db(engine, fit_id)
    tslices = [int(x) for x in fit.tslices[:-3]]
    tslices += list(range(tslices[-1] + 14, len(hosp_data) - 1 - 13, 14)) + [len(hosp_data)]

    fit_count = fitted_tc_count
    fixed_efs = [float(x) for x in fit.efs[:(len(tslices) - 1 - fit_count)]]

    # run fits
    for i in range(len(tslices) - fit_count, len(tslices) - batch_size + 1):
        fit = CovidModelFit(tslices=tslices[:(i+batch_size)], fixed_efs=fixed_efs.copy(), actual_hosp=hosp_data, fit_params=vars(fit_params))

        t1_start = perf_counter()
        fit.run(engine)
        t1_stop = perf_counter()
        print(f'Transmission control fitting completed in {t1_stop - t1_start} seconds.')
        fixed_efs.append(fit.efs[i - 1])
        fit.write_to_db(engine)

    # run model with best_efs, and plot total hosps
    fit.model.solve_seir()
    print('t-slices: ', fit.model.tslices)
    print('TC by t-slice:', fit.efs)
    fit.model.write_to_db(db_engine())

    actual_hosps(engine)
    total_hosps(fit.model)
    plt.show()


if __name__ == '__main__':
    run()
