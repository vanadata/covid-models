from db import db_engine
from model import CovidModelFit, CovidModel
from data_imports import get_hosps
from charts import actual_hosps, total_hosps
import matplotlib.pyplot as plt
import datetime as dt
from time import perf_counter
import argparse


def run_fit(engine, fit_id, look_back, batch_size, model_params='input/params.json', fit_params={}):
    # load actual hospitalization data for fitting
    hosp_data = get_hosps(engine, dt.datetime(2020, 1, 24))

    # fetch external parameters to use for tslices and fixed efs
    fit = CovidModelFit.from_db(engine, fit_id)
    tslices = [int(x) for x in fit.tslices[:-3]]
    tslices += list(range(tslices[-1] + 14, len(hosp_data) - 1 - 13, 14)) + [len(hosp_data)]
    efs = [float(x) for x in fit.efs]

    # run fits
    for i in range(len(tslices) - look_back, len(tslices) - batch_size + 1):
        fit = CovidModelFit(tslices=tslices[:(i + batch_size)], fixed_efs=efs[:(i-1)], actual_hosp=hosp_data, fit_params={'efs0': (efs + [0.75]*999)[i:(i + batch_size)], **fit_params})

        t1_start = perf_counter()
        fit.run(engine, model_params=model_params)
        t1_stop = perf_counter()

        print(f'Transmission control fitting completed in {t1_stop - t1_start} seconds.')
        efs = fit.efs.copy()
        fit_id = fit.write_to_db(engine)

    # run model with best_efs
    fit.model.solve_seir()
    print('t-slices: ', fit.model.tslices)
    print('TC by t-slice:', fit.efs)

    return fit_id, fit


def run():
    # get fit params
    parser = argparse.ArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (14-day) windows to look back and refit; default to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    fit_params = parser.parse_args()
    look_back = fit_params.look_back if fit_params.look_back is not None else 3
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865

    engine = db_engine()
    fit_id, fit = run_fit(engine, fit_id, look_back, batch_size, fit_params=vars(fit_params))

    actual_hosps(engine)
    total_hosps(fit.model)
    plt.show()

    # actual_hosps(engine)
    #
    # fit1 = CovidModelFit.from_db(conn=engine, fit_id=fit_id)
    # model1 = CovidModel(fit1.model_params, [0, 600], engine=engine)
    # model1.set_ef_from_db(fit_id)
    # model1.prep()
    # model1.solve_seir()
    # total_hosps(model1)

    # plt.show()

    # load actual hospitalization data for fitting
    # engine = db_engine()
    # hosp_data = get_hosps(engine, dt.datetime(2020, 1, 24))
    #
    # # fetch external parameters to use for tslices and fixed efs
    # fit = CovidModelFit.from_db(engine, fit_id)
    # tslices = [int(x) for x in fit.tslices[:-3]]
    # tslices += list(range(tslices[-1] + 14, len(hosp_data) - 1 - 13, 14)) + [len(hosp_data)]
    #
    # fit_count = fitted_tc_count
    # fixed_efs = [float(x) for x in fit.efs[:(len(tslices) - 1 - fit_count)]]
    #
    # # run fits
    # for i in range(len(tslices) - fit_count, len(tslices) - batch_size + 1):
    #     fit = CovidModelFit(tslices=tslices[:(i+batch_size)], fixed_efs=fixed_efs.copy(), actual_hosp=hosp_data, fit_params=vars(fit_params))
    #
    #     t1_start = perf_counter()
    #     fit.run(engine)
    #     t1_stop = perf_counter()
    #     print(f'Transmission control fitting completed in {t1_stop - t1_start} seconds.')
    #     fixed_efs.append(fit.efs[i - 1])
    #     fit.write_to_db(engine)
    #
    # # run model with best_efs, and plot total hosps
    # fit.model.solve_seir()
    # print('t-slices: ', fit.model.tslices)
    # print('TC by t-slice:', fit.efs)
    # fit.model.write_to_db(db_engine())
    #
    # actual_hosps(engine)
    # total_hosps(fit.model)
    # plt.show()


if __name__ == '__main__':
    run()
