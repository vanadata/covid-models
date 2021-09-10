from db import db_engine
from model import CovidModelFit, CovidModel
from data_imports import get_hosps
from charts import actual_hosps, total_hosps
import matplotlib.pyplot as plt
import datetime as dt
from time import perf_counter
import argparse


def run_fit(engine, fit_id, look_back=0, batch_size=0, look_back_date=None, tags=None, fit_params={}, **model_params):
    # load actual hospitalization data for fitting
    hosp_data = get_hosps(engine, dt.datetime(2020, 1, 24))

    # fetch external parameters to use for tslices and fixed efs
    fit = CovidModelFit.from_db(engine, fit_id)
    tslices = [int(x) for x in fit.tslices[:-3]]
    tslices += list(range(tslices[-1] + 14, len(hosp_data) - 1 - 13, 14)) + [len(hosp_data)]
    efs = [float(x) for x in fit.efs]

    # if_look_back_to_date is not None then set look_back
    if look_back_date is not None:
        look_back = sum(1 for tslice in tslices if tslice > (look_back_date - CovidModel.datemin).days)

    # run fits
    tags = tags if tags is not None else {}
    for i in range(len(tslices) - look_back, len(tslices) - min(batch_size, look_back) + 1):
        fit_type = 'final' if i == len(tslices) - batch_size else 'intermediate'
        fit = CovidModelFit(tslices=tslices[:(i + batch_size)], fixed_efs=efs[:(i-1)], actual_hosp=hosp_data
                            , fit_params={'efs0': (efs + [0.75]*999)[i:(i + batch_size)], **fit_params}
                            , tags={'fit_type': fit_type, **(tags if tags is not None else {})})

        t1_start = perf_counter()
        fit.run(engine, **model_params)
        t1_stop = perf_counter()

        print(f'Transmission control fitting completed in {t1_stop - t1_start} seconds.')
        efs = fit.efs.copy()
        fit_id = fit.write_to_db(engine)

    # run model with best_efs
    fit.model.solve_seir()
    print('t-slices:', fit.model.tslices)
    print('TC by t-slice:', fit.efs)

    return fit_id, fit


def run():
    # get fit params
    parser = argparse.ArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (14-day) windows to look back and refit; default to 3")
    parser.add_argument("-lbd", "--look_back_date", type=str, help="the date (YYYY-MM-DD) from which we want to refit; default to using -lb, which defaults to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    fit_params = parser.parse_args()
    look_back = fit_params.look_back if fit_params.look_back is not None else 3
    look_back_date = dt.datetime.strptime(fit_params.look_back_date, '%Y-%m-%d') if fit_params.look_back_date else None
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865
    params = fit_params.params if fit_params.params is not None else 'input/params.json'

    engine = db_engine()
    fit_id, fit = run_fit(engine, fit_id, look_back, batch_size, params=params, look_back_date=look_back_date, fit_params=vars(fit_params))

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
