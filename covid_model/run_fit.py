from db import db_engine
from model import CovidModelFit, CovidModel
from data_imports import get_hosps
from analysis.charts import actual_hosps, modeled
import matplotlib.pyplot as plt
import datetime as dt
from time import perf_counter
import argparse


def run_fit(engine, fit_id, look_back=None, batch_size=1, look_back_date=None, tags=None, fit_params={}, tc_window_size=14, **model_params):
    # load actual hospitalization data for fitting
    hosp_data = get_hosps(engine, dt.datetime(2020, 1, 24))

    # fetch external parameters to use for tslices and fixed efs
    base_fit = CovidModelFit.from_db(engine, fit_id)
    tslices = [int(x) for x in base_fit.tslices[:(-look_back) if look_back is not None else 1]]
    tslices += list(range(tslices[-1] + tc_window_size, len(hosp_data) - tc_window_size, tc_window_size)) + [len(hosp_data)]
    efs = [float(x) for x in base_fit.efs]

    # if_look_back_to_date is not None then set look_back
    if look_back_date is not None:
        look_back = sum(1 for tslice in tslices if tslice > (look_back_date - CovidModel.datemin).days)
    if look_back is None:
        look_back = len(tslices) - 1

    # run fits
    tags = tags if tags is not None else {}

    mock_model = CovidModel(tslices=tslices, engine=engine)
    mock_model.prep(**model_params)

    for i in range(len(tslices) - look_back, len(tslices) - batch_size + 1):
        fit_type = 'final' if i == len(tslices) - batch_size else 'intermediate'
        base_fit = CovidModelFit(tslices=tslices[:(i + batch_size)], fixed_efs =efs[:(i-1)], actual_hosp=hosp_data
                            , fit_params={'efs0': (efs + [0.75]*999)[i:(i + batch_size)], **fit_params}
                            , tags={'fit_type': fit_type, **(tags if tags is not None else {})})

        t1_start = perf_counter()
        base_fit.run(engine, base_model=mock_model, **model_params)
        t1_stop = perf_counter()

        print(f'Transmission control fit {i - len(tslices) + look_back + 1}/{look_back - batch_size + 1} completed in {t1_stop - t1_start} seconds.')
        efs = base_fit.efs.copy()
        fit_id = base_fit.write_to_db(engine)

    base_fit.model.solve_seir()
    base_fit.model.write_to_db(engine)
    print('t-slices:', base_fit.model.tslices)
    print('TC by t-slice:', base_fit.model.efs)

    return fit_id, base_fit


def run():
    # get fit params
    parser = argparse.ArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; default to 3")
    parser.add_argument("-lbd", "--look_back_date", type=str, help="the date (YYYY-MM-DD) from which we want to refit; default to using -lb, which defaults to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window")
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    look_back_date = dt.datetime.strptime(fit_params.look_back_date, '%Y-%m-%d') if fit_params.look_back_date else None
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    params = fit_params.params if fit_params.params is not None else 'input/params.json'

    engine = db_engine()
    fit_id, fit = run_fit(engine, fit_id, look_back, batch_size, params=params, look_back_date=look_back_date, fit_params=vars(fit_params), tc_window_size=window_size)

    actual_hosps(engine)
    modeled(fit.model, 'Ih')
    plt.show()


if __name__ == '__main__':
    run()
