from model import CovidModelFit
from run_fit import run_fit
from db import db_engine
from charts import actual_hosps, modeled
import json
import matplotlib.pyplot as plt
import datetime as dt
import argparse
import pandas as pd

def run_sensitivity_tests(fit_id, alternate_params, refit_from_date=dt.datetime(2020, 4, 10), batch_size=3, add_base=False):
    engine = db_engine()
    base_params = CovidModelFit.from_db(engine, fit_id).model_params

    scens = json.load(open(alternate_params, 'r'))
    scen_params = {k: {**base_params, **v} for k, v in scens.items()}
    if add_base:
        scen_params['base'] = base_params

    fits = {}
    fit_ids = {}
    for label, params in scen_params.items():
        print(f'Running fit for scenario {label} with altered parameters {scens[label]}.')
        fit_ids[label], fits[label] = run_fit(engine, fit_id, batch_size=batch_size, look_back_date=refit_from_date, model_params=params)
        modeled(fits[label].model, compartments=['Ih'], label=label)

    pd.DataFrame.from_dict(fit_ids, orient='index').to_csv('output/sensitivity_tests_fit_ids.csv', header=False)
    actual_hosps(engine)
    plt.legend(loc='best')
    plt.savefig('output/sensitivity_tests_chart.png')


def main():
    # get fit params
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    parser.add_argument("-ap", "--alternate_params", type=str, help="file path for json indicating which parameters to vary for which scenarios")
    parser.add_argument("-lbd", "--look_back_date", type=str, help="the date (YYYY-MM-DD) from which we want to refit; default to using -lb, which defaults to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (14-day) windows to fit in each batch; default to running everything in one batch")
    fit_params = parser.parse_args()
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865
    alternate_params = fit_params.alternate_params if fit_params.alternate_params is not None else 'input/alternate_params.json'
    look_back_date = dt.datetime.strptime(fit_params.look_back_date, '%Y-%m-%d')
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else 3

    run_sensitivity_tests(fit_id, alternate_params, refit_from_date=look_back_date, batch_size=batch_size)


if __name__ == '__main__':
    main()


