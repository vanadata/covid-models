import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from covid_model.db import db_engine
from covid_model.model import CovidModelFit, CovidModel
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates
from time import perf_counter
from timeit import timeit
from vacc_scen_comparison import build_mab_prevalence

if __name__ == '__main__':
    engine = db_engine()

    fig, ax = plt.subplots()
    print('Prepping model...')
    model = CovidModel([0, 800], engine=engine)
    model.set_ef_from_db(1961)
    print(timeit('model.prep()', number=1, globals=globals()), 'seconds to prep model.')

    # for method in ['RK45', 'RK23', 'BDF', 'LSODA', 'Radau', 'DOP853']:
    for method in ['RK45']:
        print(timeit('model.solve_seir(method=method)', number=1, globals=globals()), 'seconds to run model.')

    # mmwr_infections_growth_rate(model)
    # re_estimates(model)
    # model.efs[11] -= 0.03
    # model.set_ef_by_t(model.efs)
    # model.set_param('betta', mult=0.95, trange=range(200, 230))
    # model.rebuild_ode_with_new_tc()
    # model.solve_seir()
    # mmwr_infections_growth_rate(model)
    # re_estimates(model)

    modeled(model, 'Ih')
    actual_hosps(engine)
    plt.show()
