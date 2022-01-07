import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates
from time import perf_counter
from timeit import timeit

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications

if __name__ == '__main__':
    engine = db_engine()

    model = CovidModel()

    print('Prepping model...')
    print(timeit('model.prep(417, engine=engine)', number=1, globals=globals()), 'seconds to prep model.')
    # model.prep(417, engine=engine)

    # , vacc_proj_params = json.load(open('input/vacc_proj_params.json'))['current trajectory'],
    # vacc_immun_params = 'input/vacc_immun_params.json',
    # param_multipliers = 'input/param_multipliers.json',
    # variant_prevalence = 'input/variant_prevalence.csv',
    # mab_prevalence = 'input/mab_prevalence.csv'

    # for method in ['RK45', 'RK23', 'BDF', 'LSODA', 'Radau', 'DOP853']:
    for method in ['RK45']:
        print(timeit('model.solve_seir(method=method)', number=1, globals=globals()), 'seconds to run model.')

    # print(timeit('model.write_to_db(engine)', number=1, globals=globals()), 'seconds to write to database.')

    vals_json_attr = 'seir'
    cmpts_json_attrs = ('age', 'vacc')
    print(model.solution_sum([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs).index.droplevel('t').to_frame().to_dict(orient='records'))
    exit()

    model.write_to_db(engine)

    fig, ax = plt.subplots()
    modeled(model, 'Ih')
    actual_hosps(engine)
    plt.show()
