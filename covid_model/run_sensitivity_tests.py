from run_fit import run_fit
from db import db_engine
from charts import actual_hosps, modeled
import json
import matplotlib.pyplot as plt


def main():
    base_params = json.load(open('input/params.json', 'r'))

    scens = json.load(open('input/alternate_fit_params.json', 'r'))
    scen_params = {k: {**base_params, **v} for k, v in scens.items()}
    scen_params['base'] = base_params

    engine = db_engine()
    fits = {}
    fit_ids = {}
    for label, params in scen_params.items():
        fit_ids[label], fits[label] = run_fit(engine, 2937, 33, 3, model_params=params)
        print(fit_ids)

        modeled(fits[label].model, compartments=['Ih'], label=label)

    actual_hosps(engine)

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()


