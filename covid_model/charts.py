from hospitalizations import get_hosps
from model import CovidModel, CovidModelFit
from db import db_engine
import scipy.stats as sps
import matplotlib.pyplot as plt
import copy


def actual_hosps(engine, **plot_params):
    hosps = get_hosps(engine)
    plt.plot(range(0, len(hosps)), get_hosps(engine), **{'color': 'red', 'label': 'Actual Hosps.', **plot_params})


def total_hosps(model, **plot_params):
    plt.plot(model.trange, model.total_hosps(), {'c': 'blue', 'label': 'Modeled Hosps.', **plot_params})


# UQ sqaghetti plot
def uq_spaghetti(fit, sample_n=100, tmax=600, **plot_params):
    # get sample TC values
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
    samples = fitted_efs_dist.rvs(sample_n)

    # for each sample, solve the model and add a line to the plot
    model = copy.copy(fit.model)
    model.add_tslice(tmax, 0)
    model.prep()
    for sample_fitted_efs in samples:
        model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [sample_fitted_efs[-1]])
        model.solve_seir()
        plt.plot(model.trange, model.total_hosps(), **{'color': 'darkblue', 'alpha': 0.03, **plot_params})


if __name__ == '__main__':
    engine = db_engine()
    # model = CovidModel.from_fit(engine, 1133)
    # mp = ModelPlotter(trange=range(0, 600))

    actual_hosps(engine)
    uq_spaghetti(CovidModelFit.from_db(engine, 1133), sample_n=10)

    plt.legend(loc='best')
    plt.xlabel('Days')
    plt.grid()
    plt.show()
