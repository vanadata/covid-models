import numpy as np
import pandas as pd
from collections import OrderedDict
# from model import CovidModel
from db import db_engine
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
from time import perf_counter
import scipy.integrate as spi


class ODEFlowTerm:
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t):
        self.from_cmpt_idx = from_cmpt_idx
        self.to_cmpt_idx = to_cmpt_idx
        self.coef_by_t = coef_by_t
        self.scale_by_cmpt_idxs = scale_by_cmpts_idxs
        self.scale_by_cmpts_coef_by_t = scale_by_cmpts_coef_by_t if scale_by_cmpts_coef_by_t is not None else np.array([1]*len(self.scale_by_cmpt_idxs))

        self.solution = None
        self.solution_y = None
        self.solution_ydf = None

    def dy(self, t, y):
        dy = np.zeros(len(y))
        # if self.coef_by_t[t] > 1:
        #     print(self.coef_by_t[t])
        # if (self.coef_by_t[t] * sum(y.take(self.scale_by_cmpt_idxs) * self.scale_by_cmpts_coef_by_t[t]) if self.scale_by_cmpt_idxs else 1) > 1:
        #     print(self.coef_by_t[t] * sum(y.take(self.scale_by_cmpt_idxs) * self.scale_by_cmpts_coef_by_t[t]) if self.scale_by_cmpt_idxs else 1)
        dy[self.from_cmpt_idx] -= y[self.from_cmpt_idx] * self.coef_by_t[t] * (sum(y.take(self.scale_by_cmpt_idxs) * self.scale_by_cmpts_coef_by_t[t]) if self.scale_by_cmpt_idxs else 1)
        dy[self.to_cmpt_idx] += y[self.from_cmpt_idx] * self.coef_by_t[t] * (sum(y.take(self.scale_by_cmpt_idxs) * self.scale_by_cmpts_coef_by_t[t]) if self.scale_by_cmpt_idxs else 1)
        return dy


class ODEBuilder:

    def __init__(self, trange, attributes: OrderedDict, params=None):
        self.trange = trange

        self.attributes = attributes
        idx = pd.MultiIndex.from_product(attributes.values(), names=attributes.keys())
        self.indices = pd.Series(index=idx, data=range(len(idx)))

        self.params = params

        # self.jacobians = {t: np.zeros((self.length, self.length)) for t in self.trange}
        self.linear_term_matrix_dict = {t: np.zeros((self.length, self.length)) for t in self.trange}
        self.linear_term_matrix_dict = {t: np.zeros((self.length, self.length)) for t in self.trange}
        self.terms = []

    @property
    def length(self):
        return len(self.indices)

    def attr_level(self, attr_name):
        return list(self.attributes.keys()).index(attr_name)

    def calc_coef_by_t(self, coef, param_cmpt):
        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in self.trange}
        elif callable(coef):
            return {t: coef(t) for t in self.trange}
        elif isinstance(coef, str):
            coef_by_t = {}
            expr = parse_expr(coef)
            relevant_params = [str(s) for s in expr.free_symbols]
            func = sym.lambdify(relevant_params, expr)
            for t in self.trange:
                coef_by_t[t] = func(**{k: v for k, v in self.params[t][param_cmpt].items() if k in relevant_params})
            return coef_by_t
        else:
            return {t: coef for t in self.trange}

    def add_flow(self, from_cmpt, to_cmpt, coef, scale_by_cmpts=None, scale_by_cmpts_coef=None):
        if len(from_cmpt) < len(self.attributes.keys()):
            raise ValueError(f'Origin compartment `{from_cmpt}` does not have the right number of attributes.')
        if len(to_cmpt) < len(self.attributes.keys()):
            raise ValueError(f'Destination compartment `{to_cmpt}` does not have the right number of attributes.')
        if scale_by_cmpts is not None:
            for cmpt in scale_by_cmpts:
                if len(cmpt) < len(self.attributes.keys()):
                    raise ValueError(f'Scaling compartment `{cmpt}` does not have the right number of attributes.')

        self.terms.append(ODEFlowTerm(
            from_cmpt_idx=self.indices[from_cmpt],
            to_cmpt_idx=self.indices[to_cmpt],
            coef_by_t=self.calc_coef_by_t(coef, from_cmpt),
            scale_by_cmpts_idxs=[self.indices[cmpt] for cmpt in scale_by_cmpts] if scale_by_cmpts is not None else [],
            scale_by_cmpts_coef_by_t=pd.DataFrame([self.calc_coef_by_t(c, from_cmpt) for c in scale_by_cmpts_coef]).to_dict(orient='list') if scale_by_cmpts_coef is not None else {t: 1 for t in self.trange}))

        # for t in self.trange:
        #     if scale_by_cmpts is None:
        #         self.linear_term_matrix_dict[t][self.indices.loc[to_cmpt]][self.indices.loc[from_cmpt]] += coef_dict[t]
        #         self.linear_term_matrix_dict[t][self.indices.loc[from_cmpt]][self.indices.loc[from_cmpt]] -= coef_dict[t]
        #     else:


    # def add_nonlinear_flow(self, attr_name, from_attr, to_attr, f):
    #
    #     self.nonlinear_terms.append(None)

    # def add_linear_flow_for_attr(self, attr_name, from_attr, to_attr, coef):
    #     from_cmpts = self.indices.xs(from_attr, level=self.attr_level(attr_name), drop_level=False).index
    #     to_cmpts = self.indices.xs(to_attr, level=self.attr_level(attr_name), drop_level=False).index
    #     for from_cmpt, to_cmpt in zip(from_cmpts, to_cmpts):
    #         self.add_flow(from_cmpt, to_cmpt, coef)

    def jacobian(self):
        pass

    def y0_from_dict(self, y0_dict):
        y0 = np.zeros(self.length)
        for cmpt, n in y0_dict.items():
            y0[self.indices[cmpt]] = n
        return y0

    def ode(self, t, y):
        t = min(np.floor(t), len(self.trange) - 1)
        dy = np.zeros(self.length)

        # print(t, y)
        for term in self.terms:
            # print(t, self.indices.index[term.from_cmpt_idx], self.indices.index[term.to_cmpt_idx])
            # print(term.dy(t, y))
            dy += term.dy(t, y)
        # print(t, dy)

        return dy

    def solve_ode(self, y0_dict):
        self.solution = spi.solve_ivp(
            fun=self.ode,
            t_span=[min(self.trange), max(self.trange)],
            y0=self.y0_from_dict(y0_dict),
            t_eval=self.trange)
        if not self.solution.success:
            raise RuntimeError(f'ODE solver failed with message: {self.solution.message}')
        self.solution_y = np.transpose(self.solution.y)
        self.solution_ydf = pd.concat([self.y_to_series(self.solution_y[t]) for t in self.trange], axis=1, keys=self.trange, names=['t']).transpose()

    def y_to_series(self, y):
        return pd.Series(index=self.indices.index, data=y)

    def solution_sum(self, group_by_attr_levels):
        return self.solution_ydf.groupby(group_by_attr_levels, axis=1).sum()





# if __name__ == '__main__':
#     engine = db_engine()
#     model = CovidModel([0, 400], engine=engine)
#     model.set_ef_from_db(5324)
#     model.prep()
#
#     ode_builder = seir_model(model)
#     t0 = perf_counter()
#     dy = ode_builder.ode(300, np.array([1] * ode_builder.length))
#     t1 = perf_counter()
#     print(f'Single calc of seir took {t1 - t0} seconds.')
#     print(ode_builder.y_to_series(dy))





    # attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'R', 'RA', 'Ih', 'D'], 'age': ['0-19', '20-39', '40-64', '65+'], 'vacc': ['unvacc', 'mrna', 'jnj']})
    # attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'R', 'RA', 'Ih', 'D'], 'age': ['0-19', '20-39', '40-64', '65+']})
    # # attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'R', 'RA', 'Ih', 'D']})
    #
    # gparams_lookup = {t: {(seir, age): {k.replace('gamma', 'gamm').replace('beta', 'bet').replace('N', 'total_pop'): v for k, v in params.items()} for age, params in model.gparams_lookup[t].items() for seir in attr['seir']} for t in model.trange}
    #
    # ode_builder = ODEBuilder(range(400), attr, params=gparams_lookup)
    # # ob.add_linear_flow_for_attr('seir', 'E', 'I', model.gparams_lookup[t][])
    #
    # for age in ode_builder.attributes['age']:
    #     ode_builder.add_flow(('S', age), ('E', age), 'bet * lamb / total_pop', scale_by_cmpts=[('I', a) for a in ode_builder.attributes['age']])
    #     ode_builder.add_flow(('S', age), ('E', age), 'bet / total_pop', scale_by_cmpts=[('A', a) for a in ode_builder.attributes['age']])
    #     ode_builder.add_flow(('E', age), ('I', age), '1 / alpha * pS')
    #     ode_builder.add_flow(('I', age), ('Ih', age), 'gamm * hosp')
    #     ode_builder.add_flow(('I', age), ('D', age), 'gamm * dnh')
    #     ode_builder.add_flow(('I', age), ('R', age), 'gamm * (1 - hosp - dnh) * immune_rate_I')
    #     ode_builder.add_flow(('I', age), ('S', age), 'gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
    #     ode_builder.add_flow(('A', age), ('RA', age), 'gamm * immune_rate_A')
    #     ode_builder.add_flow(('A', age), ('S', age), 'gamm * (1 - immune_rate_A)')
    #     ode_builder.add_flow(('Ih', age), ('D', age), '1 / hlos * dh')
    #     ode_builder.add_flow(('Ih', age), ('R', age), '1 / hlos * (1 - dh) * immune_rate_I')
    #     ode_builder.add_flow(('Ih', age), ('S', age), '1 / hlos * (1 - dh) * (1 - immune_rate_I)')
    #     ode_builder.add_flow(('R', age), ('S', age), '1 / dimmuneI')
    #     ode_builder.add_flow(('RA', age), ('S', age), '1 / dimmuneA')


    # I * (gamma * (1 - hosp - dnh)) * immune_rate_I



    # solution = spi.solve_ivp(fun=ode_builder.ode, t_span=[model.tmin, model.tmax], y0=model.y0(),
    #                               t_eval=range(model.tmin, model.tmax))

