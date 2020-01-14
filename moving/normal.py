import numpy as np
import pandas as pd
from scipy.special import factorial
import scipy
import itertools
import copy
from qutip import tensor, qeye, destroy


def gen_c_matrix(operator, threshold=None):
    dims = operator.dims[0]
    elements = operator.data.tocoo()

    c_limit_a = dims[0]
    c_limit_b = dims[1]

    c_matrix = np.zeros([c_limit_a, c_limit_b, c_limit_a, c_limit_b], dtype=complex)

    for row, col, el in zip(elements.row, elements.col, elements.data):
        mi_l = np.unravel_index(row, dims, order='C')
        mi_r = np.unravel_index(col, dims, order='C')
        mi = np.hstack([mi_l, mi_r])
        for delta_a in range(0, c_limit_a - max(mi[0], mi[2])):
            for delta_b in range(0, c_limit_b - max(mi[1], mi[3])):
                coeff = el * (-1) ** (delta_a + delta_b)
                coeff /= factorial(delta_a) * factorial(delta_b)
                coeff /= np.sqrt(factorial(mi[0]) * factorial(mi[1]) * factorial(mi[2]) * factorial(mi[3]))
                multi_index = [mi[0] + delta_a, mi[1] + delta_b, mi[2] + delta_a, mi[3] + delta_b]
                c_matrix[mi[0] + delta_a, mi[1] + delta_b, mi[2] + delta_a, mi[3] + delta_b] += coeff

    if threshold is not None:
        mask = np.abs(c_matrix) > threshold
        c_matrix *= mask

    return c_matrix


def gen_amp_funcs(component_idx, weight):
    product = itertools.product(range(component_idx[0] + 1), range(component_idx[1] + 1),
                                range(component_idx[2] + 1), range(component_idx[3] + 1))

    generated_indices = []
    amp_funcs_list = []

    for generated_idx in list(product):
        binom_coeffs = [scipy.special.comb(c_idx, g_idx, exact=True)
                        for c_idx, g_idx in zip(component_idx, generated_idx)]
        coeff = weight * np.prod(binom_coeffs)
        powers = [(c_idx - g_idx) for c_idx, g_idx in zip(component_idx, generated_idx)]

        amp_func = amp_func_factory(coeff, powers)

        generated_indices.append(generated_idx)
        amp_funcs_list.append(amp_func)

    multi_index = pd.MultiIndex.from_tuples(generated_indices, names=['a_l', 'b_l', 'a_r', 'b_r'])
    amp_funcs_frame = pd.DataFrame(amp_funcs_list, index=multi_index)

    return amp_funcs_frame


def amp_func_factory(coeff_in, powers):
    def amp_func(disp=[0, 0]):
        coeff = coeff_in
        coeff *= np.conjugate(disp[0]) ** powers[0]
        coeff *= np.conjugate(disp[1]) ** powers[1]
        coeff *= disp[0] ** powers[2]
        coeff *= disp[1] ** powers[3]
        return coeff

    return amp_func


def combine(frame1, frame2):
    combined = copy.deepcopy(frame1)
    for idx in frame2.index:
        if idx not in combined.index:
            combined.loc[idx] = frame2.loc[idx]
        else:
            combined.loc[idx] = function_adder(frame1.loc[idx, 0], frame2.loc[idx, 0])
    return combined


def function_adder(func1, func2):
    def combined_func(x):
        total = func1(x) + func2(x)
        return total

    return combined_func


def combine_dependences(amp_dependence_funcs, dims, time_func):
    pairs = []
    a = tensor(destroy(dims[0]), qeye(dims[1]))
    b = tensor(qeye(dims[0]), destroy(dims[1]))
    for idx in amp_dependence_funcs.index:
        amp_func = amp_dependence_funcs.loc[idx].iloc[0]
        dependence_func = copy.deepcopy(function_multiplier(time_func, amp_func))
        component = a.dag() ** idx[0] * b.dag() ** idx[1] * a ** idx[2] * b ** idx[3]
        pairs.append([component, dependence_func])
    pairs = pd.DataFrame(pairs, columns=['component', 'func'], index=amp_dependence_funcs.index)
    return pairs


def function_multiplier(time_func, amp_func):

    def combined_func(t, psi, args, disp, vel):
        total = time_func(t, psi, args) * amp_func(disp)
        return total

    return combined_func


def function_multiplier_2(coeff, func):
    def new_func(t, psi, args, disp, vel):
        out = coeff * func(t, psi, args, disp, vel)
        return out

    return new_func


def gen_comm_pairs(ham_pairs, dims, mode_idx):
    assert mode_idx == 0 or mode_idx == 1

    a = tensor(destroy(dims[0]), qeye(dims[1]))
    b = tensor(qeye(dims[0]), destroy(dims[1]))

    commutator_funcs = []
    commutator_components = []
    commutator_indices = []

    for idx in ham_pairs.index:
        if idx[mode_idx] > 0:
            vel_sub = np.array([0, 0])
            func = function_multiplier_2(idx[mode_idx], ham_pairs.func.loc[idx])
            if mode_idx == 0:
                component = a.dag() ** (idx[0] - 1) * b.dag() ** idx[1] * a ** idx[2] * b ** idx[3]
            else:
                component = a.dag() ** idx[0] * b.dag() ** (idx[1]-1) * a ** idx[2] * b ** idx[3]
            commutator_funcs.append(func)
            idx_out = np.array(idx)
            idx_out[mode_idx] -= 1
            commutator_indices.append(idx_out)
            commutator_components.append(component)

    multi_index = pd.MultiIndex.from_tuples(commutator_indices, names=['a_l', 'b_l', 'a_r', 'b_r'])
    commutator_pairs = pd.DataFrame(np.array([commutator_funcs, commutator_components]).T, index=multi_index,
                                    columns=['func', 'component'])

    return commutator_pairs


def gen_ham_pairs(ham_pair):
    ham_component = ham_pair[0]
    time_func = ham_pair[1]
    c_matrix = gen_c_matrix(ham_component, threshold=None)

    flat_dim = c_matrix.shape[0] * c_matrix.shape[1]
    dims = ham_component.dims[0]
    c_coo = scipy.sparse.coo_matrix(c_matrix.reshape(flat_dim, flat_dim))

    amp_funcs = None

    for el, row, col in zip(c_coo.data, c_coo.row, c_coo.col):
        mi_l = np.unravel_index(row, dims, order='C')
        mi_r = np.unravel_index(col, dims, order='C')
        mi = np.hstack([mi_l, mi_r])
        new_amp_funcs = gen_amp_funcs(mi, el)
        if amp_funcs is None:
            amp_funcs = new_amp_funcs
        else:
            amp_funcs = combine(amp_funcs, new_amp_funcs)

    pairs = combine_dependences(amp_funcs, dims, time_func)

    return pairs