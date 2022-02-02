from sympy import Matrix
from coeff_calculator import CoeffCalculator
from functional import Functional
from model import Model
from plotter import Plotter
from numpy.random import normal
import math

adj_matrix = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]

input_data = {
    'a1': 0.7,
    'a2': 0.4,
    'a3': 0.6,
    'disc': 0.6,
    'x_hat': 1,
    'gamma': 2.1,
    'ag': 3
}
a1 = input_data['a1']
a2 = input_data['a2']
a3 = input_data['a3']
A = [[1 - a1 / 2, a1 / 2, 0],
     [a2 / 3, 1 - 2 * a2 / 3, a2 / 3],
     [0, a3 / 2, 1 - a3 / 2]]

calc_control = lambda c0, co: (Matrix(co).T *
                               Matrix([j.opinion for j in m.agents_list]))[0] + c0

coeffs = CoeffCalculator(input_data)
solutions = coeffs.solutions_dict
num_of_sols = coeffs.num_of_solutions
del CoeffCalculator

initial_ops = [round(i, 2) for i in normal(0.5,  0.2, 3)]

for i in range(1, num_of_sols + 1):
    m = Model(input_data, adj_matrix, [0.1, 0.4, 0.4])
    copt = solutions[f'{i}copt']
    c0opt = solutions[f'{i}c0opt']
    with open('output.txt', 'a') as out:
        out.write(f"{i}\tc0 = {c0opt}\tc = {copt}\n")
        while not m.stopped:
            control_value = calc_control(c0opt, copt)
            m.control.append(control_value)
            stroka = f"T = {m.T}\t\t{control_value}\t\t{m.agents_list[0].opinion}\t{m.agents_list[1].opinion}" \
                     f"\t{m.agents_list[2].opinion}\n"
            out.write(stroka)
            m.iterate_model(control_value)
        if m.exitcode == 1:
            out.write(f"Goal reached --> NICE SOLUTION\n")
            J = Functional(input_data, m.opinions_dict, m.control)
            out.write(f"J = {J.calculate_functional()}\n")
            p = Plotter(m.control, m.opinions_dict, m.x_hat, 'ex1.png')
        if m.exitcode == 2:
            out.write("Too many iterations --> WRONG SOLUTION\n\n\n")
            J = Functional(input_data, m.opinions_dict, m.control)
            out.write(f"J = {J.calculate_functional()}\n")
            p = Plotter(m.control, m.opinions_dict, m.x_hat, 'wrong_sol.png')
        if m.exitcode == 3:
            out.write("Too big opinions --> WRONG SOLUTION\n\n\n")
            J = Functional(input_data, m.opinions_dict, m.control)
            out.write(f"J = {J.calculate_functional()}\n")
            p = Plotter(m.control, m.opinions_dict, m.x_hat, 'too_big_ops.png')
        if m.exitcode == 4:
            out.write("Control value almost ~ 0 --> WRONG SOLUTION\n\n\n")
            J = Functional(input_data, m.opinions_dict, m.control)
            out.write(f"J = {J.calculate_functional()}\n")
            p = Plotter(m.control, m.opinions_dict, m.x_hat, 'wrong_sol_control_almost_zero.png')
        if m.exitcode == 5:
            out.write(f"No sufficient changes in the model (mean = {sum(m.opinions_dict[m.T])/len(m.agents_list)})"
                      f" --> WRONG SOLUTION\n")
            J = Functional(input_data, m.opinions_dict, m.control)
            out.write(f"J = {J.calculate_functional()}\n")
            p = Plotter(m.control, m.opinions_dict, m.x_hat, 'no_suff_chganges.png')
            print(math.log(max(m.opinions_dict.keys())) * max(m.opinions_dict.keys()) * 0.5 * sum([pow(i, 2) for i in m.control]))
    del m
