"""
Module calculates all significant values and
collect solutions for Riccati system of equations
"""
from wolframclient.evaluation.kernel.localsession import WolframLanguageSession


class CoeffCalculator:
    """
    Class of calculator

    a* - confidence parameter of each agent
    disc - discount coefficient
    x_hat - goal opinion for each agent
    gamma - cost for control
    ag - number of agents
    solutions_dict - dictionary of real solutions for Riccati system of equations
    num_of_solutions - number of these solutions
    """

    def __init__(self, input_data):
        """Fill all class fields"""
        self.a1 = input_data['a1']
        self.a2 = input_data['a2']
        self.a3 = input_data['a3']
        self.disc = input_data['disc']
        self.x_hat = input_data['x_hat']
        self.gamma = input_data['gamma']
        self.ag = input_data['ag']
        self.solutions_dict = dict()
        self.__calculate_coefficients()

    def __calculate_coefficients(self):
        """Calculate needed parameters and collect them in the class fields"""
        with WolframLanguageSession() as ses:
            ses.evaluate(f'a1:={self.a1};'
                         f'a2:={self.a2};'
                         f'a3:={self.a3};'
                         f'd:={self.disc};'
                         f'xHat:={self.x_hat};'
                         f'gamma:={self.gamma};'
                         f'ag:={self.ag};')
            ses.evaluate('A:={{1- 2*a1/3, a1/3, a1/3},{a2/3, 1 - 2*a2/3, a2/3},{a3/3, a3/3 ,1 - 2*a3/3}};')
            ses.evaluate('B:={0,1,0};')
            ses.evaluate('Q:={{1,0,0},{0,1,0},{0,0,1}};')
            ses.evaluate('q:={-2*xHat,-2*xHat,-2*xHat};')
            ses.evaluate('k:={k1,k2,k3};')
            ses.evaluate('K:={{k11,k12,k13},{k12,k22,k23},{k13,k23,k33}};')
            ses.evaluate('k0:=k0;')
            ses.evaluate('c0:=-(0.5*gamma*k.B)/(gamma+d*B.K.B);')
            ses.evaluate('c:=-d*B.K.A/(gamma+d*B.K.B);')
            ses.evaluate('eq1:= k0-3*xHat^2-gamma*c0^2-d*k.B*c0-d*(B*c0).K.B*c0-d*k0==0;'
                         'eq2:=K-Q-gamma*c.c-d*Transpose[A+B.c].K.(A+B.c)==0;'
                         'eq3:=k-q-2*gamma*c*c0-d*k.(A+B.c)-2*d*c0*Transpose[(A+B.c)].K.B==0;')
            solutions = ses.evaluate(
                'sols=NSolve[Join[eq1 ,eq2 ,eq3], {k0,k1,k2,k3,k11,k22,k33,k12,k13,k23}, Reals]')
            self.num_of_solutions = len(solutions)
            for i in range(1, len(solutions) + 1):
                ses.evaluate(f'Kopt:=K/.sols[[{i}]];')
                ses.evaluate(f'kopt:=k/.sols[[{i}]];')
                ses.evaluate(f'k0opt:=k0/.sols[[{i}]];')
                ses.evaluate('c0opt:=-(0.5*d*kopt.B)/(gamma+d*B.Kopt.B);')
                c0opt = ses.evaluate('c0opt')
                ses.evaluate('copt:=-d*B.Kopt.A/(gamma+d*B.Kopt.B);')
                copt = ses.evaluate('copt')
                self.solutions_dict[f'{i}c0opt'] = c0opt
                self.solutions_dict[f'{i}copt'] = list(copt)
            ses.stop()
