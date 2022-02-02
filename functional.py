"""
Calculate value of the Bellman functional
"""


class Functional:
    """It's a class"""

    def __init__(self, input_data, opinions_dict, control):
        """Fill all class fields with needed data"""
        self.disc = input_data['disc']
        self.x_hat = input_data['x_hat']
        self.gamma = input_data['gamma']
        self.ag = input_data['ag']
        self.opinions_dict = opinions_dict
        self.control = control

    def calculate_functional(self):
        """Return the functional value"""
        J = 0
        for i in range(1, max(self.opinions_dict.keys())):
            p_j = 0
            for j in range(0, 3):
                p_j += pow((self.opinions_dict[i][j] - self.x_hat), 2)
            p_j += self.gamma * pow(self.control[i - 1], 2)
            J += p_j * pow(self.disc, i)
        return J
