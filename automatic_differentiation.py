import numpy as np


class Variable:
    def __init__(self, value: float, parents: list, parent_derivative_multiplier: list):
        self.value = value
        self.parents: list[Variable] = parents
        self.parent_derivative_multipliers = parent_derivative_multiplier
        self.derivative = 0

    def go_backwards(self):
        for i in range(len(self.parents)):
            self.parents[i].derivative += self.derivative * self.parent_derivative_multipliers[i]


def add_var(v1: Variable, v2: Variable, vars_list):
    new_var = Variable(value=(v1.value + v2.value),
                       parents=[v1, v2],
                       parent_derivative_multiplier=[1, 1])
    vars_list.append(new_var)
    return new_var


def add_const(v1: Variable, const, vars_list):
    new_var = Variable(value=(v1.value + const),
                       parents=[v1],
                       parent_derivative_multiplier=[1])
    vars_list.append(new_var)
    return new_var


def multiply_by_var(v1: Variable, v2: Variable, vars_list):
    new_var = Variable(value=(v1.value * v2.value),
                       parents=[v1, v2],
                       parent_derivative_multiplier=[v2.value, v1.value])
    vars_list.append(new_var)
    return new_var


def multiply_by_const(v1: Variable, const, vars_list):
    new_var = Variable(value=(v1.value * const),
                       parents=[v1],
                       parent_derivative_multiplier=[const])
    vars_list.append(new_var)
    return new_var


def apply_sin(v1: Variable, vars_list):
    new_var = Variable(value=np.sin(v1.value),
                       parents=[v1],
                       parent_derivative_multiplier=[np.cos(v1.value)])
    vars_list.append(new_var)
    return new_var


def exponentiate_by_const(v1: Variable, const, vars_list):
    new_var = Variable(value=v1.value ** const,
                       parents=[v1],
                       parent_derivative_multiplier=[const * v1.value ** (const - 1)])
    vars_list.append(new_var)
    return new_var


x1 = Variable(5, [], [])
vars_list: list[Variable] = []
y = apply_sin(exponentiate_by_const(x1, 3, vars_list), vars_list)
y.derivative = 1
for i in range(len(vars_list) - 1, -1, -1):
    vars_list[i].go_backwards()
print(f"dx1={x1.derivative}")
