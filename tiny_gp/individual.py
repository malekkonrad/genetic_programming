import math
from functools import cached_property

import numpy as np
from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema
from sympy import simplify

from tiny_gp.operations import *


class Individual(BaseModel):
    """
    Individual.
    """
    individual_raw: list[str]
    individual: list[str | float]
    operations: SkipJsonSchema[
                    dict[str, int]] | None = None  # allowed operations cause they can change, set manually, ugh

    def __init__(self, individual: list[str], x: list[float] = None, var_number: int = None, **kwargs):
        if kwargs:  # it's so ugly
            super().__init__(
                individual_raw=kwargs["individual_raw"],
                individual=individual
            )
        else:
            super().__init__(
                individual_raw=individual,
                individual=Individual._parse_individual(individual, x, var_number)
            )

    @property
    def var_number(self):
        return len({i for i in self.individual if isinstance(i, str) and ord(i) < FSET_START})

    @staticmethod
    def _parse_individual(individual: list[str], x: list[float], constant_count: int) -> list[str | float]:
        parsed: list[str | float] = list()
        for s in individual:
            if constant_count <= ord(s) < FSET_START:  # it's a number
                parsed.append(x[ord(s)])
            else:
                parsed.append(s)  # it's an operation or variable
        return parsed

    @cached_property
    def minimal_form(self):
        return simplify(self.__str__())

    def simplify(self) -> "Individual":
        stack: list[str | float] = list()
        FSET_2ARG_END = max(v for k, v in self.operations.items() if k in FUN_1ARG)
        for symbol in self.individual:
            if isinstance(symbol, str):
                # it's a variable or an operation
                stack.append(symbol)
            elif isinstance(symbol, float):  # it's a number
                stack.append(symbol)
                while len(stack) > 2:  # TODO not good, cause it could be one operation
                    # 2 argument functions
                    if (isinstance(stack[-3], str)
                            and isinstance(stack[-2], float)
                            and FSET_START <= (s := ord(stack[-3])) <= FSET_2ARG_END):
                        num2 = stack.pop()
                        num1 = stack.pop()
                        if s == self.operations["ADD"]:
                            stack[-1] = num2 + num1
                        elif s == self.operations["SUB"]:
                            stack[-1] = num2 - num1
                        elif s == self.operations["MUL"]:
                            stack[-1] = num2 * num1
                        elif s == self.operations["DIV"]:
                            stack[-1] = num1 if abs(num2) <= DIVISION_CUT_OUT else num1 / num2
                        else:
                            raise ValueError(f"Unrecognized operation: {s}")
                    elif (isinstance(stack[-2], str)
                            and FSET_2ARG_END < (s := ord(stack[-2]))):
                        num = stack.pop()
                        if s == self.operations["EXP"]:
                            stack[-1] = math.e ** num if abs(num) <= EXPONENT_CUT_OUT else num
                        elif s == self.operations["SIN"]:
                            stack[-1] = math.sin(math.radians(num))
                        elif s == self.operations["COS"]:
                            stack[-1] = math.cos(math.radians(num))
                        else:
                            raise ValueError(f"Unrecognized operation: {s}")
                    else:
                        break
            else:
                raise ValueError(f"Unrecognized type: {symbol} of type: {type(symbol)}, provide str or float.")

        instance = Individual(stack, None, individual_raw=self.individual_raw)
        instance.operations = self.operations
        return instance

    def __str__(self):
        pos: int = 0

        def print_individual() -> str:
            nonlocal pos
            symbol = self.individual[pos]
            pos += 1
            if isinstance(symbol, str):
                s = ord(symbol)
                if s < FSET_START:  # it's a variable
                    return f"X{s + 1}"
                else:  # it's an operation
                    if s == self.operations["ADD"]:
                        return f"({print_individual()} + {print_individual()})"
                    if s == self.operations["SUB"]:
                        return f"({print_individual()} - {print_individual()})"
                    if s == self.operations["MUL"]:
                        return f"({print_individual()} * {print_individual()})"
                    if s == self.operations["DIV"]:
                        return f"({print_individual()} / {print_individual()})"
                    if s == self.operations["EXP"]:
                        return f"EXP({print_individual()})"
                    if s == self.operations["SIN"]:
                        return f"SIN({print_individual()})"
                    if s == self.operations["COS"]:
                        return f"COS({print_individual()})"
                    raise ValueError(f"Unrecognized operation: {s}")
            elif isinstance(symbol, float):  # it's a number
                return str(symbol)
            else:
                raise ValueError(f"Unrecognized type: {symbol} of type: {type(symbol)}, provide str or float.")

        return print_individual()

    def __repr__(self):
        return self.__str__()

    def _evaluate_one(self, variables: list[float]) -> float:
        # FIXME just suppress it lol
        # if self.var_number != len(variables):
        #     raise IndexError(f"There are {self.var_number} variables but you provided: {len(variables)}")

        pos: int = 0

        def run() -> float:
            nonlocal pos
            symbol = self.individual[pos]
            pos += 1
            if isinstance(symbol, str):
                s = ord(symbol)
                if s < FSET_START:  # it's a variable
                    if s < len(variables):
                        return variables[s]
                    else:
                        raise IndexError(f"There are {s + 1} variables but you provided: {len(variables)}")
                else:  # it's an operation
                    if s == self.operations["ADD"]:
                        return run() + run()
                    if s == self.operations["SUB"]:
                        return run() - run()
                    if s == self.operations["MUL"]:
                        return run() * run()
                    if s == self.operations["DIV"]:
                        num1 = run()
                        den = run()
                        if abs(den) <= DIVISION_CUT_OUT:
                            return num1
                        else:
                            return num1 / den
                    if s == self.operations["EXP"]:
                        num = run()
                        if num <= EXPONENT_CUT_OUT:
                            return math.e ** num
                        else:
                            return num
                    if s == self.operations["SIN"]:
                        return math.sin(math.radians(run()))
                    if s == self.operations["COS"]:
                        return math.cos(math.radians(run()))
                    raise ValueError(f"Unrecognized operation: {s}")
            elif isinstance(symbol, float):  # it's a number
                return symbol
            else:
                raise ValueError(f"Unrecognized type: {symbol} of type: {type(symbol)}, provide str or float.")

        return run()

    def evaluate(self, variables: list[float] | np.ndarray) -> float | np.ndarray:
        if isinstance(variables, list):
            return self._evaluate_one(variables)
        if isinstance(variables, np.ndarray):
            # ind = self.simplify() TODO evaluation is wrong with simplified
            ind = self
            if variables.ndim == 1:
                # if self.var_number != variables.shape[0]:
                #     raise IndexError(
                #         f"There are {self.var_number} variables but you provided: {variables.shape[0]}")
                return np.asarray(ind._evaluate_one(variables.tolist()))
            elif variables.ndim == 2:
                # FIXME this error is wrong, mainly the self.var_number calculation is wrong, the function can be constant
                # if self.var_number != variables.shape[1]:
                #     raise IndexError(
                #         f"There are {self.var_number} variables but you provided: {variables.shape[1]}. Maybe use reshape(-1, 1)?")
                return np.asarray([ind._evaluate_one(vs) for vs in variables])
            else:
                raise ValueError(f"Wrong input dimension: {variables.ndim}, input must be of dimension 1 or 2.")
        else:
            raise ValueError(f"Wrong input type: {type(variables)}")
