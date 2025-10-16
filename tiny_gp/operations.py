from enum import StrEnum, IntEnum, auto

FSET_START = 110
# ADD = 110
# SUB = 111
# MUL = 112
# DIV = 113
# EXP = 114
# SIN = 115
# COS = 116
# FSET_2ARG_END = 113
# FSET_END = 114

DIVISION_CUT_OUT = 0.001
EXPONENT_CUT_OUT = 100


class FitnessFunction(StrEnum):
    MAE = "Math.abs(result - actual)"
    MSE = "(result - actual) * (result - actual)"


class Operation(IntEnum):
    ADD = 0
    SUB = auto()
    MUL = auto()
    DIV = auto()
    EXP = auto()
    SIN = auto()
    COS = auto()


FUN_2ARG: set[str] = {Operation.ADD.name, Operation.SUB.name, Operation.MUL.name, Operation.DIV.name}
FUN_1ARG: set[str] = {Operation.EXP.name, Operation.SIN.name, Operation.COS.name}
