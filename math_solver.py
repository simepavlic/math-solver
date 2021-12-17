import operator
import math_parser

operations = {
    math_parser.TokenType.T_PLUS: operator.add,
    math_parser.TokenType.T_MINUS: operator.sub,
    math_parser.TokenType.T_MULT: operator.mul,
    math_parser.TokenType.T_DIV: operator.truediv
}


def compute(node):
    if node.token_type == math_parser.TokenType.T_NUM:
        return node.value
    left_result = compute(node.children[0])
    right_result = compute(node.children[1])
    operation = operations[node.token_type]
    return operation(left_result, right_result)
