from .context import assert_equal
from sympy import asinh

# x = Symbol('x', real=True);

# latex = "\\sinh(x)"
# math = process_sympy(latex)
# print("latex: %s to math: %s" %(latex,math))
#
# latex = "\\arcsinh(x)"
# math = process_sympy(latex)
# print("latex: %s to math: %s" %(latex,math))
#
# latex = "\\arsinh(x)"
# math = process_sympy(latex)
# print("latex: %s to math: %s" %(latex,math))


def test_arcsinh():
    assert_equal("\\operatorname{arcsinh}\\left(1\\right)", asinh(1, evaluate=False))
