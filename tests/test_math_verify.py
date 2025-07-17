# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify.errors import TimeoutException

math_comparer = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


class TestMathVerify(unittest.TestCase):
    def test_timeout(self):
        reference1 = "Rudolf claims the distance to Neustadt is greater than 5 km, while Emil claims it is less than 5 km. Robert states that one of them is right, but we know Robert's statement is false. \n\nTo analyze:\n1. If Robert's statement is false, then the opposite must be true. The opposite of \"exactly one is right\" is \"both are right or both are wrong.\"\n2. Rudolf and Emil's statements are contradictory, so they cannot both be right. Therefore, they must both be wrong.\n3. If both are wrong, the distance cannot be greater than 5 km (Rudolf's claim) or less than 5 km (Emil's claim). Thus, the distance must be exactly 5 km.\n\nVerification:\n- If the distance were greater than 5 km, Rudolf would be right and Emil wrong, making Robert's statement true (invalid).\n- If the distance were less than 5 km, Emil would be right and Rudolf wrong, again making Robert's statement true (invalid).\n- If the distance is exactly 5 km, both are wrong, making Robert's statement false (valid).\n\nThus, the actual distance is \\boxed{5} kilometers."
        to_be_evaluated1 = "\n\nSince exactly one of Rudolf and Emil's statements is true, and Robert said one of them is right when in fact both statements contradict each other, Robert must have said that Emil is right while Emil is in fact wrong. The distance cannot be less than $5 \\mathrm{~km}$, so it needs to be greater or equal to $5 \\mathrm{~km}$."
        reference2 = "To prove that the number 11...1 (1986 ones) has at least 8 and 32 different divisors, we consider the properties of repunit numbers and their divisors.\n\n### Part a) At least 8 different divisors\n\n1. **Repunit Divisors**: The number \\( R_{1986} \\) (which is 11...1 with 1986 ones) is a repunit number. Repunit numbers have the property that \\( R_n \\) is divisible by \\( R_d \\) for each divisor \\( d \\) of \\( n \\).\n2. **Divisors of 1986**: The divisors of 1986 are 1, 2, 3, 6, 331, 662, 993, and 1986.\n3. **Corresponding Repunits**: Each divisor \\( d \\) of 1986 corresponds to a repunit number \\( R_d \\). These repunits are distinct and divide \\( R_{1986} \\).\n4. **List of Divisors**: The divisors of \\( R_{1986} \\) include 1, \\( R_2 \\), \\( R_3 \\), \\( R_6 \\), \\( R_{331} \\), \\( R_{662} \\), \\( R_{993} \\), and \\( R_{1986} \\). This gives us 8 distinct divisors.\n\nThus, the number 11...1 (1986 ones) has at least 8 different divisors.\n\n### Part b) At least 32 different divisors\n\n1. **Repunit Factorization**: The repunit number \\( R_6 \\) (111111) is a divisor of \\( R_{1986} \\) because 6 divides 1986.\n2. **Prime Factorization of \\( R_6 \\)**: \\( R_6 \\) factors into prime components: \\( R_6 = 111111 = 3 \\times 7 \\times 11 \\times 13 \\times 37 \\). This gives 5 distinct prime factors.\n3. **Number of Divisors**: The number of divisors of a number with prime factorization \\( p_1 \\times p_2 \\times \\cdots \\times p_n \\) is \\( 2^n \\). For \\( R_6 \\), this is \\( 2^5 = 32 \\).\n4. **Divisors of \\( R_{1986} \\)**: Since \\( R_6 \\) is a divisor of \\( R_{1986} \\), all 32 divisors of \\( R_6 \\) are also divisors of \\( R_{1986} \\).\n\nThus, the number 11...1 (1986 ones) has at least 32 different divisors.\n\n### Final Answers\n\na) \\boxed{8}\n\nb) \\boxed{32}"
        # This test case can trigger TimeoutException.
        to_be_evaluated2 = "\n\nI think that I should use the fact that if we have a number with n digits, then there exist a unique decomposition in base 10 of the number as the sum of n different powers of 10, and the number of divisors of the number is the number of non-negative integer solutions to the equation\n\nx₁ + x₂ + ... + xₙ = n.\nI am not able to prove that the number of non-negative integer solutions for the above equation with 1986 variables is greater or equal than 8 or 32.\n\nI also know that there exist a relation between the number of divisors of an integer and its number of prime factors, but I don't know how this could help me in this problem.\n\nEDIT:\nI have proved that the above equation has at least 32 non-negative integer solutions by showing that it has at least 32 integer solutions, but I am not able to prove that 32 is greater or equal than the number of non-negative integer solutions.\n\nEDIT 2:\nI have found the following inequality (see http://www.math.uconn.edu/~kconrad/blurbs/ugradnumthy/binomial/binomial2.pdf):\n\n$$\\binom{n}{k} \\geq \\binom{n-1}{k} + \\binom{n-1}{k-1}$$\n\nBy using it we can say that the number of non-negative integer solutions to the above equation with 1986 variables is at least the number of non-negative integer solutions to the same equation with 1985, and the number of non-negative integer solutions to the equation with 1985 is at least the number of non-negative integer solutions to the equation with 1984, and so on. So the number of non-negative integer solutions is greater than or equal to the sum of the number of non-negative integer solutions to the equation with 1 variables, with 2 variables, with 3 variables, and so on. It should be easy to see that the number of solutions to the equation with 1 variable is 1, and the number of solutions to the equation with 2 variables is 2, and so we get 1 + 2 + 3 + ... + 1986 >= 32.\n\nCan you tell me if the reasoning I made is correct, and if it is a common way to solve this kind of problems?\n\nEDIT 3:\nI have found that the number of solutions to the equation\n\n$$x_1 + x_2 + ... + x_n = k$$\n\nwith the condition $0 \\leq x_1 \\leq x_2 \\leq ... \\leq x_n \\leq k$\n\nis given by the Catalan number $C_k$ (which is the number of ways to arrange $n$ parentheses to make a parenthesized expression with $2n$ symbols, in which no parenthesis should be nested).\n\nBy considering all possible values of $k$ between 1 and 1986, we can see that the sum of the Catalan numbers between $C_1$ and $C_{1986}$ is 8334474, which is greater than 32.\n\nI wonder if there is an easier way to solve the problem.\n\nEDIT 4:\nI have found this: http://www.math.uconn.edu/~kconrad/blurbs/ugradnumthy/combin10.pdf\n\nFrom the above page I can see that the number of non-negative integer solutions to the equation\n\n$$x_1 + x_2 + ... + x_n = n$$\n\nwith no condition on the $x_i$'s is equal to $2^{n - 1}$.\n\nFrom the page I linked above I also find that if we have n variables and we impose that $x_1 \\leq x_2 \\leq ... \\leq x_n$ the number of solutions is equal to\n\n$$\\sum_{k=0}^{n} {n \\choose k} \\big(\\frac{1}{k+1}\\big)^n.$$\n\nSince in our case we need $x_1 = x_2 = ... = x_{1986} = 1$, we see that the number of solutions is equal to\n\n$$1 + \\sum_{k=1}^{1986} {1986 \\choose k} \\frac{1}{k+1}$$\n\nwhich, by using the inequality $k^2 +"
        reference_list = [reference1, reference2]
        to_be_evaluated_list = [to_be_evaluated1, to_be_evaluated2]
        for reference, to_be_evaluated in zip(reference_list, to_be_evaluated_list):
            score = -1.0
            try:
                score, _ = math_comparer([reference], [to_be_evaluated])
            except TimeoutException as e:
                print(f"TimeoutException: {e}")
                assert str(e) == "Operation timed out!"
                score = 0.0
            assert score >= 0.0
            assert score <= 1.0


if __name__ == "__main__":
    unittest.main()
