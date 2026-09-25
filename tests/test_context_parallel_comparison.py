# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for the native context-parallel test's numerical gate."""

import ast
import logging
from pathlib import Path

import pytest
import torch


@pytest.fixture
def compare_tensor():
    # Extract only the comparison helper: importing the GPU integration test
    # initializes unrelated model dependencies. Exercise its actual source,
    # rather than a second implementation of the acceptance criteria.
    source = Path(__file__).with_name("test_context_parallel.py")
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compare_tensor"
    ]
    assert len(functions) == 1
    namespace = {"torch": torch, "logger": logging.getLogger(__name__)}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["compare_tensor"]


@pytest.mark.parametrize("is_forward", [False, True])
def test_close_values_pass(compare_tensor, is_forward):
    values = torch.tensor([0.0, 1.0, -2.0])
    assert compare_tensor(values, values.clone(), is_forward)


@pytest.mark.parametrize("is_forward", [False, True])
def test_existing_cosine_fallback_is_preserved(compare_tensor, is_forward):
    # Intentionally outside assert_close tolerance, but collinear. Preserve
    # this existing test criterion; this repair is not a tolerance redesign.
    assert compare_tensor(
        torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0]), is_forward
    )


@pytest.mark.parametrize("is_forward", [False, True])
@pytest.mark.parametrize(
    "actual, expected",
    [
        ([1.0, 0.0], [0.0, 1.0]),
        ([1.0, 2.0], [-1.0, -2.0]),
        ([float("nan"), 1.0], [1.0, 1.0]),
        ([float("inf"), 1.0], [1.0, 1.0]),
    ],
)
def test_failed_comparison_raises_even_if_caller_ignores_result(
    compare_tensor, is_forward, actual, expected
):
    with pytest.raises(AssertionError, match="comparison failed"):
        compare_tensor(torch.tensor(actual), torch.tensor(expected), is_forward)
