import os
import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

sys.path.insert(0, os.path.abspath("../cosmos_rl/policy"))

# Import the declarative schema without running cosmos_rl.__init__, which
# initializes rollout imports and requires the full GPU runtime installation.
package = module_from_spec(ModuleSpec("cosmos_rl", loader=None, is_package=True))
package.__path__ = [os.path.abspath("../cosmos_rl")]
sys.modules.setdefault("cosmos_rl", package)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cosmos-rl"
copyright = "2025, NVIDIA"
author = "NVIDIA"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxarg.ext",
    "sphinx-jsonschema",
    "recommonmark",
    "sphinx_markdown_tables",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
