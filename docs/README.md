# Usage

1. `git clone git@github.com:nvidia-cosmos/cosmos-rl.git && cd cosmos-rl`
2. Install `sphinx-...` packages
    ``` bash
    pip install sphinx-autobuild  sphinx_rtd_theme recommonmark sphinx_markdown_tables sphinx-argparse sphinx-jsonschema
    ```

3. In `docs` folder, run `make clean && make html` to build the static html files
4. Either open static file located at `./_build/html/index.html` or host it with
    ``` bash
    python -m http.server -d _build/html/
    ```