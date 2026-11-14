> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/nvidia/cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-RL, Cosmos 3 unifies reasoning, world prediction, simulation, transfer, and action generation within a single model family and ecosystem.
>
> Rather than relying on separate models for reasoning, prediction, transfer, and policy learning, a single Cosmos 3 model can understand the world, reason about physical interactions, predict future outcomes, transform observations across domains, and generate actions for embodied agents. This unified architecture enables stronger performance across a broad range of Physical AI applications, including robotics, autonomous vehicles, and smart spaces.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/nvidia/cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-RL. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

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
