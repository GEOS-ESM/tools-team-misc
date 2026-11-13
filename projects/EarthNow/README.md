# EarthNow

Initial files are from the conversion of IDL scripts to Python for the GEOS forecast experiments site:
https://g6dev.smce.nasa.gov/GEOS_WxMaps/geos_py2.html

The goal for this project is to update/add to these scripts to generate imagery for the EarthNow site:
https://g6dev.smce.nasa.gov/WebGL/geos_earth_now.html

## Installation and Environment

EarthNow requires Python 3.11 or newer. Install [uv](https://docs.astral.sh/uv/) if it is not already available, then run this command from the repository root:

```sh
uv sync --group dev
```

This creates or updates the project's `.venv`, installs EarthNow and its dependencies, and includes the development tools in the `dev` dependency group. There is no need to run `uv venv` or install optional extras separately.

Run commands in the project environment with `uv run`; activation is optional:

```sh
uv run python bin/plotall.py --help
```

To activate the environment for a shell session instead, run `source .venv/bin/activate`.

## Pre-commit

The pre-commit hooks format staged Python files with Black and lint staged YAML files with yamllint. If Black changes files, stage the formatted changes and commit again. The GitHub check also runs Black for pull requests; local hooks provide an early check and do not replace that required merge check.

From the EarthNow project directory, install the hook into the parent Git repository:

```sh
uv run pre-commit install --config "$PWD/.pre-commit-config.yaml"
```

To manually run only the YAML lint check on staged changes:

```sh
uv run pre-commit run yamllint --config "$PWD/.pre-commit-config.yaml"
```

The commit hook also checks only staged, changed files. Do not pass `--all-files` unless you intentionally want to lint every matching YAML file.

The hooks pin Black and yamllint versions in `.pre-commit-config.yaml`. Update those `rev` values when upgrading either linter.



