# EarthNow

Initial files are from the conversion of IDL scripts to Python for the GEOS forecast experiments site:
https://g6dev.smce.nasa.gov/GEOS_WxMaps/geos_py2.html

The goal for this project is to update/add to these scripts to generate imagery for the EarthNow site:
https://g6dev.smce.nasa.gov/WebGL/geos_earth_now.html

## Installation/Environment
The EarthNow code has a pyproject.toml script to help install the package locally.
You'll need uv installed to build the environment

### Installation
In the root dir for EarthNow, run
`uv venv`

If .venv/ already exists, it won’t destroy it.

Activate it:

`source .venv/bin/activate`

Install the env:

`uv pip install -e ".[plotting,images,dev]"`

(sorry for the optional list, I'm testing out how to handle the package)

### Environment
After installation, you can activate/use the environment by either sourcing it
`source .env/bin/activate`

or running 'uv run' before python calls:
`uv run python plotall.py --product basemap [other args]`

### Pre-commit hooks 
Optional to the EarthNow workflow is a pre-commit hook to run the black formatter automatically before each git commit. If you wish to activate this, simply install the hook with
`pre-commit install`

More details: 
    - This commit hook is defined with `.pre-commit-config.yaml` in the root directory of the git repo.
    - Commits will fail automatically it black reformats any files, and you will need to run `git commit` again to confirm  your changes.



