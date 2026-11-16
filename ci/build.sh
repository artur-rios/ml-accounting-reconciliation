#!/bin/sh
# The `build` check: builds the package's sdist and wheel into dist/ (python -m build, from pyproject.toml), installs
# the wheel, with its dependencies, a pinned pytest and a pinned pytest-cov, into a throwaway virtual environment, then
# runs the test suite against it under coverage, with the floor pyproject.toml's [tool.coverage.report] sets (its
# integration tests included: they run the pipeline and experiment 3 end to end on small configurations, in temporary
# directories). yggdrasil.yml runs it from the repository's root in the pinned Python
# image, so it runs the same on your own machine:
#   docker run --rm -v "$PWD:/src" -w /src <yggdrasil.yml's build image> ci/build.sh
#
# The package is src/-laid out, so the tests import the installed wheel, not the checkout. The dev extra (jupyter,
# the rest of it) is not installed: the notebooks are not run here. build, pytest and pytest-cov are pinned to exact
# versions; the build backend is pyproject.toml's [build-system], and the dependencies resolve from its ranges.
set -eu

# The repository's root, wherever this is run from.
cd "$(dirname "$0")/.."

# renovate: datasource=pypi depName=build
BUILD_VERSION=1.6.1
# renovate: datasource=pypi depName=pytest
PYTEST_VERSION=9.2.0
# renovate: datasource=pypi depName=pytest-cov
PYTEST_COV_VERSION=7.1.0

venv=$(mktemp -d)
trap 'rm -rf "$venv"' EXIT
python3 -m venv "$venv"
"$venv/bin/pip" install --quiet --disable-pip-version-check "build==$BUILD_VERSION"

rm -rf dist
"$venv/bin/python" -m build --outdir dist .
"$venv/bin/pip" install --quiet --disable-pip-version-check dist/*.whl \
  "pytest==$PYTEST_VERSION" "pytest-cov==$PYTEST_COV_VERSION"

# No cache directory or coverage data file in the checkout. --cov takes its source and floor from pyproject.toml.
export COVERAGE_FILE="$venv/.coverage"
"$venv/bin/python" -m pytest -p no:cacheprovider --cov --cov-report=term
