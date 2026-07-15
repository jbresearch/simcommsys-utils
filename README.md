# Simcommsys Utilities

This is a Python package that aids in building Simcommsys systems, timers and
simulation files, and running the corresponding simulations either locally or on
a cluster. We recommend that you use this package to run simulations using
Simcommsys rather than directly interacting with the binaries.

## Prerequisites

This package requires Python and the [Poetry](https://python-poetry.org) build
system in order to run. Supported versions are:
* Python 3.10 or later
* Poetry 1.8.2 or later
These can be installed on Ubuntu (22.04 or later) from the standard repos using:

```bash
apt-get install python3 python3-poetry
```

As for the main Simcommsys software, the main target platform is the latest LTS
version of Ubuntu.

## Installation

Before running this tool for the first time, or after pulling from the Git
remote, you should install/refresh the tool:

```bash
poetry install
```

This will create a virtual environment at `./.venv` (if one does not already
exist), and install any project dependencies inside it.

## Usage

The tool can be run using the following (template) command:

```bash
poetry run simcommsys-utils [subcommand ...]
```

In order to see the list of subcommands offered by the tool, run:

```bash
poetry run simcommsys-utils --help
```

For full usage instructions of a `<subcommand>` offered by the tool, run:

```bash
poetry run simcommsys-utils <subcommand> --help
```

## Examples

The `Scripts/` directory contains some test cases for the tool which also serve
as examples of how it could be used.
