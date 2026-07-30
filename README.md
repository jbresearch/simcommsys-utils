# Simcommsys Utilities

This is a Python package that aids in building Simcommsys systems, timers and
simulation files, and running the corresponding simulations either locally or on
a cluster. We recommend that you use this package to run simulations using
Simcommsys rather than directly interacting with the binaries.

## Prerequisites

The reference platform for `simcommsys-utils` is the same as for `simcommsys`,
that is, an Ubuntu 24.04 LTS (64-bit) server system. In addition to the
`simcommsys` requirements, the following packages are also necessary:

- Python (`python3`)
- [Poetry](https://python-poetry.org) (`python3-poetry`)

These are all available as packages on the standard Ubuntu repositories.
Support for Ubuntu 22.04 LTS is also currently maintained.

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
