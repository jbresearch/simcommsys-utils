# Copyright (c) 2024-2026 Mark Mizzi, Johann A. Briffa
#
# This file is part of SimCommSys.
#
# SimCommSys is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SimCommSys is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class SimcommsysBuildConfig(str, Enum):
    DEBUG = "debug"
    RELEASE = "release"
    PROFILE = "profile"


@dataclass
class SimcommsysBuildSettings:
    working_dir: Path
    install_dir: Path
    simcommsys_repo: str
    simcommsys_branch: str
    build_config: SimcommsysBuildConfig
    build_arch: str
    build_use_omp: bool
    build_use_mpi: bool
    build_use_gmp: bool
    build_use_cuda: bool


class SimcommsysOutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


@dataclass
class SimcommsysJob:
    name: str
    inputfile: Path
    outputfile: Path
    logfile: Path
    param_ranges: list[str]
    confidence: float
    relative_error: float
    floor_min: float
    output_format: SimcommsysOutputFormat
