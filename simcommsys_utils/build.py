# Copyright (c) 2026 Johann A. Briffa
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


from enum import Enum
import logging
from pathlib import Path
import subprocess


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


def _get_build_cmd(
    build_settings: SimcommsysBuildSettings,
    build_target: str,
    workers: int,
):
    # determine build arguments
    build_args = []
    build_args.append(f"-j{workers}")
    build_args.append(f"DOTARGET=install")
    build_args.append(f"BINDIR={build_settings.install_dir}")
    build_args.append(f"CONFIG={build_settings.build_config.value}")
    build_args.append(f"USE_ARCH={build_settings.build_arch}")
    if not build_settings.build_use_omp:
        build_args.append("USE_OMP=0")
    if not build_settings.build_use_mpi:
        build_args.append("USE_MPI=0")
    if not build_settings.build_use_gmp:
        build_args.append("USE_GMP=0")
    if not build_settings.build_use_cuda:
        build_args.append("USE_CUDA=0")
    # command sequence to check out, build, and install the target
    cmd = f"""
#!/bin/bash
set -euo pipefail
exec 3>&1 # save original stdout
exec 1>&2 # redirect stdout to stderr for logging
# Make sure work directory exists
mkdir -p {build_settings.working_dir}
## CRITICAL SECTION START
exec {{lock_fd}}>"{build_settings.working_dir}/LOCK" # open file descriptor for lock
flock -x "$lock_fd" # acquire the lock
# Clone the repo and build the binaries
if [ -d {build_settings.working_dir}/.git/ ]; then
    echo Existing working copy found, updating
    git -C {build_settings.working_dir} fetch origin
else
    echo No working copy found, initializing new one
    git -C {build_settings.working_dir} init
    git -C {build_settings.working_dir} remote add origin {build_settings.simcommsys_repo}
    git -C {build_settings.working_dir} fetch origin
fi
echo Checking out {build_settings.simcommsys_branch}
git -C {build_settings.working_dir} checkout {build_settings.simcommsys_branch}
echo Updating to the latest version of {build_settings.simcommsys_branch}
git -C {build_settings.working_dir} reset --hard origin/{build_settings.simcommsys_branch}
echo Removing any extraneous files
git -C {build_settings.working_dir} clean -fde LOCK
echo "Building binaries into {build_settings.install_dir}..."
make --no-print-directory -C {build_settings.working_dir} {" ".join(build_args)} {build_target}
echo Done building binaries
BINARY_FILENAME=$(make --no-print-directory -C {build_settings.working_dir} {" ".join(build_args[2:])} DOTARGET=showfinal {build_target} 2>/dev/null |tail -1)
echo Binary filename: $BINARY_FILENAME
## CRITICAL SECTION END
rm -f "{build_settings.working_dir}/LOCK" # first remove the file (unlinks with file descriptor)
flock -u "$lock_fd" # release the lock
exec {{lock_fd}}>&- # close the file descriptor, deleting the unlinked file
# print binary filename to original stdout
echo "$BINARY_FILENAME" >&3
exec 3>&- # close original stdout
"""
    return cmd


def build_binary(
    build_settings: SimcommsysBuildSettings,
    build_target: str,
    workers: int
) -> Path:
    build_cmd = _get_build_cmd(build_settings, build_target, workers)
    try:
        res = subprocess.run(build_cmd,
                             shell=True, check=True, capture_output=True,
                             text=True, executable="/bin/bash")
    except subprocess.CalledProcessError as e:
        logging.error("Build script failed with a non-zero exit code.")
        logging.error(f"stdout was: {e.stdout}")
        logging.error(f"stderr was: {e.stderr}")
        exit(-1)
    binary_filename = Path(res.stdout.strip()).resolve()
    if not binary_filename.is_file():
        logging.error("Could not identify binary from build script.")
        logging.error(f"stdout was: {res.stdout}")
        logging.error(f"stderr was: {res.stderr}")
        exit(-1)
    return binary_filename
