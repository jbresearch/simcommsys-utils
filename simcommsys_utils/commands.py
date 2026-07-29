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

from simcommsys_utils.models import SimcommsysBuildSettings, SimcommsysJob


# internal helper methods


def _get_build_cmd(
    build_settings: SimcommsysBuildSettings,
    build_target: str,
    workers: int,
) -> str:
    """
    Return commands to build simcommsys binaries in a parallel-safe way.
    """

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
"""
    return cmd


def _get_ulimit_cmd(memlimit_gb: int | None) -> str:
    if memlimit_gb is None:
        return ""
    return f"ulimit -v {memlimit_gb * 1024 * 1024}"


def _get_simcommsys_cmd(
    binary_filename: Path,
    job: SimcommsysJob,
    endpoint: str = None,
):
    """
    Returns simcommsys command for a single simulation.
    """

    cmd = str(f"{binary_filename}"
              f" -i {job.inputfile}"
              f" -o {job.outputfile}"
              "".join(f" -r {pr}" for pr in job.param_ranges) +
              f" --confidence {job.confidence}"
              f" --relative-error {job.relative_error}"
              f" --floor-min {job.floor_min}"
              f" -f {job.output_format.value}"
              f" -e {endpoint}" if endpoint is not None else ""
              f" -q")

    return cmd


# helper methods used by clients


def _safe_wrap(cmd: str) -> str:
    wrapped_cmd = f"""
#!/bin/bash
set -euo pipefail
{cmd}
"""
    return wrapped_cmd

# main methods used by clients


def _get_local_cmd(
    binary_filename: Path,
    job: SimcommsysJob,
    memlimit_gb: int,
) -> str:
    """
    Returns commands for a single simulation run in local mode.
    """

    # build command to submit to shell.
    cmd = f"""
{_get_ulimit_cmd(memlimit_gb)}
{_get_simcommsys_cmd(binary_filename, job)}
"""
    return cmd


def _get_masterslave_cmd(
    binary_filename: Path,
    job: SimcommsysJob,
    port: int,
    update_port_if_in_use: bool,
    workers: int,
    memlimit_gb: int | None,
) -> str:
    """
    Returns commands for a single simulation run in master-slave mode.
    """

    # logfile where output from workers will be stored.
    worker_logfile = job.logfile.with_name(f"{job.logfile.stem}-w$i.log")
    # build command to submit to shell.
    # 1. sets a memory limit using ulimit -v.
    # 2. starts the simcommsys master in a screen session
    # 3. starts the simcommsys slaves in the background, with stderr and stdout suppressed.
    # 4. Registers a handler for Ctrl+C that can kill the simcommsys server screen session
    # 5. Waits for the simcommsys server screen session to terminate or for the user to press Ctrl+C
    cmd = f"""
echo "Starting simulation of {job.name} with {workers} workers."

{_get_ulimit_cmd(memlimit_gb)}

port={port}
while true; do
    echo "Attempting to use port $port"
    # Launch the server in the background
    {_get_simcommsys_cmd(binary_filename, job, ':$port')} 1>"{job.logfile}" 2>&1 &
    SERVER_PID=$!
    # Wait a moment for the server to attempt to bind to the port
    sleep 1
    # Check if the server is running and started correctly
    if ps -p $SERVER_PID > /dev/null && timeout 30 sh -c "until netcat -z localhost $port; do sleep 1; done"; then
        echo "Server started successfully on port $port"
        break
    fi
    echo "Failed to start server on port $port"
    # Stop here if we were instructed not to update the port
    if ! {str(update_port_if_in_use).lower()}; then
        exit 1
    fi
    # Ensure the failed server process is killed before retrying
    echo "Killing failed server process..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Trying next port..."
    port=$((port+1))
done

# Launch the workers in the background
for i in $(seq 1 {workers}); do
    echo "Starting worker $i"
    {binary_filename} -q -e localhost:$port 1>"{worker_logfile}" 2>&1 &
done

handler () {{
    echo "Killing {job.name} server..."
    kill $SERVER_PID
}}

trap handler INT
echo "Press Ctrl+C to interrupt..."

wait $SERVER_PID
echo "Finished simulation of {job.name} with {workers} workers."
"""
    return cmd


def _get_slurm_cmd(
    build_settings: SimcommsysBuildSettings,
    job: SimcommsysJob,
) -> str:
    """
    Returns commands for a single simulation run in SLURM mode.

    Note: this command first builds simcommsys, then runs the simulation.
    """

    # build command to submit to SLURM.
    cmd = f"""
{_get_build_cmd(build_settings, "SimCommsys/SimCommsys", 1)}
{_get_simcommsys_cmd("$BINARY_FILENAME", job)}
"""
    return cmd


def _get_slurm_masterslave_cmd(
    build_settings: SimcommsysBuildSettings,
    job: SimcommsysJob,
    port: int,
    update_port_if_in_use: bool,
    workers: int,
) -> str:
    """
    Returns commands for a single simulation run in SLURM master-slave mode.

    Note: this command first builds simcommsys, then runs the simulation.
    """

    # build command to submit to SLURM.
    cmd = f"""
{_get_build_cmd(build_settings, "SimCommsys/SimCommsys", 1)}
{_get_masterslave_cmd("$BINARY_FILENAME", job, port, update_port_if_in_use, workers, None)}
"""
    return cmd
