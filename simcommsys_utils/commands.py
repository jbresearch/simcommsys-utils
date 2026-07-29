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

from simcommsys_utils.executors import SimcommsysJob


def _get_ulimit_cmd(memlimit_gb):
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

    simcommsys_cmd = str(f"{binary_filename}"
                         f" -i {job.inputfile}"
                         f" -o {job.outputfile}"
                         "".join(f" -r {pr}" for pr in job.param_ranges) +
                         f" --confidence {job.confidence}"
                         f" --relative-error {job.relative_error}"
                         f" --floor-min {job.floor_min}"
                         f" -f {job.output_format.value}"
                         f" -e {endpoint}" if endpoint is not None else ""
                         f" -q")

    return simcommsys_cmd


def _get_masterslave_cmd(
    binary_filename: Path,
    job: SimcommsysJob,
    port: int,
    update_port_if_in_use: bool,
    workers: int,
    memlimit_gb: int | None,
) -> str:
    # build command to submit to shell.
    ulimit_cmd = _get_ulimit_cmd(memlimit_gb)
    simcommsys_cmd = _get_simcommsys_cmd(binary_filename, job, ':$port')
    # logfile where output from workers will be stored.
    worker_logfile = job.logfile.with_name(f"{job.logfile.stem}-w$i.log")
    # the command:
    # 1. sets a memory limit using ulimit -v.
    # 2. starts the simcommsys master in a screen session
    # 3. starts the simcommsys slaves in the background, with stderr and stdout suppressed.
    # 4. Registers a handler for Ctrl+C that can kill the simcommsys server screen session
    # 5. Waits for the simcommsys server screen session to terminate or for the user to press Ctrl+C
    cmd = f"""set -e
echo "Starting simulation of {job.name} with {workers} workers."

{ulimit_cmd}

port={port}
while true; do
    echo "Attempting to use port $port"
    # Launch the server in the background
    {simcommsys_cmd} 1>"{job.logfile}" 2>&1 &
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
