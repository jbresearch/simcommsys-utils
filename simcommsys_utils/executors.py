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

import abc
from pathlib import Path
import subprocess
from dataclasses import dataclass
import logging
import multiprocessing
from multiprocessing import cpu_count
from enum import Enum
from typing import Any

from simcommsys_utils.build import SimcommsysBuildConfig, SimcommsysBuildSettings, build_binary


def _validate_param_as_int(value: Any, param_name: str) -> int:
    """
    Validates if a given value can be converted to an integer and returns it.
    Logs an error and exits if the conversion fails.
    """
    try:
        return value if isinstance(value, int) else int(value)
    except (ValueError, TypeError):
        logging.error(
            f"Invalid value supplied for {param_name}: {value}, must be an integer."
        )
        exit(-1)


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


class SimcommsysExecutor(abc.ABC):
    def _get_simcommsys_cmd(
        self,
        binary_filename: Path,
        job: SimcommsysJob,
    ):
        """
        Returns simcommsys command for a single simulation.

        NOTE that the -e flag is not given and must be provided by a subclass
        """
        param_ranges = ""
        for prange in job.param_ranges:
            param_ranges += f"--param-range {prange}"

        return f"{binary_filename} \
                    -i {job.inputfile} \
                    -o {job.outputfile} \
                    {param_ranges} \
                    --confidence {job.confidence} \
                    --relative-error {job.relative_error} \
                    --floor-min {job.floor_min} \
                    -f json \
                    -q"

    @abc.abstractmethod
    def run(
        self,
        jobs: list[SimcommsysJob],
        build_settings: SimcommsysBuildSettings,
        dry_run: bool = False,
        **jobparams,
    ):
        """
        Runs several simcommsys simulations.
        """
        raise NotImplementedError


def _run_local_cmd_wrapper(job_command: tuple[str, SimcommsysJob]):
    cmd, job = job_command
    _run_local_cmd(cmd, job)


def _run_local_cmd(cmd: str, job: SimcommsysJob):
    """
    Run a command using subprocess. Command is run in a shell and we error out on failure.
    """
    logging.debug(cmd)
    try:
        print(f"Starting local simulation of {job.name}.")
        with open(job.logfile, "a") as fid:
            subprocess.run(cmd, shell=True, check=True,
                           stdout=fid, stderr=subprocess.STDOUT)
        print(f"Finished local simulation of {job.name}.")
    except subprocess.CalledProcessError as e:
        # catch error in subprocess and print useful human readable info.
        # TODO: check if this is still useful now that we redirect output
        logging.error(str(e))
        if e.stderr:
            logging.error(f"stderr of failed process is:")
            for line in e.stderr.split("\n"):
                logging.error(f"\t{line}")
        if e.output:
            logging.error(f"stdout of failed process is:")
            for line in e.output.split("\n"):
                logging.error(f"\t{line}")

        # crash since one of the jobs failed.
        exit(-1)


class LocalSimcommsysExecutor(SimcommsysExecutor):
    def __init__(self, workers: int | None = None):
        self.workers = workers or cpu_count()

    def run(  # type:ignore
        self,
        jobs: list[SimcommsysJob],
        build_settings: SimcommsysBuildSettings,
        dry_run: bool = False,
        *,
        memlimit_gb: int | None = None,
    ):
        # first compile the simulator
        binary_filename = build_binary(
            build_settings, "SimCommsys/SimCommsys", self.workers)

        # prepare list of simulation runs to do
        job_commands = []
        for job in jobs:
            # build command to submit to shell.
            simcommsys_cmd = self._get_simcommsys_cmd(binary_filename, job)
            cmd = ""
            if memlimit_gb is not None:
                cmd += f"ulimit -v {memlimit_gb * 1024 * 1024} && "
            cmd += f"{simcommsys_cmd} -e local"
            # add this job
            job_commands.append((cmd, job))

        if dry_run:
            for cmd in job_commands:
                print(cmd)
        elif self.workers == 1 or len(job_commands) == 1:
            # There is no point in using multiprocessing as there is only one command or one worker.
            for cmd, job in job_commands:
                _run_local_cmd(cmd, job)
        else:
            with multiprocessing.Pool(processes=self.workers) as p:
                p.map(_run_local_cmd_wrapper, job_commands)


class MasterSlaveSimcommsysExecutor(SimcommsysExecutor):

    def _get_masterslave_cmd(
        self,
        binary_filename: Path,
        job: SimcommsysJob,
        port: int,
        update_port_if_in_use: bool,
        workers: int,
        memlimit_gb: int | None,
    ) -> str:
        # build command to submit to shell.
        simcommsys_cmd = self._get_simcommsys_cmd(binary_filename, job)
        # logfile where output from workers will be stored.
        worker_logfile = job.logfile.with_name(f"{job.logfile.stem}-w$i.log")
        ulimit_cmd = ""
        if memlimit_gb is not None:
            ulimit_cmd = f"ulimit -v {memlimit_gb * 1024 * 1024}"
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
    {simcommsys_cmd} -e :$port 1>"{job.logfile}" 2>&1 &
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

    def run(  # type:ignore
        self,
        jobs: list[SimcommsysJob],
        build_settings: SimcommsysBuildSettings,
        dry_run: bool = False,
        *,
        memlimit_gb: int | None = None,
        port: int | str,
        update_port_if_in_use: bool = False,
        workers: int | str = cpu_count(),
    ):
        workers = _validate_param_as_int(workers, "workers")
        port = _validate_param_as_int(port, "port")

        # first compile the simulator
        binary_filename = build_binary(
            build_settings, "SimCommsys/SimCommsys", workers)
        # start simulations
        for job in jobs:
            # build command to submit to shell.
            cmd = self._get_masterslave_cmd(
                binary_filename, job,
                port, update_port_if_in_use, workers, memlimit_gb
            )
            if dry_run:
                print(cmd)
            else:
                logging.debug(cmd)
                subprocess.run(cmd, shell=True, check=True)


class SlurmSimcommsysExecutor(SimcommsysExecutor):
    def __init__(self, *, email: str, account: str, cancel_on_failure: bool = True):
        # SLURM specific global options.
        self.email = email
        self.account = account
        self.cancel_on_failure = cancel_on_failure

    def _run_slurm(
        self,
        job_name: str,
        job_logfile: Path,
        wrapped_command: str,
        node_index: int,
        dry_run: bool = False,
        *,
        partition: str,
        gres: str | None,
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | None,
        cpus_per_task: int = 1,
    ):
        """
        Run a command as a slurm job using --wrap
        """

        # build sbatch command for this job
        nodename = nodelist[node_index] if nodelist else None
        sbatch_opts = f"--mail-user={self.email} \
                        --job-name={job_name} \
                        --partition={partition} \
                        --ntasks=1 \
                        --cpus-per-task={cpus_per_task} \
                        --mem-per-cpu={memlimit_gb}G \
                        --time={timeout_mins} \
                        --output={job_logfile} \
                        --error={job_logfile} \
                        --account={self.account} \
                        --mail-type=all"

        if gres:
            sbatch_opts += f" --gres={gres}"
        if nodename:
            sbatch_opts += f" --nodelist={nodename}"
        cmd = f"sbatch {sbatch_opts} --wrap='{wrapped_command}'"

        if dry_run:
            print(cmd)
        else:
            logging.debug(cmd)
            subprocess.run(cmd, shell=True, check=self.cancel_on_failure)

    def run(  # type:ignore
        self,
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        partition: str,
        gres: str | None = None,
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | None = None,
    ):
        nodelist = nodelist.split(",") if isinstance(
            nodelist, str) else nodelist

        index = 0
        for job in jobs:
            _run_slurm_args = dict(
                job_name=job.name,
                job_logfile=job.logfile.with_name(
                    job.logfile.stem + '-slurm.log'),
                wrapped_command=self._get_simcommsys_cmd(
                    build_version, build_variant, build_config, job
                )
                + " -e local",
                node_index=index,
                dry_run=dry_run,
                partition=partition,
                gres=gres,
                memlimit_gb=memlimit_gb,
                timeout_mins=timeout_mins,
                nodelist=nodelist,
            )
            if self.cancel_on_failure:
                try:
                    self._run_slurm(**_run_slurm_args)
                except Exception as e:
                    # cancel all jobs in this group which have already been scheduled
                    for job in jobs:
                        subprocess.run(
                            f"scancel --name={job.name}", shell=True, check=False
                        )
                    raise e
            else:
                self._run_slurm(**_run_slurm_args)

            # update node index so that we allocate to every node in nodelist.
            if nodelist:
                index = (index + 1) % len(nodelist)


class SlurmMasterSlaveSimcommsysExecutor(
    SlurmSimcommsysExecutor, MasterSlaveSimcommsysExecutor
):
    def run(  # type:ignore
        self,
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        partition: str,
        gres: str | None = None,
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | None = None,
        port: int | str,
        update_port_if_in_use: bool = False,
        workers: int | str,
    ):  # type: ignore
        workers = _validate_param_as_int(workers, "workers")
        port = _validate_param_as_int(port, "port")

        index = 0
        for job in jobs:
            self._run_slurm(
                job.name,
                job.logfile.with_name(job.logfile.stem + '-slurm.log'),
                self._get_masterslave_cmd(job, build_version, build_variant,
                                          build_config,
                                          port, update_port_if_in_use,
                                          workers, memlimit_gb),
                index,
                dry_run=dry_run,
                partition=partition,
                gres=gres,
                memlimit_gb=memlimit_gb,
                timeout_mins=timeout_mins,
                nodelist=nodelist,
                cpus_per_task=workers,
            )
            if nodelist:
                index = (index + 1) % len(nodelist)


class SimcommsysExecutorType(str, Enum):
    """
    String tag for Simcommsys executor

    Update as more executors are added.
    We subclass from str so we can use enum values as strings without
    needing to access .value prop or call str():
    https://stackoverflow.com/questions/58608361/string-based-enum-in-python
    """

    LOCAL = "local"
    MASTERSLAVE = "masterslave"
    SLURM = "slurm"
    SLURM_MASTERSLAVE = "slurm_masterslave"

    @property
    def executor_type(self) -> type[SimcommsysExecutor]:
        match self:
            case self.LOCAL:
                return LocalSimcommsysExecutor
            case self.MASTERSLAVE:
                return MasterSlaveSimcommsysExecutor
            case self.SLURM:
                return SlurmSimcommsysExecutor
            case self.SLURM_MASTERSLAVE:
                return SlurmMasterSlaveSimcommsysExecutor
