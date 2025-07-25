# Copyright (c) 2024 Mark Mizzi
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
import subprocess
from dataclasses import dataclass
from typing import Literal
import logging
from multiprocessing import cpu_count
import os
from enum import Enum


@dataclass
class SimcommsysJob:
    name: str
    inputfile: str
    outputfile: str
    start: float
    stop: float
    step: float | None
    mul: float | None
    confidence: float
    relative_error: float
    floor_min: float


class SimcommsysExecutor(abc.ABC):
    def _get_simcommsys_cmd(
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        job: SimcommsysJob,
    ):
        """
        Returns simcommsys command for a single simulation.

        NOTE that the -e flag is not given and must be provided by a subclass
        """
        assert job.step or job.mul, "Must specify step or mul in SimmcommsysJob object."
        step_or_mul_opt: str
        if job.step:
            step_or_mul_opt = f"--step={job.step}"
        else:
            step_or_mul_opt = f"--mul={job.mul}"

        return f"simcommsys.{simcommsys_tag}.{simcommsys_type} \
                    -i {job.inputfile} -o {job.outputfile} \
                    --start {job.start} --stop {job.stop} {step_or_mul_opt} \
                    --confidence {job.confidence} --relative-error {job.relative_error} \
                    --floor-min {job.floor_min} \
                    -f json"

    @abc.abstractmethod
    def run(
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        **jobparams,
    ):
        """
        Runs several simcommsys simulations on a remote.
        """
        pass


class SlurmSimcommsysExecutor(SimcommsysExecutor):
    def __init__(
        self,
        *,
        email: str,
        gpu_partition: str,
        cpu_partition: str,
        account: str,
    ):
        # SLURM specific global options.
        self.email = email
        self.gpu_partition = gpu_partition
        self.cpu_partition = cpu_partition
        self.account = account

    def _run_slurm(
        self,
        job_name: str,
        job_outputfile: str,
        wrapped_command: str,
        node_index: int,
        dry_run: bool = False,
        *,
        needs_gpu: bool,
        # NOTE: Defaults are provided so that we don't need to specify them in the case that needs_gpu = False.
        n_gpus: int = 1,
        gpuarch: str = "ampere",
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | str | None = None,
        cpus_per_task: int = 1,
    ):
        """
        Run a command as a slurm job using --wrap
        """

        # build sbatch command for this job
        nodename = nodelist[node_index] if nodelist else None
        sbatch_opts = f"--mail-user={self.email} \
                        --job-name={job_name} \
                        --partition={self.gpu_partition if needs_gpu else self.cpu_partition} \
                        --ntasks=1 \
                        --cpus-per-task={cpus_per_task} \
                        --mem-per-cpu={memlimit_gb}G \
                        --time={timeout_mins} \
                        --output={os.path.basename(job_outputfile).removesuffix('.json')}.out \
                        --error={os.path.basename(job_outputfile).removesuffix('.json')}.err \
                        --account={self.account} \
                        --mail-type=all"
        if needs_gpu:
            sbatch_opts += f" --gres=gpu:{gpuarch}:{n_gpus}"
        if nodename:
            sbatch_opts += f" --nodelist={nodename}"
        cmd = f"sbatch {sbatch_opts} --wrap='{wrapped_command}'"

        if dry_run:
            print(cmd)
        else:
            logging.debug(cmd)
            subprocess.run(cmd, shell=True)

    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        needs_gpu: bool,
        # NOTE: Defaults are provided so that we don't need to specify them in the case that needs_gpu = False.
        n_gpus: int = 1,
        gpuarch: str = "ampere",
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | str | None = None,
    ):
        nodelist = nodelist.split(",") if isinstance(nodelist, str) else nodelist

        index = 0
        for job in jobs:
            self._run_slurm(
                job.name,
                job.outputfile,
                self._get_simcommsys_cmd(simcommsys_tag, simcommsys_type, job)
                + " -e local",
                index,
                dry_run=dry_run,
                needs_gpu=needs_gpu,
                n_gpus=n_gpus,
                gpuarch=gpuarch,
                memlimit_gb=memlimit_gb,
                timeout_mins=timeout_mins,
                nodelist=nodelist,
            )
            if nodelist:
                index = (index + 1) % len(nodelist)


class LocalSimcommsysExecutor(SimcommsysExecutor):
    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        memlimit_gb=100,
    ):
        for job in jobs:
            # build command to submit to shell.
            simcommsys_cmd = self._get_simcommsys_cmd(
                simcommsys_tag, simcommsys_type, job
            )
            cmd = f"ulimit -v {memlimit_gb * 1024 * 1024} && {simcommsys_cmd} -e local"

            if dry_run:
                print(cmd)
            else:
                logging.debug(cmd)
                subprocess.run(cmd, shell=True)


class MasterSlaveSimcommsysExecutor(SimcommsysExecutor):

    def _get_cmd(
        self,
        job: SimcommsysJob,
        simcommsys_tag: str,
        simcommsys_type: str,
        port: int,
        workers: int,
        memlimit_gb: int,
    ) -> str:
        # build command to submit to shell.
        simcommsys_cmd = self._get_simcommsys_cmd(simcommsys_tag, simcommsys_type, job)

        # logfile where output from workers will be stored.
        logfile = os.path.join(
            os.path.dirname(job.outputfile), f"worker-$i.{job.name}.log"
        )
        launch_slaves = f"""
# Wait for server to open its port
timeout 30 sh -c 'until netcat -z localhost {port}; do sleep 1; done'

# Launch the workers in the bg
for i in $(seq 1 {workers})
do
    echo "Starting worker $i"
    simcommsys.{simcommsys_tag}.{simcommsys_type} -e localhost:{port} 1>"{logfile}" 2>&1 &
done
"""
        # the command:
        # 1. sets a memory limit using ulimit -v.
        # 2. starts the simcommsys master in a screen session
        # 3. starts the simcommsys slaves in the background, with stderr and stdout suppressed.
        # 4. Registers a handler for Ctrl+C that can kill the simcommsys server screen session
        # 5. Waits for the simcommsys server screen session to terminate or for the user to press Ctrl+C
        cmd = f"""set -e
ulimit -v {memlimit_gb * 1024 * 1024}
screen -d -m -S "{port}.{simcommsys_tag}.{simcommsys_type}" {simcommsys_cmd} -e :{port}
SERVER_PID=$!

{launch_slaves}

handler () {{
    echo 'Killing {job.name} server...'
    screen -XS "{port}.{simcommsys_tag}.{simcommsys_type}" quit
}}

trap handler INT

echo "Started simulation of {job.name} with {workers} workers."
echo "Press Ctrl+C to interrupt..."
while screen -list | grep -q "{port}.{simcommsys_tag}.{simcommsys_type}"
do
    sleep 1
done
"""
        return cmd

    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        memlimit_gb=100,
        port: int | str = 3008,
        workers: int | str = cpu_count(),
    ):
        try:
            workers = workers if isinstance(workers, int) else int(workers)
        except ValueError:
            logging.error(
                f"Invalid value supplied for workers: {workers}, must be integer."
            )
            exit(-1)

        try:
            port = port if isinstance(port, int) else int(port)
        except ValueError:
            logging.error(f"Invalid value supplied for port: {port}, must be integer.")
            exit(-1)

        for job in jobs:
            # build command to submit to shell.
            cmd = self._get_cmd(job, simcommsys_tag, simcommsys_type, port, workers)
            if dry_run:
                print(cmd)
            else:
                logging.debug(cmd)
                subprocess.run(cmd, shell=True)


class SlurmMasterSlaveSimcommsysExecutor(
    SlurmSimcommsysExecutor, MasterSlaveSimcommsysExecutor
):
    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        needs_gpu: bool,
        # NOTE: Defaults are provided so that we don't need to specify them in the case that needs_gpu = False.
        n_gpus: int = 1,
        gpuarch: str = "ampere",
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | str | None = None,
        port: int | str = 3008,
        workers: int | str = cpu_count(),
    ):
        try:
            workers = workers if isinstance(workers, int) else int(workers)
        except ValueError:
            logging.error(
                f"Invalid value supplied for workers: {workers}, must be integer."
            )
            exit(-1)

        try:
            port = port if isinstance(port, int) else int(port)
        except ValueError:
            logging.error(f"Invalid value supplied for port: {port}, must be integer.")
            exit(-1)

        index = 0
        for job in jobs:
            self._run_slurm(
                job.name,
                job.outputfile,
                self._get_cmd(job, simcommsys_tag, simcommsys_type, port, workers),
                index,
                dry_run=dry_run,
                needs_gpu=needs_gpu,
                n_gpus=n_gpus,
                gpuarch=gpuarch,
                memlimit_gb=memlimit_gb,
                timeout_mins=timeout_mins,
                nodelist=nodelist,
                cpus_per_task=workers + 1,
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

    SLURM = "slurm"
    LOCAL = "local"
    MASTERSLAVE = "masterslave"
    # master slave within a single SLURM job
    SLURM_MASTERSLAVE = "slurm_masterslave"

    @property
    def executor_type(self) -> type[SimcommsysExecutor]:
        match self:
            case self.SLURM:
                return SlurmSimcommsysExecutor
            case self.LOCAL:
                return LocalSimcommsysExecutor
            case self.MASTERSLAVE:
                return MasterSlaveSimcommsysExecutor
            case self.SLURM_MASTERSLAVE:
                return SlurmMasterSlaveSimcommsysExecutor
