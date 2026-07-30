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
import logging
import multiprocessing
from multiprocessing import cpu_count
from enum import Enum
from typing import Any

from simcommsys_utils.commands import (
    _critical_wrap,
    _get_build_cmd,
    _get_checkout_cmd,
    _get_local_cmd,
    _get_masterslave_cmd,
    _get_slurm_cmd,
    _get_slurm_masterslave_cmd,
    _safe_wrap,
)
from simcommsys_utils.models import (
    SimcommsysBuildSettings,
    SimcommsysJob,
)


def _validate_param_as_int(value: Any, param_name: str) -> int:
    """
    Validates if a given value can be converted to an integer and returns it.
    Logs an error and exits if the conversion fails.
    """
    try:
        return value if isinstance(value, int) else int(value)
    except (ValueError, TypeError):
        logging.error(f"Invalid value supplied for {param_name}: "
                      f"{value}, must be an integer.")
        exit(-1)


class SimcommsysExecutor(abc.ABC):
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


def _run_checkout_cmd(
    build_settings: SimcommsysBuildSettings,
) -> Path:
    cmd = _critical_wrap(_get_checkout_cmd(build_settings),
                         build_settings.working_dir)
    logging.debug(f"CHECKOUT: {cmd}")
    try:
        subprocess.run(_safe_wrap(cmd),
                       shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Checkout script failed with a non-zero exit code.")
        logging.error(f"stdout was: {e.stdout}")
        logging.error(f"stderr was: {e.stderr}")
        exit(-1)


def _run_build_cmd(
    build_settings: SimcommsysBuildSettings,
    build_target: str,
    workers: int
) -> Path:
    newline = "\n"
    cmd = f"""
exec 3>&1 # save original stdout
exec 1>&2 # redirect stdout to stderr for logging
# do the actual build
{_critical_wrap(newline.join(
        [_get_checkout_cmd(build_settings),
         _get_build_cmd(build_settings, build_target, workers)]),
        build_settings.working_dir)}
# print binary filename to original stdout
echo "$BINARY_FILENAME" >&3
exec 3>&- # close original stdout
"""
    logging.debug(f"BUILD: {cmd}")
    try:
        res = subprocess.run(_safe_wrap(cmd),
                             shell=True, executable="/bin/bash", check=True,
                             capture_output=True, text=True)
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


def _run_local_cmd_wrapper(job_command: tuple[str, SimcommsysJob]):
    cmd, job = job_command
    _run_local_cmd(cmd, job)


def _run_local_cmd(cmd: str, job: SimcommsysJob):
    """
    Run a command using subprocess. Command is run in a shell and we error out on failure.
    """
    logging.debug(f"RUN: {cmd}")
    try:
        print(f"Starting local simulation of {job.name}.")
        with open(job.logfile, "a") as fid:
            subprocess.run(_safe_wrap(cmd),
                           shell=True, executable="/bin/bash", check=True,
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
        binary_filename = _run_build_cmd(
            build_settings, "SimCommsys/SimCommsys", self.workers)

        # prepare list of simulation runs to do
        job_commands = []
        for job in jobs:
            # build command to submit to shell.
            cmd = _get_local_cmd(binary_filename, job, memlimit_gb)
            # add this job
            job_commands.append((cmd, job))

        if dry_run:
            for cmd, job in job_commands:
                logging.info(f"DRY RUN: {cmd}")
        elif self.workers == 1 or len(job_commands) == 1:
            # There is no point in using multiprocessing as there is only one command or one worker.
            for cmd, job in job_commands:
                _run_local_cmd(cmd, job)
        else:
            with multiprocessing.Pool(processes=self.workers) as p:
                p.map(_run_local_cmd_wrapper, job_commands)


class MasterSlaveSimcommsysExecutor(SimcommsysExecutor):

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
        binary_filename = _run_build_cmd(
            build_settings, "SimCommsys/SimCommsys", workers)

        # start simulations
        for job in jobs:
            # build command to submit to shell.
            cmd = _get_masterslave_cmd(
                binary_filename, job,
                port, update_port_if_in_use, workers, memlimit_gb
            )
            if dry_run:
                logging.info(f"DRY RUN: {cmd}")
            else:
                logging.debug(f"RUN: {cmd}")
                subprocess.run(_safe_wrap(cmd),
                               shell=True, executable="/bin/bash", check=True)


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
        cmd = str("sbatch " +
                  f"--job-name={job_name} " +
                  f"--partition={partition} " +
                  f"--account={self.account} " +
                  (f"--nodelist={nodename} " if nodename else "") +
                  f"--output={job_logfile} " +
                  f"--error={job_logfile} " +
                  f"--mail-user={self.email} " +
                  f"--mail-type=all " +
                  f"--ntasks=1 " +
                  f"--cpus-per-task={cpus_per_task} " +
                  f"--mem-per-cpu={memlimit_gb}G " +
                  (f"--gres={gres} " if gres else "") +
                  f"--time={timeout_mins} " +
                  f"--wrap='{wrapped_command}'")

        if dry_run:
            logging.info(f"DRY RUN: {cmd}")
        else:
            logging.debug(f"RUN: {cmd}")
            subprocess.run(_safe_wrap(cmd),
                           shell=True, executable="/bin/bash",
                           check=self.cancel_on_failure)

    def run(  # type:ignore
        self,
        jobs: list[SimcommsysJob],
        build_settings: SimcommsysBuildSettings,
        dry_run: bool = False,
        *,
        partition: str,
        gres: str | None = None,
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | None = None,
    ):
        # preprocess arguments
        nodelist = nodelist.split(",") if isinstance(
            nodelist, str) else nodelist

        # first checkout the simulator
        _run_checkout_cmd(build_settings)

        # start simulations
        index = 0
        for job in jobs:
            _run_slurm_args = dict(
                job_name=job.name,
                job_logfile=job.logfile.with_name(
                    job.logfile.stem + '-slurm.log'),
                wrapped_command=_get_slurm_cmd(build_settings, job),
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
                        subprocess.run(f"scancel --name={job.name}",
                                       shell=True, check=False)
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
        jobs: list[SimcommsysJob],
        build_settings: SimcommsysBuildSettings,
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
        # preprocess arguments
        workers = _validate_param_as_int(workers, "workers")
        port = _validate_param_as_int(port, "port")

        # first checkout the simulator
        _run_checkout_cmd(build_settings)

        # start simulations
        index = 0
        for job in jobs:
            self._run_slurm(
                job_name=job.name,
                job_logfile=job.logfile.with_name(
                    job.logfile.stem + '-slurm.log'),
                wrapped_command=_get_slurm_masterslave_cmd(
                    build_settings, job,
                    port, update_port_if_in_use, workers
                ),
                node_index=index,
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
