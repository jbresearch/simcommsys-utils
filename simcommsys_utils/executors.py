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
from subprocess import CalledProcessError
from dataclasses import dataclass
import logging
import multiprocessing
from multiprocessing import cpu_count
import os
from enum import Enum


class SimcommsysBuildConfig(str, Enum):
    DEBUG = "debug"
    RELEASE = "release"
    PROFILE = "profile"


@dataclass
class SimcommsysJob:
    name: str
    inputfile: str
    outputfile: str
    logfile: str
    param_ranges: list[str]
    confidence: float
    relative_error: float
    floor_min: float


class SimcommsysExecutor(abc.ABC):
    def _get_build_cmd(
        self,
        build_version: str,
        plain_build: bool,
        build_config: SimcommsysBuildConfig,
        source_url: str,
        target_work_dir: str,
        target_bin_dir: str,
        workers: int = cpu_count()
    ):
        build_target = "plain-" if plain_build else ""
        build_target += f"install-main-{build_config.value}"
        cmd = f"""
set -euo pipefail
# Make sure work directory exists
mkdir -p {target_work_dir}
# Open file descriptor for lock
exec {{lock_fd}}>"{target_work_dir}/LOCK"
## CRITICAL SECTION START
flock -x "$lock_fd"
# Clone the repo and build the binaries
if [ -d {target_work_dir}/.git/ ]; then
    echo Existing working copy found, updating
    git -C {target_work_dir} fetch origin
else
    echo No working copy found, cloning
    git -C {target_work_dir} clone -n {source_url} .
fi
echo Checking out {build_version}
git -C {target_work_dir} checkout {build_version}
echo Building binaries into {target_bin_dir}
BINDIR={target_bin_dir} make -C {target_work_dir} -j {workers} {build_target}
echo Done building binaries
BUILD_VARIANT=$(make -C {target_work_dir} showbuildid 2>/dev/null)
echo Build variant is $BUILD_VARIANT
## CRITICAL SECTION END
flock -u "$lock_fd"
"""
        return cmd

    def _get_simcommsys_cmd(
        self,
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        job: SimcommsysJob,
    ):
        """
        Returns simcommsys command for a single simulation.

        NOTE that the -e flag is not given and must be provided by a subclass
        """
        param_ranges = ""
        for prange in job.param_ranges:
            param_ranges += f"--param-range {prange}"

        return f"simcommsys.{build_version}.{build_variant}.{build_config.value} \
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
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        jobs: list[SimcommsysJob],
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
    except CalledProcessError as e:
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
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        memlimit_gb: int | None = None,
    ):
        job_commands = []
        for job in jobs:
            # build command to submit to shell.
            simcommsys_cmd = self._get_simcommsys_cmd(
                build_version, build_variant, build_config, job
            )
            cmd = ""
            if memlimit_gb is not None:
                cmd += f"ulimit -v {memlimit_gb * 1024 * 1024} && "
            cmd += f"{simcommsys_cmd} -e local"
            job_commands.append((cmd, job))

        if dry_run:
            for cmd in job_commands:
                print(cmd)
        elif self.workers == 1 or len(job_commands) == 1:
            # There is no point in using multiprocessing as there is only one command or one worker.
            for cmd, job in job_commands:
                _run_local_cmd(cmd, job.logfile)
        else:
            with multiprocessing.Pool(processes=self.workers) as p:
                p.map(_run_local_cmd_wrapper, job_commands)


class MasterSlaveSimcommsysExecutor(SimcommsysExecutor):

    def _get_masterslave_cmd(
        self,
        job: SimcommsysJob,
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        port: int,
        workers: int,
        memlimit_gb: int | None,
    ) -> str:
        # build command to submit to shell.
        simcommsys_cmd = self._get_simcommsys_cmd(
            build_version, build_variant, build_config, job)
        # logfile where output from workers will be stored.
        worker_logfile = os.path.join(
            os.path.dirname(job.logfile),
            f"{os.path.splitext(os.path.basename(job.logfile))[0]}-w$i.log"
        )
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

# Launch the server in the background
{simcommsys_cmd} -e :{port} 1>"{job.logfile}" 2>&1 &
SERVER_PID=$!

# Wait for server to open its port
timeout 30 sh -c 'until netcat -z localhost {port}; do sleep 1; done'

# Launch the workers in the background
for i in $(seq 1 {workers}); do
    echo "Starting worker $i"
    simcommsys.{build_version}.{build_variant}.{build_config.value} -q -e localhost:{port} 1>"{worker_logfile}" 2>&1 &
done

handler () {{
    echo 'Killing {job.name} server...'
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
        build_version: str,
        build_variant: str,
        build_config: SimcommsysBuildConfig,
        jobs: list[SimcommsysJob],
        dry_run: bool = False,
        *,
        memlimit_gb: int | None = None,
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
            logging.error(
                f"Invalid value supplied for port: {port}, must be integer.")
            exit(-1)

        for job in jobs:
            # build command to submit to shell.
            cmd = self._get_masterslave_cmd(
                job, build_version, build_variant, build_config,
                port, workers, memlimit_gb
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
        job_logfile: str,
        wrapped_command: str,
        node_index: int,
        dry_run: bool = False,
        *,
        partition: str,
        gres: list[str] | None,
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | str | None,
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
            sbatch_opts += f" --gres={','.join(gres)}"
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
        gres: list[str] | None = None,
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | str | None = None,
    ):
        nodelist = nodelist.split(",") if isinstance(
            nodelist, str) else nodelist

        index = 0
        for job in jobs:
            _run_slurm_args = dict(
                job_name=job.name,
                job_logfile=job.logfile,
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
        gres: list[str],
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
            logging.error(
                f"Invalid value supplied for port: {port}, must be integer.")
            exit(-1)

        index = 0
        for job in jobs:
            self._run_slurm(
                job.name,
                job.logfile,
                self._get_masterslave_cmd(job, build_version, build_variant,
                                          build_config, port, workers),
                index,
                dry_run=dry_run,
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
