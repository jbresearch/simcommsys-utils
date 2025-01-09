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

from typing_extensions import Self
import re
import os
from glob import glob

from pydantic import BaseModel, model_validator

from simcommsys_utils.executors import SimcommsysExecutorType, SimcommsysJob


class JobBatchSpec(BaseModel):
    """
    Specification for a batch of jobs

    These are paired with executors in YAML config files to give
    complete way of running jobs
    """

    # Absolute path of the config file from which this job batch was parsed
    config_dir: str
    # Path of the base directory relative to which we search for input files
    # If not given we use config_dir as base_dir
    # Base directory should be absolute or specified relative to config_dir
    base_dir: str | None
    # Python regex specifying set of simulators to be run
    # Paths must be specified relative to config_dir
    # Ignored if glob is specified
    rgx: str | None
    # Glob pattern specifying set of simulators to be run
    # Paths must be specified relative to config_dir
    glob: str | None
    # Output directory where results of simulations will be placed
    # Must be relative to config_dir
    output_dir: str
    # If specified, a simulation with parameter set to each float in
    # the list. Ignored if start/stop/step are given
    # This is useful mostly when we want to run timers for a range of
    # system params
    params: list[float] | None
    # starting param of each simulation
    start: float | None
    # ending param of each simulation
    stop: float | None
    # step used for each simulation
    step: float | None
    # multiplicative step used for each simulation
    mul: float | None
    # Level of confidence for each simulation
    confidence: float
    # Relative error required for each simulation
    relative_error: float
    # Each simulation ends when error floor is lower than this value
    floor_min: float
    # Specify tag of Simcommsys binary to use, e.g. development-cuda86
    simcommsys_tag: str
    # Specify type of Simcommsys binary to use, should be release|debug|profile
    simcommsys_type: str
    executor_kwargs: dict[str, any]

    @property
    def jobs(self) -> list[SimcommsysJob]:
        """
        Return all simcommsys jobs specified by this batch spec.
        """
        input_files: list[str] = []
        # base directory relative to which we search for input files
        base_dir: str
        if self.base_dir is not None:
            if os.path.isabs(self.base_dir):
                base_dir = self.base_dir
            else:
                base_dir = os.path.join(self.config_dir, self.base_dir)
        else:
            base_dir = self.config_dir

        # find absolute path to output_dir
        output_dir: str
        if os.path.isabs(self.output_dir):
            output_dir = self.output_dir
        else:
            output_dir = os.path.join(self.config_dir, self.output_dir)

        # get input files using either regex or glob
        if self.rgx is not None:
            r = re.compile(self.rgx)
            for root, _, files in os.walk(base_dir):
                for name in files:
                    if r.match(name):
                        input_files.append(os.path.join(root, name))

        else:
            input_files += [
                os.path.join(base_dir, name)
                for name in glob(self.glob, root_dir=base_dir)
            ]

        if all(x is not None for x in [self.start, self.stop]):
            # get list of jobs using start/stop/step or mul
            return [
                SimcommsysJob(
                    name=os.path.basename(jobfile.removesuffix(".txt")),
                    inputfile=jobfile,
                    outputfile=os.path.join(
                        output_dir,
                        os.path.basename(jobfile).removesuffix(".txt") + ".json",
                    ),
                    start=self.start,
                    stop=self.stop,
                    step=self.step,
                    mul=self.mul,
                    confidence=self.confidence,
                    relative_error=self.relative_error,
                    floor_min=self.floor_min,
                )
                for jobfile in input_files
            ]
        else:
            # get list of jobs using params
            return [
                SimcommsysJob(
                    name=os.path.basename(jobfile.removesuffix(".txt")),
                    inputfile=jobfile,
                    outputfile=os.path.join(
                        output_dir,
                        os.path.basename(jobfile).removesuffix(".txt") + ".json",
                    ),
                    start=param,
                    stop=param,
                    step=1,
                    mul=self.mul,
                    confidence=self.confidence,
                    relative_error=self.relative_error,
                    floor_min=self.floor_min,
                )
                for jobfile in input_files
                for param in self.params
            ]

    ### Validators

    @model_validator(mode="after")
    def check_inputs_specified(self) -> Self:
        """
        Check that a set of input files has been specified using either rgx or glob
        Also check validity of regex if this is given
        """
        if self.rgx is not None:
            try:
                re.compile(self.rgx)
            except re.error as e:
                raise ValueError(f"'rgx' given for input files is invalid: {e}")
        elif self.glob is None:
            raise ValueError(
                "Must specify set of input files using either 'rgx' or 'glob'"
            )
        return self

    @model_validator(mode="after")
    def check_dirs_exist(self) -> Self:
        """
        Check that directories specified using 'output_dir' and 'config_dir' exist
        Also checks 'base_dir' if this is specified
        """
        if self.base_dir is not None and not os.path.isdir(self.base_dir):
            raise ValueError(
                f"Base directory {self.base_dir} does not exist or is not a directory"
            )
        if not os.path.isdir(self.config_dir):
            raise ValueError(
                f"Config directory {self.config_dir} does not exist or is not a directory"
            )
        if not os.path.isdir(self.output_dir):
            raise ValueError(
                f"Output directory {self.output_dir} does not exist or is not a directory"
            )
        return self

    @model_validator(mode="after")
    def check_params_specified(self) -> Self:
        """
        Check that a range of params for use with simulation have
        been specified, either through params or start/step/mul/stop
        """
        if all(x is None for x in [self.start, self.stop, self.step, self.mul]):
            if self.params is None:
                raise ValueError(
                    "Must specify range of simulation params using 'params' or 'start'/'step'/'stop'"
                )
        elif any(x is None for x in [self.start, self.stop, self.step or self.mul]):
            raise ValueError(
                "If specifying range of simulation params using 'start'/('step'|'mul')/'stop', all these fields are required"
            )
        return self


class RunJobsSpec(BaseModel):
    """
    Specification for running a batch of Simcommsys jobs

    This is parsed from a YAML file
    """

    executor_type: SimcommsysExecutorType
    executor_kwargs: dict[str, any]
    jobs: dict[str, JobBatchSpec]
