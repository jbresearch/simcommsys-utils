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

from typing import Any
from typing_extensions import Self
import re
import os
from glob import glob

from pydantic import BaseModel, model_validator
from pydantic import ConfigDict

from simcommsys_utils.executors import (
    SimcommsysExecutorType,
    SimcommsysJob,
    SimcommsysExecutor,
)


class JobBatchSpec(BaseModel):
    """
    Specification for a batch of jobs

    These are paired with executors in YAML config files to give
    complete way of running jobs
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Absolute path of the config file from which this job batch was parsed
    config_dir: str
    # "Base" directory of the input files to Simcommsys.
    # glob or rgx will match files relative to this directory.
    # If base_dir is not an absolute directory, then it will be considered relative to config_dir.
    # This parameter is optional and if not specified, it defaults to config_dir.
    base_dir: str | None
    #  Python regex string which is used to find simulator files underneath base_dir.
    # One of glob or rgx must be specified.
    # If both are specified, rgx takes precedent and glob is ignored.
    rgx: str | None
    # UNIX glob regex string which is used to find simulator files underneath base_dir.
    # One of glob or rgx must be specified.
    # If both are specified, rgx takes precedent and glob is ignored.
    glob: str | None
    # Output directory for Simcommsys runs.
    # If not absolute, the output directory is considered relative to config_dir.
    # Must be specified.
    output_dir: str
    # List of parameter combinations.
    # If specified, then for every simulator file matched by rgx or glob,
    # an individual Simcommsys run is invoked with every parameter combination in params.
    # Exactly one of params or param_ranges must be specified.
    params: list[list[float]] | None
    # List of strings containing parameter ranges.
    # Simcommsys accepts parameter ranges which obey the syntax "<start>:<step>:<stop>:arithmetic" or "<start>:<step>:<stop>:geometric"
    # where <start>, <step>, and <stop> are floats.
    # If specified, then each simulator file matched by rgx or glob is invoked a single time with the parameters specified in param_ranges.
    # Exactly one of params or param_ranges must be specified.
    param_ranges: list[str] | None
    # Value of the --confidence Simcommsys parameter that each simulation in the group is invoked with.
    # Must be specified.
    confidence: float
    # Value of the --relative-error Simcommsys parameter that each simulation in the group is invoked with.
    # Must be specified.
    relative_error: float
    # Value of the --floor-min Simcommsys parameter that each simulation in the group is invoked with.
    # Must be specified.
    floor_min: float
    # Simcommsys binary tag.
    # When Simcommsys is compiled, the produced binaries have the format simcommsys.<tag>.<build-type>,
    # where <tag> depends on the branch and certain build parameters.
    # simcommsys_tag should be set to the value of <tag> of the Simcommsys binary
    # that you wish to run each simulation with.
    # Must be specified.
    simcommsys_tag: str
    # Simcommsys build type.
    # When Simcommsys is compiled, the produced binaries have the format simcommsys.<tag>.<build-type>,
    # where <build-type> is either debug, release or profile.
    # simcommsys_tag should be set to the value of <tag> of the Simcommsys binary
    # that you wish to run each simulation with.
    # Must be specified.
    simcommsys_type: str
    executor_kwargs: dict[str, Any]

    @property
    def jobs(self) -> list[SimcommsysJob]:
        """
        Return all simcommsys jobs specified by this batch spec.
        """
        input_files: list[str] = []
        # base directory relative to which we search for input files
        base_dir: str = self.base_dir or self.config_dir

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

        if self.param_ranges is not None:
            # get list of jobs using param_ranges
            return [
                SimcommsysJob(
                    name=os.path.basename(jobfile.removesuffix(".txt")),
                    inputfile=jobfile,
                    outputfile=os.path.join(
                        self.output_dir,
                        os.path.basename(jobfile).removesuffix(".txt") + ".json",
                    ),
                    param_ranges=self.param_ranges,
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
                    name=os.path.basename(jobfile.removesuffix(".txt"))
                    + "."
                    + ":".join(map(lambda p: f"{p:e}", pset)),
                    inputfile=jobfile,
                    outputfile=os.path.join(
                        self.output_dir,
                        os.path.basename(jobfile).removesuffix(".txt")
                        + "."
                        + ":".join(map(lambda p: f"{p:e}", pset))
                        + ".json",
                    ),
                    param_ranges=[f"{p}:1:{p}:arithmetic" for p in pset],
                    confidence=self.confidence,
                    relative_error=self.relative_error,
                    floor_min=self.floor_min,
                )
                for jobfile in input_files
                for pset in self.params
            ]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Parse a JobBatchSpec object from a dict

        Once a YAML file has been parsed into a dict, that dict can
        be converted into models using this method
        """
        return cls(
            config_dir=d.pop("config_dir"),
            base_dir=d.pop("base_dir", None),
            rgx=d.pop("rgx", None),
            glob=d.pop("glob", None),
            output_dir=d.pop("output_dir"),
            params=d.pop("params", None),
            param_ranges=d.pop("param_ranges", None),
            confidence=d.pop("confidence"),
            relative_error=d.pop("relative_error"),
            floor_min=d.pop("floor_min"),
            simcommsys_tag=d.pop("simcommsys_tag"),
            simcommsys_type=d.pop("simcommsys_type"),
            # remaining elements in d will be passed to executor
            executor_kwargs=d,
        )

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

        We also make 'base_dir' and 'output_dir' absolute if we can
        """
        if self.base_dir is not None:
            # make base_dir absolute if it is not
            if not os.path.isabs(self.base_dir):
                self.base_dir = os.path.join(self.config_dir, self.base_dir)
            # check that base_dir exists
            if not os.path.isdir(self.base_dir):
                raise ValueError(
                    f"Base directory {self.base_dir} does not exist or is not a directory"
                )

        # check that config_dir exists
        if not os.path.isdir(self.config_dir):
            raise ValueError(
                f"Config directory {self.config_dir} does not exist or is not a directory"
            )

        # make output_dir absolute if it is not
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(self.config_dir, self.output_dir)
        # check that output_dir exists
        if not os.path.isdir(self.output_dir):
            raise ValueError(
                f"Output directory {self.output_dir} does not exist or is not a directory"
            )
        return self

    @model_validator(mode="after")
    def check_params_or_param_ranges_specified(self) -> Self:
        """
        Check that a range of params for use with simulation have
        been specified, either through params or param_ranges
        """
        if self.params is None and self.param_ranges is None:
            raise ValueError(
                "Must specify range of simulation params using 'params' or 'param_ranges'"
            )
        elif self.params is not None and self.param_ranges is not None:
            raise ValueError(
                "If specifying range of simulation params using 'param_ranges', you cannot specify 'params'"
            )
        return self

    @model_validator(mode="after")
    def check_param_ranges_valid(self) -> Self:
        """
        Check that parameter ranges specified using param_ranges have a valid syntax accepted by Simcommsys.
        """
        if self.param_ranges is not None:
            prange_rgx_str = r"^((\d+(.\d*)?)|(\d*.\d+)):((\d+(.\d*)?)|(\d*.\d+)):((\d+(.\d*)?)|(\d*.\d+)):((arithmetic)|(geometric))$"
            prange_rgx = re.compile(prange_rgx_str)
            for prange in self.param_ranges:
                if prange_rgx.match(prange) is None:
                    raise ValueError(
                        f"Invalid parameter range {prange} specified in param_ranges field, range must follow syntax {prange_rgx_str}"
                    )


class RunJobsSpec(BaseModel):
    """
    Specification for running a batch of Simcommsys jobs

    This is parsed from a YAML file
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    executor_type: SimcommsysExecutorType
    executor_kwargs: dict[str, Any]
    jobs: dict[str, JobBatchSpec]

    @property
    def executor(self) -> SimcommsysExecutor:
        """
        Obtain executor specified by this RunJobsSpec object
        """
        return self.executor_type.executor_type(**self.executor_kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Parse a RunJobSpec object from a dict

        Once a YAML file has been parsed into a dict, that dict can
        be converted into models using this method
        """
        return cls(
            executor_type=d["executor"].pop("type"),
            # remaining elements in executor dict passed as args to executor constructor
            executor_kwargs=d["executor"],
            jobs={
                groupname: JobBatchSpec.from_dict(
                    group | {"config_dir": d["config_dir"]}
                )
                for groupname, group in d["jobs"].items()
            },
        )
