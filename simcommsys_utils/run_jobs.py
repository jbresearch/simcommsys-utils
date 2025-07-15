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

from typing import Any, Annotated, Optional
from typing_extensions import Self
import re
import os
from glob import glob
import logging

from pydantic import BaseModel, model_validator, StringConstraints, AfterValidator
from pydantic import ConfigDict

from simcommsys_utils.executors import (
    SimcommsysExecutorType,
    SimcommsysJob,
    SimcommsysExecutor,
    SimcommsysBuildType,
)


def _is_valid_rgx(rgx: str | None):
    """
    Check that given rgx is valid.

    Returns rgx if it is valid (compiles correctly)
    Throws ValueError if it is incorrect.
    """
    if rgx is not None:
        try:
            re.compile(rgx)
        except re.error as e:
            raise ValueError(f"'rgx' is invalid: {e}")
    return rgx


def _is_dir(dir: str):
    """
    Check that given directory exists.

    Converts it to absolute dir if it does.
    """
    if not os.path.isdir(dir):
        raise ValueError(f"Config directory {dir} does not exist or is not a directory")
    return os.path.realpath(dir)


class JobBatchSpec(BaseModel):
    """
    Specification for a batch of jobs

    These are paired with executors in YAML config files to give
    complete way of running jobs
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # groupname given in the YAML file.
    groupname: str

    # Absolute path of the config file from which this job batch was parsed
    config_dir: Annotated[str, AfterValidator(_is_dir)]
    # "Base" directory of the input files to Simcommsys.
    # glob or rgx will match files relative to this directory.
    # If base_dir is not an absolute directory, then it will be considered relative to config_dir.
    # This parameter is optional and if not specified, it defaults to config_dir.
    base_dir: str | None = None
    #  Python regex string which is used to find simulator files underneath base_dir.
    # One of glob or rgx must be specified.
    # If both are specified, rgx takes precedent and glob is ignored.
    rgx: Annotated[Optional[str], AfterValidator(_is_valid_rgx)] = None
    # UNIX glob regex string which is used to find simulator files underneath base_dir.
    # One of glob or rgx must be specified.
    # If both are specified, rgx takes precedent and glob is ignored.
    glob: str | None = None
    # Output directory for Simcommsys runs.
    # If not absolute, the output directory is considered relative to config_dir.
    # Must be specified.
    output_dir: str
    # List of parameter combinations.
    # If specified, then for every simulator file matched by rgx or glob,
    # an individual Simcommsys run is invoked with every parameter combination in params.
    # Exactly one of params or param_ranges must be specified.
    params: list[list[float]] | None = None
    # List of strings containing parameter ranges.
    # Simcommsys accepts parameter ranges which obey the syntax "<start>:<step>:<stop>:arithmetic" or "<start>:<step>:<stop>:geometric"
    # where <start>, <step>, and <stop> are floats.
    # If specified, then each simulator file matched by rgx or glob is invoked a single time with the parameters specified in param_ranges.
    # Exactly one of params or param_ranges must be specified.
    param_ranges: (
        list[
            Annotated[
                str,
                StringConstraints(
                    # NOTE that this pattern enforces that strings given here are parseable by Simcommsys
                    # Must update when Simcommsys definition changes
                    pattern=r"^((\d+(.\d*)?)|(\d*.\d+)):((\d+(.\d*)?)|(\d*.\d+)):((\d+(.\d*)?)|(\d*.\d+)):((arithmetic)|(geometric))$"
                ),
            ]
        ]
        | None
    ) = None
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
    simcommsys_type: SimcommsysBuildType
    executor_kwargs: dict[str, Any] = {}

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
                    # make sure that filename we try to match is relative path to base_dir
                    if r.match(
                        os.path.relpath(os.path.join(root, name), start=base_dir)
                    ):
                        input_files.append(os.path.join(root, name))

        else:
            input_files += [
                os.path.join(base_dir, name)
                for name in glob(self.glob, root_dir=base_dir)
            ]

        if len(input_files) == 0:
            logging.warning(
                f"In group {self.groupname} given rgx or glob pattern did not match any input files."
            )

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
                    param_ranges=[f"{p}:1:{p+1}:arithmetic" for p in pset],
                    confidence=self.confidence,
                    relative_error=self.relative_error,
                    floor_min=self.floor_min,
                )
                for jobfile in input_files
                for pset in self.params
            ]

    @classmethod
    def model_validate(
        cls, obj: dict[str, Any], *, strict=None, from_attributes=None, context=None
    ):
        """
        Construct pydantic model from a dict
        """
        if "executor_kwargs" in obj:
            raise ValueError("Cannot have executor_kwargs as field in job group.")
        # apply reflection, and place any additional keys in obj that are not a part of model into executor_kwargs
        flds = set(cls.model_fields.keys())
        obj["executor_kwargs"] = {k: obj.pop(k) for k in set(obj.keys()) - flds}
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )

    ### Validators

    @model_validator(mode="after")
    def check_inputs_specified(self) -> Self:
        """
        Check that a set of input files has been specified using either rgx or glob
        """
        if self.rgx is None and self.glob is None:
            raise ValueError(
                "Must specify set of input files using either 'rgx' or 'glob'"
            )

        return self

    @model_validator(mode="after")
    def check_base_dir_exists(self) -> Self:
        """
        Checks that 'base_dir' exists and convert it to absolute path.

        Note that if 'base_dir' is relative, we consider it relative to 'config_dir'.
        If 'base_dir' is not given, we set it to 'config_dir'.
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
        else:
            self.base_dir = self.config_dir
        return self

    @model_validator(mode="after")
    def check_output_dir_exist(self) -> Self:
        """
        Check that 'output_dir' exists and turn it into an absolute path if it does

        If 'output_dir' is relative, we consider it relative to 'config_dir'
        """

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
    def model_validate(
        cls, obj: dict[str, Any], *, strict=None, from_attributes=None, context=None
    ):
        """
        Construct model from Python dict.

        We override the default method as we need to restructure the dictionary obtained from YAML before using it.
        """
        obj["executor_type"] = obj["executor"].pop("type")
        obj["executor_kwargs"] = obj.pop("executor")
        # pass config_dir and groupname to every dict in objs[jobs]
        for groupname, group in obj["jobs"].items():
            obj["jobs"][groupname]["config_dir"] = obj["config_dir"]
            obj["jobs"][groupname]["groupname"] = groupname
            obj["jobs"][groupname] = JobBatchSpec.model_validate(
                group, strict=strict, from_attributes=from_attributes, context=context
            )
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )
