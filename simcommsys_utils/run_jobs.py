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
from pathlib import Path
from typing_extensions import Self
import re
from glob import glob
import logging

from pydantic import BaseModel, ConfigDict, Field, model_validator, StringConstraints, AfterValidator

from simcommsys_utils.executors import (
    SimcommsysExecutorType,
    SimcommsysJob,
    SimcommsysExecutor,
    SimcommsysBuildConfig,
)
from simcommsys_utils.executors import SimcommsysBuildSettings


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


class BuildSettingsSpec(BaseModel):
    """
    Specification for Simcommsys build settings
    """
    # Working directory where Simcommsys will be cloned and built.
    # Normally this will be either an ignored subfolder of your project repo, or
    # somewhere in your home directory, common to all projects using Simcommsys.
    # Must be specified.
    working_dir: Path
    # The directory where Simcommsys binaries will be installed.
    # Normally this will be either a temporary folder (i.e. under /tmp) or
    # somewhere on the computational node's local storage.
    # Must be specified, and must be in your path.
    install_dir: Path
    # The source URL for Simcommsys.
    # Defaults to the official public repository, but can be set manually to the
    # development repo or a personal fork.
    simcommsys_repo: str = "git@github.com:jbresearch/simcommsys.git"
    # The branch (or commit reference) to be checked out.
    # Defaults to master, but can be set to another branch, tag, or hash of a
    # specific commit.
    simcommsys_branch: str = "master"
    # Simcommsys build configuration (release | debug | profile).
    # Used to specify which configuration of the simulator libraries and binary
    # need to be built and used.
    # Defaults to release.
    build_config: SimcommsysBuildConfig = SimcommsysBuildConfig.RELEASE
    # Simcommsys build architecture.
    # Used to specify which architecture to optimise for when building the
    # simulator libraries and binary.
    # Defaults to native (i.e. the architecture of the node in use).
    build_arch: str = "native"
    # Allow use of OpenMP features in Simcommsys.
    # Defaults to True.
    build_use_omp: bool = True
    # Allow use of OpenMPI features in Simcommsys (if available).
    # Defaults to True.
    build_use_mpi: bool = True
    # Allow use of GnuMP features in Simcommsys (if available).
    # Defaults to True.
    build_use_gmp: bool = True
    # Allow use of CUDA features in Simcommsys (if available).
    # Defaults to True.
    build_use_cuda: bool = True


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
    config_dir: Path
    # "Base" directory of the input files to Simcommsys.
    # glob or rgx will match files relative to this directory.
    # If base_dir is not an absolute directory, then it will be considered relative to config_dir.
    # This parameter is optional and if not specified, it defaults to config_dir.
    base_dir: Path | None = None
    #  Python regex string which is used to find simulator files underneath base_dir.
    # One of glob or rgx must be specified.
    # If both are specified, rgx takes precedent and glob is ignored.
    rgx: Annotated[Optional[str], AfterValidator(_is_valid_rgx)] = None
    # UNIX glob regex string which is used to find simulator files underneath base_dir.
    # One of glob or rgx must be specified.
    # If both are specified, rgx takes precedent and glob is ignored.
    glob: str | None = None
    # Output directory for Simcommsys results.
    # If not absolute, the output directory is considered relative to config_dir.
    # Must be specified.
    output_dir: Path
    # Output directory for Simcommsys logs.
    # If not absolute, the output directory is considered relative to config_dir.
    # Must be specified.
    log_dir: Path
    # List of parameter combinations.
    # If specified, then for every simulator file matched by rgx or glob,
    # an individual Simcommsys run is invoked with every parameter combination in params.
    # Exactly one of params or param_ranges must be specified.
    params: list[list[float]] | None = None
    # List of strings containing parameter ranges.
    # Simcommsys accepts parameter ranges which obey the syntax "arithmetic:<start>:<step>:<stop>" or "geometric:<start>:<step>:<stop>"
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
                    pattern=r"^((arithmetic)|(geometric)):((\d+(.\d*)?)|(\d*.\d+)):((\d+(.\d*)?)|(\d*.\d+)):((\d+(.\d*)?)|(\d*.\d+))$"
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
    # Simcommsys build settings
    build_settings_spec: BuildSettingsSpec = Field(alias="build_settings")

    @property
    def build_settings(self) -> SimcommsysBuildSettings:
        """
        Creates a SimcommsysBuildSettings object from the build_settings specification.
        """
        return SimcommsysBuildSettings(
            **self.build_settings_spec.model_dump()
        )

    executor_kwargs: dict[str, Any] = {}

    @property
    def jobs(self) -> list[SimcommsysJob]:
        """
        Return all simcommsys jobs specified by this batch spec.
        """
        input_files: list[Path] = []

        # get input files using either regex or glob
        if self.rgx is not None:
            r = re.compile(self.rgx)
            for path in self.base_dir.rglob("*"):
                if path.is_file() and r.match(str(path.relative_to(self.base_dir))):
                    input_files.append(path)

        else:
            input_files += [
                *self.base_dir.glob(self.glob)
            ]

        if len(input_files) == 0:
            logging.warning(
                f"In group {self.groupname} given rgx or glob pattern did not match any input files."
            )

        if self.param_ranges is not None:
            # get list of jobs using param_ranges
            return [
                SimcommsysJob(
                    name=jobfile.stem,
                    inputfile=jobfile,
                    outputfile=self.output_dir / (jobfile.stem + ".json"),
                    logfile=self.log_dir / (jobfile.stem + ".log"),
                    param_ranges=self.param_ranges,
                    confidence=self.confidence,
                    relative_error=self.relative_error,
                    floor_min=self.floor_min,
                )
                for jobfile in input_files
            ]
        else:
            # get list of jobs using params
            jobs = []
            for jobfile in input_files:
                for pset in self.params:
                    pset_str = "_".join(f"{p:g}" for p in pset)
                    name = f"{jobfile.stem}-{pset_str}"
                    jobs.append(SimcommsysJob(
                        name=name,
                        inputfile=jobfile,
                        outputfile=self.output_dir / f"{name}.json",
                        logfile=self.log_dir / f"{name}.log",
                        param_ranges=[f"arithmetic:{p}:1:{p+1}" for p in pset],
                        confidence=self.confidence,
                        relative_error=self.relative_error,
                        floor_min=self.floor_min,
                    ))
            return jobs

    @classmethod
    def model_validate(
        cls, obj: dict[str, Any], *, strict=None, from_attributes=None, context=None
    ):
        """
        Construct pydantic model from a dict
        """
        if "executor_kwargs" in obj:
            raise ValueError(
                "Cannot have executor_kwargs as field in job group.")
        # apply reflection, and place any additional keys in obj that are not a part of model into executor_kwargs
        flds = set(cls.model_fields.keys())
        obj["executor_kwargs"] = {k: obj.pop(k)
                                  for k in set(obj.keys()) - flds}
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )

    # Validators

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
    def check_dirs_exists(self) -> Self:
        """
        Checks that the required directories exist and convert them to absolute
        paths. These include 'base_dir', 'output_dir', 'log_dir', 'working_dir',
        and 'install_dir'.

        Note that if 'base_dir' is relative, we consider it relative to
        'config_dir', and if 'base_dir' is not given, we set it to 'config_dir'.
        All other directories are considered relative to 'base_dir', unless they
        are absolute.
        """

        # make sure config_dir is resolvable and absolute
        self.config_dir = self.config_dir.resolve()

        # set base_dir if necessary, and make it absolute
        if self.base_dir is None:
            self.base_dir = self.config_dir
        else:
            # first expand any user homedir
            self.base_dir = self.base_dir.expanduser()
            # make base_dir absolute if it is not
            if not self.base_dir.is_absolute():
                self.base_dir = (self.config_dir / self.base_dir).resolve()
            # check that base_dir exists
            if not self.base_dir.is_dir():
                raise ValueError(
                    f"Base directory {self.base_dir} does not exist or is not a directory"
                )

        # first expand any user homedir
        self.output_dir = self.output_dir.expanduser()
        # make output_dir absolute if it is not
        if not self.output_dir.is_absolute():
            self.output_dir = (self.base_dir / self.output_dir).resolve()
        # check that output_dir exists
        if not self.output_dir.is_dir():
            raise ValueError(
                f"Output directory {self.output_dir} does not exist or is not a directory"
            )

        # first expand any user homedir
        self.log_dir = self.log_dir.expanduser()
        # make log_dir absolute if it is not
        if not self.log_dir.is_absolute():
            self.log_dir = (self.base_dir / self.log_dir).resolve()
        # check that log_dir exists
        if not self.log_dir.is_dir():
            raise ValueError(
                f"Log directory {self.log_dir} does not exist or is not a directory"
            )

        # first expand any user homedir
        self.build_settings_spec.working_dir = self.build_settings_spec.working_dir.expanduser()
        # make working_dir absolute if it is not
        if not self.build_settings_spec.working_dir.is_absolute():
            self.build_settings_spec.working_dir = (
                self.base_dir / self.build_settings_spec.working_dir).resolve()

        # first expand any user homedir
        self.build_settings_spec.install_dir = self.build_settings_spec.install_dir.expanduser()
        # make install_dir absolute if it is not
        if not self.build_settings_spec.install_dir.is_absolute():
            self.build_settings_spec.install_dir = (
                self.base_dir / self.build_settings_spec.install_dir).resolve()

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
            obj["jobs"][groupname]["config_dir"] = Path(obj["config_dir"])
            obj["jobs"][groupname]["groupname"] = groupname
            obj["jobs"][groupname] = JobBatchSpec.model_validate(
                group, strict=strict, from_attributes=from_attributes, context=context
            )
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )
