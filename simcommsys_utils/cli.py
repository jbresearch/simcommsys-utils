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

import logging
import os
import subprocess
from typing import Annotated, Literal, cast, List
import re
import sys
from enum import Enum
import json

import typer
from yaml import load
from pydantic import ValidationError

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type:ignore

from simcommsys_utils.run_jobs import RunJobsSpec
from simcommsys_utils.pchk import PchkMatrix, PchkMatrixFormat, ValuesMethod
from simcommsys_utils.executors import SimcommsysBuildType

app = typer.Typer()


@app.command()
def copy_binaries(
    remote: Annotated[
        str,
        typer.Argument(
            help="Remote host to copy binaries to. Format: <username>@<host>"
        ),
    ],
    simcommsys_tag: Annotated[
        str, typer.Option(help="Build tag of Simcommsys binaries to copy.")
    ] = "development",
    arch: Annotated[
        str, typer.Option(help="Architecture of target machine.")
    ] = "x86_64",
    binary: Annotated[
        List[str] | None, typer.Option(help="List of binaries to copy to remote.")
    ] = None,
    simcommsys_build_type: Annotated[
        List[SimcommsysBuildType] | None,
        typer.Option(help="List of Simcommsys types to copy to remote."),
    ] = None,
):
    """
    This is a utility command which can be used to copy Simcommsys binaries to a remote.
    """

    logging.info(f"Making directory {remote}:~/bin.{arch}...")
    subprocess.run(f"ssh {remote} 'mkdir -p bin.{arch}'", shell=True)

    # check simcommsys build types are valid
    simcommsys_build_type = set(simcommsys_build_type or [])
    for b in binary or []:
        for t in simcommsys_build_type:
            if os.path.isfile(f"~/bin.{arch}/{b}.{simcommsys_tag}.{t}"):
                logging.info(f"Copying {b}.{simcommsys_tag}.{t}...")
                subprocess.run(
                    f"scp ~/bin.{arch}/{b}.{simcommsys_tag}.{t} {remote}:~/bin.{arch}",
                    shell=True,
                )
            else:
                logging.warning(f"{b}.{simcommsys_tag}.{t} not found, skipping...")


class InputMode(Enum):
    """
    Specifies how a Simcommsys simulator should generate input data for the communication system.
    """

    ZERO = "zero"
    RANDOM = "random"
    USER = "user"

    @property
    def code(self) -> int:
        match self:
            case self.ZERO:
                return 0
            case self.RANDOM:
                return 1
            case self.USER:
                return 2
            case _:
                raise RuntimeError(
                    f"Unrecognized input mode for simulator specified: {self}"
                )


@app.command()
def make_simulators(
    input_dir: Annotated[
        str, typer.Argument(help="Input directory containing Simcommsys system files.")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Output directory where Simcommsys simulator files will be generated."
        ),
    ],
    resultscollector: Annotated[
        str, typer.Option(help="Results collector for the simulators.")
    ],
    input_mode: Annotated[
        InputMode,
        typer.Option(help="Specify input mode for the simulator."),
    ],
    analyze_decode_iters: Annotated[
        bool,
        typer.Option(
            help="Should results be collected for each seperate decode iteration? Ignored for stream simulators (effectively always set to true)."
        ),
    ] = True,
    stream_reset_n: Annotated[
        List[int] | None,
        typer.Option(
            help="List of stream lengths to work with for reset. Only applicable to commsys_stream. Default is [1]."
        ),
    ] = None,
    stream_term_n: Annotated[
        List[int] | None,
        typer.Option(
            help="List of stream lengths to work with for termination. Only applicable to commsys_stream. Default is [1]."
        ),
    ] = None,
    simcommsys_tag: Annotated[
        str,
        typer.Option(
            help="Build tag of Simcommsys binary used to get input alphabetsize."
        ),
    ] = "development",
    simcommsys_build_type: Annotated[
        SimcommsysBuildType | None,
        typer.Option(
            help="Build type of Simcommsys binary used to get input alphabetsize."
        ),
    ] = None,
):
    """
    Create Simcommsys simulator files from a set of Simcommsys system files in --input-dir.
    """

    simcommsys_build_type = simcommsys_build_type or SimcommsysBuildType.RELEASE

    if not os.path.isdir(input_dir):
        print(f"Input directory given {input_dir} does not exist.")
        exit(-1)
    if not os.path.isdir(output_dir):
        print(f"Output directory given {output_dir} does not exist.")
        exit(-1)

    input_mode_code = input_mode.code

    for sysfile in os.listdir(input_dir):
        commsys: str
        with open(os.path.join(input_dir, sysfile), "r") as fl:
            commsys = fl.read()

        if match := re.match(r"commsys_stream[^<]*<(.*),vector,(.*)>", commsys):

            # Simulators without parameters: open-ended streams
            with open(
                os.path.join(
                    output_dir, f"{resultscollector}-{input_mode.value}-{sysfile}"
                ),
                "w",
            ) as fl:

                fl.write(
                    f"""commsys_stream_simulator<{match.group(1)},{resultscollector},{match.group(2)}>
# Version
2
# Streaming mode (0=open, 1=reset, 2=terminated)
0
# Version
2
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                )

                if input_mode == InputMode.USER:
                    fl.write(
                        """#: input symbols - count
1
#: input symbols - values
1
"""
                    )

                fl.write(
                    f"""# Communication system
{commsys}"""
                )

            # Simulators with parameters: streams with reset
            for reset_frames in [int(x) for x in stream_reset_n or [1]]:
                with open(
                    os.path.join(
                        output_dir,
                        f"{resultscollector}-{input_mode.value}-stream-reset-{reset_frames}-{sysfile}",
                    ),
                    "w",
                ) as fl:
                    fl.write(
                        f"""commsys_stream_simulator<{match.group(1)},{resultscollector},{match.group(2)}>
# Version
2
# Streaming mode (0=open, 1=reset, 2=terminated)
1
# Number of frames to reset
{reset_frames}
# Version
2
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                    )

                    if input_mode == InputMode.USER:
                        fl.write(
                            """#: input symbols - count
1
#: input symbols - values
1
"""
                        )

                    fl.write(
                        f"""# Communication system
{commsys}"""
                    )

            # Simulators with parameters: streams with termination
            for term_frames in [int(x) for x in stream_term_n or [1]]:
                with open(
                    os.path.join(
                        output_dir,
                        f"{resultscollector}-{input_mode.value}-stream-term-{term_frames}-{sysfile}",
                    ),
                    "w",
                ) as fl:
                    fl.write(
                        f"""commsys_stream_simulator<{match.group(1)},{resultscollector},{match.group(2)}>
# Version
2
# Streaming mode (0=open, 1=reset, 2=terminated)
2
# Number of frames to terminate
{term_frames}
# Version
2
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                    )

                    if input_mode == InputMode.USER:
                        fl.write(
                            """#: input symbols - count
1
#: input symbols - values
1
"""
                        )

                    fl.write(
                        f"""# Communication system
{commsys}"""
                    )

        elif match := re.match(r"commsys[^<]*<(.*),(.*)>", commsys):
            alphabetsize: int
            # read alphabetsize from system file (we use simcommsys for this)
            with os.popen(
                f"getsystemparams.{simcommsys_tag}.{simcommsys_build_type.value} --param input-alphabetsize --system-file {os.path.join(input_dir, sysfile)} --type {match.group(1)} --container {match.group(2)}",
                mode="r",
            ) as pipe:
                data = json.load(pipe)
                alphabetsize = data["input-alphabetsize"]

            with open(
                os.path.join(
                    output_dir, f"{resultscollector}-{input_mode.value}-{sysfile}"
                ),
                "w",
            ) as fl:
                source: str
                match input_mode:
                    case InputMode.RANDOM:
                        source = f"""# Input data source
uniform<int, vector>
## Alphabet size
{alphabetsize}"""
                    case InputMode.ZERO:
                        source = """# Input data source
zero<int, vector>"""

                    case InputMode.USER:
                        source = """# Input data source
sequential<int, vector>
## Version
1
#: input symbols - count
1
#: input symbols - values
1"""

                fl.write(
                    f"""commsys_simulator<{match.group(1)},{resultscollector}>
# Version
4
# Analyze all decode iterations?
{1 if analyze_decode_iters else 0}
{source}
"""
                )
                if input_mode == InputMode.USER:
                    fl.write(
                        """#: input symbols - count
1
#: input symbols - values
1
"""
                    )

                fl.write(
                    f"""# Communication system
{commsys}"""
                )
        else:
            logging.warning(
                f"Skipping {sysfile} as it does not appear to be a valid system file."
            )


@app.command()
def make_timers(
    input_dir: Annotated[
        str, typer.Argument(help="Input directory containing Simcommsys system files.")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Output directory where Simcommsys timer files will be generated."
        ),
    ],
):
    """
    Create Simcommsys timer files from a set of Simcommsys system files in --input-dir.
    """

    if not os.path.isdir(input_dir):
        print(f"Input directory given {input_dir} does not exist.")
        exit(-1)
    if not os.path.isdir(output_dir):
        print(f"Output directory given {output_dir} does not exist.")
        exit(-1)

    for sysfile in os.listdir(input_dir):
        commsys: str
        with open(os.path.join(input_dir, sysfile), "r") as fl:
            commsys = fl.read()

        if match := re.match(r"commsys[^<]*<([^,>]*)", commsys):
            with open(
                os.path.join(output_dir, sysfile),
                "w",
            ) as fl:
                fl.write(
                    f"""commsys_timer<{match.group(1)}>
# Version
4
# Analyze all decode iterations?
0
# Input mode (0=zero, 1=random, 2=user[seq])
1
# Communication system
{commsys}"""
                )
        else:
            logging.warning(
                f"Skipping {sysfile} as it does not appear to be a valid system file."
            )


class ExitChartSimulatorType(Enum):
    """
    Type of exit chart simulator that user wants to generate through command make-exit-chart-simulators.
    """

    PARALLEL_CODEC = "parallel/codec"
    SERIAL_CODEC = "serial/codec"
    SERIAL_MODEM = "serial/modem"

    @property
    def code(self) -> int:
        match self:
            case self.PARALLEL_CODEC:
                return 0
            case self.SERIAL_CODEC:
                return 1
            case self.SERIAL_MODEM:
                return 2
            case _:
                raise RuntimeError(f"Unrecognized EXIT chart type {self}.")


@app.command()
def make_exit_chart_simulators(
    input_dir: Annotated[
        str, typer.Argument(help="Input directory containing Simcommsys system files.")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Output directory where Simcommsys exit chart simulator files will be generated."
        ),
    ],
    exit_chart_type: Annotated[
        ExitChartSimulatorType,
        typer.Option(help="Type of exit chart produced by the simulator."),
    ],
    system_param: Annotated[
        float,
        typer.Option(help="Specify system parameter for the exit chart simulator."),
    ],
):
    """
    Create Simcommsys exit chart simulator files from a set of Simcommsys system files in --input-dir.
    """

    if not os.path.isdir(input_dir):
        print(f"Input directory given {input_dir} does not exist.")
        exit(-1)
    if not os.path.isdir(output_dir):
        print(f"Output directory given {output_dir} does not exist.")
        exit(-1)

    exit_chart_type_code = exit_chart_type.code

    for sysfile in os.listdir(input_dir):
        commsys: str
        with open(os.path.join(input_dir, sysfile), "r") as fl:
            commsys = fl.read()

        if match := re.match(r"commsys[^<]*<([^,>]*)", commsys):
            with open(
                os.path.join(
                    output_dir,
                    f"exit_computer-{exit_chart_type_code}-{system_param}-{sysfile}",
                ),
                "w",
            ) as fl:
                # use exit chart computer simulator
                fl.write(
                    f"""exit_computer<{match.group(1)}>
# Version
2
# EXIT chart type (0=parallel/codec, 1=serial/codec, 2=serial/modem)
{exit_chart_type_code}
# Compute binary LLR statistics?
0
# System parameter
{system_param}
# Communication system
{commsys}
"""
                )
        else:
            logging.warning(
                f"Skipping {sysfile} as it does not appear to be a valid system file."
            )


class RealType(Enum):
    """
    Type of real to be used with LDPC code in invocation of make-ldpc-systems.
    """

    DOUBLE = "double"
    FLOAT = "float"


class SPAType(Enum):
    """
    SPA implementation to be used with LDPC code in invocation of make-ldpc-systems.
    """

    GDL = "gdl"
    GDL_CUDA = "gdl_cuda"
    TRAD = "trad"


@app.command()
def make_ldpc_systems(
    codes_dir: Annotated[
        str,
        typer.Argument(
            help="Path to directory of Simcommsys TXT files with LDPC codes."
        ),
    ],
    templates_dir: Annotated[
        str,
        typer.Argument(
            help="Path to directory with system template which the generated system files will be based on."
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Path to directory where generated system files will be placed."
        ),
    ],
    real_type: Annotated[
        List[RealType],
        typer.Option(
            help="Floating point types to use in LDPC instantiations. An instantiation will be generated for each of the specified types."
        ),
    ],
    gf: Annotated[
        List[str],
        typer.Option(
            help="Galois fields to use in LDPC instantiations. An instantiation will be generated for each of the specified fields."
        ),
    ],
    spa_type: Annotated[
        List[SPAType],
        typer.Option(
            help="SPA types to use in LDPC instantiations. An instantian will be generated for each of the specified SPA types."
        ),
    ],
):
    """
    Creates LDPC system files from the templates specified by --templates-dir and the LDPC codes
    specified by --codes-dir

    One system file is created for each combination of template file, code file, real type
    (specified by --real-type), Galois field (specified by --gf) and SPA type (specified by --spa-type).

    The system files are generated in the output directory specified by --output-dir and have a filename
    following the scheme <template-filename>.<gf>.<ldpc-code-filename>.<real-type>.<spa-type>.txt

    The template file should contain all system components up until the codec.

    If there are components which depend on the Galois field, the Galois field can be specified as
    {gf} in the template; this will then be substituted for the actual field when instantiating the
    template.
    """

    if not os.path.isdir(templates_dir):
        print(f"Templates directory given {templates_dir} does not exist.")
        exit(-1)
    if not os.path.isdir(codes_dir):
        print(f"Codes directory given {codes_dir} does not exist.")
        exit(-1)

    if not os.path.isdir(output_dir):
        print(f"Output directory given {output_dir} does not exist.")
        exit(-1)

    for template_file in os.listdir(templates_dir):
        if not template_file.endswith(".txt"):
            logging.warning(
                f"{template_file} does not appear to be a template file (as it does not have a .txt suffix), skipping..."
            )
            continue

        commsys_template: str
        with open(os.path.join(templates_dir, template_file), "r") as fl:
            commsys_template = fl.read()

        for code_file in os.listdir(codes_dir):
            # check if code_file is a valid parity check matrix file by parsing it.
            try:
                with open(os.path.join(codes_dir, code_file), "r") as fl:
                    PchkMatrix.read(fl, PchkMatrixFormat.SIMCOMMSYS)
            except Exception as e:
                logging.warning(
                    f"{code_file} does not appear to be a parity check matrix file (in Simcommsys format), skipping..."
                )
                continue

            code: str
            with open(os.path.join(codes_dir, code_file), "r") as fl:
                code = fl.read()

            for r in real_type:
                for s in spa_type:
                    for g in gf:
                        commsys = (
                            commsys_template.format(gf=g).removesuffix("\n")
                            + "\n"
                            + f"""## Codec
ldpc<{g},{r.value}>
# Version
5
# SPA type (trad|gdl|gdl_cuda)
{s.value}
# Number of iterations
100
# Clipping method (zero=replace only zeros, clip=replace values below almostzero)
clip
# Value of almostzero
1e-38
# Reduce generator matrix to REF? (true|false)
0
{code}
"""
                        )

                        with open(
                            os.path.join(
                                output_dir,
                                os.path.basename(template_file).removesuffix(".txt")
                                + "."
                                + g
                                + "."
                                + code_file.removesuffix(".txt")
                                + "."
                                + r.value
                                + "."
                                + s.value
                                + ".txt",
                            ),
                            "w",
                        ) as fl:
                            fl.write(commsys)


@app.command()
def check_pchk(
    *,
    input: Annotated[
        str,
        typer.Argument(help="Input file"),
    ],
    format: Annotated[
        PchkMatrixFormat,
        typer.Option(help="Format of the input file."),
    ],
    delimiter: Annotated[
        str,
        typer.Option(
            help="Delimiter of input. This is ignored if the input format is not 'flat'"
        ),
    ] = ",",
    transpose: Annotated[
        bool,
        typer.Option(
            help="Is the input transposed? This is ignored if the input format is not 'flat'"
        ),
    ] = False,
):
    """
    This command can be used to check whether a parity check matrix file is valid.

    \b
    The currently supported formats are:
    - alist: This is the format introduced by MacKay in http://www.inference.org.uk/mackay/codes/alist.html and subsequently extended for non-binary codes.
    - simcommsys: This is the format used internally by Simcommsys. It can accomodate binary/non-binary codes, and also specifies how non-binary values are generated if a binary code is to be used in a non-binary context.
    - flat: Matrix is specified in full with some delimiter seperating values in each row

    The command will automatically determine whether the input specifies a binary or non-binary LDPC code.
    """

    if not os.path.isfile(input):
        print(f"Input file given {input} does not exist.")
        exit(-1)

    try:
        with open(input) as fl:
            # passing fl in as Iterable[str] allows for lazy loading, which when reading large flat files off of disk is handy.
            PchkMatrix.read(fl, format, delimiter=delimiter, transpose=transpose)
    except Exception as e:
        print(f"Error while reading input file {input}: {e}")
        print(f"Ensure the input is a valid {format.value} file.")
        exit(-1)


@app.command()
def convert_pchk(
    *,
    input: Annotated[
        str,
        typer.Argument(help="Input file"),
    ],
    output: Annotated[
        str | None,
        typer.Argument(
            help="Output file. If no file is specified, the output is printed to stdout."
        ),
    ] = None,
    from_format: Annotated[
        PchkMatrixFormat,
        typer.Option(help="Format of the input file."),
    ],
    to_format: Annotated[
        PchkMatrixFormat, typer.Option(help="Required format for the output.")
    ],
    values_method: Annotated[
        ValuesMethod,
        typer.Option(
            help="Which values method should be used in the output? Ignored if the output format is not 'simcommsys'"
        ),
    ] = ValuesMethod.RANDOM,
    random_seed: Annotated[
        int | None,
        typer.Option(
            help="Random seed to be included in output if needed. Ignored if the output format is not 'simcommsys'"
        ),
    ] = None,
    in_delimiter: Annotated[
        str,
        typer.Option(
            help="Delimiter of input. This is ignored if the input format is not 'flat'"
        ),
    ] = ",",
    in_transpose: Annotated[
        bool,
        typer.Option(
            help="Is the input transposed? This is ignored if the input format is not 'flat'"
        ),
    ] = False,
    out_delimiter: Annotated[
        str,
        typer.Option(
            help="Delimiter of output. This is ignored if the output format is not 'flat'"
        ),
    ] = ",",
    out_transpose: Annotated[
        bool,
        typer.Option(
            help="Should the output be transposed? This is ignored if the output format is not 'flat'"
        ),
    ] = False,
):
    """
    This command can be used to convert one parity check matrix format to another.

    \b
    The currently supported formats are:
    - alist: This is the format introduced by MacKay in http://www.inference.org.uk/mackay/codes/alist.html and subsequently extended for non-binary codes.
    - simcommsys: This is the format used internally by Simcommsys. It can accomodate binary/non-binary codes, and also specifies how non-binary values are generated if a binary code is to be used in a non-binary context.
    - flat: Matrix is specified in full with some delimiter seperating values in each row

    The command tries to accomodate the format conversions specified by the user as best as possible
    given the input file.

    The command will automatically determine whether the input specifies a binary or non-binary LDPC code.
    """

    if not os.path.isfile(input):
        print(f"Input file given {input} does not exist.")
        exit(-1)
    if output is not None and not os.path.isdir(os.path.dirname(output)):
        print(f"Directory for output file given {output} does not exist.")
        exit(-1)

    try:
        with open(input) as fl:
            # passing fl in as Iterable[str] allows for lazy loading, which when reading large flat files off of disk is handy.
            pchk = PchkMatrix.read(
                fl, from_format, delimiter=in_delimiter, transpose=in_transpose
            )
    except Exception as e:
        print(f"Error while reading input file {input}: {e}")
        print(f"Ensure the input is a valid {from_format.value} file.")
        exit(-1)

    with open(output, "w") if output else sys.stdout as fl:
        fl.write(
            pchk.write(
                to_format,
                delimiter=out_delimiter,
                transpose=out_transpose,
                values_method=values_method,
                random_seed=random_seed,
            )
        )


@app.command()
def run_jobs(
    config_file: Annotated[
        str,
        typer.Option(
            help="Path to configuration file with details of Simcommsys jobs to run."
        ),
    ],
    group: Annotated[
        List[str] | None,
        typer.Option(
            help="Select groups of jobs to run from configuration file. All groups are run by default."
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Print commands that would be run to start jobs, but do not actually run them."
        ),
    ] = False,
):
    """
    This command can be used to run a batch of Simcommsys simulations.

    The user specifies a path to a configuration file (in YAML format) using --config-file. This file specifies the simulations to be run and must be in the following format:

    \b
    # specify executor (how Simcommsys simulations are run) and its details
    executor:
        type: slurm | local | masterslave
        # additional dynamic params that are specific to an executor type.
        [key: value]*
    # Specify simulations to be run.
    jobs:
        [
        <group-name>:
            # "Base" directory of the input files to Simcommsys.
            # glob or rgx will match files relative to this directory.
            # If base_dir is not an absolute directory, then it will be
            # considered relative to the directory containing the YAML job file.
            # This parameter is optional and if not specified, it defaults to
            # the directory containing YAML job file.
            base_dir: <str>
            # UNIX glob regex string which is used to find simulator files underneath
            # base_dir.
            # One of glob or rgx must be specified.
            # If both are specified, rgx takes precedent and glob is ignored.
            glob: <glob str>
            #  Python regex string which is used to find simulator files underneath
            # base_dir.
            # One of glob or rgx must be specified.
            # If both are specified, rgx takes precedent and glob is ignored.
            rgx: <rgx str>
            # Output directory for Simcommsys runs.
            # If not absolute, the output directory is considered relative to the
            # directory containing the YAML job file.
            # Must be specified.
            output_dir: <str>
            # List of parameter combinations.
            # If specified, then for every simulator file matched by rgx or glob,
            # an individual Simcommsys run is invoked with every parameter combination
            # in params.
            # Exactly one of params or param_ranges must be specified.
            params: <list of list of float>
            # List of strings containing parameter ranges.
            # Simcommsys accepts parameter ranges which obey the syntax
            # "<start>:<step>:<stop>:arithmetic" or "<start>:<step>:<stop>:geometric"
            # where <start>, <step>, and <stop> are floats.
            # If specified, then each simulator file matched by rgx or glob is
            # invoked a single time with the parameters specified in param_ranges.
            # Exactly one of params or param_ranges must be specified.
            param_ranges: <list of str>
            # Value of the --confidence Simcommsys parameter that each simulation
            # in the group is invoked with.
            # Must be specified.
            confidence: <float>
            # Value of the --relative-error Simcommsys parameter that each simulation
            # in the group is invoked with.
            # Must be specified.
            relative_error: <float>
            # Value of the --floor-min Simcommsys parameter that each simulation
            # in the group is invoked with.
            # Must be specified.
            floor_min: <float>
            # Simcommsys binary tag.
            # When Simcommsys is compiled, the produced binaries have the format
            # simcommsys.<tag>.<build-type>, where <tag> depends on the branch and
            # certain build parameters.
            # simcommsys_tag should be set to the value of <tag> of the Simcommsys binary
            # that you wish to run each simulation with.
            # Must be specified.
            simcommsys_tag: <str>
            # Simcommsys build type.
            # When Simcommsys is compiled, the produced binaries have the format
            # simcommsys.<tag>.<build-type>, where <build-type> is either debug, release
            # or profile.
            # simcommsys_tag should be set to the value of <tag> of the Simcommsys binary
            # that you wish to run each simulation with.
            # Must be specified.
            simcommsys_type: <"release" | "debug" | "profile">
            # additional dynamic params that are specific to an executor type.
            [key: value]*
        ]*

    As shown in the above specification, Simcommsys simulations are grouped into named groups
    within the configuration file. The user can run just a select few of the groups specified
    in the configuration file using the --group option (by default all groups are run).

    For example if the configuration file contains the groups with names "cpu-eccperf",
    "gpu-eccperf", "cpu-timings" and "gpu-timings", the user can choose to run the simulations
    in the first two groups only by specifying --group cpu-eccperf --group gpu-eccperf.
    """

    if not os.path.isfile(config_file):
        print(f"Specified configuration file {config_file} does not exist.")
        exit(-1)

    config: RunJobsSpec
    try:
        with open(config_file, "r") as fl:
            d = load(fl, Loader=Loader)
            config = RunJobsSpec.model_validate(
                d | {"config_dir": os.path.realpath(os.path.dirname(config_file))}
            )

    except ValidationError as e:
        print(e)
        exit(-1)

    try:
        executor = config.executor
    except TypeError as e:
        print(f"Error while initializing executor: {e}")
        exit(-1)

    # by default all group names are selected
    selected_groups: set[str] = set(config.jobs.keys())
    if group is not None:
        selected_groups = set(group)

    for groupname, jobsspec in config.jobs.items():
        if groupname in selected_groups:
            executor.run(
                jobsspec.simcommsys_tag,
                cast(Literal["debug", "release"], jobsspec.simcommsys_type),
                jobs=jobsspec.jobs,
                dry_run=dry_run,
                **jobsspec.executor_kwargs,
            )
