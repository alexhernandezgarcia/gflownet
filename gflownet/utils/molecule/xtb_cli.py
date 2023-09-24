import contextlib
import os
import re
import subprocess
import warnings

import numpy as np


def _get_energy(file):
    normal_termination = False
    with open(file) as f:
        for l in f:
            if "TOTAL ENERGY" in l:
                try:
                    energy = float(re.search(r"[+-]?(?:\d*\.)?\d+", l).group())
                except:
                    return np.nan
            if "normal termination of xtb" in l:
                normal_termination = True
    if normal_termination:
        return energy
    else:
        return np.nan


def run_gfn_xtb(
    filepath,
    filename,
    gfn_version="gfnff",
    opt=False,
    gfn_xtb_config: str = None,
    remove_scratch=True,
):
    """
    Runs GFN_XTB/FF given a directory and either a coord file or all coord files will be run

    :param filepath: Directory containing the coord file
    :param filename: if given, the specific coord file to run
    :param gfn_version: GFN_xtb version (default is 2)
    :param opt: optimization or single point (default is opt)
    :param gfn_xtb_config: additional xtb config (default is None)
    :param remove_scratch: remove xtb files
    :return:
    """
    xyz_file = os.path.join(filepath, filename)

    # optimization vs single point
    if opt:
        opt = "--opt"
    else:
        opt = ""

    # cd to filepath
    starting_dir = os.getcwd()
    os.chdir(filepath)

    file_name = str(xyz_file.split(".")[0])
    cmd = "xtb --{} {} {} {}".format(
        str(gfn_version), xyz_file, opt, str(gfn_xtb_config or "")
    )

    # run XTB
    with open(file_name + ".out", "w") as fd:
        subprocess.run(cmd, shell=True, stdout=fd, stderr=subprocess.STDOUT)

    # check XTB results
    if os.path.isfile(os.path.join(filepath, "NOT_CONVERGED")):
        # optimization not converged
        warnings.warn(
            "xtb --{} for {} is not converged, using last optimized step instead; proceed with caution".format(
                str(gfn_version), file_name
            )
        )

        # remove files
        if remove_scratch:
            os.remove(os.path.join(filepath, "NOT_CONVERGED"))
            os.remove(os.path.join(filepath, "xtblast.xyz"))
            os.remove(os.path.join(filepath, file_name + ".out"))
        energy = np.nan

    elif opt and not os.path.isfile(os.path.join(filepath, "xtbopt.xyz")):
        # other abnormal optimization convergence
        warnings.warn(
            "xtb --{} for {} abnormal termination, likely scf issues, using initial geometry instead; proceed with caution".format(
                str(gfn_version), file_name
            )
        )
        if remove_scratch:
            os.remove(os.path.join(filepath, file_name + ".out"))
        energy = np.nan

    else:
        # normal convergence
        # get energy
        energy = _get_energy(file_name + ".out")
        if remove_scratch:
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(filepath, file_name + ".out"))
                os.remove(os.path.join(filepath, "gfnff_charges"))
                os.remove(os.path.join(filepath, "gfnff_adjacency"))
                os.remove(os.path.join(filepath, "gfnff_topo"))
                os.remove(os.path.join(filepath, "xtbopt.log"))
                os.remove(os.path.join(filepath, "xtbopt.xyz"))
                os.remove(os.path.join(filepath, "xtbtopo.mol"))
                os.remove(os.path.join(filepath, "wbo"))
                os.remove(os.path.join(filepath, "charges"))
                os.remove(os.path.join(filepath, "xtbrestart"))
    os.chdir(starting_dir)
    return energy
