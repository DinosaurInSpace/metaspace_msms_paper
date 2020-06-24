#!/usr/bin/env python

"""
Generates a psuedo-MS/MS spectra as: [(mz, int)1, ... (mz, int)n] from downloaded METASPACE
results containing annoted fragmetns and parents.

"""

import pandas as pd
import numpy as np
import argparse



__author__ = "Christopher M Baxter Rath"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def read_ms_file(ms_file):
    # Parses .ms output by series to [(mz, int)1, ... (mz, int)n]
    # Modify to write MS/MS file
    df = pd.read_csv(ms_file, sep="\t").iloc[:, 0:2]
    return list(df.apply(lambda x: (x.mz, x.intensity), axis=1))




def main():
    # Main captures input variables when called as command line script.
    parser = argparse.ArgumentParser(description='Generates cos score between 2 MS/MS spectra')
    parser.add_argument("--spec1",
                        default=None,
                        type=list,
                        help="MS/MS spectra as: [(mz, int)1, ... (mz, int)n]")
    parser.add_argument("--spec2",
                        default=None,
                        type=list,
                        help="MS/MS spectra as: [(mz, int)1, ... (mz, int)n]")
    parser.add_argument("--pm1",
                        default=None,
                        type=float,
                        help="Parent mass as float in Daltons.")
    parser.add_argument("--pm2",
                        default=None,
                        type=float,
                        help="Parent mass as float in Daltons.")
    parser.add_argument("--tolerance",
                        default=None,
                        type=float,
                        help="Mass error tolerance in Daltons..")
    parser.add_argument("--max_charge_consideration",
                        default=1,
                        type=int,
                        help="Max charge considered as int.")
    args = parser.parse_args()

    return score_alignment(args.spec1,
                           args.spec2,
                           args.pm1,
                           args.pm2,
                           args.tolerance,
                           args.max_charge_consideration
                           )


if __name__ == "__main__":
    main()