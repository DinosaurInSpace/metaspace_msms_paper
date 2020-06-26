#!/usr/bin/env python

"""
This script will a) download an existing METASPACE MS/MS report from
the server provided ds_id and db_id and then b) generate psuedo-MS/MS
spectra from the report, and c) compare them versus reference MS/MS
from core_metabolome_v3

Graph output is a) printed to stdout b) saved to local directory,
and c) returned as variable from plotting function.

This script is adapted from "metaspace_msms_mirror_spectra.ipynb"
"""

import argparse
import pandas as pd
import glob
from matplotlib import pyplot as plt
from pathlib import Path

from results_from_metaspace_msms_process import reporting_loop
from results_processing_on_dl_results import annotate_cos_parent_fragment
from results_processing_on_dl_results import spectral_encoder


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def assign_adduct(formula, polarity):
    # Assigns adduct for METASPACE MS/MS results.
    # Only supports: ['M-H-', 'M+H+', 'M+K+', 'M+Na+']
    if polarity == "negative":
        return 'M-H-'
    elif formula.find('Na') == True:
        return 'M+Na+'
    elif formula.find('K') == True:
        return 'M+K+'
    else:
        return 'M+H+'


def plot_spectra(df, out_path, psuedo_y_axis):
    # Plots mirrored MS/MS spectra!

    ref = list(df.spectra)
    test = list(df.psuedo_msms)

    mzs = [x[0] for x in ref] + [x[0] for x in test]

    y_ref = [x[1] for x in ref]
    y_test = [x[1] for x in test]

    y_ref_norm = [x / max(y_ref) * 100 for x in y_ref]
    y_test_norm = [x / max(y_test) * -100 for x in y_test]

    ys = y_ref_norm + y_test_norm

    title = df.ds_id + "_" + df.id + "_" + df.polarity + "_" + df.adduct

    Path(out_path + "/").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.stem(mzs, ys, markerfmt=' ')
    plt.title(title)
    plt.xlabel('m/z (Da)')
    plt.ylabel("top(ref.): intensity, bottom(pMSMS): " + psuedo_y_axis)
    plt.ylim((-100, 100))
    plt.savefig(out_path + "/" + title)
    plt.show()
    return fig, ax


def load_ref_df(df_or_path):
    # Can be input as df or path to df.
    if type(df_or_path) == str:
        return pd.read_pickle(df_or_path)
    elif isinstance(df_or_path, pd.DataFrame):
        return df_or_path
    else:
        exit(1)


def mirror_main(original_ds_id_on_prod, db_id_on_beta, ds_id_on_beta, path_to_reports,
                polarity, psuedo_y_axis, ref_spectra_df):
    # This is the main function, downloading the METASPACE-MS/MS report from beta,
    # and plotting reference MS/MS spectra versus psuedo-MS/MS spectra.
    ref_spectra_df = load_ref_df(ref_spectra_df)

    # Download results
    reporting_loop(original_ds_id_on_prod, db_id_on_beta, ds_id_on_beta, path_to_reports,
                   parent_and_fragment_req=True, fdr_max=0.5, save_image=False)

    # Annotate results with cosine similarity
    df_path = glob.glob(path_to_reports + ds_id_on_beta + "/*.pickle")[0]
    df = pd.read_pickle(df_path)
    df = annotate_cos_parent_fragment(path_to_reports, [ds_id_on_beta], df_path)

    ## Generate psuedo-MS/MS spectra from ISF data.
    par_df = df[df.par_or_frag == 'P'].copy(deep=True)
    par_df['psuedo_msms'] = par_df.apply(lambda x: spectral_encoder(df, x.ds_id, x.id_x, psuedo_y_axis),
                                         axis=1)
    par_df['polarity'] = polarity

    #Find matches between psuedo-MS/MS and reference MS/MS spectra
    par_df['adduct'] = par_df.apply(lambda x: assign_adduct(x.formula, x.polarity), axis=1)
    par_df = par_df.rename(columns={"id_x": 'id'})

    temp_df = ref_spectra_df[ref_spectra_df.id.isin(list(par_df.id))]
    temp_df = temp_df[temp_df.polarity == polarity]

    df1 = par_df[['id', 'adduct', 'psuedo_msms', 'polarity', 'ds_id']].copy(deep=True)
    df2 = temp_df[['id', 'adduct', 'spectra']].copy(deep=True)
    to_plot_df = df1.merge(df2, how="left", on=['id', 'adduct'])

    # Plot psuedo-MS/MS and reference MS/MS spectra
    path = path_to_reports + "spectra"
    return to_plot_df.apply(lambda x: plot_spectra(x, path, psuedo_y_axis), axis=1)


def main():
    # Main captures input variables when called as command line script.
    # Defaults should execute successfully for a test dataset.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--original_ds_id_on_prod", default="2020-03-12_17h55m21s", type=str, help="ds_id original ms1")
    parser.add_argument("--db_id_on_beta", default="2020-05-13_17h50m21s", type=str, help="database name for M-MS/MS")
    parser.add_argument("--ds_id_on_beta", default="2020-05-14_16h32m01s", type=str, help="ds_id for M-MS/MS")
    parser.add_argument("--path_to_reports", default="TEMP/reporting/", type=str, help="arbitrary path")
    parser.add_argument("--polarity", default="positive", type=str, help="Pick one: ['positive', 'negative']")
    parser.add_argument("--psuedo_y_axis", default="binary", type=str, help="Pick one: ['binary', 'fdr', 'msm', 'cos', 'intensity']")
    parser.add_argument("--ref_spectra_df", default="input/cm3_reference_spectra_df.pickle", type=str,
                        help="Can be filtered first.  For example, spotted compounds in manuscript")

    args = parser.parse_args()

    mirror_main(args.original_ds_id_on_prod,
                args.db_id_on_beta,
                args.ds_id_on_beta,
                args.path_to_reports,
                args.polarity,
                args.psuedo_y_axis,
                args.ref_spectra_df)

    print('Plotting mirrored spectra complete!')
    return 1


if __name__ == "__main__":
    main()