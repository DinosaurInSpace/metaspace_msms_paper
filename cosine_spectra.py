#!/usr/bin/env python

"""
Generates cosine similarity score for two MS/MS spectra per GNPS implementation at UCSD:
https://github.com/CCMS-UCSD/GNPS_Workflows/blob/master/shared_code/spectrum_alignment.py#L84

"""

import pandas as pd
import argparse
import math
import bisect
from collections import namedtuple


__author__ = "Christopher M Baxter Rath"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def read_ms_file(ms_file):
    # Parses .ms output by series to [(mz, int)1, ... (mz, int)n]
    df = pd.read_csv(ms_file, sep="\t").iloc[:, 0:2]
    return list(df.apply(lambda x: (x.mz, x.intensity), axis=1))


#https://github.com/CCMS-UCSD/GNPS_Workflows/blob/master/shared_code/spectrum_alignment.py#L84
Match = namedtuple('Match', ['peak1', 'peak2', 'score'])
Peak = namedtuple('Peak',['mz','intensity'])
Alignment = namedtuple('Alignment', ['peak1', 'peak2'])


def alignment_to_match(spec1_n,spec2_n,alignment):
    s1_peak = spec1_n[alignment.peak1].intensity
    s2_peak = spec2_n[alignment.peak2].intensity
    match_score = s1_peak * s2_peak
    return Match(
            peak1 = alignment.peak1,
            peak2 = alignment.peak2,
            score = match_score)


def find_match_peaks_efficient(spec1, spec2, shift, tolerance):
    adj_tolerance =  tolerance + 0.000001
    spec2_mass_list = []

    for i,peak in enumerate(spec2):
        spec2_mass_list.append(peak.mz)

    alignment_mapping = []

    for i, peak in enumerate(spec1):
        left_mz_bound = peak.mz - shift - adj_tolerance
        right_mz_bound = peak.mz - shift + adj_tolerance

        left_bound_index = bisect.bisect_left(spec2_mass_list, left_mz_bound)
        right_bound_index = bisect.bisect_right(spec2_mass_list, right_mz_bound)

        for j in range(left_bound_index,right_bound_index):
            alignment_mapping.append(Alignment(i,j))
    return alignment_mapping


def convert_to_peaks(peak_tuples):
    #using the splat we can handle both size 2 lists and tuples
    return [Peak(*p) for p in peak_tuples]


def sqrt_normalize_spectrum(spectrum):
    output_spectrum = []
    intermediate_output_spectrum = []
    acc_norm = 0.0
    for s in spectrum:
        sqrt_intensity = math.sqrt(s.intensity)
        intermediate_output_spectrum.append(Peak(s.mz,sqrt_intensity))
        acc_norm += s.intensity
    normed_value = math.sqrt(acc_norm)
    for s in intermediate_output_spectrum:
        output_spectrum.append(Peak(s.mz,s.intensity/normed_value))
    return output_spectrum


def score_alignment(spec1,spec2,pm1,pm2,tolerance,max_charge_consideration=1):
    # https://github.com/CCMS-UCSD/GNPS_Workflows/blob/master/shared_code/spectrum_alignment.py#L84
    # Spec is [(mz,int), ... (mz,int)]
    # tolerance is in Dalton
    if len(spec1) == 0 or len(spec2) == 0:
        return 0.0, []

    spec1_n = sqrt_normalize_spectrum(convert_to_peaks(spec1))
    spec2_n = sqrt_normalize_spectrum(convert_to_peaks(spec2))
    shift = (pm1 - pm2)

    #zero_shift_alignments = find_match_peaks(spec1_n,spec2_n,0,tolerance)
    #real_shift_alignments = find_match_peaks(spec1_n,spec2_n,shift,tolerance)

    zero_shift_alignments = find_match_peaks_efficient(spec1_n,spec2_n,0,tolerance)
    real_shift_alignments = []
    if abs(shift) > tolerance:
        real_shift_alignments = find_match_peaks_efficient(spec1_n,spec2_n,shift,tolerance)

        if max_charge_consideration > 1:
            for charge_considered in range(2, max_charge_consideration + 1):
                real_shift_alignments += find_match_peaks_efficient(spec1_n,spec2_n,shift/charge_considered,tolerance)

    #Making real_shift_alignments without repetition
    real_shift_alignments = list(set(real_shift_alignments))

    zero_shift_match = [alignment_to_match(spec1_n,spec2_n,alignment) for alignment in zero_shift_alignments]
    real_shift_match = [alignment_to_match(spec1_n,spec2_n,alignment) for alignment in real_shift_alignments]

    all_possible_match_scores = zero_shift_match + real_shift_match
    all_possible_match_scores.sort(key=lambda x: x.score, reverse=True)

    reported_alignments = []

    spec1_peak_used = set()
    spec2_peak_used = set()

    total_score = 0.0

    for match in all_possible_match_scores:
        if not match.peak1 in spec1_peak_used and not match.peak2 in spec2_peak_used:
            spec1_peak_used.add(match.peak1)
            spec2_peak_used.add(match.peak2)
            reported_alignments.append(Alignment(match.peak1,match.peak2))
            total_score += match.score

    return total_score, reported_alignments


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