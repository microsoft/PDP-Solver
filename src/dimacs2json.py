#!/usr/bin/env python3
"""
Auxiliary script for converting sets of DIMACS files into PDP's compact JSON format.
"""

# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import sys
import argparse

from os import listdir
from os.path import isfile, join, split, splitext

import numpy as np


class CompactDimacs:
    "Encapsulates a CNF file given in the DIMACS format."

    def __init__(self, dimacs_file, output, propagate):

        self.propagate = propagate
        self.file_name = split(dimacs_file)[1]

        with open(dimacs_file, 'r') as f:
            j = 0
            for line in f:
                seg = line.split(" ")
                if seg[0] == 'c':
                    continue

                if seg[0] == 'p':
                    var_num = int(seg[2])
                    clause_num = int(seg[3])
                    self._clause_mat = np.zeros((clause_num, var_num), dtype=np.int32)

                elif len(seg) <= 1:
                    continue
                else:
                    temp = np.array(seg[:-1], dtype=np.int32)
                    self._clause_mat[j, np.abs(temp) - 1] = np.sign(temp)
                    j += 1

        ind = np.where(np.sum(np.abs(self._clause_mat), 1) > 0)[0]
        self._clause_mat = self._clause_mat[ind, :]

        ind = np.where(np.sum(np.abs(self._clause_mat), 0) > 0)[0]
        self._clause_mat = self._clause_mat[:, ind]

        if propagate:
            self._clause_mat = self._propagate_constraints(self._clause_mat)

        self._output = output

    def _propagate_constraints(self, clause_mat):
        n = clause_mat.shape[0]
        if n < 2:
            return clause_mat

        length = np.tile(np.sum(np.abs(clause_mat), 1), (n, 1))
        intersection_len = np.matmul(clause_mat, np.transpose(clause_mat))

        temp = intersection_len == np.transpose(length)
        temp *= np.tri(*temp.shape, k=-1, dtype=bool)
        flags = np.logical_not(np.any(temp, 0))

        clause_mat = clause_mat[flags, :]

        n = clause_mat.shape[0]
        if n < 2:
            return clause_mat

        length = np.tile(np.sum(np.abs(clause_mat), 1), (n, 1))
        intersection_len = np.matmul(clause_mat, np.transpose(clause_mat))

        temp = intersection_len == length
        temp *= np.tri(*temp.shape, k=-1, dtype=bool)
        flags = np.logical_not(np.any(temp, 1))

        return clause_mat[flags, :]

    def to_json(self):
        clause_list = []
        clause_num, var_num = self._clause_mat.shape

        ind = np.nonzero(self._clause_mat)
        return [[var_num, clause_num], list((ind[1] + 1) * self._clause_mat[ind]),
                 list(ind[0] + 1), self._output, [self.file_name]]


def convert_directory(dimacs_dir, output_file, propagate, only_positive=False):
    file_list = [join(dimacs_dir, f) for f in listdir(dimacs_dir) if isfile(join(dimacs_dir, f))]

    with open(output_file, 'w') as f:
        for i in range(len(file_list)):
            if (splitext(file_list[i])[1]).lower() != '.dimacs':
                continue

            if len(file_list[i]) < 8:
                label = -1
            else:
                temp = file_list[i][-8]
                label = float(temp) if temp.isdigit() else -1

            if only_positive and label == 0:
                continue

            bc = CompactDimacs(file_list[i], label, propagate)
            f.write(str(bc.to_json()).replace("'", '"') + '\n')
            print("Generating JSON input file: %6.2f%% complete..." % (
                (i + 1) * 100.0 / len(file_list)), end='\r', file=sys.stderr)


def convert_file(file_name, output_file, propagate):
    with open(output_file, 'w') as f:
        if len(file_name) < 8:
            label = -1
        else:
            temp = file_name[-8]
            label = float(temp) if temp.isdigit() else -1

        bc = CompactDimacs(file_name, label, propagate)
        f.write(str(bc.to_json()).replace("'", '"') + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', action='store', type=str)
    parser.add_argument('out_file', action='store', type=str)
    parser.add_argument('-s', '--simplify', help='Propagate binary constraints', required=False, action='store_true', default=False)
    parser.add_argument('-p', '--positive', help='Output only positive examples', required=False, action='store_true', default=False)
    args = vars(parser.parse_args())

    convert_directory(args['in_dir'], args['out_file'], args['simplify'], args['positive'])
