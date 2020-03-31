#!/usr/bin/env python3
##########################################################################################
# Author: Scott Christley
# Date Started: 2019-10-03
# Purpose: Sequence alignment
##########################################################################################

import csv
import math

def load_similarity_matrix(filename):
    similarity_matrix = {}
    reader = csv.DictReader(open(filename, 'r'))
    entries = []
    for row in reader:
        entries.append(row)
    for k in reader.fieldnames:
        if len(k) < 1:
            continue
        similarity_matrix[k] = [ float(obj[k]) for obj in entries ]
        #print(k, similarity_matrix[k])
    return similarity_matrix

def print_matrix(m, cdr3):
    max_col = len(cdr3)
    print(' %11s' % '', end='')
    for col in range(0,max_col):
        print(' %11s' % cdr3[col], end='')
    print('')
    for row in range(0,33):
        for col in range(0,max_col+1):
            print(' %11.4f' % m[row][col], end='')
        print('')

def print_bp(bp, cdr3):
    max_col = len(cdr3)
    print(' %11s' % '', end='')
    for col in range(0,max_col):
        print(' %11s' % cdr3[col], end='')
    print('')
    for row in range(0,33):
        for col in range(0,max_col+1):
            print(' %11s' % bp[row][col], end='')
        print('')

def print_alignment(bp, cdr3):
    cdr3_align = []
    theta_align = []
    max_col = len(cdr3)
    col = max_col
    row = 32
    done = False
    while not done:
        #print(row, col, bp[row][col])
        if bp[row][col] == 'diag':
            theta_align.append(row)
            cdr3_align.append(cdr3[col - 1])
            row -= 1
            col -= 1
        elif bp[row][col] == 'up':
            theta_align.append(row)
            cdr3_align.append('.')
            row -= 1
        elif bp[row][col] == 'left':
            theta_align.append('.')
            cdr3_align.append(cdr3[col - 1])
            col -= 1
        else:
            print('ERROR')
        if row <= 0 or col <= 0:
            done = True
    if row != 0:
        for i in range(row,0,-1):
            theta_align.append(i)
            cdr3_align.append('.')
    align_str = ''
    for c in list(reversed(cdr3_align)):
        align_str += c
    #print(list(reversed(cdr3_align)))
    #print(list(reversed(theta_align)))
    return align_str

def do_alignment(sm, cdr3):
    theta_gap = 0
    cdr3_gap = -1000
    am = []
    bp = []
    for row in range(0,33):
        am.append([ 0.0 for col in range(0,33) ])
        bp.append([ None for col in range(0,33) ])

    max_col = len(cdr3) + 1
    score = 0
    for row in range(0,33):
        am[row][0] = score
        score += theta_gap
    score = 0
    for col in range(0,max_col):
        am[0][col] = score
        score += cdr3_gap

    for col in range(1,max_col):
        cdr3_pos = col - 1
        for row in range(1,33):
            theta_pos = row - 1
            up = am[row-1][col] + theta_gap
            diag = am[row-1][col-1] + sm[cdr3[cdr3_pos]][theta_pos]
            left = am[row][col-1] + cdr3_gap
            if up > diag:
                if up > left:
                    am[row][col] = up
                    bp[row][col] = 'up'
                else:
                    am[row][col] = left
                    bp[row][col] = 'left'
            else:
                if diag > left:
                    am[row][col] = diag
                    bp[row][col] = 'diag'
                else:
                    am[row][col] = left
                    bp[row][col] = 'left'
    return [am, bp]

def do_file_alignment(input, output, sm_tra, sm_trb, tag):
    reader = csv.DictReader(open(input, 'r'))
    fieldnames = reader.fieldnames.copy()
    fieldnames.append('tra_alignment_'+tag)
    fieldnames.append('tra_score_'+tag)
    fieldnames.append('trb_alignment_'+tag)
    fieldnames.append('trb_score_'+tag)
    writer = csv.DictWriter(open(output,'w'), fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        r = 32
        col = len(row['tra_cdr3'])
        tra_align = do_alignment(sm_tra, row['tra_cdr3'])
        row['tra_alignment_'+tag] = print_alignment(tra_align[1], row['tra_cdr3'])
        row['tra_score_'+tag] = tra_align[0][r][col] / math.sqrt(float(col))
        col = len(row['trb_cdr3'])
        trb_align = do_alignment(sm_trb, row['trb_cdr3'])
        row['trb_alignment_'+tag] = print_alignment(trb_align[1], row['trb_cdr3'])
        row['trb_score_'+tag] = trb_align[0][r][col] / math.sqrt(float(col))
        writer.writerow(row)

def test_alignment(sm, cdr3):
    align = do_alignment(sm, cdr3)
    print_matrix(align[0], cdr3)
    print_bp(align[1], cdr3)
    print(print_alignment(align[1], cdr3))

#treg_sm = load_similarity_matrix('similarity_table_treg.csv')
#naive_sm = load_similarity_matrix('similarity_table_naive.csv')

#do_file_alignment('data.csv', 'data_with_alignments.csv')

#test_alignment(treg_sm, 'CAWSGAGDREPDTQYF')
#test_alignment(naive_sm, 'CAWSGAGDREPDTQYF')
