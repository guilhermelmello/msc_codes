"""
MorphyNet statistics script.
"""
from collections import defaultdict

import csv
import os


def compute_derivation_stats(file_path, output_path):
    '''Extract statistics from MorphyNet's derivation file.'''
    print("Computing derivation statistcs")
    input_file = open(file_path, 'r', encoding="utf-8")
    input_reader = csv.reader(input_file, delimiter='\t')

    s_pos_count = defaultdict(int)
    t_pos_count = defaultdict(int)
    m_type_count = defaultdict(int)
    m_count = defaultdict(int)

    for row in input_reader:
        s_pos = row[2]
        t_pos = row[3]
        morpheme = row[4]
        m_type = row[5]

        s_pos_count[s_pos] += 1
        t_pos_count[t_pos] += 1
        m_type_count[m_type] += 1
        m_count[morpheme] += 1

    output_file = open(
        os.path.join(output_path, 'stats_reporter.txt'),
        'w', encoding="utf-8"
    )

    def report_print(message):
        output_file.write(message + "\n")
        print(message)

    report_print("Source POS\n")
    _print_stats_from_dict(report_print, s_pos_count)

    report_print("\nTarget POS\n")
    _print_stats_from_dict(report_print, t_pos_count)

    report_print("\nMorpheme Type\n")
    _print_stats_from_dict(report_print, m_type_count)

    def report(message):
        output_file.write(message + "\n")

    report(f"\nMorphemes: ({len(m_count)})\n")
    _print_stats_from_dict(report, m_count)

    output_file.close()


def compute_inflection_stats(file_path, output_path):
    '''Extract statistics from MorphyNet's inflection file.'''
    print("Computing inflection statistcs")
    input_file = open(file_path, 'r', encoding="utf-8")
    input_reader = csv.reader(input_file, delimiter='\t')

    pos_count = defaultdict(int)
    seg_count = defaultdict(int)

    for row in input_reader:
        m_features = row[2]
        m_segmentation = row[3].split("|")[1:]

        pos_tag = m_features[0]
        pos_count[pos_tag] += 1

        for seg in m_segmentation:
            seg_count[seg] += 1

    output_file = open(
        os.path.join(output_path, 'stats_reporter.txt'),
        'w', encoding="utf-8"
    )

    def report_print(message):
        output_file.write(message + "\n")
        print(message)

    report_print("POS Tags\n")
    _print_stats_from_dict(report_print, pos_count)
    report_print(f"\nMorphemes ({len(seg_count)})\n")
    _print_stats_from_dict(report_print, seg_count)

    output_file.close()


def _print_stats_from_dict(printer, d, maxlines=None):
    total = sum(d.values())
    sd = sorted(d.items(), key=lambda item: item[1], reverse=True)

    i = 0
    for tag, count in sd:
        printer(f"{tag} {count/total*100:5.2f}% ({count})")
        i += 1
        if(maxlines and i >= maxlines):
            return
