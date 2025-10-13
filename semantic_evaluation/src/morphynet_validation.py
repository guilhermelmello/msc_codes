"""
MorphyNet validation script.
"""
import csv
import os

import enchant

_BR_DICT = enchant.Dict("pt_BR")
_PT_DICT = enchant.Dict("pt_PT")

class Word:
    '''A class that represents a word and its information from MorphyNet data'''
    def __init__(self, lemma, word, morphemes):
        self.lemma = lemma
        self.word = word
        self.morpheme = morphemes
        self._is_ptbr = _BR_DICT.check(self.lemma) or _BR_DICT.check(self.word)
        self._is_ptpt = _PT_DICT.check(self.lemma) or _PT_DICT.check(self.word)

    @property
    def is_ptbr(self):
        '''Returns True if the word is a valid Brazilian Portuguese word.'''
        return self._is_ptbr

    @property
    def is_ptpt(self):
        '''Returns True if the word is a valid European Portuguese word.'''
        return self._is_ptpt

    @property
    def is_pt(self):
        '''Returns True if the word is a valid Portuguese word (Brazilian ou European).'''
        return self._is_ptbr or self._is_ptpt

    def __str__(self):
        return f"Word({self.word}, {self.lemma})"

    @staticmethod
    def from_derivation_row(row):
        '''Creates a new Word, based on derivation formatted row.'''
        return Word(
            lemma=row[0],
            word=row[1],
            morphemes=[4],
        )

    @staticmethod
    def from_inflection_row(row):
        """
        Creates a Word based on inflection formatted row.
        Returns None if no segmentation is available
        """
        if row[3] == '-':
            return None
        return Word(
            lemma=row[0],
            word=row[1],
            morphemes=row[3],
        )


def validate_inflection_file(file_path, output_path):
    '''Run validation on inflection formatted file'''
    print('Validating inflection file')
    _validate_file(
        file_path=file_path,
        output_path=output_path,
        row_converter=Word.from_inflection_row
    )

def validate_derivation_file(file_path, output_path):
    '''Run validation on a derivation formatted file'''
    print('Validating derivation file')
    _validate_file(
        file_path=file_path,
        output_path=output_path,
        row_converter=Word.from_derivation_row,
    )


def _validate_file(file_path, output_path, row_converter):
    w_total = 0     # processed rows
    w_skipped = 0   # skipped rows
    w_is_pt = 0     # valid PT_BR or valid PT_PT
    w_is_ptbr = 0   # valid PT_BR
    w_is_ptpt = 0   # valid PT_PT
    w_not_pt = 0    # invalid PT_BR and invalid PT_PT
    w_not_ptbr = 0  # invalid PT_BR
    w_not_ptpt = 0  # invalid PT_PT

    def success_message(num):
        return f"Success: {100*num/w_total:05.2f}% ({num})"

    def failure_message(num):
        return f"Failure: {100*num/w_total:05.2f}% ({num})"

    def report_print(f, message):
        f.write(message + "\n")
        print(message)

    input_file = open(file_path, 'r', encoding="utf-8")

    invalid_file = open(os.path.join(output_path, "val_invalidate.txt"), "w", encoding="utf-8")
    extra_file = open(os.path.join(output_path, "val_extra.txt"), "w", encoding="utf-8")

    input_reader = csv.reader(input_file, delimiter='\t')
    for row in input_reader:
        word = row_converter(row)

        # should consider this a valid row?
        if word is None:
            w_skipped += 1
            continue
        w_total +=1

        # is a valid portuguese word?
        if word.is_pt:
            w_is_pt += 1
        else:
            w_not_pt += 1

        # is a valid brazilian pt word?
        if word.is_ptbr:
            w_is_ptbr += 1
        else:
            w_not_ptbr += 1

        # is a valid european pt word?
        if word.is_ptpt:
            w_is_ptpt += 1
        else:
            w_not_ptpt += 1

        # words that are invalid for brazilian,
        if not word.is_ptbr:
            if word.is_pt:
                # but valid for european portuguese
                extra_file.write('\t'.join(row) + '\n')
            else:
                invalid_file.write('\t'.join(row) + '\n')

    input_file.close()
    invalid_file.close()
    extra_file.close()

    report_file = open(os.path.join(output_path, "val_report.txt"), "w", encoding="utf-8")

    report_print(report_file, f"Input: {file_path}")
    report_print(report_file, f"skipped: {w_skipped} rows not processed")
    report_print(report_file, f"total: {w_total} rows processed")

    report_print(report_file, "\nPortuguese")
    report_print(report_file, success_message(w_is_pt))
    report_print(report_file, failure_message(w_not_pt))

    report_print(report_file, "\nBrazilian")
    report_print(report_file, success_message(w_is_ptbr))
    report_print(report_file, failure_message(w_not_ptbr))

    report_print(report_file, "\nEuropean")
    report_print(report_file, success_message(w_is_ptpt))
    report_print(report_file, failure_message(w_not_ptpt))

    report_file.close()
