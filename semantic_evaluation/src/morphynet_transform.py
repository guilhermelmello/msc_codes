"""
Script to filter and transform the original Morphynet
entries into a sequence of surface level morphemes.
"""

import csv
import os

import enchant

_BR_DICT = enchant.Dict("pt_BR")

def transform_derivation_file(file_path, output_path):
    '''Convert derivation file in a sequence of surface morphemes'''
    print(f"Processing derivation file: {file_path}")
    _process_file(
        file_path=file_path,
        output_path=output_path,
        row_converter=_transform_derivation_row
    )

def transform_inflection_file(file_path, output_path):
    '''Convert inflection file in a sequence of surface morphemes'''
    print(f"Processing inflection file: {file_path}")
    _process_file(
        file_path=file_path,
        output_path=output_path,
        row_converter=_transform_inflection_row
    )


def _process_file(file_path, output_path, row_converter):
    rows_total = 0      # total rows in the input file
    rows_success = 0    # rows converted and ready to use
    rows_failure = 0    # row that could not process for some reason
    rows_skipped = 0    # rows that are intentionally left out

    try:
        input_file = open(file_path, 'r', encoding='utf-8')
        output_file = open(os.path.join(output_path, 'transform_base.tsv'), 'w', encoding='utf-8')
        skip_file = open(os.path.join(output_path, 'transform_skip.tsv'), 'w', encoding='utf-8')
        fail_file = open(os.path.join(output_path, 'transform_fail.tsv'), 'w', encoding='utf-8')

        input_reader = csv.reader(input_file, delimiter='\t')
        for row in input_reader:
            success, c_row = row_converter(row)
            rows_total += 1

            if success is None:
                rows_failure += 1
                fail_file.write("\t".join(row) + "\n")
            elif not success:
                rows_skipped += 1
                skip_file.write("\t".join(row) + "\n")
            else:
                rows_success += 1
                output_file.write(c_row + "\n")

        input_file.close()
        output_file.close()
        skip_file.close()
        fail_file.close()

        success_rate = rows_success / rows_total * 100
        failure_rate = rows_failure / rows_total * 100
        skipped_rate = rows_skipped / rows_total * 100

        report_file_path = os.path.join(output_path, "transform_report.txt")
        report_file = open(report_file_path, 'w', encoding="utf-8")

        def report_print(message):
            report_file.write(message + "\n")
            print(message)

        report_print(f"Function: {row_converter.__name__}")
        report_print(f"Input: {file_path}")
        report_print(f"Output: {output_path}")
        report_print(f"total: {rows_total} rows processed")
        report_print(f"success: {success_rate:05.2f}% ({rows_success} rows)")
        report_print(f"failure: {failure_rate:05.2f}% ({rows_failure} rows)")
        report_print(f"skipped: {skipped_rate:05.2f}% ({rows_skipped} rows)")

        report_file.close()

    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


def _transform_derivation_row(row):
    s_word = row[0]
    t_word = row[1]
    morpheme = row[4]
    morpheme_type = row[5]

    isbr = (
        _BR_DICT.check(s_word) or
        _BR_DICT.check(t_word)
    )

    # remove words that are not
    # in the brazilian dictionary
    if not isbr:
        return False, None

    morphemes = list()
    match morpheme_type:
        case "prefix":
            stem = t_word.removeprefix(morpheme)
            morphemes = [morpheme, stem]
            # c_row = f"{word}\t{affix}|{stem}\t{morpheme_type}"
        case "suffix":
            stem = t_word.removesuffix(morpheme)
            morphemes = [stem, morpheme]
            # c_row = f"{word}\t{stem}|{affix}\t{morpheme_type}"
        case _:
            print("ERROR: Invalid Affix :: ", row)
            return None, None

    # valida reconstrução da palavra a partir dos morfemas
    if  t_word == ''.join(morphemes):
        return True, f"{t_word}\t{'|'.join(morphemes)}\t{morpheme_type}"

    # valida alguns casos irregulares
    affix = None
    if morpheme_type == "prefix":
        affix = t_word.removesuffix(s_word)
        morphemes = [affix, s_word]
    else:
        affix = t_word.removeprefix(s_word)
        morphemes = [s_word, affix]

    accepted_affixes = [
        "i", "im", "circum", "milí", "com", "antí", "em", "ên",
        "hidr", "pre", "Euro", "nanô", "mon", "con", "mesó", "aló"
    ]
    if(t_word == ''.join(morphemes) and affix in accepted_affixes):
        return True, f"{t_word}\t{'|'.join(morphemes)}\t{morpheme_type}"
    return None, None


# success, conv_row
# True: success
# False: skip
# None: failure (wrong format)
def _transform_inflection_row(row):
    lemma = row[0]
    word = row[1]
    morphemes = row[3]

    # no morpheme segmentation
    if morphemes == '-':
        return False, None

    # remove words that are not
    # in the brazilian dictionary
    isbr = (_BR_DICT.check(lemma) or _BR_DICT.check(word))
    if not isbr:
        return False, None

    # keeps only inflection morphemes
    morphemes = morphemes.split("|")[1:]

    # fix gender tokens
    _morphemes = list(enumerate(morphemes))
    for i, m in reversed(_morphemes):
        # if femenine and previous is masculine
        if(m == 'a' and morphemes[i-1].endswith('o')):
            morphemes.pop(i)
            morphemes[i-1] = morphemes[i-1][:-1] + 'a'

    word_left = word
    tokens = []

    # create surface level tokens
    for m in reversed(morphemes):
        if word_left.endswith(m):
            word_left = word_left.removesuffix(m)
            tokens.append(m)
        else:
            print(f"ERROR: {word} -> {morphemes}")
            return None, None

    tokens.append(word_left)
    tokens = list(reversed(tokens))

    is_word = word == ''.join(tokens)
    if not is_word:
        print(f"ERROR: {is_word} : {word} -> {tokens}")
        return None, None

    return True, f"{word}\t{"|".join(tokens)}"
