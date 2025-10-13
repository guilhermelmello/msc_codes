# Semantic Evaluation

## Datasets

- MorphyNet: original portuguese files from [MorphyNet](https://github.com/kbatsuren/MorphyNet).
- SemEval: MorphyNet data converted into surface-level morpheme segmentation, prepared for semantic evaluation.

## MorphyNet Analysis

Scripts to analize, validate and transform MorphyNet data.
The `morphynet.py` file is the entrypoint to run each script.


### Validation:

For each entry on MorphyNet data, checks if it is a valid word, based
on `pyenchant` dictionaries for brazilian and european portuguese. To
be considered a valid word, the base or its derivative (or inflected)
form must be present in the dictionaries.

**How to run:**

```bash
python morphynet.py \
    --validate --derivation \
    --output-dir=./logs/derivation
```
```bash
python morphynet.py \
    --validate --inflection \
    --output-dir=./logs/inflection
```

**Generated files:**
- *val_extra.txt*: MorphyNet entries that are only valid on European
    Portuguese. Can be considered to extend Brazilian Portuguese words.
- *val_invalidate*: MorphyNet entries that could not be recognized as valid
    Portuguese words.
- *val_report*: Describes some statics of the validation process. If, for
    some reason, a row not considered for validation, like inflection entries
    that does not have the segmented word, are skipped from validation. The
    report show how many words are valid when considering different sets
    of Portuguese words (PTBR only, PT only and PTBR+PT).


### Statistics:

Extract distribution statistics from MorphyNet files.

**How to run:**

```bash
python morphynet.py \
    --stats --derivation \
    --output-dir=./logs/derivation
```

```bash
python morphynet.py \
    --stats --inflection \
    --output-dir=./logs/inflection
```

**Generated Data:**

Data distribution for POS tags, affix types and morpheme types.
Results from original MorphyNet dataset and saved on *stats_reporter.txt*.

### Transform:

Filter and transform the original Morphynet entries into a sequence of
surface level morphemes.

**How to run:**

```bash
python morphynet.py \
    --transform --derivation \
    --output-dir=./logs/derivation/
```
```bash
python morphynet.py \
    --transform --inflection \
    --output-dir=./logs/inflection/
```

**Generated Files:**
- *transform_base.tsv*: filtered and transformed data.
    This file was moved to `./data/semevam/` directory.
- *transform_fail.txt*: rows that failed to be included in the semeval dataset.
- *transform_skip.txt*: rows that were intentionally skipped, and not
    included in the semeval dataset.
- *transform_report.txt*: report the results obtained during transformation.


### Affixes:
## Semantic Evaluation: