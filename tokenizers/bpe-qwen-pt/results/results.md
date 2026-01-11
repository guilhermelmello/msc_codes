# Results

Results from the semantic evaluation.


## Unicode Normalization

First experiments, comparing the performance of unicode normalization methods. Only the affix results are considered in the following tables.

Vocabulary config:
- final size: 10k
- initial alphabet: bytes

### Derivations

| Unigram | F1     | Prec.  | Recall | words  |
|---------|--------|--------|--------|--------|
| NFC     | 59.36% | 57.94% | 60.86% |  1.96% |
| NFKC    | 59.06% | 57.58% | 60.63% |  1.97% |
| NFD     | 49.98% | 40.97% | 64.07% |  1.90% |
| NFKD    | 50.06% | 41.03% | 64.20% |  1.90% |

| BPE     | F1     | Prec.  | Recall | words  |
|---------|--------|--------|--------|--------|
| NFC     | 51.02% | 64.20% | 42.33% |  2.90% |
| NFKC    | 51.02% | 64.20% | 42.33% |  2.90% |
| NFD     | 50.19% | 51.16% | 49.26% |  2.06% |
| NFKD    | 50.19% | 51.16% | 49.26% |  2.06% |


### Inflections

| Unigram | F1     | Prec.  | Recall | words  |
|---------|--------|--------|--------|--------|
| NFC     | 35.96% | 29.25% | 46.67% |  0.60% |
| NFKC    | 36.06% | 29.32% | 46.84% |  0.60% |
| NFD     | 35.02% | 26.65% | 51.04% |  0.65% |
| NFKD    | 34.97% | 26.61% | 50.98% |  0.66% |

| BPE     | F1     | Prec.  | Recall | words  |
|---------|--------|--------|--------|--------|
| NFC     | 55.18% | 54.98% | 55.38% |  0.56% |
| NFKC    | 55.19% | 54.99% | 55.39% |  0.56% |
| NFD     | 21.24% | 20.85% | 21.64% |  0.75% |
| NFKD    | 21.23% | 20.85% | 21.63% |  0.75% |


### Comments

- Unigram have bad results for inflections, but shows a good performance on derivation. BPE lacks behind on derivations, but is more balanced on both word formation process.
- Worse results were obtained when considering different initial alphabet, including characters and derivation morphemes. Results are available for [derivation](affixes_der.txt) and [inflection](affixes_inf.txt).


## Final Results: Using NFC

Based on previous results, NFC normalization was selected for further experimentation. The following results compare the performance of different vocabulary sizes, using only bytes as initial alphabet. Only the affix results are reported on the following tables, but complete evaluation logs are available for [BPE](bpe-nfc.txt) and [Unigram](unigram-nfc.txt).

By default, the Unigram trainer uses `n_sub_iterations=2`, used to obtain the following results. Different values were tested (3 and 5), but no improvement was observed. Results are available [here](unigram10k3.md) and [here](unigram10k5.md)


### Derivations


| UNIGRAM | F1     | Prec.  | Recall |  words |
|---------|--------|--------|--------|--------|
| 5K      | 55.71% | 47.03% | 68.32% |  1.17% |
| 8K      | 61.73% | 57.53% | 66.59% |  1.70% |
| 10K     | 59.36% | 57.94% | 60.86% |  1.96% |
| 15K     | 58.39% | 59.19% | 57.62% |  2.53% |
| 30K     | 56.26% | 59.21% | 53.59% |  4.04% |
| 50K     | 54.87% | 61.07% | 49.80% |  5.29% |


| BPE     | F1     | Prec.  | Recall |  words |
|---------|--------|--------|--------|--------|
| 5K      | 52.67% | 61.77% | 45.90% |  0.77% |
| 8K      | 51.66% | 63.95% | 43.34% |  1.95% |
| 10K     | 51.02% | 64.20% | 42.33% |  2.90% |
| 15K     | 51.25% | 66.65% | 41.63% |  4.57% |
| 30K     | 52.20% | 70.34% | 41.49% |  9.57% |
| 50K     | 54.91% | 75.73% | 43.07% | 15.58% |

### Inflections

| UNIGRAM | F1     | Prec.  | Recall |  words |
|---------|--------|--------|--------|--------|
| 5K      | 38.22% | 30.00% | 52.63% |  0.31% |
| 8K      | 36.38% | 29.16% | 48.36% |  0.48% |
| 10K     | 35.96% | 29.25% | 46.67% |  0.60% |
| 15K     | 35.08% | 29.16% | 44.02% |  0.85% |
| 30K     | 34.71% | 30.60% | 40.08% |  1.26% |
| 50K     | 31.05% | 27.90% | 35.00% |  1.74% |

| BPE     | F1     | Prec.  | Recall |  words |
|---------|--------|--------|--------|--------|
| 5K      | 54.64% | 51.44% | 58.26% |  0.20% |
| 8K      | 55.04% | 53.94% | 56.20% |  0.40% |
| 10K     | 55.18% | 54.98% | 55.38% |  0.56% |
| 15K     | 55.21% | 57.25% | 53.32% |  0.94% |
| 30K     | 54.60% | 58.61% | 51.11% |  2.16% |
| 50K     | 55.48% | 62.47% | 49.89% |  3.70% |

### Comments

- BPE have a good influece by the size os its vocabulary, resulting in more activation of the whole-word route. This is an expected behavior of the BPE algorithm. The Unigram model does not follow this behavior, resulting in worse performance as more words are detected as a single token.
- On deviration process, the Unigram shows a better performance, but the results are much worse on inflections. BPE show a better balance between both processes. The question that raises is: how much each process affects the performance on downstream tasks?
