from transformers import AutoTokenizer


DERIVATION_PREFIXES = [
    'anti', 'des', 'a', 're', 'in', 'auto', 'bio', 'sub', 'agro', 'mono', 'geo',
    'multi', 'im', 'micro', 'poli', 'super', 'foto', 'aero', 'radio', 'inter',
    'di', 'cripto', 'semi', 'en', 'sobre', 'eletro', 'con', 'tele', 'bi', 'de',
    'co', 'hiper', 'pré', 'an', 'i', 'tri', 'adeno', 'neuro', 'neo', 'hidro',
    'ante', 'antropo', 'pre', 'hipo', 'psico', 'quilo', 'pós', 'ultra', 'astro',
    'es', 'intra', 'contra', 'para', 'endo', 'supra', 'termo', 'eco', 'meta',
    'acro', 'ântero', 'extra', 'arqui', 'trans', 'tetra', 'homo', 'com',
    'entre', 'paleo', 'hetero', 'so', 'per', 'socio', 'omni', 'osteo', 'bis',
    'euro', 'fosfo', 'ciber', 'vice', 'web', 'ab', 'quimio', 'uni', 'macro',
    'mili', 'ano', 'adreno', 'mini', 'nano', 'proto', 'actino', 'alo', 'ciclo',
    'ambi', 'anfi', 'antí', 'mega', 'organo', 'andro', 'orto', 'pseudo', 'tris',
    'braqui', 'aniso', 'aorto', 'as', 'cardio', 'pan', 'retro', 'rodo', 'cito',
    'anglo', 'hemo', 'penta', 'ir', 'piro', 'pluri', 'exo', 'mal', 'circun',
    'ferro', 'Euro', 'infra', 'afro', 'histo', 'anto', 'dendro', 'abdomino',
    'narco', 'hexa', 'tio', 'electro', 'iso', 'peri', 'biblio', 'pro', 'crono',
    'lacto', 'fito', 'pró', 'res', 'auri', 'kilo', 'cromo', 'circum', 'dis',
    'carbo', 'audio', 'gastro', 'milí', 'linfo', 'oni', 'viti', 'glico',
    'braquio', 'frigo', 'vini', 'luso', 'ciano', 'mon', 'hemi', 'imuno',
    'musculo', 'calco', 'em', 'cupro', 'sesqui', 'cloro', 'nanô', 'meso',
    'tres', 'anarco', 'quadri', 'mesó', 'sin', 'alelo', 'aló', 'angio',
    'aracno', 'hecto', 'centi', 'deci', 'video', 'hidr', 'ên', 'xeno',
    'sur', 'e',
]

DERIVATION_SUFFIXES = [
    'mente', 'ar', 'ção', 'ico', 'mento', 'idade', 'ismo', 'dor', 'eiro',
    'ista', 'ão', 'izar', 'ense', 'ado', 'oso', 'ada', 'al', 'eira', 'ável',
    'inho', 'a', 'ano', 'agem', 'inha', 'zinho', 'ante', 'ia', 'ear', 'ês',
    'ivo', 'aria', 'o', 'ência', 'eza', 'ível', 'ação', 'ice', 'ura', 'dura',
    'ário', 'eta', 'ento', 'udo', 'ez', 'aço', 'ático', 'ância', 'ido', 'ejar',
    'iano', 'esa', 'ança', 'logia', 'ino', 'ato', 'io', 'deira', 'tório', 'ote',
    'ente', 'ificar', 'ete', 'idão', 'são', 'vel', 'es', 'ela', 'dade',
    'íssimo', 'abilidade', 'or', 'ola', 'douro', 'ilho', 'oide', 'im', 'zinha',
    'nte', 'ador', 'ito', 'il', 'ina', 'esco', 'tura', 'ença', 'ite', 'edo',
    'metro', 'iço', 'eca', 'alho', 'ectomia', 'ístico', 'onho', 'enho', 'ejo',
    'aça', 'ida', 'ório', 'logo', 'oca', 'ópolis', 'ona', 'imento', 'emia',
    'elho', 'cida', 'tiva', 'isa', 'dora', 'diço', 'ita', 'ume', 'deiro',
    'fobo', 'essa', 'icho', 'aco', 'ial', 'ase', 'tivo', 'anço', 'teca', 'eda',
    'ata', 'acho', 'elo', 'ose', 'alha', 'ota', 'isco', 'itar', 'geno',
    'logista', 'ama', 'inte', 'oto', 'iar', 'icar', 'tico', 'ífero', 'dromo',
    'ículo', 'açal', 'ouro', 'avo', 'ango', 'ir', 'ólogo', 'eno', 'ologia',
    'imo', 'zana', 'is', 'grafia', 'asca', 'arão', 'engo', 'filia', 'ando',
    'gênico', 'dela', 'ício', 'ebre', 'diça', 'ecer', 'éu', 'ema', 'ilha',
    'oma', 'pata', 'plastia', 'cêntrico', 'iscar', 'ana', 'ardo', 'ugem',
    'áceo', 'polis', 'bol', 'arro', 'eco', 'lândia', 'ara', 'oco', 'ragia',
    'igo', 'ego', 'ation', 'ol', 'ilhar', 'dio', 'reia', 'ético', 'osa',
    'fago', 'el', 'eima', 'ficar', 'iça',
]
DERIVATION_AFFIXES = DERIVATION_PREFIXES + DERIVATION_SUFFIXES
INFLECTION_AFFIXES = [
    's', 'e', 'a', 'amos', 'emos', 'em', 'ada', 'aria', 'ado', 'ava', 'ares',
    'armos', 'ardes', 'arem', 'ara', 'asse', 'aram', 'es', 'am', 'as', 'o',
    'ais', 'eis', 'ei', 'ará', 'arei', 'aremos', 'areis', 'arão', 'arias',
    'aríamos', 'aríeis', 'ariam', 'arás', 'avas', 'ai', 'ávamos', 'áveis',
    'avam', 'ou', 'aste', 'astes', 'aras', 'áramos', 'áreis', 'asses',
    'ássemos', 'ásseis', 'assem', 'ões', 'ia', 'ido', 'ida', 'i', 'era', 'esse',
    'eram', 'is', 'eres', 'ermos', 'erdes', 'erem', 'iria', 'irmos', 'irdes',
    'íamos', 'íeis', 'eria', 'ias', 'iam', 'ires', 'irem', 'imos', 'iram',
    'ira', 'isse', 'ns', 'estes', 'eras', 'esses', 'essem', 'este', 'irei',
    'irás', 'irá', 'iremos', 'ireis', 'irão', 'irias', 'iríamos', 'iríeis',
    'iriam', 'iu', 'íramos', 'íreis', 'íssemos', 'ísseis', 'erei', 'erás',
    'erá', 'eremos', 'ereis', 'erão', 'erias', 'eríamos', 'eríeis', 'eriam',
    'eu', 'êramos', 'êreis', 'êssemos', 'êsseis', 'iste', 'istes', 'iras',
    'isses', 'issem', 'er', 'ído', 'ída', 'ía', 'í', 'ímos', 'íram', 'íra',
    'ísse', 'íres', 'írem', 'esa', 'íam', 'ías', 'ís', 'íste', 'ístes', 'íras',
    'ísses', 'íssem', 'des', 'osto', 'osta', 'oria', 'ona', 'omos', 'ores',
    'ormos', 'ordes', 'orem', 'orias', 'oríamos', 'oríeis', 'oriam', 'íssimo',
    'ói', 'oa', 'éns', 'imo', 'ir', 'ã', 'ar', 'isa', 'ando', 'és',
]


def validate_prefix(tokenizer, vocab_items, prefix):
    prefix = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(prefix)[0][0]
    return prefix in vocab_items or f'Ġ{prefix}' in vocab_items


def validate_suffix(tokenizer, vocab_items, suffix):
    try:
        sw_prefix = tokenizer.backend_tokenizer.model.continuing_subword_prefix
        if sw_prefix is not None:
            suffix = f"{sw_prefix}{suffix}"
    except:
        # model does not have continuing_subword_prefix
        pass

    try:
        sw_suffix = tokenizer.backend_tokenizer.model.end_of_word_suffix
        if sw_suffix is not None:
            suffix = f"{suffix}{sw_suffix}"
    except:
        # model does not have end_of_word_suffix
        pass

    # print(f'validando sufixo: {suffix}')
    return suffix in vocab_items


def validate_tokenizer(name):
    print(f'Validando tokenizer: {name}')
    tokenizer = AutoTokenizer.from_pretrained(name)

    vocab_items = list(tokenizer.vocab.keys())
    print(f'vocab size: {len(vocab_items)}')

    dp = [validate_prefix(tokenizer, vocab_items, a) for a in DERIVATION_PREFIXES]
    ds = [validate_suffix(tokenizer, vocab_items, a) for a in DERIVATION_SUFFIXES]
    ia = [validate_suffix(tokenizer, vocab_items, a) for a in INFLECTION_AFFIXES]

    print(f'Derivation prefixes: {sum(dp)}/{len(dp)} ({100 * sum(dp)/len(dp):.2f}%)')
    print(f'Derivation suffixes: {sum(ds)}/{len(ds)} ({100 * sum(ds)/len(ds):.2f}%)')
    print(f'Inflection suffixes: {sum(ia)}/{len(ia)} ({100 * sum(ia)/len(ia):.2f}%)')


# models = [
#     'PORTULAN/albertina-900m-portuguese-ptbr-encoder',
#     'PORTULAN/albertina-100m-portuguese-ptbr-encoder',
#     'neuralmind/bert-base-portuguese-cased',
#     'google-bert/bert-base-multilingual-cased',
#     'guilhermelmello/bpe_pt',
#     'guilhermelmello/tokenizer-bpe-pt-5k',
#     'guilhermelmello/tokenizer-bpe-pt-8k',
#     'guilhermelmello/tokenizer-bpe-pt-10k',
#     'guilhermelmello/tokenizer-bpe-pt-15k',
#     'guilhermelmello/tokenizer-bpe-pt-30k',
#     'guilhermelmello/tokenizer-bpe-pt-50k',
#     'guilhermelmello/tokenizer-unigram-pt-5k',
#     'guilhermelmello/tokenizer-unigram-pt-8k',
#     'guilhermelmello/tokenizer-unigram-pt-10k',
#     'guilhermelmello/tokenizer-unigram-pt-15k',
#     'guilhermelmello/tokenizer-unigram-pt-30k',
#     'guilhermelmello/tokenizer-unigram-pt-50k',
#     'microsoft/deberta-v2-xlarge',
#     'microsoft/deberta-v2-xxlarge',
#     'microsoft/deberta-base',
#     'Qwen/Qwen3-0.6B',
# ]

# for m in models:
#     print('='*40)
#     validate_tokenizer(m)

if __name__ == "__main__":
    validate_tokenizer('models/teste-bpe8k')
    validate_tokenizer('models/teste-unigram8k')
