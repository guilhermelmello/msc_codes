import torch
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

import utils
import trainer


# to emulate a full training
STEPS = 1
EPOCHS = 3


def main(
    model_name: str,
    tokenizer_name: str,
    batch_size: int,
    max_sequence_len: int,
    output_path: str,
    num_workers: int,
    init_mode: utils.InitMode,
):
    if(torch.cuda.is_available()):
        gpu = torch.cuda.get_device_properties()
        print('GPU:', gpu.name)
        print(f'{gpu.total_memory / 1024**3:.2f} GB\n')
    else:
        print('No GPU available!')
        return
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # load model from config
    model = utils.initialize_clm(
        model_name=model_name,
        tokenizer=tokenizer,
        init_mode=init_mode,
    )
    utils.print_model_size(model)

    # create dummy dataset
    dataset = Dataset.from_dict(
        {
            'input_ids': torch.randint(0, tokenizer.vocab_size, (batch_size*STEPS, max_sequence_len,)),
            'attention_mask': torch.ones(batch_size*STEPS, max_sequence_len),
        },
        features=Features({
            'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
            'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
        }),
    )
    dataset.set_format('torch')

    # START GPU MONITORING
    torch.cuda.memory._record_memory_history(max_entries=100000)
    trainer.train_clm(
        model=model,
        train_dataset=dataset,
        validation_dataset=dataset,
        batch_size=batch_size,
        learning_rate=0.00001,
        weight_decay=0.1,
        warmup_steps=100,
        num_epochs=EPOCHS,
        num_workers=num_workers,
    )

    # Dump memory snapshot history to a file and stop recording
    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == '__main__':
    main(
        model_name='Qwen/Qwen3-0.6B',
        tokenizer_name='guilhermelmello/tokenizer-unigram-pt-8k',
        num_workers=8,
        batch_size=32,
        max_sequence_len=1024,
        output_path="logs/gpu_profile.pkl",
        init_mode=utils.InitMode.BASE_CONFIG,
    )
