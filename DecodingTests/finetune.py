import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer

root_dir = '/mnt/infonas/data/absathe/DecodingTests/Translation/OPUS/triple_test/'
A = 'fr'
C = 'de'
A_id = 'fr_XX'
C_id = 'de_DE'
n_epochs = 150
batch_size = 16
weight_decay = 1e-5 
lr = 3e-5

PROJECT_NAME = 'mBART50-FineTune-Baseline'

class AToCTranslate(Dataset):
    def __init__(self, root_dir='/mnt/infonas/data/absathe/DecodingTests/Translation/OPUS/triple_test/', A='fr', C='de', train=True, train_frac=0.7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.src_path = os.path.join(root_dir, f'opus.de-fr-test.{A}')
        self.tgt_path = os.path.join(root_dir, f'opus.de-fr-test.{C}')
        with open(self.src_path, 'r') as f1, open(self.tgt_path, 'r') as f2:
            self.src = [line.strip() for line in f1.readlines()]
            self.tgt = [line.strip() for line in f2.readlines()]
        assert len(self.src) == len(self.tgt)
        idx = int(train_frac * len(self.src))
        if train:
            self.src = self.src[:idx]
            self.tgt = self.tgt[:idx]
        else:
            self.src = self.src[idx:]
            self.tgt = self.tgt[idx:]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx) -> dict:
        return dict(src=self.src[idx], tgt=self.tgt[idx])

training_args = TrainingArguments(
    output_dir=f'{root_dir}/{A}->{C}.finetune/',
    overwrite_output_dir=True,
    num_train_epochs=n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=1,
    prediction_loss_only=False,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    weight_decay=weight_decay,
    learning_rate=lr,
    warmup_steps=100,
    lr_scheduler_type='linear',
    metric_for_best_model='loss',
    load_best_model_at_end=True,
    fp16=True,
    eval_accumulation_steps=8,
    report_to='tensorboard',
    run_name=f'{A}->{C}.finetune'
)

tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

def collate_fn(batch):
    src = [x['src'] for x in batch]
    tgt = [x['tgt'] for x in batch]
    tokenizer.src_lang = A_id
    ret = tokenizer(src, return_tensors='pt', padding=True)
    tokenizer.src_lang = C_id
    ret['labels'] = tokenizer(tgt, return_tensors='pt', padding=True)['input_ids']
    return ret

def main():
    model = AutoModelForMaskedLM.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

    train = AToCTranslate(root_dir=root_dir, A=A, C=C, train=True)
    dev = AToCTranslate(root_dir=root_dir, A=A, C=C, train=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collate_fn,
    )

    result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    main()
