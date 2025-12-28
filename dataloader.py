from torch.utils.data import DataLoader, Dataset
from data_sequential import DataSequential
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def get_dataloader(args, tokenizer, tokenized_text, tokenized_domain, token_id_dict):
    train_dataset = DataSequential(args, tokenizer,tokenized_text,tokenized_domain, token_id_dict,'train')
    val_dataset = DataSequential(args, tokenizer,tokenized_text, tokenized_domain, token_id_dict,'valid')
    test_dataset = DataSequential(args, tokenizer, tokenized_text,tokenized_domain,token_id_dict,'test')

    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, seed=args.seed)
        valid_sampler = DistributedSampler(val_dataset, seed=args.seed)
        test_sampler = DistributedSampler(test_dataset, seed=args.seed)
    else:
        train_sampler, valid_sampler, test_sampler = None, None, None

    if args.distributed:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=False,  # shuffle is handled inside DistributedSampler
            pin_memory=True,
            sampler=train_sampler,
            # num_workers=args.num_workers,           
            worker_init_fn=seed_worker,
            generator=g
        )

        val_loader = DataLoader(val_dataset,
                                batch_size= 1, # TODO - Eval padding
                                # batch_size= 4,
                                collate_fn=train_dataset.collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                sampler=valid_sampler)
        test_loader = DataLoader(test_dataset,
                                 batch_size= 1,
                                #  batch_size=4,
                                 collate_fn=train_dataset.collate_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 sampler=test_sampler)

    else:
        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  # num_workers=args.num_works,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=True,
                                  generator=g  # ✅ Explicitly pass generator
                                  )
        val_loader = DataLoader(val_dataset,
                                # batch_size=args.batch_size * 4,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=val_dataset.collate_fn
                                )
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                #  batch_size=args.batch_size * 4,
                                 shuffle=False,
                                 collate_fn=test_dataset.collate_fn
                                 )
    return train_loader, val_loader, test_loader
