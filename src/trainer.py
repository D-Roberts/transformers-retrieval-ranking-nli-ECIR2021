# coding=utf-8
"""
Set up for cpu or 1 GPU.
"""
import logging
import math
import os
import random

import numpy as np
import torch
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.optimization import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange


logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    name,
    optimizer,
    num_warmup_steps,
    num_training_steps,
):

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


class Trainer:
    def __init__(
        self,
        module=None,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        optimizers=(None, None),  # optim and lr sched
    ):
        self.module = module
        self.model = model
        self.args = args
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.lr_scheduler = optimizers

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.local_rank == -1 or args.no_cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
            )
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device("cuda", args.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )

        logger.info(
            f"device: {self.device} n_gpu: {self.n_gpu}, distributed training: {bool(args.local_rank != -1)}"
        )
        self.model.to(self.device)
        self.args.train_batch_size = (
            self.args.train_batch_size // self.args.gradient_accumulation_steps
        )

    def _get_eval_dataloader(self, eval_dataset):
        eval_sampler = SequentialSampler(eval_dataset)
        return DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

    def _get_train_dataloader(self, train_dataset, batch_size):
        """CPU or 1 GPU"""
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    def get_optimizer(self, num_train_optimization_steps):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )

        if self.lr_scheduler is None:
            warmup_steps = math.ceil(
                num_train_optimization_steps * self.args.warmup_proportion
            )

            self.lr_scheduler = get_scheduler(
                "linear",
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps,
            )

    def get_loss(self):
        self.loss_fct = CrossEntropyLoss()

    def save_model(self):
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        # To load `from_pretrained`
        output_model_file = os.path.join(self.args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.args.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        # and tokenizer
        self.tokenizer.save_vocabulary(self.args.output_dir)
        # and training args
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))

    def training_step_nli(self, batch, num_labels):
        batch = tuple(t.to(self.device) for t in batch)
        # input_ids, input_mask, segment_ids, label_ids = batch
        guid, input_ids, input_mask, segment_ids, label_ids = batch
        logits = self.model(input_ids, segment_ids, input_mask, labels=None)

        self.get_loss()
        loss = self.loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach(), input_ids.size(0)

    def training_step_sentence(self, batch, num_labels, it):
        batch = tuple(t.to(self.device) for t in batch)
        guid, input_ids, input_mask, segment_ids, label_ids = batch

        try:
            batch_neg = tuple(t.to(self.device) for t in next(it))
        except Exception:
            it = iter(self.negative_dataloader)
            batch_neg = tuple(t.to(self.device) for t in next(it))

        (
            guid_neg,
            input_ids_neg,
            input_mask_neg,
            segment_ids_neg,
            label_ids_neg,
        ) = batch_neg

        input_ids_cat = torch.cat([input_ids, input_ids_neg], dim=0)
        segment_ids_cat = torch.cat([segment_ids, segment_ids_neg], dim=0)
        input_mask_cat = torch.cat([input_mask, input_mask_neg], dim=0)
        label_ids_cat = torch.cat([label_ids.view(-1), label_ids_neg.view(-1)], dim=0)

        logits = self.model(input_ids_cat, segment_ids_cat, input_mask_cat, labels=None)
        loss = self.loss_fct(logits.view(-1, num_labels), label_ids_cat)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach(), input_ids.size(0)

    def train(
        self, train_data, num_labels, num_eg, num_eg_neg=None, negative_train_data=None
    ):
        logger.info(f"***** Training Module {self.module} *****")
        train_dataset, _ = self._prepare_inputs(train_data, num_eg)
        train_dataloader = self._get_train_dataloader(
            train_dataset, self.args.train_batch_size
        )

        if self.module == "sentence":
            logger.info("** Prepare Negative Sentences **")
            neg_train_dataset, _ = self._prepare_inputs(negative_train_data, num_eg_neg)
            self.negative_dataloader = self._get_train_dataloader(
                neg_train_dataset, self.args.negative_batch_size
            )

        num_train_optimization_steps = (
            int(
                num_eg
                / self.args.train_batch_size
                / self.args.gradient_accumulation_steps
            )
            * self.args.num_train_epochs
        )

        global_step = 0

        self.get_optimizer(num_train_optimization_steps)
        self.get_loss()
        self.model.train()

        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if self.module == "sentence":
                it = iter(self.negative_dataloader)

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                if self.module == "nli":
                    loss, id_size = self.training_step_nli(batch, num_labels)
                elif self.module == "sentence":
                    loss, id_size = self.training_step_sentence(batch, num_labels, it)
                # should have a sanity check here

                tr_loss += loss.item()
                nb_tr_examples += id_size
                nb_tr_steps += 1
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            print("training_loss~=", tr_loss / nb_tr_steps)
        # Save the trained model, configuration and tokenizer
        self.save_model()

    def _prepare_inputs(self, eval_features, num_eg):
        logger.info(" Num examples = %d", num_eg)
        logger.info(" Batch size = %d", self.args.eval_batch_size)
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
        guid_list = [f.guid for f in eval_features]

        guid_ids_map = {k: v for k, v in enumerate(guid_list)}
        guid_ids = torch.tensor(list(guid_ids_map.keys()), dtype=torch.long)
        eval_data = TensorDataset(
            guid_ids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        return eval_data, guid_ids_map

    def predict(self, predict_inputs, num_eg):
        logger.info("***** Predicting *****")
        eval_data, guid_map = self._prepare_inputs(predict_inputs, num_eg)
        eval_dataloader = self._get_eval_dataloader(eval_data)
        return (
            (*self.prediction_loop(eval_dataloader), guid_map)
            if not self.args.onnx
            else (*self.prediction_loop_onnx(eval_dataloader), guid_map)
        )

    def prediction_loop(self, data_loader):
        model = self.model
        model.eval()
        preds, labels = [], []
        guids = []

        for guid, input_ids, input_mask, segment_ids, label_ids in tqdm(
            data_loader, desc="Predicting"
        ):
            logits, label_ids, guid_ids = self.prediction_step(
                model, input_ids, input_mask, segment_ids, label_ids, guid
            )
            labels.extend(label_ids.detach().cpu().numpy())
            guids.extend(guid_ids.detach().cpu().numpy())
            preds.append(logits.detach().cpu().numpy()[0])

        return (preds, labels, guids)

    def prediction_loop_onnx(self, data_loader):

        # the stored optimized onnx model; this is only for the sentence selection module
        model_quant = os.path.join(self.args.output_dir, "converted-optimized.onnx")

        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        session = InferenceSession(model_quant, options)

        preds, labels = [], []
        guids = []

        for guid, input_ids, input_mask, segment_ids, label_ids in tqdm(
            data_loader, desc="Predicting"
        ):
            logits, label_ids, guid_ids = self.prediction_step_onnx(
                session, input_ids, input_mask, segment_ids, label_ids, guid
            )
            labels.extend(label_ids.detach().cpu().numpy())
            guids.extend(guid_ids.detach().cpu().numpy())
            preds.append(logits[0])

        return (preds, labels, guids)

    def prediction_step(
        self, model, input_ids, input_mask, segment_ids, label_ids, guid_ids
    ):

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        label_ids = label_ids.to(self.device)
        guid_ids = guid_ids.to(self.device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        return (logits, label_ids, guid_ids)

    def prediction_step_onnx(
        self, session, input_ids, input_mask, segment_ids, label_ids, guid_ids
    ):

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        label_ids = label_ids.to(self.device)
        guid_ids = guid_ids.to(self.device)

        # tokens for session run
        tokens = {
            "input_ids": input_ids.detach().cpu().numpy(),
            "token_type_ids": segment_ids.detach().cpu().numpy(),
            "attention_mask": input_mask.detach().cpu().numpy(),
        }
        logits = session.run(None, tokens)[0]

        return (logits, label_ids, guid_ids)
