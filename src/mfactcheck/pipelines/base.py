# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipelines ONNX runtime only"""

import argparse
import os
from tqdm import tqdm
import torch
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions, RunOptions
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ..utils.model_utils import get_model_dir
from ..utils.log_helper import LogHelper

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])

class Pipeline:
    def __init__(self,
                module="",
                args=None,
                args_parser=None,
                ):

        self.device = "cpu"
        self.module = module
        self.args_parser = args_parser

        if not args:
            args = argparse.Namespace()
            self.args = args
        else:
            self.args = args
        self.args = self.args_parser(self.args)

        if not os.path.isdir(self.args.output_dir):
            get_model_dir(output_dir=self.args.output_dir, add_ro=self.args.add_ro, module=self.module, onnx=self.args.onnx) # onnx = True

        self.tokenizer = BertTokenizer.from_pretrained(self.args.output_dir, do_lower_case=False)

        self.options = SessionOptions()
        # 1 thread ensures higher throughput overall 

        # self.options.enable_profiling = True
        self.options.intra_op_num_threads = 1
        self.options.inter_op_num_threads = 1
        # self.options.log_severity_level = 1
        self.options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        # the stored optimized onnx model for the module given by the output_dir value.
        self.model_quant = os.path.join(self.args.output_dir, "converted-optimized.onnx")
        self.session = InferenceSession(self.model_quant, self.options)

    def __call__(self, eval_data, num_eg):
        preds, labels, new_guids, guids_map = self.predict(eval_data, num_eg)
        return (preds, labels, new_guids, guids_map)

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

    def _get_eval_dataloader(self, eval_dataset):
        eval_sampler = SequentialSampler(eval_dataset)
        return DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

    def predict(self, predict_inputs, num_eg):
        eval_data, guid_map = self._prepare_inputs(predict_inputs, num_eg)
        eval_dataloader = self._get_eval_dataloader(eval_data)
        preds, labels, guids = self.prediction_loop_onnx(eval_dataloader)  
        return (preds, labels, guids, guid_map)

    def prediction_loop_onnx(self, data_loader):
    
        preds, labels = [], []
        guids = []

        for guid, input_ids, input_mask, segment_ids, label_ids in tqdm(
            data_loader, desc="Predicting"
        ):
            logits, label_ids, guid_ids = self.prediction_step_onnx(
                self.session, input_ids, input_mask, segment_ids, label_ids, guid
            )
            labels.extend(label_ids.detach().cpu().numpy())
            guids.extend(guid_ids.detach().cpu().numpy())
            preds.append(logits[0])

        return (preds, labels, guids)


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