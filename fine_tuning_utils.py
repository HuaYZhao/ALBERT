# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
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
# Lint as: python3
"""Helper library for ALBERT fine-tuning.

This library can be used to construct ALBERT models for fine-tuning, either from
json config files or from TF-Hub modules.
"""

import tokenization


def create_vocab(vocab_file, do_lower_case, spm_model_file, hub_module):
    """Creates a vocab, either from vocab file or from a TF-Hub module."""
    if hub_module:
        return tokenization.FullTokenizer.from_hub_module(
            hub_module=hub_module,
            spm_model_file=spm_model_file)
    else:
        return tokenization.FullTokenizer.from_scratch(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            spm_model_file=spm_model_file)
