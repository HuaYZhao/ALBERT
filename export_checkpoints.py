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
r"""Exports a minimal module for ALBERT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import modeling
import tensorflow.compat.v1 as tf
from tensorflow.contrib.framework import list_variables, load_variable

flags.DEFINE_string(
    "albert_directory", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "checkpoint_name", "model.ckpt-best",
    "Name of the checkpoint under albert_directory to be exported.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string("export_path", None, "Path to the output module.")

FLAGS = flags.FLAGS


def cast_ckpt_vars_from_bfloat16_to_float32(ckpt_path):
    with tf.Session() as sess:
        new_var_list = []  # 新建一个空列表存储更新后的Variable变量
        for var_name, _ in list_variables(ckpt_path):  # 得到checkpoint文件中所有的参数（名字，形状）元组
            var = load_variable(ckpt_path, var_name)  # 得到上述参数的值

            # 除了修改参数名称，还可以修改参数值（var）
            new_dtype = tf.float32 if "bfloat16" in var.dtype.name else var.dtype
            fp16_var = tf.Variable(var, name=var_name, dtype=new_dtype)  # 使用加入前缀的新名称重新构造了参数
            new_var_list.append(fp16_var)  # 把赋予新名称的参数加入空列表

        print('starting to write fp16 checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)  # 构造一个保存器
        sess.run(tf.global_variables_initializer())  # 初始化一下参数（这一步必做）
        saver.save(sess, FLAGS.export_path)  # 直接进行保存
        print("finished modify float32 to float16 !")


def main(_):
    checkpoint_path = os.path.join(FLAGS.albert_directory, FLAGS.checkpoint_name)
    cast_ckpt_vars_from_bfloat16_to_float32(checkpoint_path)


if __name__ == "__main__":
    flags.mark_flag_as_required("albert_directory")
    flags.mark_flag_as_required("export_path")
    app.run(main)
