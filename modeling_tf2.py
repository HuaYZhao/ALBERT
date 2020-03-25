import tensorflow as tf

from transformers import (
    TFAlbertMainLayer,
    TFAlbertPreTrainedModel,
    AlbertConfig,
)

from transformers.modeling_tf_utils import get_initializer


class SquadQALayer(tf.keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.start_dense = tf.keras.layers.Dense(
            1,
            kernel_initializer=get_initializer(config.initializer_range),
            name="start_dense_0"
        )
        self.end_dense0 = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=tf.tanh,
            name="end_dense_0"
        )
        self.end_dense1 = tf.keras.layers.Dense(
            1,
            kernel_initializer=get_initializer(config.initializer_range),
            name="end_dense_1"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        self.answer_dense0 = tf.keras.layers.Dense(
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=get_initializer(config.initializer_range),
            name="answer_dense_0"
        )
        self.answer_dense1 = tf.keras.layers.Dense(
            1,
            kernel_initializer=get_initializer(config.initializer_range),
            name="answer_dense_1",
            use_bias=False
        )
        self.dropout = tf.keras.layers.Dropout(rate=kwargs.get("dropout_prob", config.classifier_dropout_prob))

    def forward(self,
                sequence_output,
                features,
                start_n_top,
                end_n_top,
                mode,
                **kwargs):
        is_training = mode == "train"
        max_seq_length, bsz, _ = tf.shape(sequence_output)
        print(sequence_output.shape)
        p_mask = features.get("p_mask", tf.ones([bsz, max_seq_length]))
        start_positions = features.get("start_positions", tf.zeros([bsz, max_seq_length]))
        return_dict = {}

        start_logits = self.start_dense(sequence_output)
        start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

        if is_training:
            start_positions = tf.reshape(start_positions, [-1])
            start_index = tf.one_hot(start_positions, depth=max_seq_length, axis=-1,
                                     dtype=tf.float32)
            start_features = tf.einsum("lbh,bl->bh", sequence_output, start_index)
            start_features = tf.tile(start_features[None], [max_seq_length, 1, 1])
            end_logits = self.end_dense0(tf.concat([sequence_output, start_features], axis=-1))
            tf.keras.layers.LayerNormalization()
            end_logits = self.LayerNorm(end_logits)

            end_logits = self.end_dense1(end_logits)
            end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
            end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
            end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)

            return_dict["start_log_probs"] = start_log_probs
            return_dict["end_log_probs"] = end_log_probs
        else:
            start_top_log_probs, start_top_index = tf.nn.top_k(
                start_log_probs, k=start_n_top)
            start_index = tf.one_hot(start_top_index,
                                     depth=max_seq_length, axis=-1, dtype=tf.float32)
            print(sequence_output.shape)
            print(start_index.shape)
            start_features = tf.einsum("lbh,bkl->bkh", sequence_output, start_index)
            end_input = tf.tile(sequence_output[:, :, None],
                                [1, 1, start_n_top, 1])
            start_features = tf.tile(start_features[None],
                                     [max_seq_length, 1, 1, 1])
            end_input = tf.concat([end_input, start_features], axis=-1)
            end_logits = self.end_dense0(end_input)

            end_logits = self.LayerNorm(end_logits)
            end_logits = self.end_dense1(end_logits)
            end_logits = tf.reshape(end_logits, [max_seq_length, -1, start_n_top])
            end_logits = tf.transpose(end_logits, [1, 2, 0])
            end_logits_masked = end_logits * (
                    1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
            end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
            end_top_log_probs, end_top_index = tf.nn.top_k(
                end_log_probs, k=end_n_top)
            end_top_log_probs = tf.reshape(
                end_top_log_probs,
                [-1, start_n_top * end_n_top])
            end_top_index = tf.reshape(
                end_top_index,
                [-1, start_n_top * end_n_top])

            return_dict["start_top_log_probs"] = start_top_log_probs
            return_dict["start_top_index"] = start_top_index
            return_dict["end_top_log_probs"] = end_top_log_probs
            return_dict["end_top_index"] = end_top_index

        cls_index = tf.one_hot(tf.zeros([bsz], dtype=tf.int32),
                               max_seq_length,
                               axis=-1, dtype=tf.float32)
        cls_feature = tf.einsum("lbh,bl->bh", sequence_output, cls_index)

        # get the representation of START
        start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                                name="softmax_start")
        start_feature = tf.einsum("lbh,bl->bh", sequence_output, start_p)

        # note(zhiliny): no dependency on end_feature so that we can obtain
        # one single `cls_logits` for each sample
        ans_feature = tf.concat([start_feature, cls_feature], -1)
        ans_feature = self.answer_dense0(ans_feature)

        ans_feature = self.dropout(ans_feature, training=is_training)

        cls_logits = self.answer_dense1(ans_feature)

        cls_logits = tf.squeeze(cls_logits, -1)

        return_dict["cls_logits"] = cls_logits

        return return_dict

    def call(self, sequence_output,
             features,
             start_n_top,
             end_n_top,
             mode,
             **kwargs):
        return self.forward(sequence_output,
                            features,
                            start_n_top,
                            end_n_top,
                            mode,
                            **kwargs)


class SquadTFAlbertModel(TFAlbertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(SquadTFAlbertModel, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.albert = TFAlbertMainLayer(config, name="albert")

        self.qa_layer = SquadQALayer(config, name="qa_layer")

    def call(self, input_ids, **kwargs):
        mode = kwargs.get("mode", "predict")
        start_n_top = kwargs.get("start_n_top", 5)
        end_n_top = kwargs.get("end_n_top", 5)

        input_mask = kwargs.get("input_mask", None)
        segment_ids = kwargs.get("segment_ids", None)
        print(input_ids)
        print(input_ids.shape)
        outputs = self.albert(input_ids,
                              attention_mask=input_mask,
                              token_type_ids=segment_ids,
                              inputs_embeds=None,
                              training=mode == "train")

        sequence_output = outputs[0]
        print(sequence_output.shape)

        sequence_output = tf.transpose(sequence_output, [1, 0, 2])

        return_dict = self.qa_layer(sequence_output,
                                    features=kwargs,
                                    start_n_top=start_n_top,
                                    end_n_top=end_n_top,
                                    mode=mode)

        return return_dict
