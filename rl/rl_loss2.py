import tensorflow as tf
from rl.utils import mask_to_start


def cross_entropy_loss(logits, answer_start, answer_end, project_layers_num, sample_num):
    """
    Cross entropy loss across all decoder timesteps
    """
    logits = tf.concat(logits, axis=0)

    # start_logits = tf.concat(
    #     [tf.tile(_sp, [sample_num, 1]) for _sp in tf.split(logits[:, :, 0], bs * project_layers_num)], axis=0)
    # end_logits = tf.concat(
    #     [tf.tile(_sp, [sample_num, 1]) for _sp in tf.split(logits[:, :, 1], bs * project_layers_num)], axis=0)

    answer_start = tf.tile(answer_start, [project_layers_num])
    answer_end = tf.tile(answer_end, [project_layers_num])

    # answer_start = tf.concat(
    #     [tf.tile(_sp, [sample_num]) for _sp in tf.split(answer_start, bs * project_layers_num)], axis=0)
    #
    # answer_end = tf.concat(
    #     [tf.tile(_sp, [sample_num]) for _sp in tf.split(answer_end, bs * project_layers_num)], axis=0)

    start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits[:, :, 0], labels=answer_start)
    end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits[:, :, 1], labels=answer_end)

    start_loss = tf.stack(tf.split(start_loss, project_layers_num), axis=1)
    end_loss = tf.stack(tf.split(end_loss, project_layers_num), axis=1)
    loss = tf.reduce_mean(tf.reduce_mean(
        start_loss + end_loss, axis=1), axis=0)
    return loss


def simple_tf_f1_score(tensors):
    prediction_start = tf.cast(tensors[0], dtype=tf.float32)
    prediction_end = tf.cast(tensors[1], dtype=tf.float32)
    ground_truth_start = tf.cast(tensors[2], dtype=tf.float32)
    ground_truth_end = tf.cast(tensors[3], dtype=tf.float32)

    min_end = tf.reduce_min([prediction_end, ground_truth_end])
    max_start = tf.reduce_max([prediction_start, ground_truth_start])

    overlap = tf.cond(tf.greater(max_start, min_end), lambda: 0., lambda: min_end - max_start + 1)
    precision = tf.cond(tf.equal(overlap, 0.), lambda: 0., lambda: overlap / (prediction_end - prediction_start + 1))
    recall = tf.cond(tf.equal(overlap, 0.), lambda: 1e-20,
                     lambda: overlap / (ground_truth_end - ground_truth_start + 1))

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def greedy_search_end_with_start(sps, els):
    """
    sps: guess start positions
    els: end logits
    """
    max_seq_len = tf.shape(els)[1]
    sps_mask = tf.sequence_mask(sps - 1, maxlen=max_seq_len, dtype=tf.float32)  # start end 是可以重复的
    els = els * (1 - sps_mask) - 1e30 * sps_mask
    sort_ids = tf.argsort(els, axis=-1, direction="DESCENDING")

    end_greedy = tf.cast(sort_ids[:, 0], tf.int32)

    return end_greedy


def greedy_sample_with_logits(sls, els):
    """
    sls: start logits
    els: end logits
    """
    max_seq_len = tf.shape(sls)[1]
    start_sample = tf.multinomial(sls, 1)
    sps_mask = tf.sequence_mask(tf.squeeze(start_sample) - 1, maxlen=max_seq_len, dtype=tf.float32)  # start end 是可以重复的
    els = els * (1 - sps_mask) - 1e30 * sps_mask
    end_sample = tf.multinomial(els, 1)

    return start_sample, end_sample


def reward(guess_start, guess_end, answer_start, answer_end, baseline, sample_num):
    """
    Reinforcement learning reward (i.e. F1 score) from sampling a trajectory of guesses across each decoder timestep
    """
    reward = [[]] * sample_num

    for t in range(sample_num):
        f1_score = tf.map_fn(
            simple_tf_f1_score, (guess_start[:, t], guess_end[:, t], answer_start, answer_end),
            dtype=tf.float32)  # [bs,]
        normalized_reward = tf.stop_gradient(f1_score - baseline)
        reward[t] = normalized_reward
    # r = 2 * tf.sigmoid(reward) - 1  # 分布变换，保留正负
    r = tf.transpose(reward)
    return r  # [bs, sample_num]


def surrogate_loss(start_logits, end_logits, guess_start, guess_end, r, sample_num):
    """
    The surrogate loss to be used for policy gradient updates
    """
    bsz, seq_length = start_logits.shape.as_list()

    guess_start = tf.reshape(guess_start, [-1])  # (bs * simple_num ,)
    guess_end = tf.reshape(guess_end, [-1])
    r = tf.reshape(r, [-1])
    start_logits = tf.concat([tf.tile(_sp, [sample_num, 1]) for _sp in tf.split(start_logits, bsz)], axis=0)
    end_logits = tf.concat([tf.tile(_sp, [sample_num, 1]) for _sp in tf.split(end_logits, bsz)], axis=0)

    def compute_loss(log_probs, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)

        return - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)

    start_loss = compute_loss(start_logits, guess_start)
    end_loss = compute_loss(end_logits, guess_end)

    loss = start_loss + end_loss
    # loss = tf.where(tf.greater(r, 0), loss, tf.zeros_like(loss))
    # r = tf.where(tf.greater(r, 0), r, tf.zeros_like(r))  # 防止溢出
    # r = 2 * tf.sigmoid(r) - 1.
    loss = r * loss

    loss = tf.stack(tf.split(loss, sample_num), axis=1)
    loss = tf.reduce_mean(loss, axis=1)
    return loss


def rl_loss(start_logits, end_logits, answer_start, answer_end, sample_num=1):
    """
    Reinforcement learning loss
    """
    start_log_probs = tf.nn.log_softmax(start_logits, -1)

    end_log_probs = tf.nn.log_softmax(end_logits, -1)

    guess_start_greedy = tf.argmax(start_logits, axis=1, output_type=tf.int32)

    guess_end_greedy = greedy_search_end_with_start(guess_start_greedy, end_logits)
    f1_baseline = tf.map_fn(simple_tf_f1_score, (guess_start_greedy, guess_end_greedy,
                                                 answer_start, answer_end), dtype=tf.float32)
    em = tf.logical_and(tf.equal(guess_start_greedy, answer_start), tf.equal(guess_end_greedy, answer_end))
    has_no_answer = tf.logical_and(tf.equal(0, answer_start), tf.equal(0, answer_end))

    guess_start_sample = []
    guess_end_sample = []
    for _ in range(sample_num):
        start_sample, end_sample = greedy_sample_with_logits(start_logits, end_logits)
        guess_start_sample.append(start_sample)
        guess_end_sample.append(end_sample)

    guess_start_sample = tf.concat(guess_start_sample, axis=1)
    guess_end_sample = tf.concat(guess_end_sample, axis=1)

    r = reward(guess_start_sample, guess_end_sample, answer_start, answer_end, f1_baseline, sample_num)  # [bs,4]
    surr_loss = surrogate_loss(start_log_probs, end_log_probs, guess_start_sample,
                               guess_end_sample, r, sample_num)

    # This function needs to return the value of loss in the forward pass so that theta_rl gets the right parameter update
    # However, this needs to have the gradient of surr_loss in the backward pass so the model gets the right policy gradient update
    loss = surr_loss + tf.stop_gradient(-tf.reduce_mean(r + f1_baseline, axis=-1) - surr_loss)

    cond_loss = tf.where(tf.logical_or(em, has_no_answer), tf.zeros_like(loss), loss)  # 只有预测错误的才做rl
    return tf.reduce_mean(loss)
