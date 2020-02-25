import tensorflow as tf
from rl.utils import mask_to_start


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


def reward(guess_start, guess_end, answer_start, answer_end, baseline, project_layers_num, simple_num):
    """
    Reinforcement learning reward (i.e. F1 score) from sampling a trajectory of guesses across each decoder timestep
    """
    reward = [[]] * simple_num

    print("answer_start_shape:", answer_start.shape)
    answer_start = tf.tile(answer_start, [project_layers_num])
    answer_end = tf.tile(answer_end, [project_layers_num])
    baseline = tf.tile(baseline, [project_layers_num])

    for t in range(simple_num):
        f1_score = tf.map_fn(
            simple_tf_f1_score, (guess_start[:, t], guess_end[:, t], answer_start, answer_end),
            dtype=tf.float32)  # [bs,]
        normalized_reward = tf.stop_gradient(f1_score - baseline)
        reward[t] = normalized_reward
    return tf.stack(reward)  # [bs * project_layers_num, 4]


def surrogate_loss(logits, guess_start, guess_end, r, project_layers_num, sample_num):
    """
    The surrogate loss to be used for policy gradient updates
    """

    logits = logits.concat()
    guess_start = tf.reshape(guess_start, [-1])  # (bs * project_layers_num * simple_num ,)
    guess_end = tf.reshape(guess_end, [-1])
    r = tf.reshape(r, [-1])

    bs = logits.read(0).shape.as_list()[0]

    start_logits = tf.concat(
        [tf.tile(_sp, [sample_num, 1]) for _sp in tf.split(logits[:, :, 0], bs * project_layers_num)], axis=0)
    end_logits = tf.concat(
        [tf.tile(_sp, [sample_num, 1]) for _sp in tf.split(logits[:, :, 1], bs * project_layers_num)], axis=0)

    start_loss = r * \
                 tf.nn.sparse_softmax_cross_entropy_with_logits(
                     logits=start_logits, labels=guess_start)
    end_loss = r * \
               tf.nn.sparse_softmax_cross_entropy_with_logits(
                   logits=end_logits, labels=guess_end)
    start_loss = tf.stack(tf.split(start_loss, sample_num), axis=1)
    end_loss = tf.stack(tf.split(end_loss, sample_num), axis=1)
    loss = tf.reduce_mean(tf.reduce_mean(
        start_loss + end_loss, axis=1), axis=0)
    return loss


def rl_loss(logits, answer_start, answer_end, project_layers_num=4, sample_num=1):
    """
    Reinforcement learning loss
    """
    final_logits = logits.read(project_layers_num - 1)
    final_start_logits = final_logits[:, :, 0]
    final_end_logits = final_logits[:, :, 1]
    guess_start_greedy = tf.argmax(final_start_logits, axis=1)
    # end_logits = tf.argmax(mask_to_start(
    #     end_logits, guess_start_greedy), axis=1)
    guess_end_greedy = tf.argmax(final_end_logits, axis=1)
    baseline = tf.map_fn(simple_tf_f1_score, (guess_start_greedy, guess_end_greedy,
                                              answer_start, answer_end), dtype=tf.float32)

    guess_start = []
    guess_end = []

    for t in range(project_layers_num):
        logits_t = logits.read(t)
        start_logits = logits_t[:, :, 0]
        end_logits = logits_t[:, :, 1]
        guess_start.append(tf.multinomial(start_logits, sample_num))
        guess_end.append(tf.multinomial(end_logits, sample_num))
    guess_start = tf.concat(guess_start, axis=0)
    guess_end = tf.concat(guess_end, axis=0)
    print("guess_start_shape", guess_start.shape)
    r = reward(guess_start, guess_end, answer_start, answer_end, baseline, project_layers_num,
               sample_num)  # [bs*project_layers,4]
    # print("reward_shape:", r.shape)
    surr_loss = surrogate_loss(logits, guess_start, guess_end, r, project_layers_num, sample_num)
    loss = tf.reduce_mean(-r)

    # This function needs to return the value of loss in the forward pass so that theta_rl gets the right parameter update
    # However, this needs to have the gradient of surr_loss in the backward pass so the model gets the right policy gradient update
    return surr_loss + tf.stop_gradient(loss - surr_loss)
