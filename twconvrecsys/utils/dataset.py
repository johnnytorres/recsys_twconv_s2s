

import tensorflow as tf


def main():

    dataset = tf.data.TFRecordDataset([])

    iterator = dataset.make_one_shot_iterator()

    get_next = iterator.get_next()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        batch = session.run(get_next)
        print(batch)


if __name__ == '__main__':
    main()