import os
import tarfile

import numpy as np
import gdown
import tensorflow_datasets as tfds
import tensorflow as tf

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")

options = tf.data.Options()
options.deterministic = True


def datasets_to_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    drop_remainder=True,
    transform_train=None,
    transform_test=None,
):
    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(10_000, reshuffle_each_iteration=True)
    val_dataset = val_dataset.shuffle(10_000, reshuffle_each_iteration=True)
    test_dataset = test_dataset.shuffle(10_000, reshuffle_each_iteration=True)

    # Transform
    if transform_train is not None:
        train_dataset = train_dataset.map(
            transform_train, num_parallel_calls=tf.data.AUTOTUNE
        )
        val_dataset = val_dataset.map(
            transform_train, num_parallel_calls=tf.data.AUTOTUNE
        )

    if transform_test is not None:
        test_dataset = test_dataset.map(
            transform_test, num_parallel_calls=tf.data.AUTOTUNE
        )

    train_dataset = train_dataset.batch(batch_size, drop_remainder=drop_remainder)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=drop_remainder)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Prefetch
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # Convert to NumPy for JAX
    return (
        tfds.as_numpy(train_dataset),
        tfds.as_numpy(val_dataset),
        tfds.as_numpy(test_dataset),
    )


def get_mnist_dataloaders(
    data_dir: str = "~/data", batch_size: int = 1, drop_remainder: bool = True
):
    """
    Returns dataloaders for the MNIST dataset (computer vision, multi-class classification)

    Information about the dataset: https://www.tensorflow.org/datasets/catalog/mnist
    """
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset, val_dataset, test_dataset = tfds.load(
        name="mnist",
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
    )
    train_dataset, val_dataset, test_dataset = (
        train_dataset.with_options(options),
        val_dataset.with_options(options),
        test_dataset.with_options(options),
    )
    print(
        "Cardinalities (train, val, test):",
        train_dataset.cardinality().numpy(),
        val_dataset.cardinality().numpy(),
        test_dataset.cardinality().numpy(),
    )

    def normalize_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return (image - 0.1307) / 0.3081, label

    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        drop_remainder=drop_remainder,
        transform_train=normalize_image,
        transform_test=normalize_image,
    )


def get_cifar10_dataloaders(
    data_dir: str = "~/data", batch_size: int = 1, drop_remainder: bool = True
):
    """
    Returns dataloaders for the MNIST dataset (computer vision, multi-class classification)

    Information about the dataset: https://www.tensorflow.org/datasets/catalog/mnist
    """
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset, val_dataset, test_dataset = tfds.load(
        name="cifar10",
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
    )
    train_dataset, val_dataset, test_dataset = (
        train_dataset.with_options(options),
        val_dataset.with_options(options),
        test_dataset.with_options(options),
    )
    print(
        "Cardinalities (train, val, test):",
        train_dataset.cardinality().numpy(),
        val_dataset.cardinality().numpy(),
        test_dataset.cardinality().numpy(),
    )

    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        drop_remainder=drop_remainder,
    )


def get_image_net_dataloaders(
    data_dir: str = "~/data", batch_size: int = 1, drop_remainder: bool = True
):
    # Load datasets
    train_dataset, val_dataset, test_dataset = tfds.load(
        name="imagenet2012",
        split=["train", "validation", "test"],
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
        download=True,
    )

    print(
        "Cardinalities (train, val, test):",
        train_dataset.cardinality().numpy(),
        val_dataset.cardinality().numpy(),
        test_dataset.cardinality().numpy(),
    )

    def normalize_train_image(image, label):
        # Resize to 32x32
        image = tf.image.resize(image, [32, 32])
        
        
        # Convert image to float32 and normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Mean and std for ImageNet
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        
        # Normalize
        image = (image - mean) / std
        return image, label

    def normalize_test_image(image, label):
        # Resize to 32x32
        image = tf.image.resize(image, [32, 32])

        # Convert image to float32 and normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Mean and std for ImageNet
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        
        # Normalize
        image = (image - mean) / std
        return image, label

    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        drop_remainder=drop_remainder,
        transform_train=normalize_train_image,
        transform_test=normalize_test_image,
    )


def get_cifar100_dataloaders(
    data_dir: str = "~/data", batch_size: int = 1, drop_remainder: bool = True
):
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset, val_dataset, test_dataset = tfds.load(
        name="cifar100",
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
    )
    train_dataset, val_dataset, test_dataset = (
        train_dataset.with_options(options),
        val_dataset.with_options(options),
        test_dataset.with_options(options),
    )
    print(
        "Cardinalities (train, val, test):",
        train_dataset.cardinality().numpy(),
        val_dataset.cardinality().numpy(),
        test_dataset.cardinality().numpy(),
    )

    def normalize_image(image, label):
        # Cast image to float32
        image = tf.cast(image, tf.float32) / 255.0

        # Normalize using mean and standard deviation for CIFAR-100
        mean = tf.constant([0.5071, 0.4867, 0.4408], dtype=tf.float32)
        std = tf.constant([0.2675, 0.2565, 0.2761], dtype=tf.float32)

        image = (image - mean) / std
        return image, label

    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        drop_remainder=drop_remainder,
        transform_train=normalize_image,
        transform_test=normalize_image,
    )


def get_imdb_dataloaders(
    data_dir: str = "~/data",
    batch_size: int = 1,
    drop_remainder: bool = True,
    max_vocab_size: int = 20_000,
    max_seq_len: int = 512,
):
    """
    Returns dataloaders for the IMDB sentiment analysis dataset (natural language processing, binary classification),
    as well as the vocabulary and tokenizer.

    Information about the dataset: https://www.tensorflow.org/datasets/catalog/imdb_reviews
    """
    import tensorflow_text as tf_text
    from tensorflow_text.tools.wordpiece_vocab.bert_vocab_from_dataset import (
        bert_vocab_from_dataset,
    )

    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset, val_dataset, test_dataset = tfds.load(
        name="imdb_reviews",
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
    )
    train_dataset, val_dataset, test_dataset = (
        train_dataset.with_options(options),
        val_dataset.with_options(options),
        test_dataset.with_options(options),
    )
    print(
        "Cardinalities (train, val, test):",
        train_dataset.cardinality().numpy(),
        val_dataset.cardinality().numpy(),
        test_dataset.cardinality().numpy(),
    )

    # Build vocabulary and tokenizer
    bert_tokenizer_params = dict(lower_case=True)
    vocab = bert_vocab_from_dataset(
        train_dataset.batch(10_000).prefetch(tf.data.AUTOTUNE).map(lambda x, _: x),
        vocab_size=max_vocab_size,
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"],
        bert_tokenizer_params=bert_tokenizer_params,
    )
    vocab_lookup_table = tf.lookup.StaticVocabularyTable(
        num_oov_buckets=1,
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.range(len(vocab), dtype=tf.int64)
        ),  # setting tf.int32 here causes an error
    )
    tokenizer = tf_text.BertTokenizer(vocab_lookup_table, **bert_tokenizer_params)

    def preprocess(text, label):
        # Tokenize
        tokens = tokenizer.tokenize(text).merge_dims(-2, -1)
        # Cast to int32 for compatibility with JAX (note that the vocabulary size is small)
        tokens = tf.cast(tokens, tf.int32)
        # Pad (all sequences to the same length so that JAX jit compiles the model only once)
        padded_inputs, _ = tf_text.pad_model_inputs(tokens, max_seq_length=max_seq_len)
        return padded_inputs, label

    return (
        datasets_to_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size,
            drop_remainder=drop_remainder,
            transform_train=preprocess,
            transform_test=preprocess,
        ),
        vocab,
        tokenizer,
    )
