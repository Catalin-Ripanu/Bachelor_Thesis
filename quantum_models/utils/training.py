from typing import Optional
import time

import bentoml
from functools import partial
import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import jaxlib
import flax.linen
from flax.training import train_state, checkpoints
import optax
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from tqdm import tqdm


TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


class TrainState(flax.training.train_state.TrainState):
    key: jax.random  # type: ignore


@partial(jax.jit, static_argnames=["src", "trg", "src_mask", "trg_mask"])
def train_step(
    state: TrainState,
    inputs: jax.Array,
    labels: jax.Array,
    key,
    src,
    trg,
    src_mask,
    trg_mask,
) -> TrainState:
    """
    Performs a single training step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.
        key: The random key to use.

    Returns:
        The updated training state.
    """
    key, dropout_key = jax.random.split(key=key)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            x=inputs,
            src=src,
            trg=trg,
            src_mask=src_mask,
            trg_mask=trg_mask,
            train=True,
            rngs={"dropout": dropout_train_key},
        )
        if logits.shape[1] <= 2:
            if logits.shape[1] == 2:
                logits = logits[:, 1]
            loss = optax.sigmoid_binary_cross_entropy(
                logits=logits, labels=labels
            ).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=labels
            ).mean()
        # return loss, logits
        return loss

    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, logits), grads = grad_fn(state.params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@partial(jax.jit, static_argnames=["src", "trg", "src_mask", "trg_mask"])
def eval_step(
    state: TrainState,
    inputs: jax.Array,
    labels: jax.Array,
    src,
    trg,
    src_mask,
    trg_mask,
) -> tuple[float, jax.Array]:
    """
    Performs a single evaluation step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.

    Returns:
        loss: The loss on the given batch.
        logits: The logits on the given batch.
    """
    logits = state.apply_fn(
        {"params": state.params},
        x=inputs,
        src=src,
        trg=trg,
        src_mask=src_mask,
        trg_mask=trg_mask,
        train=False,
        rngs={"dropout": state.key},
    )
    if logits.shape[1] <= 2:
        if logits.shape[1] == 2:
            logits = logits[:, 1]
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
    return loss, logits


def evaluate(
    state: TrainState,
    eval_dataloader,
    num_classes: int,
    src,
    trg,
    src_mask,
    trg_mask,
    tqdm_desc: Optional[str] = None,
    debug: bool = False,
) -> tuple[float, float, npt.ArrayLike, npt.ArrayLike]:
    """
    Evaluates the model given the current training state on the given dataloader.

    Args:
        state: The current training state.
        eval_dataloader: The dataloader to evaluate on.
        num_classes: The number of classes.
        tqdm_desc: The description to use for the tqdm progress bar. If None, no progress bar is shown.
        debug: Whether to print extra information for debugging.

    Returns:
        eval_loss: The loss.
        eval_auc: The AUC.
    """
    logits, labels = [], []
    eval_loss = 0.0
    with tqdm(
        total=len(eval_dataloader),
        desc=tqdm_desc,
        unit="batch",
        bar_format=TQDM_BAR_FORMAT,
        disable=tqdm_desc is None,
    ) as progress_bar:
        for inputs_batch, labels_batch in eval_dataloader:
            loss_batch, logits_batch = eval_step(
                state, inputs_batch, labels_batch, src, trg, src_mask, trg_mask
            )
            logits.append(logits_batch)
            labels.append(labels_batch)
            eval_loss += loss_batch
            progress_bar.update(1)
        eval_loss /= len(eval_dataloader)
        logits = jnp.concatenate(logits)  # type: ignore
        y_true = jnp.concatenate(labels)  # type: ignore
        if debug:
            print(f"logits = {logits}")
        if num_classes == 2:
            y_pred = [jax.nn.sigmoid(l) for l in logits]
        else:
            y_pred = [jax.nn.softmax(l) for l in logits]
        if debug:
            print(f"y_pred = {y_pred}")
            print(f"y_true = {y_true}")
        if num_classes == 2:
            eval_fpr, eval_tpr, _ = roc_curve(y_true, y_pred)
            eval_auc = auc(eval_fpr, eval_tpr)
        else:
            eval_fpr, eval_tpr = [], []
            eval_auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
        progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, AUC = {eval_auc:.3f}")
        if num_classes != 2:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = jnp.array(y_pred)

            # Define the threshold
            threshold = 0.5

            # Apply the threshold to get binary predictions
            binary_pred = y_pred > threshold

            # Convert boolean array to int
            y_pred = binary_pred.astype(int)

    return eval_loss, eval_auc, eval_fpr, eval_tpr, y_pred, y_true


def nopeak_mask(size):
    np_mask = jnp.triu(jnp.ones((1, size, size)), k=1).astype("uint8")
    np_mask = np_mask == 0
    return np_mask


def create_masks(src, trg, src_pad, trg_pad):
    """
    Create source and target masks.

    Args:
        src: Source input tensor.
        trg: Target input tensor.
        src_pad: Padding token for the source input.
        trg_pad: Padding token for the target input.

    Returns:
        src_mask: Source mask tensor.
        trg_mask: Target mask tensor.
    """
    src_mask = (src != src_pad).astype(jnp.float32)[:, jnp.newaxis, :]

    if trg is not None:
        trg_mask = (trg != trg_pad).astype(jnp.float32)[:, jnp.newaxis, :]
        size = trg.shape[1]  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask


def yesno(response):
    while True:
        if response != "y" and response != "n":
            response = input("command not recognised, enter y or n : ")
        else:
            return response


def train_and_evaluate(
    model: flax.linen.Module,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    num_classes: int,
    num_epochs: int,
    src_mask_flag: bool,
    trg_mask_flag: bool,
    src: float,
    trg: float,
    lrs_peak_value: float = 1e-3,
    lrs_warmup_steps: int = 5_000,
    lrs_decay_steps: int = 50_000,
    seed: int = 42,
    use_ray: bool = False,
    debug: bool = False,
):
    """
    Trains the given model on the given dataloaders for the given hyperparameters.

    The progress and evaluation results are printed to stdout.

    Args:
        model: The model to train.
        train_dataloader: The dataloader for the training set.
        val_dataloader: The dataloader for the validation set.
        num_classes: The number of classes.
        num_epochs: The number of epochs to train for.
        learning_rate: The learning rate to use.
        seed: The seed to use for reproducibility.
        use_ray: Whether to use Ray for logging.
        debug: Whether to print extra information for debugging.

    Returns:
        None
    """

    root_key = jax.random.PRNGKey(seed=seed)
    root_key, params_key, train_key = jax.random.split(key=root_key, num=3)

    dummy_batch = next(iter(train_dataloader))[0]
    input_shape = dummy_batch[0].shape
    input_dtype = dummy_batch[0].dtype
    batch_size = len(dummy_batch)
    root_key, input_key = jax.random.split(key=root_key)
    if jnp.issubdtype(input_dtype, jnp.floating):
        dummy_batch = jax.random.uniform(
            key=input_key, shape=(batch_size,) + tuple(input_shape), dtype=input_dtype
        )
    elif jnp.issubdtype(input_dtype, jnp.integer):
        dummy_batch = jax.random.randint(
            key=input_key,
            shape=(batch_size,) + tuple(input_shape),
            minval=0,
            maxval=100,
            dtype=input_dtype,
        )
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    src_mask = None
    trg_mask = None

    inputs_batch = dummy_batch[0]

    variables = model.init(
        params_key, dummy_batch, src, trg, src_mask, trg_mask, train=False
    )

    if debug:
        print(jax.tree_map(lambda x: x.shape, variables))
    print(
        f"Number of parameters = {sum(x.size for x in jax.tree_util.tree_leaves(variables))}"
    )

    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lrs_peak_value,
        warmup_steps=lrs_warmup_steps,
        decay_steps=lrs_decay_steps,
        end_value=0.0,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_schedule),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        key=train_key,
        tx=optimizer,
    )

    wish = yesno(input("Want to load a model? : "))

    if wish == "y":
        print("Type the model's name")
        name = input()
        model, _ = bentoml.flax.load_model(f"{name}:latest")
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=f"./checkpoints_quantum_ml/{name}/checkpoint_{num_epochs}/",
            target=state,
        )
        state = restored_state

    best_val_auc, best_epoch, best_state = 0.0, 0, None
    total_train_time = 0.0
    start_time = time.time()

    metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_aucs": [],
        "val_aucs": [],
        "test_loss": 0.0,
        "test_auc": 0.0,
        "test_fpr": [],
        "test_tpr": [],
        "best_auc": 0.0,
    }

    step = 0

    for epoch in range(num_epochs):
        step = step + 1
        with tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1:3}/{num_epochs}",
            unit="batch",
            bar_format=TQDM_BAR_FORMAT,
        ) as progress_bar:
            epoch_train_time = time.time()
            for inputs_batch, labels_batch in train_dataloader:

                if src_mask_flag and trg_mask_flag:
                    src_mask, trg_mask = create_masks(inputs_batch, inputs_batch, 0, 0)

                state = train_step(
                    state,
                    inputs_batch,
                    labels_batch,
                    train_key,
                    src,
                    trg,
                    src_mask,
                    trg_mask,
                )
                progress_bar.update(1)
            epoch_train_time = time.time() - epoch_train_time
            total_train_time += epoch_train_time

            train_loss, train_auc, _, _, _, _ = evaluate(
                state,
                train_dataloader,
                num_classes,
                src,
                trg,
                src_mask,
                trg_mask,
                tqdm_desc=None,
                debug=debug,
            )
            val_loss, val_auc, _, _, _, _ = evaluate(
                state,
                val_dataloader,
                num_classes,
                src,
                trg,
                src_mask,
                trg_mask,
                tqdm_desc=None,
                debug=debug,
            )

            progress_bar.set_postfix_str(
                f"Loss = {val_loss:.4f}, AUC = {val_auc:.3f}, Train time = {epoch_train_time:.2f}s"
            )

            metrics["train_losses"].append(train_loss)
            metrics["val_losses"].append(val_loss)
            metrics["train_aucs"].append(train_auc)
            metrics["val_aucs"].append(val_auc)

            if val_auc > best_val_auc:
                metrics["best_auc"] = val_auc
                best_val_auc = val_auc
                best_epoch = epoch + 1
                best_state = state

    metrics["train_losses"] = jnp.array(metrics["train_losses"])
    metrics["val_losses"] = jnp.array(metrics["val_losses"])
    metrics["train_aucs"] = jnp.array(metrics["train_aucs"])
    metrics["val_aucs"] = jnp.array(metrics["val_aucs"])

    print(f"Best validation AUC = {best_val_auc:.3f} at epoch {best_epoch}")
    print(
        f"Total training time = {total_train_time:.2f}s, total time (including evaluations) = {time.time() - start_time:.2f}s"
    )

    # Evaluate on test set using the best model
    assert best_state is not None
    test_loss, test_auc, test_fpr, test_tpr, pred, true = evaluate(
        best_state,
        test_dataloader,
        num_classes,
        src,
        trg,
        src_mask,
        trg_mask,
        tqdm_desc="Testing",
    )

    print(50 * "*")

    print(classification_report(true, pred))

    metrics["test_loss"] = test_loss
    metrics["test_auc"] = test_auc
    metrics["test_fpr"] = test_fpr
    metrics["test_tpr"] = test_tpr
    
    print("\n" + 50 * "*" + "Training Loss" + 50 * "*" + "\n")
    print(metrics["train_losses"])

    print("\n" + 50 * "*" + "Validation Loss" + 50 * "*" + "\n")
    print(metrics["val_losses"])

    print("\n" + 50 * "*" + "Training AUCs" + 50 * "*" + "\n")
    print(metrics["train_aucs"])

    print("\n" + 50 * "*" + "Validation AUCs" + 50 * "*" + "\n")
    print(metrics["test_auc"])

    print("\n" + 50 * "*" + "Test TPR" + 50 * "*" + "\n")
    print(metrics["test_tpr"])
    
    print("\n" + 50 * "*" + "Test FPR" + 50 * "*" + "\n")
    print(metrics["test_fpr"])

    wish = yesno(input("Want to save the model? : "))

    if wish == "y":
        name = input("Enter the name of the model: ")
        print("Saving the model in BentoML store and the checkpoint...")

        bentoml.flax.save_model(f"{name}", model, state)

        checkpoints.save_checkpoint(
            ckpt_dir=f"./checkpoints_quantum_ml/{name}", target=state, step=num_epochs
        )
        
    print(50 * "*")

    return metrics
