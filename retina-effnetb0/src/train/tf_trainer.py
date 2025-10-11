import os
import tensorflow as tf
import tf_keras as keras
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard

def enable_perf():
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    tf.config.optimizer.set_jit(has_gpu)
    if has_gpu:
        try:
            from tf_keras.mixed_precision import set_global_policy
            set_global_policy("mixed_float16")
        except Exception:
            pass

def compile_model(model, lr=1e-3, w_grade=0.7, w_edema=0.3):
    losses = [
        keras.losses.SparseCategoricalCrossentropy(),  # for grade
        keras.losses.BinaryCrossentropy(),             # for edema
    ]
    metrics = [
        [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        [keras.metrics.BinaryAccuracy(name="accuracy"),
         keras.metrics.AUC(name="auc")],
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=losses,
        loss_weights=[w_grade, w_edema],
        metrics=metrics,
    )
    return model

def cbs(out_best_path, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    return [
        ModelCheckpoint(out_best_path, monitor="val_grade_accuracy", mode="max",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_grade_accuracy", mode="max",
                          factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_grade_accuracy", mode="max",
                      patience=6, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(run_dir, "history.csv"), append=False),
        TensorBoard(log_dir=os.path.join(run_dir, "tb"), profile_batch=0),
    ]

def two_stage_finetune(model, base, train_ds, val_ds, out_best_path, run_dir,
                       epochs1=5, epochs2=20, unfreeze_ratio=0.34,
                       lr1=1e-3, lr2=3e-5):
    if hasattr(base, "trainable"):
        base.trainable = False
    compile_model(model, lr=lr1)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs1,
              callbacks=cbs(out_best_path, run_dir))

    if tf.io.gfile.exists(out_best_path):
        model.load_weights(out_best_path)

    if hasattr(base, "layers") and isinstance(getattr(base, "layers", None), list) and base.layers:
        base.trainable = True
        n = len(base.layers)
        for layer in base.layers[: int((1 - unfreeze_ratio) * n)]:
            layer.trainable = False
    else:
        if hasattr(base, "trainable"):
            base.trainable = True

    compile_model(model, lr=lr2)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs2,
              callbacks=cbs(out_best_path, run_dir))
    return model
