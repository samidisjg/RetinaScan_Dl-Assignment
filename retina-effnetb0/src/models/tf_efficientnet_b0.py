import tf_keras as keras

def build_effnet_b0(input_shape=(380, 380, 3), dropout=0.3, pretrained=True):
    inputs = keras.Input(shape=input_shape, name="image")

    x = inputs
    if input_shape[-1] == 1:
        x = keras.layers.Concatenate(axis=-1)([x, x, x])

    x = keras.layers.Resizing(224, 224, interpolation="bilinear", name="resize_to_224")(x)

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights=("imagenet" if pretrained else None),
        pooling="avg",
    )

    feats = base(x, training=False)
    feats = keras.layers.Dropout(dropout)(feats)
    grade = keras.layers.Dense(5, activation="softmax", name="grade")(feats)
    edema = keras.layers.Dense(1, activation="sigmoid", name="edema")(feats)

    # <<< IMPORTANT: list (positional) outputs, but layer names remain "grade" and "edema"
    model = keras.Model(inputs=inputs, outputs=[grade, edema])
    return model, base
