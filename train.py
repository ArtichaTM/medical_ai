from pathlib import Path
from io import StringIO
import sys
sys.stderr = StringIO()

import pandas as pd
import keras
import tensorflow as tf
from general import (
    load_info,
    drop_other_diagnoses,
    prepare_datasets,
    build_feature_space,
    diagnoses_to_digits
)


__all__ = (
    'train'
)


def train(
    name: str,
    exclude_columns: list[str] = None,
    epochs: int = 20,
    sample_fraction: float = 0.2
):
    print('- - - Starting training')
    print('- - Preparing dataset')
    data: pd.DataFrame = load_info()
    data = drop_other_diagnoses(data, exclude_columns=exclude_columns)
    print(f"- Using columns {', '.join(list(data.columns))}")
    diagnoses_to_digits(data)
    ds_train, ds_validation = prepare_datasets(data, sample_fraction=sample_fraction)

    print('- - Adapting feature_space')
    feature_space = build_feature_space()
    train_ds_with_no_labels = ds_train.map(lambda x, _: x)
    feature_space.adapt(train_ds_with_no_labels)

    print('- - Preprocessing datasets')
    ds_train_preprocessed = ds_train.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train_preprocessed = ds_train_preprocessed.prefetch(tf.data.AUTOTUNE)
    ds_validation_preprocessed = ds_validation.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_validation_preprocessed = ds_validation_preprocessed.prefetch(tf.data.AUTOTUNE)
    dict_inputs = feature_space.get_inputs()
    encoded_features = feature_space.get_encoded_features()

    print('- - Creating models')
    x = keras.layers.Dense(32, activation="relu")(encoded_features)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(1, activation="sigmoid")(x)

    training_model = keras.Model(inputs=encoded_features, outputs=predictions)
    training_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

    print('- - Training')

    history = training_model.fit(
        ds_train_preprocessed,
        epochs=epochs,
        validation_data=ds_validation_preprocessed,
        verbose=2,
    )

    path = Path() / 'models'
    path.mkdir(exist_ok=True)
    if not name.endswith('.keras'):
        name = f'{name}.keras'
    print(f'- Saving to {name}')
    training_model.save(path / name, save_format='keras')

    print('- - Finished!')


def main():
    train(name='20.keras', epochs=50, sample_fraction=0.2)


if __name__ == '__main__':
    main()
