import os
from pathlib import Path
from io import StringIO
import sys
sys.stderr = StringIO()


os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import keras
import tensorflow as tf
from keras.utils import FeatureSpace

from main import load_info


def drop_other_diagnoses(
    data: pd.DataFrame,
    exclude_columns: list[str] = None
) -> pd.DataFrame:
    indexes = []
    for index, row in enumerate(data['diagnose']):
        if row not in {'норма', 'астма ремиссия'}:
            indexes.append(index)
    data.drop(indexes, inplace=True)

    if exclude_columns is not None:
        data.drop(columns=exclude_columns, inplace=True)

    feature_space = build_feature_space()
    features = set(feature_space.features)
    features.add('diagnose')
    features = set(data.columns).difference(features)
    if features:
        data.drop(columns=list(features), inplace=True)

    return data


def diagnoses_to_digits(df: pd.DataFrame) -> None:
    diagnoses = df['diagnose'].unique()
    dictionary = {diagnose: index for index, diagnose in enumerate(diagnoses)}
    df.replace({'diagnose': dictionary}, inplace=True)


def dataframe_to_dataset(dataframe: pd.DataFrame) -> tf.data.Dataset:
    dataframe = dataframe.copy()
    labels = dataframe.pop("diagnose")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def prepare_datasets(
    data: pd.DataFrame,
    sample_fraction: float
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    :return: Train and validation dataset
    """
    df_validation = data.sample(frac=sample_fraction, random_state=1658179)
    df_train = data.drop(df_validation.index)
    ds_validation = dataframe_to_dataset(df_validation)
    ds_train = dataframe_to_dataset(df_train)
    ds_validation = ds_validation.batch(32)
    ds_train = ds_train.batch(32)
    return ds_train, ds_validation


def build_feature_space():
    return FeatureSpace(
        features={
            'sex'   : FeatureSpace.integer_categorical(),
            'age'   : FeatureSpace.float_rescaled(),
            'IN T'  : FeatureSpace.float_normalized(),
            'IN P'  : FeatureSpace.float_normalized(),
            'IN P1' : FeatureSpace.float_normalized(),
            'IN P2' : FeatureSpace.float_normalized(),
            'IN P3' : FeatureSpace.float_normalized(),
            'OUT T' : FeatureSpace.float_normalized(),
            'OUT P'  : FeatureSpace.float_normalized(),
            'OUT P1': FeatureSpace.float_normalized(),
            'OUT P2': FeatureSpace.float_normalized(),
            'OUT P3': FeatureSpace.float_normalized(),
        },
        output_mode='concat'
    )


def build_model(inputs: int) -> keras.Model:
    model = keras.Sequential(
        layers=[
            keras.layers.Input((inputs,)),
            # keras.layers.Dense(inputs * 4),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(inputs * 6),
            # keras.layers.Dropout(0.2),
            keras.layers.Dropout(0.5),
            # keras.layers.Dense(inputs * 4),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    model.build()
    model.compile(optimizer="adam")
    return model


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

    training_model.fit(
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
