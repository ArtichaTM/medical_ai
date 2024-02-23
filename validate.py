from keras.saving import load_model

from general import load_info


def main(model: str):
    model = load_model(model)


def build_confusion_matrix(model: str):
    model = load_model('models/Test.keras')


if __name__ == '__main__':
    build_confusion_matrix('models/Test.keras')
