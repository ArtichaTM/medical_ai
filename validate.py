from keras.saving import load_model


def main():
    model = load_model('models/Test.keras')


if __name__ == '__main__':
    main()
