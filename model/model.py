import tensorflow as tf
import matplotlib.pyplot as plt

CLASSES = 2

input_size = 244



def build_feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    return x


def build_model_adaptor(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return x


def build_classifier_head(inputs):
    return tf.keras.layers.Dense(CLASSES, activation='softmax', name='classifier_head')(inputs)


def build_regressor_head(inputs):
    return tf.keras.layers.Dense(units='4', name='regressor_head')(inputs)

def compile_model(input_size=244):
    def build_model(inputs):
        feature_extractor = build_feature_extractor(inputs)

        model_adaptor = build_model_adaptor(feature_extractor)

        classification_head = build_classifier_head(model_adaptor)

        regressor_head = build_regressor_head(model_adaptor)

        model = tf.keras.Model(inputs=inputs, outputs=[classification_head, regressor_head])

        return model

    model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'classifier_head': 'categorical_crossentropy', 'regressor_head': 'mse'},
                  metrics={'classifier_head': 'accuracy', 'regressor_head': 'mse'})

    return model


def plot_accuracy(training_history):
    plt.plot(training_history.history['classifier_head_accuracy'])
    plt.plot(training_history.history['val_classifier_head_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_classification_loss(training_history):
    plt.plot(training_history.history['classifier_head_loss'])
    plt.plot(training_history.history['val_classifier_head_loss'])
    plt.title('Classification Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_regression_loss(training_history):
    plt.plot(training_history.history['regressor_head_loss'])
    plt.plot(training_history.history['val_regressor_head_loss'])
    plt.title('Bounding Box Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()