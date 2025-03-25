import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, input_shape, num_classes=6):
        self.model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the LSTM model."""
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return history

    def predict(self, X):
        """Predict motion sequences."""
        return self.model.predict(X)

    def save(self, path):
        """Save the trained model."""
        self.model.save(path)

if __name__ == "__main__":
    # Example usage
    tracker = MotionTracker()
    sequences = tracker.preprocess()
    y_dummy = np.random.randint(0, 6, (sequences.shape[0],))  # Dummy labels
    y_dummy = tf.keras.utils.to_categorical(y_dummy, num_classes=6)

    model = LSTMModel(input_shape=(50, 17))  # 50 timesteps, 17 features
    model.train(sequences, y_dummy)
    model.save("lstm_model.h5")