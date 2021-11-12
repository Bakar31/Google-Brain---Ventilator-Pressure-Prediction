from global_vars import *
from processing import train, test, X_train, y_train, X_valid, y_valid

model_2 = keras.models.Sequential([
keras.layers.Input(shape=train.shape[-2:]), 
keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
keras.layers.Dense(64, activation='selu'),
keras.layers.Dense(64, activation='selu'),
keras.layers.Dense(1),
])

optimizer = keras.optimizers.Adam()
model_2.compile(optimizer=optimizer, loss="mae")
model_2.save('model_2.h5')

history_2 = model_2.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                    epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es])

model_2_preds = model_2.predict(test).squeeze().reshape(-1, 1).squeeze()