from global_vars import *

model_1 = keras.models.Sequential([
keras.layers.Input(shape=train.shape[-2:]),    
keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
keras.layers.Dense(64, activation='selu'),
keras.layers.Dense(1),
])

optimizer = keras.optimizers.Adam()
model_1.compile(optimizer=optimizer, loss="mae")
model_1.save('model_1.h5')

history_1 = model_1.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                    epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es])

model_1_preds = model_1.predict(test).squeeze().reshape(-1, 1).squeeze()