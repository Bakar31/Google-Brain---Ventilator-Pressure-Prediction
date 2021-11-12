from global_vars import *
from processing_2nd import train, test, X_train, y_train, X_valid, y_valid

model_5 = keras.models.Sequential([
keras.layers.Input(shape=train.shape[-2:]),    
keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
keras.layers.Dense(128, activation='selu'),
keras.layers.Dense(64, activation='selu'),
keras.layers.Dense(1),
])

optimizer = keras.optimizers.Adam()
model_5.compile(optimizer=optimizer, loss="mae")
model_5.save('model_5.h5')

history_5 = model_5.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                    epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es])

model_5_preds = model_5.predict(test).squeeze().reshape(-1, 1).squeeze()