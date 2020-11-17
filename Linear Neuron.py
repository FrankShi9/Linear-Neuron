
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

path = 'cereal.csv'
data = pd.read_csv(path)
filtered_data = data.dropna(axis = 0)

y = filtered_data.calories
features = ['protein','fat','sugars']
x = filtered_data[features]

#create a network with 1 linear unit
model = keras.Sequential([layers.Dense(units=1, input_shape=[3])])
                               #how many output  #num of features

from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split (x, y, random_state = 0)

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit(train_x, train_y)

prediction = model.predict(valid_x)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(valid_y, prediction))
