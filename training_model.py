# train_captcha_model.py

import os
import cv2
import numpy as np
np.bool = bool  # 修補 imgaug 用到已移除的 np.bool

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, LSTM, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import imgaug.augmenters as iaa

# 讀取資料路徑
label_path = "labels_training.csv"
image_dir = "captchas_training"

characters = list("abcdefghijklmnopqrstuvwxyz")

# 將標籤轉為 one-hot
char_to_vec = lambda c: [1 if i == ord(c) - ord('a') else 0 for i in range(26)]
def text_to_onehot(text):
    return [char_to_vec(c) for c in text]

# 資料增強 pipeline（模糊、扭曲、雜訊）
augmenter = iaa.Sequential([
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-5, 5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    ])
])

# 載入資料
print("📂 載入圖片與標籤...")
df = pd.read_csv(label_path)
X, y = [], []

for _, row in df.iterrows():
    img_path = os.path.join(image_dir, row["filename"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 40))
    X.append(img)
    y.append(text_to_onehot(row["label"]))

X = np.array(X, dtype=np.uint8)
X = np.expand_dims(X, -1)

# 製作資料增強版本
print("🧪 進行資料增強...")
X_aug = augmenter(images=X.squeeze())
X_aug = np.expand_dims(X_aug, -1)

# 合併資料
X_full = np.concatenate([X, X_aug], axis=0)
y_full = y + y  # 標籤直接複製

# 切分資料
X_full = X_full.astype(np.float32) / 255.0
y_split = list(zip(*y_full))
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
y_train_split = [np.array(t) for t in zip(*y_train)]
y_val_split = [np.array(t) for t in zip(*y_val)]

# 建立 CNN + LSTM 模型
print("🧠 建立 CNN + LSTM 模型...")
input_layer = Input(shape=(40, 100, 1))
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
print("⚠️ 卷積後的 x shape:", x.shape)
x = Reshape((25, 640))(x)  # 或者根據前面卷積後的 shape 改成對的數值
x = LSTM(128, return_sequences=True)(x)
x = TimeDistributed(Dense(64, activation='relu'))(x)

# 切割成 4 個 Dense 輸出
outputs = [Dense(26, activation='softmax', name=f'char_{i}')(x[:, i, :]) for i in range(4)]

model = Model(inputs=input_layer, outputs=outputs)
model.compile(
    loss={
        'char_0': 'categorical_crossentropy',
        'char_1': 'categorical_crossentropy',
        'char_2': 'categorical_crossentropy',
        'char_3': 'categorical_crossentropy'
    },
    optimizer='adam',
    metrics={
        'char_0': 'accuracy',
        'char_1': 'accuracy',
        'char_2': 'accuracy',
        'char_3': 'accuracy'
    }
)

model.summary()

# 加入 EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 開始訓練
print("🚀 開始訓練...")
history = model.fit(
    X_train, y_train_split,
    validation_data=(X_val, y_val_split),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop]
)

# 儲存模型
model.save("captcha_model_cnn_lstm.keras")
print("✅ 模型已儲存為 captcha_model_cnn_lstm.keras")
