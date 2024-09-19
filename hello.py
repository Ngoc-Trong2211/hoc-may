import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Đổi từ tqdm.notebook import tqdm
import time

# Đọc dữ liệu từ file CSV
df = pd.read_csv('bmi_data.csv')
df.columns = ['Sex', 'Age', 'Height', 'Weight', 'BMI']

# Drop missing rows
df = df.dropna()

print(df.head(5))  # Hiển thị 5 hàng đầu tiên

# Get an overview of data
print(df.describe())  # Hiển thị thống kê mô tả của dữ liệu

# Tạo màu sắc dựa trên BMI
colors = [(1 - (BMI - 13) / 14, 0, 0) for BMI in df.BMI.values]
fig, ax = plt.subplots()
ax.scatter(df['Weight'].values, df['Height'].values, c=colors, picker=True)
ax.set_xlabel('Weight')
ax.set_ylabel('Height')
ax.set_title('BMI Distribution')
plt.show()

# Phân chia dữ liệu thành tập train và validation
train_pct = 0.8
train_index = int(len(df) * train_pct)

train_data = df.iloc[:train_index].copy()
validation_data = df.iloc[train_index:].copy()
print(f'train = {len(train_data)},\nvalidation = {len(validation_data)}')

# Chọn chỉ các cột số để tính toán trung bình và độ lệch chuẩn
numeric_train_data = train_data.select_dtypes(include=[np.number])

# Khởi tạo các trọng số ngẫu nhiên
def reset():
    global w1, w2, w3, bias
    w1 = np.random.randn()
    w2 = np.random.randn()
    w3 = np.random.randn()
    bias = np.random.randn()

reset()

print_weight = lambda: print(f'w1 = {w1},\nw2 = {w2},\nw3 = {w3},\nbias = {bias}')
print_weight()

# Chuẩn hóa dữ liệu
def normalize(df, means, stds):
    df['Weight'] = (df['Weight'] - means.Weight) / stds.Weight
    df['Height'] = (df['Height'] - means.Height) / stds.Height
    df['Age'] = (df['Age'] - means.Age) / stds.Age
    if 'BMI' in df.columns:
        df['BMI'] = (df['BMI'] - means.BMI) / stds.BMI

# Khôi phục dữ liệu sau khi chuẩn hóa
def de_normalize(df, means, stds):
    df = df.copy()
    df['Weight'] = df['Weight'] * stds.Weight + means.Weight
    df['Height'] = df['Height'] * stds.Height + means.Height
    df['Age'] = df['Age'] * stds.Age + means.Age
    if 'BMI' in df.columns:
        df['BMI'] = df['BMI'] * stds.BMI + means.BMI
    if 'predictionBMI' in df.columns:
        df['predictionBMI'] = df['predictionBMI'] * stds.BMI + means.BMI
    return df

means = numeric_train_data.mean()
stds = numeric_train_data.std()
normalize(train_data, means, stds)
print('Normalized train data')
print(train_data.head())

normalize(validation_data, means, stds)
print('Normalized validation data')
print(validation_data.head())

# Hàm dự đoán BMI
def predict_BMI(df):
    pred = w1 * df['Height'] + w2 * df['Weight'] + w3 * df['Age'] + bias
    df['predictionBMI'] = pred
    return df

print('Random weights predictions')
preddf = predict_BMI(train_data)
print(preddf.head())

# Hàm tính toán loss (mức độ sai số)
def calculate_loss(df):
    return np.square(df['predictionBMI'] - df['BMI'])

preddf = predict_BMI(train_data)
print('loss = ', calculate_loss(preddf).mean())

# Hàm tính gradient
def calculate_gradients(df):
    diff = df['predictionBMI'] - df['BMI']
    dw1 = 2 * diff * df['Height']
    dw2 = 2 * diff * df['Weight']
    dw3 = 2 * diff * df['Age']
    dbias = 2 * diff
    dw1, dw2, dw3, dbias = dw1.mean(), dw2.mean(), dw3.mean(), dbias.mean()
    return dw1, dw2, dw3, dbias

# Hàm train mô hình
def train(learning_rate=0.01):
    global w1, w2, w3, bias, preddf
    dw1, dw2, dw3, dbias = calculate_gradients(preddf)
    w1 -= dw1 * learning_rate
    w2 -= dw2 * learning_rate
    w3 -= dw3 * learning_rate
    bias -= dbias * learning_rate
    preddf = predict_BMI(train_data)
    return calculate_loss(preddf).mean()

print('\nPrediction on validation set before training')
print(de_normalize(predict_BMI(validation_data), means, stds).head(10))

learning_rate = 0.01

for i in tqdm(range(300)):
    loss = train(learning_rate)
    time.sleep(0.01)
    if i % 20 == 0:
        print(f'epoch: {i}, loss = {loss}')

print('After training:')
print_weight()

print('\nPrediction on validation set after training')
print(de_normalize(predict_BMI(validation_data), means, stds).head(10))

# Hàm dự đoán cho dữ liệu mới
def predictBMI_real(data):
    df = pd.DataFrame(data)
    normalize(df, means, stds)
    df = predict_BMI(df)
    return de_normalize(df, means, stds)

# Dự đoán cho dữ liệu mới
new_data = [{'name': 'Krishan', 'Age': 30, 'Height': 68, 'Weight': 157.63}]
predicted_df = predictBMI_real(new_data)
print(predicted_df)
