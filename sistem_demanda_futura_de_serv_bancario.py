import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Dados simulados
ti_data = np.random.rand(100, 1) * 10  # Dados de TI (carga de servidores)
bank_data = np.random.rand(100, 1) * 100  # Dados bancários (transações)

# Normalizando os dados
ti_data = (ti_data - np.mean(ti_data)) / np.std(ti_data)
bank_data = (bank_data - np.mean(bank_data)) / np.std(bank_data)

# Criando a variável(demanda futura)
future_demand = np.roll(bank_data, -1)
future_demand[-1] = np.random.rand() * 100  # Último valor simulado

# Dividindo os dados em conjuntos
train_size = int(len(ti_data) * 0.8)
ti_train, ti_test = ti_data[:train_size], ti_data[train_size:]
bank_train, bank_test = bank_data[:train_size], bank_data[train_size:]
demand_train, demand_test = future_demand[:train_size], future_demand[train_size:]

# Convertendo para TensorFlow
ti_train_tf = tf.convert_to_tensor(ti_train, dtype=tf.float32)
bank_train_tf = tf.convert_to_tensor(bank_train, dtype=tf.float32)
demand_train_tf = tf.convert_to_tensor(demand_train, dtype=tf.float32)

# Ajustes caso apresente erro
x_train_tf = tf.concat([ti_train_tf, bank_train_tf], axis=1)

# TensorFlow
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mse')

# Treinamento do modelo TensorFlow
model_tf.fit(x=x_train_tf, y=demand_train_tf, epochs=50)

# Convertendo para PyTorch
ti_train_pt = torch.from_numpy(ti_train).float()
bank_train_pt = torch.from_numpy(bank_train).float()
demand_train_pt = torch.from_numpy(demand_train).float()

# Modelo PyTorch
class DemandPredictor(nn.Module):
    def __init__(self):
        super(DemandPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model_pt = DemandPredictor()

criterion = nn.MSELoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.01)

# Treinamento do PyTorch
for epoch in range(50):
    inputs = torch.cat([ti_train_pt, bank_train_pt], dim=1)
    outputs = model_pt(inputs)
    loss = criterion(outputs, demand_train_pt.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Avaliação do modelo TensorFlow
ti_test_tf = tf.convert_to_tensor(ti_test, dtype=tf.float32)
bank_test_tf = tf.convert_to_tensor(bank_test, dtype=tf.float32)

# Ajuste para corrigir erro no PyTorch
x_test_tf = tf.concat([ti_test_tf, bank_test_tf], axis=1)

predictions_tf = model_tf.predict(x_test_tf)

# Avaliação do modelo PyTorch
ti_test_pt = torch.from_numpy(ti_test).float()
bank_test_pt = torch.from_numpy(bank_test).float()

inputs_pt = torch.cat([ti_test_pt, bank_test_pt], dim=1)
predictions_pt = model_pt(inputs_pt).detach().numpy()

# Comparação dos resultados
print("Resultados TensorFlow:")
print(predictions_tf.flatten())

print("\nResultados PyTorch:")
print(predictions_pt.flatten())

