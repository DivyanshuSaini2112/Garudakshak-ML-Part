import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load Dataset
data = pd.read_csv("D:\logged_data.csv")

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Verify the columns in the dataset
print("Dataset Columns:")
print(data.columns)

# Create a class based on Signal Strength
def classify_signal_strength(signal_strength):
    if signal_strength > -40:
        return 0  # High signal strength
    elif signal_strength > -60:
        return 1  # Medium signal strength
    else:
        return 2  # Low signal strength

# Apply the function to create a new 'CLASS' column
data['CLASS'] = data['Signal Strength'].apply(classify_signal_strength)

# Now proceed with the rest of the code
print("Dataset Columns after CLASS creation:")
print(data.columns)

# Visualize Important Parameters
plt.figure(figsize=(16, 8))

# Signal Strength Distribution
plt.subplot(2, 2, 1)
sns.histplot(data['Signal Strength'], kde=True, bins=30, color='blue')
plt.title("Signal Strength Distribution")
plt.xlabel("Signal Strength (dBm)")
plt.ylabel("Frequency")

# Modulation Types
plt.subplot(2, 2, 2)
sns.countplot(y='Modulation', data=data, order=data['Modulation'].value_counts().index, palette="viridis")
plt.title("Count of Modulation Types")
plt.xlabel("Count")
plt.ylabel("Modulation Type")

# Bandwidth vs. Signal Strength
plt.subplot(2, 2, 3)
sns.scatterplot(x='Bandwidth', y='Signal Strength', hue='CLASS', data=data, palette="coolwarm")
plt.title("Bandwidth vs Signal Strength")
plt.xlabel("Bandwidth (MHz)")
plt.ylabel("Signal Strength (dBm)")

# Temperature and Humidity Distribution
plt.subplot(2, 2, 4)
sns.scatterplot(x='Temperature', y='Humidity', hue='CLASS', data=data, palette="Set2")
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")

plt.tight_layout()
plt.show()

# Preprocess Data
relevant_columns = [
    "Frequency", "Signal Strength", "Modulation", "Bandwidth", 
    "Antenna Type", "Temperature", "Humidity", "I/Q Data", "CLASS"
]

# Filter relevant columns
data = data[relevant_columns]

# Handle missing values
data = data.dropna()

# Encode categorical columns
label_encoder = LabelEncoder()
data['Modulation'] = label_encoder.fit_transform(data['Modulation'])
data['Antenna Type'] = label_encoder.fit_transform(data['Antenna Type'])
data['CLASS'] = label_encoder.fit_transform(data['CLASS'])

# Extract features and target
X = data.drop(columns=["CLASS"])
y = data["CLASS"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop(columns=["I/Q Data"]))

# One-hot encode the target
y_encoded = to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build Neural Network Model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Plot Training History
plt.figure(figsize=(10, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# Test Model with New Data
new_data = {
 "Frequency": 5.8,  # Example: 5.8 GHz (common frequency for drones)
    "Signal Strength": -45,  # Example: -45 dBm (realistic for drone)
    "Modulation": "QPSK",  # Quadrature Phase Shift Keying (common in wireless communication)
    "Bandwidth": 40,  # Example: 40 MHz (typical bandwidth for wireless communication)
    "Antenna Type": "Omni",  # Example: Omni antenna type
    "Temperature": 30,  # Example: 30 °C (realistic temperature)
    "Humidity": 50,  # Example: 50% Humidity (realistic humidity)
    "I/Q Data": "0.3,0.5,0.6,0.8,0.9,1.0,1.1"  # Example I/Q Data (simplified)
}

# Convert new data to DataFrame
new_data_df = pd.DataFrame([new_data])

# Extract features for prediction (excluding I/Q Data)
features_to_predict = ["Frequency", "Signal Strength", "Modulation", "Bandwidth", "Antenna Type", "Temperature", "Humidity"]
new_data_to_predict = new_data_df[features_to_predict]

# Encode 'Modulation' and 'Antenna Type' based on previously fitted LabelEncoder
new_data_to_predict['Modulation'] = label_encoder.transform(new_data_to_predict["Modulation"])
new_data_to_predict['Antenna Type'] = label_encoder.transform(new_data_to_predict["Antenna Type"])

# Standardize the new data
new_data_scaled = scaler.transform(new_data_to_predict)

# Predict the class of the new data
prediction = model.predict(new_data_scaled)
predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
confidence_score = np.max(prediction)

print(f"Predicted Class: {predicted_class}")
print(f"Confidence Score: {confidence_score:.2f}")