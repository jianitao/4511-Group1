import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


# Load the dataset from an Excel file into a pandas DataFrame
file_path = './SP_7JXQ_A_no-H2O_1cons_M1-div_arm_hb_16rota_new-smiles_dedup_full_columns_sample.xlsx'
df = pd.read_excel(file_path)
print(f"Original dataset size:",df.shape)
# Remove duplicate entries
df = df.drop_duplicates()
print(f"Dataset size after removing duplicates: {df.shape}")
threshold = 0.8  # Threshold of 80% missing values
df_cleaned = df.dropna(axis='columns', thresh=int(threshold * len(df)))
# Shape of the new dataset
print(f"Dataset size after handling missing values:", df_cleaned.shape)


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # SMILES parsing check
        return [None] * 4  # Adjust the number if you change the number of descriptors
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]

# Function to compute ECFP (Morgan) fingerprints
def compute_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # SMILES parsing check
        return [None] * 2048  # Adjust if you change the fingerprint length
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

# Ensure all SMILES data are strings and filter out any rows where SMILES might be NaN or improper format
df['SMILES'] = df['SMILES'].astype(str)
df = df[df['SMILES'] != 'nan']

# Creating new DataFrame columns for descriptors
descriptor_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']
descriptors = df['SMILES'].apply(lambda x: pd.Series(compute_descriptors(x), index=descriptor_names))

# Creating new DataFrame columns for fingerprints
fingerprint_names = [f'Bit_{i}' for i in range(2048)]  # Naming bits for clarity
fingerprints = df['SMILES'].apply(lambda x: pd.Series(compute_fingerprints(x), index=fingerprint_names))

# Concatenating all features (descriptors and fingerprints) with the original DataFrame
full_features_df = pd.concat([df, descriptors, fingerprints], axis=1)
# Print the first few rows of the descriptors dataframe
print("First few rows of molecular descriptors:")
print(descriptors.head())

# Print the first few rows of the fingerprints dataframe
print("First few rows of molecular fingerprints:")
print(fingerprints.head())
# Assuming 'df' contains your original data including 'Docking Score'
# and 'descriptors' and 'fingerprints' are your feature DataFrames
combined_df = pd.concat([df, descriptors, fingerprints], axis=1)

# Remove columns with all NaN values
numeric_features = combined_df.select_dtypes(include=[np.number])
numeric_features = numeric_features.dropna(axis=1, how='all')

# Impute missing values in numeric_features
# Here, we use the median for imputation, but you can choose another strategy
imputer = SimpleImputer(strategy='median')
numeric_features_imputed = imputer.fit_transform(numeric_features)
numeric_features_imputed_df = pd.DataFrame(numeric_features_imputed, columns=numeric_features.columns)

# Separate the target variable
if 'docking score' in combined_df.columns:
    targets = combined_df['docking score']
else:
    targets = None
    print("Target column 'docking score' was not found or defined, please verify your dataset.")

# Normalize the numeric features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(numeric_features_imputed_df)
normalized_features_df = pd.DataFrame(normalized_features, columns=numeric_features_imputed_df.columns)

# Normalize the 'docking score', if it exists
if targets is not None:
    # Impute missing values in targets if necessary
    targets_imputed = imputer.fit_transform(targets.values.reshape(-1, 1)).flatten()
    target_scaler = MinMaxScaler()
    normalized_targets = target_scaler.fit_transform(targets_imputed.reshape(-1, 1)).flatten()
# Display the first few rows of the normalized features
print("Normalized Features:")
print(normalized_features_df.head())

# If the target variable exists and was normalized, display it as well
if 'docking score' in combined_df.columns:
    # Create a DataFrame for the normalized targets for easy viewing
    normalized_targets_df = pd.DataFrame(normalized_targets, columns=['Normalized Docking Score'])
    print("\nNormalized Target Variable ('docking score'):")
    print(normalized_targets_df.head())
else:
    print("Target column 'docking score' was not found or defined, please verify your dataset.")
    
train, test = train_test_split(normalized_features_df, test_size = 0.2, random_state=20)
valid, test = train_test_split(test, test_size = 0.5, random_state=20)

# Determine the number of input features. This should match the number of columns in your training data.
n_features = normalized_features_df.shape[1]

# Initialize the model
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(units=1024, activation='relu', input_shape=(n_features,)))

# Adding the second hidden layer
model.add(Dense(units=512, activation='relu'))

# Adding the output layer - since it's a regression problem, we have one output and no activation function
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Split your normalized_features_df and normalized_targets into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(normalized_features_df, normalized_targets, test_size=0.2, random_state=20)

# Define a simple schedule to decrease the learning rate gradually
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    LearningRateScheduler(scheduler, verbose=1)
]

# Assuming you have the Sequential model defined as 'model' and compiled

# Train the model with the enhancements
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100, batch_size=32, verbose=1,
                    callbacks=callbacks)

# Save the model: 
save_dir = './models/'

# If it is not created yet, create the directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model.save(os.path.join(save_dir, 'baseline.h5'))

# Plotting the training history

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss Progression')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['lr'], label='Learning Rate')
plt.title('Learning Rate Progression')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.savefig('./figures/training_val_learning_rate.png')