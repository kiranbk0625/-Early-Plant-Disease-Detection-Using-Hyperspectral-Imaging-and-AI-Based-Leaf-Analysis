# model_app.py
"""
Single-file Plant Disease classifier app:
- Hardcoded 38 class labels (so no missing .npy issues)
- Saves class_labels.npy automatically
- Trains a Keras CNN using ImageDataGenerator.flow_from_directory if dataset present
- Saves/loads model to MODEL_PATH
- Streamlit app: upload image -> top-3 predictions, show classes, optionally train
"""
import kagglehub
import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
import traceback
model = load_model("plant_disease_model.h5")
model.summary()





# Step 1: Download dataset
path = kagglehub.dataset_download("emmarex/plantdisease")
print("Path to dataset files:", path)

data_dir = os.path.join(path, "PlantVillage")

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# --------- CONFIG ----------
DATASET_PATH = "."    # <-- set to your dataset directory (folders per class)
MODEL_PATH = "plant_disease_model.h5"
CLASS_NPY = "class_labels.npy"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8   # change to larger (20-30) for better accuracy; small for quick tests
# ---------------------------

# Hardcoded class list (38 classes from the PlantVillage dataset)
CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Save class labels array if not present (saved as plain numpy array of strings)
if not os.path.exists(CLASS_NPY):
    np.save(CLASS_NPY, np.array(CLASS_LABELS))
    # no pickle used; this file can be loaded with np.load without allow_pickle

# Helper: build the CNN model (simple but practical)
def build_model(num_classes):
    model = Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Try to load model & labels
model = None
loaded_labels = None
load_errors = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        load_errors = traceback.format_exc()

# Load labels from the .npy we wrote (fallback to CLASS_LABELS)
try:
    loaded_labels = np.load(CLASS_NPY)
    # If it's an ndarray of dtype <U... convert to list
    if isinstance(loaded_labels, np.ndarray):
        loaded_labels = loaded_labels.tolist()
except Exception:
    loaded_labels = CLASS_LABELS

num_classes = len(loaded_labels)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌱 Plant Disease Detector (38 classes)")
st.write("This app can train a CNN from a `flow_from_directory` dataset and predict diseases from uploaded leaf images.")
st.markdown("**Dataset folder structure:** `DATASET_PATH/<class_name>/*.jpg` (one subfolder per class)")

col1, col2 = st.columns([1,1])
with col1:
    st.write(f"**Model file:** `{MODEL_PATH}`")
    st.write(f"**Class labels:** {num_classes} classes")
with col2:
    if model is None:
        st.warning("No trained model found. You can train one (requires dataset).")
    else:
        st.success("Model loaded successfully ✅")
        if load_errors:
            st.text("Model load logged errors (check console).")

# Show class list (collapsible)
with st.expander("Show class labels (click)"):
    st.write(loaded_labels)

# Check dataset presence
dataset_exists = os.path.exists(DATASET_PATH) and any(os.scandir(DATASET_PATH))
if not dataset_exists:
    st.error(f"Dataset not found at `{DATASET_PATH}` or directory is empty.")
    st.info("Place your dataset in the folder and restart the app, or use 'Upload & Quick Train' below for a tiny demo.")
else:
    st.success(f"Dataset found at `{DATASET_PATH}`")

# Option: quick upload demo training (small)
st.write("---")
st.header("Quick demo (optional)")
st.write("If you don't have the full dataset ready, you can upload a few labeled images (per class) to quickly train a tiny model for testing. This is only for experimentation and not recommended for production accuracy.")

with st.form("quick_train_form"):
    demo_class_name = st.text_input("Demo class name (e.g., 'demo_healthy')", value="demo_healthy")
    demo_files = st.file_uploader("Upload a few images (jpg/png) for this demo class (multiple allowed)", accept_multiple_files=True)
    submit_demo = st.form_submit_button("Create demo folder & quick-train")
if submit_demo and demo_files:
    demo_folder = os.path.join("demo_data", demo_class_name)
    os.makedirs(demo_folder, exist_ok=True)
    # Save uploaded files to demo folder
    saved_count = 0.0
    for f in demo_files:
        try:
            img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            # save as JPG
            save_path = os.path.join(demo_folder, f.filename)
            img.save(save_path)
            saved_count += 1
        except Exception as e:
            st.write("Failed to save", f.filename, e)
    st.success(f"Saved {saved_count} images to {demo_folder}")
    st.info("You can use this demo folder as part of a small dataset by putting multiple demo class folders inside `demo_data/` and retraining.")

st.write("---")
st.header("Train / Retrain model (use only if dataset is ready)")
st.write("Training will use `ImageDataGenerator.flow_from_directory` and expects each class in a separate subfolder under the dataset path.")
train_button = st.button("Start Training (this runs in the same process)")

if train_button:
    if not dataset_exists:
        st.error(f"No dataset found at `{DATASET_PATH}`. Place your dataset there first.")
    else:
        try:
            st.info("Preparing data generator...")
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )
            train_gen = datagen.flow_from_directory(
                DATASET_PATH,
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                subset="training",
                shuffle=True
            )
            val_gen = datagen.flow_from_directory(
                DATASET_PATH,
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                subset="validation",
                shuffle=False
            )
            # if class indices differ, create mapping and save labels
            classes_from_data = sorted(list(train_gen.class_indices.keys()))
            if classes_from_data != loaded_labels:
                st.warning("Class labels from dataset differ from embedded list. Using labels from dataset.")
                loaded_labels = classes_from_data
                np.save(CLASS_NPY, np.array(loaded_labels))
            num_classes = len(loaded_labels)
            st.write(f"Detected {num_classes} classes. Starting training for {EPOCHS} epochs...")
            # Build model
            model = build_model(num_classes)
            # Train - show progress in streamlit
            history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
            model.save(MODEL_PATH)
            st.success("Training finished and model saved.")
            st.write("Training history keys:", list(history.history.keys()))
        except Exception as e:
            st.error("Training failed.")
            st.text(traceback.format_exc())

st.write("---")
st.header("Upload & Predict")
st.write("Upload a leaf image and get top-3 predictions (probabilities).")

uploaded_file = st.file_uploader("Upload a leaf image to classify", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        x = np.array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)

        # Ensure model exists
        if model is None:
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
            else:
                st.error("No trained model available. Train a model first.")
                st.stop()

        preds = model.predict(x)[0]  # vector of probabilities
        # get top 3
        top3_idx = preds.argsort()[-3:][::-1]
        st.subheader("Top 3 predictions")
        for i in top3_idx:
            label = loaded_labels[i] if i < len(loaded_labels) else f"Class_{i}"
            st.write(f"- **{label}** — {preds[i]*100:.2f}%")
    except Exception:
        st.error("Prediction failed. See console for details.")
        st.text(traceback.format_exc())

st.write("---")
st.markdown("**Notes & tips:**")
st.markdown("""
- For production/generalization you should train longer (more epochs) and use more augmentation, or fine-tune a pre-trained model (transfer learning).  
- The dataset folder must contain subfolders where each subfolder name is a class label and contains images for that class.  
- If your dataset is very large, training on CPU will be slow — use a GPU or Google Colab.  
""")
