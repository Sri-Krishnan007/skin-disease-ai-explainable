import numpy as np
import cv2
import tensorflow as tf
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# -------------------------------
# GRAD-CAM (your existing method)
# -------------------------------
def get_gradcam(model, image, last_conv_layer_name="conv_pw_13"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -------------------------------
# GRAD-CAM++ (improved version)
# -------------------------------
def get_gradcam_plus_plus(model, image, layer_name="conv_pw_13"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3
    sum_grads = tf.reduce_sum(conv_outputs * grads_power_3, axis=(1,2))
    alpha = grads_power_2 / (2 * grads_power_2 + sum_grads[..., None, None, None] + 1e-10)
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1,2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam)
    return cam[0].numpy()

# -------------------------------
# HEATMAP OVERLAY
# -------------------------------
def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Ensure heatmap is 2D, float in [0,1]
    if heatmap.ndim == 3:
        heatmap = np.mean(heatmap, axis=-1)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = np.uint8(255 * heatmap)  # Convert to uint8

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (224, 224))  # Ensure same size

    # Ensure both are 3-channel
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result

# -------------------------------
# LIME EXPLANATION
# -------------------------------
def lime_explanation(model, image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image[0].astype("double"),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5
    )
    lime_img = mark_boundaries(temp/255.0, mask)
    # Convert to uint8 for saving
    lime_img = img_as_ubyte(lime_img)
    return lime_img

# -------------------------------
# SALIENCY EXPLANATION
# -------------------------------
def saliency_explanation(model, image):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.cast(image_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        preds = model(image_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, image_tensor)
    # Take the absolute value and max over channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    # Normalize to [0, 255]
    saliency = saliency.numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = (saliency * 255).astype(np.uint8)
    # Convert to 3-channel for saving
    saliency = cv2.cvtColor(saliency, cv2.COLOR_GRAY2BGR)
    return saliency

# -------------------------------
# OCCLUSION SENSITIVITY MAP
# -------------------------------
def occlusion_map(model, image, size=20):
    img = image.copy()
    heatmap = np.zeros((224,224))
    for y in range(0,224,size):
        for x in range(0,224,size):
            occluded = img.copy()
            occluded[:,y:y+size,x:x+size,:] = 0
            pred = model.predict(occluded)
            score = np.max(pred)
            heatmap[y:y+size,x:x+size] = score
    heatmap = heatmap / np.max(heatmap)
    # Convert to uint8 for saving
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# -------------------------------
# ABCDE FEATURE DETECTION
# -------------------------------
def detect_abcde_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    asymmetry = np.sum(edges[:, :112] != edges[:, 112:])
    border_irregularity = np.std(edges)
    color_var = np.std(img)
    diameter = np.count_nonzero(edges)
    features = {
        "Asymmetry": asymmetry,
        "Border Irregularity": border_irregularity,
        "Color Variation": color_var,
        "Diameter": diameter
    }
    return features

# -------------------------------
# SIMILAR IMAGE EXPLANATION
# -------------------------------
def compute_similarity(img1, img2):
    img1 = cv2.resize(img1,(224,224))
    img2 = cv2.resize(img2,(224,224))
    diff = np.mean((img1 - img2)**2)
    similarity = 1 / (1 + diff)
    return similarity

# -------------------------------
# NATURAL LANGUAGE EXPLANATION
# -------------------------------
def generate_text_explanation(label, confidence):
    return f"""
The model predicts the lesion as {label} with confidence {confidence:.2f}%.
The highlighted regions indicate areas with abnormal pigmentation
and irregular borders which influenced the AI decision.
This explanation helps clinicians understand how the model arrived
at the diagnosis.
"""

# -------------------------------
# PREDICTION UNCERTAINTY
# -------------------------------
def prediction_uncertainty(model, image, n=10):
    preds = []
    for _ in range(n):
        pred = model(image, training=True)
        preds.append(pred.numpy())
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)
    return mean_pred, uncertainty