from dotenv import load_dotenv
import os

load_dotenv()
from flask import Flask, render_template, request, flash
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from gradcam import (
    get_gradcam, overlay_heatmap, get_gradcam_plus_plus,
    lime_explanation, saliency_explanation, occlusion_map,
    detect_abcde_features, prediction_uncertainty
)
import os
import requests


# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = "medvision_secret_key"  # Needed for flashing messages

# --- Model and Classes ---
model = None

def get_model():
    global model
    if model is None:
        model = load_model('mobilenet_model.h5', compile=False)
    return model
classes = [
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis-like lesions",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma"
]

# --- Disease Details ---
disease_details = {
    "Melanocytic nevi": {
        "info": "Commonly known as moles, these are usually benign skin growths formed by clusters of melanocytes. They are generally harmless but should be monitored for any changes.",
        "risk": "Low",
        "precautions": [
            "Monitor for changes in size, color, or shape.",
            "Use sunscreen regularly to prevent new moles.",
            "Consult a dermatologist if a mole becomes irregular, bleeds, or itches."
        ]
    },
    "Melanoma": {
        "info": "A serious and aggressive form of skin cancer that originates in melanocytes. Melanoma can spread rapidly to other parts of the body if not detected early.",
        "risk": "High",
        "precautions": [
            "Seek immediate medical attention for suspicious lesions.",
            "Avoid excessive sun exposure and use broad-spectrum sunscreen.",
            "Schedule regular skin checks with a dermatologist.",
            "Be aware of the ABCDEs (Asymmetry, Border, Color, Diameter, Evolving) of melanoma."
        ]
    },
    "Benign keratosis-like lesions": {
        "info": "These are non-cancerous skin growths that include seborrheic keratoses and lichen planus-like keratoses. They are generally harmless and common in older adults.",
        "risk": "Low",
        "precautions": [
            "No specific treatment is usually required.",
            "Monitor for rapid changes or irritation.",
            "Consult a dermatologist if lesions become painful, bleed, or change appearance."
        ]
    },
    "Basal cell carcinoma": {
        "info": "The most common type of skin cancer, arising from the basal cells in the epidermis. It grows slowly and rarely spreads but can cause local tissue damage.",
        "risk": "Medium",
        "precautions": [
            "Seek prompt medical evaluation for persistent, non-healing sores.",
            "Protect skin from UV exposure with clothing and sunscreen.",
            "Follow up regularly with a dermatologist after treatment."
        ]
    },
    "Actinic keratoses": {
        "info": "Rough, scaly patches on the skin caused by long-term sun exposure. These lesions are considered precancerous and can develop into squamous cell carcinoma.",
        "risk": "Medium",
        "precautions": [
            "Limit sun exposure and use sunscreen daily.",
            "Have regular skin exams to monitor for changes.",
            "Consult a dermatologist for treatment options such as cryotherapy or topical medications."
        ]
    },
    "Vascular lesions": {
        "info": "A group of benign skin conditions involving blood vessels, such as hemangiomas and angiomas. Most are harmless but some may require treatment for cosmetic or medical reasons.",
        "risk": "Low",
        "precautions": [
            "Monitor for rapid growth, bleeding, or pain.",
            "Protect from trauma to prevent bleeding.",
            "Consult a healthcare provider if lesions change or cause concern."
        ]
    },
    "Dermatofibroma": {
        "info": "A common benign skin nodule, often firm and raised, usually resulting from minor skin injuries. Dermatofibromas are harmless and do not require treatment unless symptomatic.",
        "risk": "Low",
        "precautions": [
            "No treatment is usually necessary.",
            "Avoid picking or scratching the lesion.",
            "Consult a dermatologist if the nodule changes, becomes painful, or bleeds."
        ]
    }
}


# --- Utility Functions ---
def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32)
    x_train_mean = 159.98024  # Replace with your actual mean
    x_train_std = 46.52079    # Replace with your actual std
    img_array = (img_array - x_train_mean) / x_train_std
    img_array = np.expand_dims(img_array, axis=0)
    pred = get_model().predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    return class_idx, pred

def get_confidence_explanation(confidence):
    if confidence > 90:
        return "The AI is highly confident in this prediction."
    elif confidence > 70:
        return "The AI is reasonably confident, but a professional review is recommended."
    else:
        return "The AI is uncertain. Please consult a dermatologist for further evaluation."

def get_warning(risk_level, confidence):
    if risk_level == "High" or confidence < 60:
        return "Warning: High risk detected or low confidence. Immediate medical attention is advised."
    return None

def get_similar_images(label, n=5):
    """Get similar images from static/uploads/{label}/ directory"""
    import random
    
    similar_images_dir = os.path.join('static/uploads', label)
    
    # Return empty list if directory doesn't exist
    if not os.path.exists(similar_images_dir):
        return []
    
    # Get all image files in the disease-specific directory
    image_files = [f for f in os.listdir(similar_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Return up to n random images
    selected = random.sample(image_files, k=min(n, len(image_files)))
    return [f'static/uploads/{label}/{img}' for img in selected]

def get_probability_chart(pred):
    probabilities = (pred[0] * 100).tolist()
    return probabilities

def get_groq_explanation(label, info, precautions):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Replace with your actual key
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = (
        f"Patient has a skin lesion classified as '{label}'.\n"
        f"Disease info: {info}\n"
        f"Precautions: {', '.join(precautions)}\n"
        "Provide a clear, safe, and concise medical explanation for a non-expert."
    )
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a safe skin medical assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        return "AI explanation is currently unavailable."

# --- Main Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            flash("No file uploaded. Please select an image.", "error")
            return render_template(
                "index.html",
                label=None,
                confidence=None,
                risk_level=None,
                disease_info=None,
                ai_explanation=None,
                precautions=[],
                probabilities=[],
                class_names=[],
                image=None,
                gradcam_result=None,
                gradcampp_result=None,
                lime_result=None,
                shap_result=None,
                occlusion_result=None,
                abcde_features=None,
                uncertainty=None,
                similar_images=[],
                warning=None
            )
        path = os.path.join("static/uploads", file.filename)
        file.save(path)

        try:
            class_idx, pred = predict_image(path)
            label = classes[class_idx]
            confidence = round(float(np.max(pred)) * 100, 2)

            # Prepare image array for explanations
            img_array = Image.open(path).resize((224, 224))
            img_array = np.asarray(img_array, dtype=np.float32)
            x_train_mean = 159.98024
            x_train_std = 46.52079
            img_array = (img_array - x_train_mean) / x_train_std
            img_array = np.expand_dims(img_array, axis=0)

            # --- Generate All Explanations ---
            gradcam_result = os.path.join("static/uploads", "gradcam_result.jpg")
            gradcampp_result = os.path.join("static/uploads", "gradcampp_result.jpg")
            lime_result = os.path.join("static/uploads", "lime_result.jpg")
            shap_result = os.path.join("static/uploads", "shap_result.jpg")
            occlusion_result = os.path.join("static/uploads", "occlusion_result.jpg")

            # Grad-CAM
            heatmap = get_gradcam(get_model(), img_array)
            gradcam_img = overlay_heatmap(path, heatmap)
            cv2.imwrite(gradcam_result, gradcam_img)

            # Grad-CAM++
            heatmappp = get_gradcam_plus_plus(get_model(), img_array)
            gradcampp_img = overlay_heatmap(path, heatmappp)
            cv2.imwrite(gradcampp_result, gradcampp_img)

            # LIME
            from skimage.io import imsave
            lime_img = lime_explanation(get_model(), img_array)
            imsave(lime_result, lime_img)

            # Saliency Map
            saliency_result = os.path.join("static/uploads", "saliency_result.jpg")
            saliency_img = saliency_explanation(get_model(), img_array)
            cv2.imwrite(saliency_result, saliency_img)

            # Occlusion Map
            occ_map = occlusion_map(get_model(), img_array)
            cv2.imwrite(occlusion_result, (occ_map * 255).astype("uint8"))

            # ABCDE Features
            abcde_features = detect_abcde_features(path)

            # Prediction Uncertainty
            mean_pred, uncertainty_val = prediction_uncertainty(get_model(), img_array)

            # Disease info, risk, precautions
            details = disease_details.get(label, {})
            disease_info = details.get("info", "No information available.")
            risk_level = details.get("risk", "Unknown")
            precautions = details.get("precautions", ["No precautions available."])

            # Confidence explanation
            confidence_explanation = get_confidence_explanation(confidence)

            # AI explanation (Groq)
            ai_explanation = get_groq_explanation(label, disease_info, precautions)

            # Similar images
            similar_images = get_similar_images(label)

            # Probability chart
            probabilities = get_probability_chart(pred)
            class_names = classes

            # Warning system
            warning = get_warning(risk_level, confidence)

            return render_template(
                "index.html",
                label=label,
                confidence=confidence,
                image=path,
                gradcam_result=gradcam_result,
                gradcampp_result=gradcampp_result,
                lime_result=lime_result,
                saliency_result=saliency_result,
                occlusion_result=occlusion_result,
                abcde_features=abcde_features,
                uncertainty=uncertainty_val.tolist() if uncertainty_val is not None else None,
                disease_info=disease_info,
                risk_level=risk_level,
                confidence_explanation=confidence_explanation,
                ai_explanation=ai_explanation,
                similar_images=similar_images,
                precautions=precautions,
                probabilities=probabilities,
                class_names=class_names,
                warning=warning
            )
        except Exception as e:
            flash(f"An error occurred during analysis: {str(e)}", "error")
            return render_template("index.html")

    return render_template("index.html")

# --- Run App ---
if __name__ == "__main__":

    upload_dir = "static/uploads"

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    app.run(host="0.0.0.0", port=10000, debug=False)