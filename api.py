from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load the trained model
model = tf.keras.models.load_model('C:/Users/admin/Documents/plant_disease_project/plant_disease_data/trained_plant_disease_model.keras')

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___late_blight',
    'Tomato___leaf_mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

recommendations_map = {
    'Apple___Apple_scab': [
        'Remove and destroy infected leaves',
        'Apply appropriate fungicides regularly',
        'Avoid overhead watering to reduce moisture'
    ],
    'Apple___Black_rot': [
        'Prune out and destroy infected branches',
        'Use fungicides containing captan or copper',
        'Maintain good sanitation in orchard'
    ],
    'Apple___Cedar_apple_rust': [
        'Remove nearby cedar trees if possible',
        'Apply fungicides during infection periods',
        'Avoid planting apples near cedars'
    ],
    'Apple___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Blueberry___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Cherry_(including_sour)___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Cherry_(including_sour)___Powdery_mildew': [
        'Apply sulfur-based fungicides',
        'Prune affected areas to improve airflow',
        'Avoid excess nitrogen fertilization'
    ],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        'Rotate crops to reduce inoculum',
        'Apply fungicides when disease appears',
        'Use resistant varieties if available'
    ],
    'Corn_(maize)___Common_rust_': [
        'Apply fungicides early when rust appears',
        'Use resistant hybrids if possible',
        'Maintain good field sanitation'
    ],
    'Corn_(maize)___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        'Use resistant hybrids',
        'Apply fungicides when symptoms first appear',
        'Practice crop rotation'
    ],
    'Grape___Black_rot': [
        'Remove mummified fruit and infected canes',
        'Apply fungicides regularly during growing season',
        'Maintain good air circulation'
    ],
    'Grape___Esca_(Black_Measles)': [
        'Remove and destroy infected wood',
        'Avoid mechanical injuries to vines',
        'Apply protective fungicides if recommended'
    ],
    'Grape___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        'Apply fungicides regularly',
        'Remove fallen leaves and debris',
        'Ensure proper spacing for airflow'
    ],
    'Orange___Haunglongbing_(Citrus_greening)': [
        'Control insect vectors (Asian citrus psyllid)',
        'Remove infected trees promptly',
        'Use disease-free nursery stock'
    ],
    'Peach___Bacterial_spot': [
        'Apply copper-based bactericides',
        'Remove and destroy infected plant parts',
        'Avoid working in wet orchards to reduce spread'
    ],
    'Peach___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Pepper,_bell___Bacterial_spot': [
        'Use disease-free seed',
        'Apply copper sprays as preventive',
        'Avoid overhead irrigation'
    ],
    'Pepper,_bell___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Potato___Early_blight': [
        'Apply fungicides containing chlorothalonil or mancozeb',
        'Practice crop rotation',
        'Remove and destroy infected plant debris'
    ],
    'Potato___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Potato___Late_blight': [
        'Apply appropriate fungicides early',
        'Remove infected plants immediately',
        'Avoid overhead watering'
    ],
    'Raspberry___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Soybean___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Squash___Powdery_mildew': [
        'Apply sulfur or potassium bicarbonate sprays',
        'Ensure good air circulation',
        'Avoid overhead watering'
    ],
    'Strawberry___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Strawberry___Leaf_scorch': [
        'Remove infected leaves',
        'Apply appropriate fungicides',
        'Avoid wet foliage and improve drainage'
    ],
    'Tomato___Bacterial_spot': [
        'Use certified disease-free seeds',
        'Apply copper sprays',
        'Remove and destroy infected plant debris'
    ],
    'Tomato___Early_blight': [
        'Remove affected leaves and stems',
        'Use fungicides containing chlorothalonil or mancozeb',
        'Practice crop rotation'
    ],
    'Tomato___healthy': [
        'No treatment needed; maintain regular care'
    ],
    'Tomato___late_blight': [
        'Apply fungicides promptly',
        'Remove and destroy infected plants',
        'Avoid overhead watering'
    ],
    'Tomato___leaf_mold': [
        'Improve airflow and reduce humidity',
        'Use fungicides as recommended',
        'Avoid watering foliage'
    ],
    'Tomato___Septoria_leaf_spot': [
        'Remove infected leaves',
        'Apply fungicides regularly',
        'Avoid overhead watering'
    ],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        'Use miticides or insecticidal soaps',
        'Introduce natural predators (e.g., predatory mites)',
        'Maintain adequate irrigation to reduce stress'
    ],
    'Tomato___Target_Spot': [
        'Remove and destroy infected leaves',
        'Apply fungicides on time',
        'Avoid dense planting'
    ],
    'Tomato___Tomato_mosaic_virus': [
        'Use virus-free seeds and transplants',
        'Control insect vectors',
        'Remove infected plants promptly'
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        'Control whitefly populations',
        'Use resistant varieties if available',
        'Remove and destroy infected plants'
    ],
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        file_stream = BytesIO(file.read())
        image = tf.keras.preprocessing.image.load_img(file_stream, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        predictions = model.predict(input_arr)
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions))

        recommendations = recommendations_map.get(predicted_class, ['No specific recommendations available.'])

        if confidence >= 0.8:
            severity = 'High'
        elif confidence >= 0.5:
            severity = 'Moderate'
        else:
            severity = 'Low'

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': predictions.tolist(),
            'recommendations': recommendations,
            'severity': severity
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
