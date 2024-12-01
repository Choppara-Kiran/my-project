from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Constants
IMG_SIZE = (224, 224)  # Adjust this based on your desired input size
THRESHOLD = 0.5  # Threshold for risk classification

# Create a dummy Keras model
def create_dummy_model():
    """
    Creates a simple Keras model for demonstration purposes.
    Replace this with your trained model when ready.
    """
    model = Sequential([
        Flatten(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Flatten input image
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Single output for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize the dummy model
model = create_dummy_model()
print("Dummy model created and ready for predictions!")

@app.route('/')
def index():
    """
    Render the home page with an upload form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction logic.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    try:
        # Preprocess the image
        img = load_img(image, target_size=IMG_SIZE)  # Resize the image
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Dummy prediction (random output for testing)
        prediction = model.predict(img_array)
        confidence = round(float(prediction[0][0]) * 100, 2)
        risk_level = 'Low Risk' if prediction[0][0] < THRESHOLD else 'High Risk'

        # Return the results
        return jsonify({
            'risk_level': risk_level,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
