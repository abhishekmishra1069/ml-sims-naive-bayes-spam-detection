# Import Flask components: Flask creates the web app, jsonify converts Python objects to JSON,
# request handles incoming HTTP requests, render_template loads HTML files with dynamic content
from flask import Flask, jsonify, request, render_template
# Import joblib for loading saved machine learning models and vectorizers from disk
import joblib
# Import sys module for system-level operations like exiting the program
import sys

# Create a Flask application instance with the current module's name
app = Flask(__name__)

# Define the filename of the saved Naive Bayes model pickle file
MODLE_FILE = "nb_spam_classifier_model.pkl"

# Define the filename of the saved TF-IDF vectorizer pickle file
VECTOR_FILE = "vectorizer.pkl"



# Function to load the pre-trained model and vectorizer from disk
def load_model():
    # Declare that we're using global variables so we can assign to them
    global model
    global vectorizer
    try:
        # Load the Naive Bayes classifier model from the pickle file
        model = joblib.load(MODLE_FILE)
        # Load the TF-IDF vectorizer from the pickle file
        vectorizer = joblib.load(VECTOR_FILE)
    except FileNotFoundError as e:
        # Handle case where model or vectorizer files don't exist
        print(f"Error loading model or vectorizer: {e}")
        # Set model and vectorizer to None if files not found
        model = None
        vectorizer = None
        # Exit the application with error code 1
        sys.exit(1)
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # Set model and vectorizer to None if error occurs
        model = None
        vectorizer = None
        # Exit the application with error code 1
        sys.exit(1)

# Create an application context and load the model when the app starts
with app.app_context():
    load_model()
    


# Define a route for the home page that accepts GET requests
@app.route('/')
def home():
    # Render and return the index.html template for the main prediction interface
    return render_template('index.html')

# Define a route for predictions that only accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # This function handles spam prediction when users submit messages
    try:
        # Extract the 'message' parameter from the form data; default to empty string if not provided
        message = request.form.get('message', '')
        
        # Check if the message is empty
        if not message:
            # Return an error response if no message was provided
            return jsonify({
                'prediction': 'Error',
                'probability': 0.0,
                'status': 'Please enter a message.'
            })

        # Transform the input message into a numerical vector using the TF-IDF vectorizer
        message_count = vectorizer.transform([message])
        
        # Use the trained Naive Bayes model to predict if message is spam (1) or ham (0)
        # [0] gets the first (and only) prediction from the result array
        prediction_code = model.predict(message_count)[0]
        # Get the probability that the message is spam (class 1); [0][1] gets the probability of class 1
        prob_spam = model.predict_proba(message_count)[0][1]
        
        # Convert the prediction code to a human-readable label
        result_label = 'SPAM' if prediction_code == 1 else 'HAM (Not Spam)'
        
        # Build the response dictionary with prediction results
        response = {
            'prediction': result_label,
            'probability': f"{prob_spam*100:.2f}%",  # Convert probability to percentage with 2 decimal places
            'message': message,
            'status': 'Success'
        }
        
        # Render the index.html template with the prediction results
        return render_template('index.html', prediction_result=response)

    except Exception as e:
        # Catch any errors that occur during prediction and return an error response
        return render_template('index.html', prediction_result={
            'prediction': 'Internal Error',
            'probability': 0.0,
            'status': f'Prediction failed: {e}'
        })

# Check if this script is being run directly (not imported as a module)
if __name__ == '__main__':
    # Start the Flask development server with debug mode enabled for hot-reload
    # host='0.0.0.0' makes the server accessible from any IP address
    # port=5000 runs the server on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)