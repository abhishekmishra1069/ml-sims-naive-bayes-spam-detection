# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all essential files: the app, the HTML template, and the model components
COPY app.py .
COPY templates templates/
COPY nb_spam_classifier_model.pkl .
COPY vectorizer.pkl .

# Expose the port the application runs on
EXPOSE 5000

# Command to run the application using a production-ready WSGI server
# gunicorn is generally preferred over flask run for production environments
# Note: Since gunicorn is not in requirements.txt, we will stick to the simple 'flask run'
CMD ["flask", "run", "--host", "0.0.0.0"]