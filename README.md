# ml-sims-naive-bayes-spam-detection

A machine learning-based SMS/Email spam detection application using Naive Bayes classifier. This repository includes training scripts, a Flask web application, and Docker containerization for easy deployment.

## Overview

This project implements a spam filter using:
- **Machine Learning**: Multinomial Naive Bayes classifier
- **Feature Extraction**: TF-IDF and Count Vectorization
- **Web Framework**: Flask for the prediction interface
- **Containerization**: Docker/Podman for reproducible deployment
- **Frontend**: HTML with Tailwind CSS for user-friendly UI

## Project Structure

```
.
├── spam.csv                           # Training dataset (from Kaggle)
├── spam_filter_app.py                # Training script to build the model
├── app.py                            # Flask web application
├── requirements.txt                  # Python dependencies
├── Containerfile                     # Container image definition
├── README.md                         # This file
├── templates/
│   └── index.html                   # Web UI template
├── nb_spam_classifier_model.pkl     # Trained model (generated after training)
└── vectorizer.pkl                   # TF-IDF vectorizer (generated after training)
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Docker or Podman (for containerization)
- Git (for version control)
- Kaggle account (to download training data)

## Step 1: Prepare Your Training Data

### 1.1 Download Data from Kaggle

1. Create a Kaggle account at https://www.kaggle.com if you don't have one
2. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

3. Configure Kaggle credentials:
   - Go to your Kaggle account settings → API section
   - Click "Create New API Token" which downloads `kaggle.json`
   - Place `kaggle.json` in `~/.kaggle/` directory:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. Download a spam dataset (example: SMS Spam Collection):
   ```bash
   kaggle datasets download -d uciml/sms-spam-collection-dataset
   unzip sms-spam-collection-dataset.zip
   ```

5. This will create a `spam.csv` file in your working directory

### 1.2 Verify Dataset Format

The `spam.csv` file should have at least these columns:
- **Category**: Label indicating if the message is 'spam' or 'ham' (not spam)
- **Message**: The actual text content to classify

Example format:
```
Category,Message
ham,Go until jurong point crazy.. Available only in bugis n great world la e buffet.. Interesting!!
spam,Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121. Standard txt rates apply...
```

## Step 2: Update Training Script with Your Data

### 2.1 Modify `spam_filter_app.py`

If your dataset has different column names or is located in a different path, update the script:

```python
# Line 10-13 in spam_filter_app.py
# Change this:
df = pd.read_csv('spam.csv')
X = df['Message']
y = df['Category']

# To match your dataset:
# If your columns are named differently:
df = pd.read_csv('path/to/your/spam.csv')
X = df['your_message_column_name']      # e.g., df['text'] or df['content']
y = df['your_category_column_name']     # e.g., df['label'] or df['classification']

# Or if your dataset is in a different format (TSV, Excel, etc.):
df = pd.read_csv('path/to/your/data.tsv', sep='\t')  # For tab-separated values
# df = pd.read_excel('path/to/your/data.xlsx')       # For Excel files
```

### 2.2 Example: Using Different Column Names

```python
# If your Kaggle dataset uses different column names
df = pd.read_csv('spam.csv', encoding='latin-1')  # Some datasets need encoding specification
# Rename columns to match expected format
df = df.rename(columns={'v1': 'Category', 'v2': 'Message'})
X = df['Message']
y = df['Category']
```

## Step 3: Train Your Personalized Model

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Run the Training Script

```bash
python spam_filter_app.py
```

This script will:
1. Load and split your training data (80% train, 20% test)
2. Extract features using CountVectorizer
3. Train a Multinomial Naive Bayes classifier
4. Evaluate model accuracy on test data
5. Save two files:
   - `nb_spam_classifier_model.pkl` - Trained model
   - `vectorizer.pkl` - Feature vectorizer

Example output:
```
Model Accuracy: 0.98
```

## Step 4: Test Locally

### 4.1 Run Flask Application Locally

```bash
python app.py
```

This starts the Flask development server at `http://localhost:5000`

### 4.2 Access Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Enter a message in the text area
3. Click "Predict" to see if it's classified as Spam or Ham
4. The model will display the classification and confidence probability

## Step 5: Create Your Container

### 5.1 Build Docker/Podman Image

Using Docker:
```bash
docker build -t spam-filter-flask:latest .
```

Using Podman:
```bash
podman build -t spam-filter-flask:latest .
```

This reads the `Containerfile` and creates an image with:
- Python environment
- All dependencies from `requirements.txt`
- Your trained model files
- Flask application code

### 5.2 Run Container Locally

Using Docker:
```bash
docker run -d -p 8080:5000 --name spam_predictor spam-filter-flask:latest
```

Using Podman:
```bash
podman run -d -p 8080:5000 --name spam_predictor spam-filter-flask:latest
```

Then access the application at `http://localhost:8080`

### 5.3 Stop and Remove Container

```bash
# Using Docker
docker stop spam_predictor
docker rm spam_predictor

# Using Podman
podman stop spam_predictor
podman rm spam_predictor
```

## Step 6: Upload to GitHub

### 6.1 Initialize Git Repository (if not already done)

```bash
cd /path/to/ml-sims-naive-bayes-spam-detection
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 6.2 Create `.gitignore`

```bash
# Create .gitignore to exclude large files and unnecessary items
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data (optional: exclude or include based on your preference)
# spam.csv

# Model files are committed for reproducibility
# nb_spam_classifier_model.pkl
# vectorizer.pkl
EOF
```

### 6.3 Add Files to Git

```bash
git add .
git status  # Review what will be committed
```

### 6.4 Create Initial Commit

```bash
git commit -m "Initial commit: SMS spam detection with Naive Bayes classifier"
```

### 6.5 Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository named `ml-sims-naive-bayes-spam-detection`
3. Choose "Private" or "Public" based on your preference
4. Do NOT initialize with README, .gitignore, or license (you already have them locally)
5. Click "Create repository"

### 6.6 Connect Local Repository to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/ml-sims-naive-bayes-spam-detection.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### 6.7 Verify Upload

1. Go to your GitHub repository URL
2. Confirm all files are visible
3. Check that the model files are included (`.pkl` files)

## Step 7: Push Container to Docker/GitHub Registry (Optional)

### 7.1 Push to Docker Hub

```bash
# Log in to Docker Hub
docker login

# Tag your image with Docker Hub username
docker tag spam-filter-flask:latest YOUR_DOCKER_USERNAME/spam-filter-flask:latest

# Push to Docker Hub
docker push YOUR_DOCKER_USERNAME/spam-filter-flask:latest
```

### 7.2 Push to GitHub Container Registry

```bash
# Log in to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Tag your image
docker tag spam-filter-flask:latest ghcr.io/YOUR_USERNAME/ml-sims-naive-bayes-spam-detection:latest

# Push to GitHub Container Registry
docker push ghcr.io/YOUR_USERNAME/ml-sims-naive-bayes-spam-detection:latest
```

## Key Configuration Files Explained

### `requirements.txt`
Lists all Python dependencies needed for training and running the application.

### `Containerfile` / `Dockerfile`
Defines how to build the container image, including:
- Base Python image
- Dependencies installation
- File copying
- Exposed ports
- Startup command

### `spam_filter_app.py`
Training script that:
- Loads your data
- Trains the Naive Bayes model
- Saves model artifacts

### `app.py`
Flask web application that:
- Loads the pre-trained model
- Provides web interface routes
- Handles predictions

## Customization Tips

### Update Model Parameters

In `spam_filter_app.py`, you can customize:

```python
# Change train/test split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use TF-IDF instead of Count Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Adjust Naive Bayes parameters
model = MultinomialNB(alpha=0.5)  # Smoothing parameter
```

### Update Web Interface

Edit `templates/index.html` to:
- Change colors and styling
- Add new input fields
- Display additional metrics

### Update Flask Routes

Edit `app.py` to:
- Add new prediction routes
- Implement batch predictions
- Add model retraining endpoints

## Troubleshooting

### Model Not Found Error
- Ensure `spam_filter_app.py` was executed successfully
- Check that `.pkl` files exist in the project root directory

### Column Name Error
- Verify your CSV column names match the script
- Use `df.head()` to inspect your data: `pandas.read_csv('spam.csv').head()`

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
kill -9 <PID>
```

### Container Build Issues
```bash
# Check Containerfile for syntax errors
docker build --no-cache -t spam-filter-flask:latest .
```

## Performance Metrics

After training, the model typically achieves:
- **Accuracy**: 95-99% (depending on dataset)
- **Inference Time**: <10ms per message
- **Model Size**: ~500KB

## License

This project is open source and available under the MIT License.

## Contributing

To contribute improvements:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add improvement"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Support

For issues or questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and environment details