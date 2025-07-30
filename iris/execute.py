import logging
import json
import os
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Use output directories created in entrypoint.sh
# output_dir = Path("/tmp/outputs")
# logs_dir = Path("/tmp/logs")

# Configure logging to write to file
# log_file = logs_dir / "iris_classifier.log"
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler(log_file)
    ]
)

def load_data():
    logging.info("Loading Iris dataset...\n")
    iris = load_iris()
    X, y = iris.data, iris.target
    logging.info(
        f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.\n"
    )
    return X, y, iris.target_names


def train_model(X_train, y_train):
    logging.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete.\n")
    return model


def evaluate_model(model, X_test, y_test, target_names):
    logging.info("Evaluating the trained model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    report = classification_report(y_test, y_pred, target_names=target_names)
    logging.info("Classification Report:")
    logging.info("\\n" + report)

def classify_instances(model, input_data, target_names):
    logging.info(f"Classifying instances from input data...\
")
    try:
        if isinstance(input_data, str):
            data = json.loads(input_data)  # Parse JSON string
        elif isinstance(input_data, dict):
            data = input_data  # Already a dictionary
        else:
            raise ValueError("Invalid input data format.")
    except Exception as e:
        logging.error(f"Failed to parse input data: {e}")
        sys.exit(1)

    instances = data.get("instances", [])
    if not instances:
        logging.warning("No instances found in the input data.")
        return

    logging.info(f"\
Classifying {len(instances)} instances...\
")
    predictions = model.predict(instances)
    results = []
    
    for i, prediction in enumerate(predictions):
        predicted_class = target_names[prediction]
    logging.info(
            f"{i + 1} - Instance {instances[i]} -> Predicted class: {predicted_class}"
    )
    # Save to file
    # save_output_to_file(predictions, instances, target_names)

# def save_output_to_file(predictions, instances, target_names):
#     results = []
#     for i, prediction in enumerate(predictions):
#         results.append({
#             "instance": instances[i],
#             "predicted_class": target_names[prediction]
#         })
    
#     # Save results to outputs directory
#     with open(f"{output_dir}/results.json", "w") as f:
#         json.dump({"status": "success", "results": results}, f, indent=2)
    
#     logging.info(f"Predictions saved to {output_dir}/results.json")


if __name__ == "__main__":
    logging.info("Starting Iris Classifier execution...")
    # logging.info(f"Using output directory: {output_dir}")
    # logging.info(f"Using logs directory: {logs_dir}")

    # Step 1: Load data
    X, y, target_names = load_data()

    # Step 2: Split into training and testing sets
    logging.info("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )
    logging.info(
        f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}"
    )

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test, target_names)

    # Step 5: Process environment variables
    env_var_from_coe = os.getenv("test_env")
    env_var_from_coe_2 = os.getenv("another_one")
    
    # Log environment variables
    logging.info(f"Testing env_var_from_coe: {env_var_from_coe}")
    if 'test_env' in os.environ:
        logging.info(f"Testing env_var_from_coe os.environ.get('test_env'): {os.environ.get('test_env')}")
        logging.info(f"Testing env_var_from_coe os.environ['test_env']: {os.environ['test_env']}")
    else:
        logging.info("Environment variable 'test_env' is not set")
    logging.info(f"Testing env_var_from_coe_2: {env_var_from_coe_2}")
    
    # Get input file from environment variable
    input_dir = os.getenv("KIT_INPUTS_FILE")
    input_data = None
    
    if not input_dir:
        logging.error("KIT_INPUTS_FILE environment variable is not set")
        sys.exit(1)

    logging.info(f"Using input directory from KIT_INPUTS_FILE: {input_dir}")
    
    if os.path.exists(input_dir):
        if os.path.isdir(input_dir):
            # List all files in the input directory for debugging
            try:
                files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
                logging.info(f"Found {len(files)} files in {input_dir}: {', '.join(files) if files else 'No files found'}")
            except Exception as e:
                logging.error(f"Error reading input directory {input_dir}: {str(e)}")
                sys.exit(1)
            
            # Look for JSON files in the directory
            json_files = [f for f in files if f.lower().endswith('.json')]
            
            if json_files:
                # Use the first JSON file found
                input_file = os.path.join(input_dir, json_files[0])
                logging.info(f"Reading input from file: {input_file}")
                try:
                    with open(input_file, "r") as f:
                        input_data = json.load(f)
                except Exception as e:
                    logging.error(f"Error reading input file {input_file}: {str(e)}")
            else:
                logging.warning(f"No JSON files found in {input_dir}")
        elif os.path.isfile(input_dir):  # KIT_INPUTS_FILE might be a direct file path
            logging.info(f"KIT_INPUTS_FILE points to a file: {input_dir}")
            try:
                with open(input_dir, "r") as f:
                    input_data = json.load(f)
            except Exception as e:
                logging.error(f"Error reading input file {input_dir}: {str(e)}")
    else:
        logging.error(f"Input path {input_dir} does not exist")

    if not input_data:
        logging.error("No input data provided. Using default test data.")
        # Provide some default test data for demonstration
        input_data = {
            "instances": [
                [5.1, 3.5, 1.4, 0.2],  # Example of setosa
                [6.3, 3.3, 4.7, 1.6],  # Example of versicolor
                [6.5, 3.0, 5.2, 2.0]   # Example of virginica
            ]
        }

    # Step 6: Classify instances
    classify_instances(model, input_data, target_names)

    logging.info("Iris classification execution completed successfully")


