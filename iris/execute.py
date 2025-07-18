import logging
import json
import os
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from pathlib import Path

# Get the directory where the script is located and create output directory
# script_dir = Path(__file__).parent.absolute()
# output_dir = script_dir / "outputs"
# try:
#     os.makedirs(output_dir, exist_ok=True, mode=0o755)
#     # Debug logging
#     logging.info(f"Script directory: {script_dir}")
#     logging.info(f"Output directory: {output_dir}")
#     logging.info(f"Output directory exists: {output_dir.exists()}")
#     if output_dir.exists():
#         logging.info(f"Output directory permissions: {oct(output_dir.stat().st_mode)[-3:]}")
# except Exception as e:
#     logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
#     raise

# Configure logging to write to file
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
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
    logging.info("\n" + report)


def classify_instances(model, input_data, target_names):
    logging.info(f"Classifying instances from input data...\n")
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

    logging.info(f"\nClassifying {len(instances)} instances...\n")
    predictions = model.predict(instances)
    for i, prediction in enumerate(predictions):
        logging.info(
            f"{i + 1} - Instance {instances[i]} -> Predicted class: {target_names[prediction]}"
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
#     output_file = output_dir / "results.json"
#     try:
#         with open(output_file, "w") as f:
#             json.dump({"status": "success", "results": results}, f, indent=2)
#         # Set file permissions explicitly
#         os.chmod(output_file, 0o644)
#         logging.info(f"Predictions saved to {output_file}")
#     except Exception as e:
#         logging.error(f"Failed to save predictions to {output_file}: {str(e)}")
#         raise

if __name__ == "__main__":
    # Test save_output_to_file function
    # save_output_to_file([], [], [])  # Just calls the function for test
    logging.info("Starting Iris Classifier execution...")

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

    # Step 5: Classify new instances from input
    env_var_from_coe = os.getenv("test_env")
    env_var_from_coe_2 = os.getenv("another_one")
    logging.info(
        f"Testing env_var_from_coe:  {env_var_from_coe}"
    )
    logging.info(
        f"Testing env_var_from_coe os.environ.get('test_env'):  {os.environ.get('test_env')}"
    )
    logging.info(
        f"Testing env_var_from_coe os.environ['test_env']:  {os.environ['test_env']}"
    )
    logging.info(
        f"Testing env_var_from_coe_2:  {env_var_from_coe_2}"
    )
    # Get input directory from environment variable
    input_dir = os.getenv("KIT_INPUTS_FILE")
    input_data = None
    
    if not input_dir:
        logging.error("KIT_INPUTS_FILE environment variable is not set")
        sys.exit(1)
        
    logging.info(f"Using input directory from KIT_INPUTS_FILE: {input_dir}")
    
    if os.path.exists(input_dir) and os.path.isdir(input_dir):
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
    else:
        logging.error(f"Input directory {input_dir} does not exist or is not a directory")


    if not input_data:
        logging.error(
            "No input data provided. Provide a compatible input file."
        )
        sys.exit(1)

    classify_instances(model, input_data, target_names)
