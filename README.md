# Intelligent-Object-Recognition-and-Query-System-for-Warehouse-Robotics

Here included-
Setup instructions
Dependencies (requirements.txt)
How to run each component
Challenges faced and how solved

Part 1: Computer Vision Module (OpenCV)
Objective: Build an object detection and tracking system

## Project Setup and Execution Instructions

### Dependencies
This project primarily relies on standard Python libraries for image processing and visualization. The key dependencies are:

*   `opencv-python`: For image loading, preprocessing (grayscale, Gaussian blur), Canny edge detection, contour finding, and drawing annotations.
*   `matplotlib`: For displaying images and plots at various stages of the pipeline.
*   `numpy`: Used implicitly by OpenCV for image array manipulation.
*   `pandas-gbq`: This was used for loading data from Google BigQuery in other previous tasks.

To ensure all necessary libraries are installed, you would typically use `pip`:

```bash
pip install opencv-python matplotlib numpy pandas-gbq
```



### How to Run Each Component

The notebook is structured to be executed sequentially, demonstrating a step-by-step computer vision pipeline.

1.  **Mount Google Drive and List Contents**: Execute the first code cell to mount your Google Drive and list the contents of the specified `Dataset/test` directory. This is necessary to access the image files.
2.  **Select and Load Image**: The subsequent cells will filter for image files, select one, and load it into memory using OpenCV. An initial display confirms successful loading.
3.  **Preprocess Image**: Run the cells dedicated to preprocessing. This involves converting the image to grayscale and applying a Gaussian blur to reduce noise. Intermediate images (grayscale, blurred) are displayed for verification.
4.  **Perform Canny Edge Detection**: Execute the cell to apply the Canny algorithm to the blurred image. The resulting edge map is displayed.
5.  **Find Contours**: The next set of cells finds contours based on the Canny edge map and draws them on a copy of the original image, displaying the result.
6.  **Analyze Contours**: This section iterates through the found contours, filters them by area, approximates their shapes, calculates bounding boxes and centroids, and then annotates a fresh copy of the original image with this information. The final annotated image is displayed.
7.  **Display Results**: The final display cell consolidates all key intermediate and final images into a single figure, providing a visual summary of the entire pipeline.
8.  **Explanation**: Review the markdown cell providing a written explanation of the computer vision approach.

Simply run each code cell in order, from top to bottom, to see the entire process in action.

### Challenges Faced and Solutions

1.  **Noise and Spurious Edge Detection**: Raw images often contain noise that can lead to false edges or fragmented true edges. This was addressed by:
    *   **Grayscale Conversion**: Simplifying the image data from three color channels to one intensity channel reduces complexity and noise.
    *   **Gaussian Blurring**: Applying a Gaussian filter before edge detection smooths out high-frequency noise, making the Canny algorithm more robust and less prone to detecting insignificant edges.

2.  **Identifying Relevant Objects from Contours**: The `cv2.findContours()` function can detect many contours, including small noise artifacts or the image border itself, which are not relevant objects. This was solved by:
    *   **Contour Area Filtering**: By calculating `cv2.contourArea()` for each contour and setting `min_contour_area` and `max_contour_area` thresholds, very small (noise) and excessively large (background/border) contours were effectively filtered out.

3.  **Complex and Irregular Contour Shapes**: Contours can be complex, making it difficult to define clear bounding boxes or analyze geometric properties directly. This was addressed by:
    *   **Polygon Approximation (`cv2.approxPolyDP`)**: This function simplifies the contour by reducing the number of vertices while maintaining its general shape. This makes it easier to compute accurate bounding rectangles and reduces computational overhead.

4.  **Parameter Tuning for Edge Detection and Filtering**: Optimal performance often depends on selecting appropriate parameters (e.g., Canny thresholds, Gaussian blur kernel size, contour area ranges). This was managed by:
    *   **Iterative Adjustment**: The initial `threshold1=50`, `threshold2=150` for Canny and `(5, 5)` kernel for Gaussian blur, along with `min_contour_area=100` and `max_contour_area=50000`, were chosen as reasonable starting points. These values can be fine-tuned based on the specific characteristics of different images to achieve better object detection results.

      
  
Part 2: Machine Learning Model (ML)
Objective: Train a classifier to categorize objects

## Final Task

### Comprehensive Project Report

This report details the process of building and training a CNN model using transfer learning for image classification on a modified CIFAR-10 dataset, tailored for warehousing-relevant categories. It covers data preparation, model development, evaluation, inference demonstration, performance summary, and a discussion of limitations and future improvements.

---

### 1. Project Setup and Dependencies

#### Setup Instructions:

This project was developed and executed in a Google Colab environment. The primary steps to set up and run the project are as follows:

1.  **Open the Colab Notebook**: Load this notebook into a Google Colab environment.
2.  **Mount Google Drive (Optional)**: If you need to save/load files from Google Drive, ensure it's mounted (though not strictly necessary for this self-contained notebook).
3.  **Run Cells Sequentially**: Execute each code cell in the notebook in order, from top to bottom. The notebook is structured with markdown explanations preceding code blocks, guiding through each stage of the process.

#### Dependencies (Python Libraries):

The following Python libraries were used in this project. They can typically be installed using `pip` if running outside of Colab (Colab often has these pre-installed).

```text
# requirements.txt
numpy
pickle
sklearn
tensorflow
matplotlib
seaborn
kagglehub
```

---

### 2. How to Run Each Component (Notebook Flow)

1.  **Data Acquisition**: The CIFAR-10 dataset is downloaded using `kagglehub.dataset_download("pankrzysiu/cifar10-python")` at the beginning of the notebook.
2.  **Load and Prepare CIFAR-10 Data**: (Cells `aKhUB7gcCVKY` to `f6de167f`)
    *   Imports necessary libraries (`numpy`, `pickle`, `os`, `sklearn.model_selection`, `tensorflow.keras.preprocessing.image`).
    *   Defines a function `load_cifar_batch` to load CIFAR-10 batch files.
    *   Loads raw CIFAR-10 training and test data.
    *   Reshapes image data (from flat array to 32x32x3) and normalizes pixel values (0-255 to 0.0-1.0).
    *   Splits training data into training and validation sets (`train_test_split`).
    *   Sets up `ImageDataGenerator` for data augmentation (rotation, shifts, flips, brightness) on the training set.
3.  **Define and Map Warehousing Categories**: (Cells `c5023031` to `eea49367`)
    *   Defines original CIFAR-10 class names.
    *   Defines new warehousing-relevant categories (`Vehicles`, `Small/Fragile Animals`, `Large/Farm Animals`).
    *   Creates a mapping from CIFAR-10 class indices to warehousing category indices.
    *   Applies this mapping to `y_train`, `y_val`, `y_test` to create new label arrays (`y_train_new`, `y_val_new`, `y_test_new`).
    *   One-hot encodes the new labels (`y_train_one_hot`, `y_val_one_hot`, `y_test_one_hot`).
4.  **Build and Train CNN with Transfer Learning**: (Cells `a78c62d8` to `21f2779a`)
    *   Imports TensorFlow and Keras components (`MobileNetV2`, `Dense`, `Flatten`, `Dropout`, `Model`, `Adam`).
    *   **Image Resizing for MobileNetV2**: Resizes `X_train`, `X_val`, `X_test` from 32x32 to 96x96 to be compatible with MobileNetV2's minimum input size requirements.
    *   Re-creates data generators with the resized images.
    *   Loads pre-trained `MobileNetV2` (without top classifier, ImageNet weights, `input_shape=(96, 96, 3)`).
    *   Freezes the `base_model` layers.
    *   Adds a custom classification head (Flatten, Dense, Dropout, Dense with softmax).
    *   Compiles the model (`Adam` optimizer, `categorical_crossentropy` loss, `accuracy` metric).
    *   Trains only the classification head for 10 epochs.
    *   Unfreezes some `base_model` layers (all but the first 100) for fine-tuning.
    *   Re-compiles the model with a lower learning rate (`Adam(learning_rate=1e-5)`).
    *   Fine- tunes the model for an additional 5 epochs.
5.  **Evaluate Model on Test Set**: (Cell `63110022`)
    *   Evaluates the model's performance on the `X_test_resized` and `y_test_one_hot` using `model.evaluate()`.
    *   Prints test loss and accuracy.
6.  **Generate Classification Report**: (Cells `5da4da6a` to `53f95eb7`)
    *   Makes predictions on the test data (`X_test_resized`).
    *   Converts predictions and true labels to class labels.
    *   Generates and prints a classification report using `sklearn.metrics.classification_report`, specifying `target_names` and `zero_division=0`.
7.  **Plot Confusion Matrix**: (Cells `9296342e` to `b46b90d4`)
    *   Generates a confusion matrix using `sklearn.metrics.confusion_matrix`.
    *   Visualizes the confusion matrix as a heatmap using `seaborn.heatmap`, with `warehousing_categories` as labels.
8.  **Demonstrate Inference**: (Cells `52708fc9` to `3b195467`)
    *   Randomly selects 5 sample images from `X_test_resized` and their true labels.
    *   Displays these sample images with their true labels.
    *   Makes predictions on these sample images using the trained model.
    *   Displays the sample images again, this time showing both true and predicted labels.

---

### 3. Challenges Faced and Solutions

During the development of this project, several challenges were encountered and addressed:

1.  **MobileNetV2 Input Shape Incompatibility**: Initially, `MobileNetV2` (pre-trained on ImageNet) expected input images larger than CIFAR-10's native 32x32 resolution, leading to a `UserWarning`. If not addressed, this could cause incorrect feature extraction or model instability.
    *   **Solution**: All images (training, validation, and test sets) were resized from 32x32 to 96x96 pixels using `tf.image.resize` before being fed into the MobileNetV2 base model. This ensured compatibility with the pre-trained model's architecture.

2.  **`Index out of range` Error during Sample Image Selection**: When attempting to select sample images using `random_indices` directly on `X_test_resized` (which was a TensorFlow tensor), an `InvalidArgumentError` occurred because TensorFlow tensors do not support direct advanced indexing with a list of integers in the same way as NumPy arrays.
    *   **Solution**: Before indexing, `X_test_resized` was explicitly converted to a NumPy array using `.numpy()`: `sample_images = X_test_resized.numpy()[random_indices]`. This allowed for correct selection of sample images.

3.  **`UndefinedMetricWarning` in Classification Report**: The initial attempt to generate a classification report resulted in an `UndefinedMetricWarning`. This indicated that for some classes (specifically 'Large/Farm Animals'), there were no true samples or no predicted samples, making metrics like precision or recall undefined.
    *   **Solution**: The `zero_division=0` parameter was added to the `classification_report` function call. This explicitly tells `sklearn` to set undefined metrics to 0.0, suppressing the warning and providing clear values in the report.

4.  **Severe Class Imbalance and Poor Performance on 'Large/Farm Animals'**: As highlighted in the performance summary, the model completely failed to classify the 'Large/Farm Animals' category (0.00 precision, recall, f1-score). This suggests a strong class imbalance in the custom warehousing categories, where this category might have fewer distinctive features or fewer examples than others after mapping from CIFAR-10 classes.
    *   **Solution (Proposed for Future Improvement)**: This issue was identified and discussed as a key limitation. Potential solutions include implementing weighted loss functions during training, resampling techniques (oversampling minority classes or undersampling majority classes), exploring more robust data augmentation strategies, or investigating alternative pre-trained models or custom architectures better suited for the dataset's characteristics and class distribution.

---

### 4. Model Performance Summary

*   **Overall Accuracy**: The model achieved a test accuracy of approximately **47%** and a test loss of **1.1350** after fine-tuning. This indicates relatively low overall performance.
*   **Category-Specific Performance**: 
    *   **'Vehicles'**: Showed the highest recall (**0.74**), meaning it correctly identified a large portion of actual vehicles. However, its precision was lower (**0.45**), suggesting frequent misclassifications of other objects as vehicles.
    *   **'Small/Fragile Animals'**: Demonstrated moderate performance with a precision of **0.51** and a recall of **0.43**.
    *   **'Large/Farm Animals'**: A critical weakness, with **0.00** precision, recall, and f1-score. The model *never* predicted an image to belong to this class.
*   **Confusion Matrix Insights**: The confusion matrix visually confirmed that the 'Large/Farm Animals' category was never predicted. True 'Large/Farm Animals' were misclassified, primarily as 'Vehicles' (1343 out of 2000) or 'Small/Fragile Animals' (657 out of 2000). Significant confusion also existed between 'Vehicles' and 'Small/Fragile Animals'.
*   **Inference Demonstration**: Sample images demonstrated both correct and incorrect predictions, clearly illustrating the model's struggle with the 'Large/Farm Animals' category.

---

### 5. Model Limitations and Areas for Improvement

*   **Severe Class Imbalance**: The complete failure to identify 'Large/Farm Animals' is the most significant limitation, likely due to an imbalanced dataset where this category is underrepresented or lacks distinct features after mapping from CIFAR-10. This would lead to major logistical errors in a warehousing scenario.
*   **Data Augmentation and Image Size Challenges**: Upscaling low-resolution (32x32) CIFAR-10 images to 96x96 for MobileNetV2 compatibility does not add detail, making feature extraction challenging for a model designed for higher-resolution inputs. The applied data augmentation might not be sufficient to overcome this.
*   **Domain Mismatch in Transfer Learning**: MobileNetV2, pre-trained on diverse ImageNet images, may not effectively transfer its learned features to the simpler, smaller, and less detailed CIFAR-10 dataset, especially for fine-grained classification within the new categories.
*   **Real-World Implications**: The model's limitations suggest it is not robust enough for critical decision-making in a dynamic warehousing environment, potentially leading to increased manual intervention and operational costs.

#### Areas for Improvement:

1.  **Address Class Imbalance**: Implement weighted loss functions or resampling techniques (e.g., oversampling minority classes, synthetic data generation using GANs).
2.  **Experiment with Different Architectures**: Explore other pre-trained CNNs (e.g., EfficientNet variants) or develop custom CNNs more suitable for small, low-resolution images.
3.  **Advanced Data Augmentation**: Utilize sophisticated augmentation methods like Mixup, CutMix, AutoAugment, or RandAugment to increase data diversity and model robustness.
4.  **Refine Transfer Learning Strategy**: Revisit which layers of the base model are frozen/unfrozen during fine-tuning, potentially adjusting learning rates with advanced schedules, or exploring progressive resizing during training.

---

This concludes the comprehensive report on the CNN model for warehousing-relevant image classification.



Part 3: RAG System
Objective: Build a retrieval system for robotics documentation

## Final Task: Project Summary

### Project Description
This project implements a Retrieval-Augmented Generation (RAG) system for robotics documentation using a hypothetical robot, the 'OmniBot 7000'. The system generates synthetic documentation, processes it into a searchable knowledge base, and then uses an LLM to answer user queries based on the retrieved information.

### Setup Instructions

1.  **Clone this Notebook**: Save a copy of this notebook to your Google Drive.
2.  **Google API Key**: Obtain a Google API Key from [Google AI Studio](https://makersuite.google.com/key). Add this key to Google Colab secrets. Click on the 'key' icon in the left sidebar (Secrets panel), click '+ New secret', enter `GOOGLE_API_KEY` as the name and paste your API key as the value. Ensure 'Notebook access' is checked.

### Dependencies (`requirements.txt`)
The following Python libraries are required:

```
sentence-transformers
chromadb
langchain-text-splitters
tenacity
google-generativeai
```

These can be installed in your Colab environment using `pip install -r requirements.txt` after saving the above content to a file named `requirements.txt`, or by running `!pip install <package-name>` for each package.

### How to Run Each Component

1.  **Generate Synthetic Robotics Documentation**: Run the code cells under the 'Generate Synthetic Robotics Documentation' section (`087654cc`, `18f1367a`). This will create `.txt` files in your Colab environment that serve as the knowledge base.

2.  **Initialize RAG Components**: Run the code cells under the 'Initialize RAG Components' section (`d8003a59`, `aa7d3422`). This installs necessary libraries (`sentence-transformers`, `chromadb`) and initializes the `SentenceTransformer` embedding model.

3.  **Chunk and Embed Documents**: Execute the code cells under the 'Chunk and Embed Documents' section (`1bc584bd`, `4c8bccb6`). This installs `langchain-text-splitters`, chunks the synthetic documents, generates embeddings, and stores them in the ChromaDB vector store.

4.  **Implement Retrieval and Response Generation**: Run the code cells under the 'Implement Retrieval and Response Generation' section (`1711a715`, `3f525792`, `cf641b96`, `b37a626b`, `6649708a`, `75445c7f`, `13478abf`, `fe5d5f5d`, `80a15f0f`, `66073cc5`). This defines the `rag_query` function, which is the core RAG logic for retrieving relevant document chunks and generating LLM responses. It also defines the `evaluate_llm_response` and `evaluate_rag_system` functions.

5.  **Demonstrate RAG System with Example Queries**: Re-run the demonstration code cell (`776305c2`). This executes several example queries through the `rag_query` function, printing the generated responses and their sources. This part will now also trigger the LLM-based evaluation.

6.  **Analyze and Present Performance Matrix**: The `evaluation_results` generated from step 5 can be further analyzed. This project focused on setting up the evaluation framework, but actual quantitative analysis (e.g., calculating average factual accuracy) would be the next logical step.

### Challenges Faced and Solutions

*   **Missing API Key**: Initially, the `GOOGLE_API_KEY` was not configured, leading to errors in LLM calls. This was resolved by instructing the user to add the API key to Colab secrets and re-configuring `google.generativeai` with `userdata.get('GOOGLE_API_KEY')`.
*   **'404 Model Not Found' Errors**: The LLM calls consistently failed with a '404 Model Not Found' error for different model names (`gemini-pro`, `gemini-1.0-pro`, `text-bison-001`). This was addressed by dynamically listing available models (`genai.list_models()`) and using a supported model (e.g., `gemini-pro-latest`) that supports `generateContent`.
*   **'Quota Exceeded' Errors**: Even with the correct model, frequent API calls quickly hit the free-tier quota limits. This was mitigated by implementing a retry mechanism with exponential backoff (`tenacity.retry` decorator for `ResourceExhausted` exceptions) in both `rag_query` and `evaluate_llm_response` functions, making the system more robust to transient rate limits.
*   **Scope Issues with Functions and Variables**: During iterative development, `NameError` exceptions occurred when functions (`rag_query`, `evaluate_rag_system`) or variables (`embedding_model`, `collection`, `evaluation_dataset`) were not defined in the current execution scope. This was resolved by ensuring all necessary components and functions were either re-initialized or re-defined within the same code block before their usage.
*   **Markdown Cell Syntax Error**: An attempt to generate a markdown cell with a Python `cell_type` resulted in a `SyntaxError`. This was corrected by ensuring that markdown content is generated with `cell_type: markdown`.

Part 4: Integration
Objective: Connect the three components
### Project Dependencies

Upon reviewing the provided notebook cells, it's clear that the core RAG system implementation relies exclusively on Python's built-in functionalities and standard data structures (strings, lists, and dictionaries).

**Conclusion:**

No external Python libraries are strictly required for the core logic of this project. All functionalities, including object detection simulation, classification, instruction retrieval, and RAG response generation, are implemented using native Python features.

**For a `requirements.txt` file (if desired for environment setup):**

Given the absence of external dependencies, a `requirements.txt` file is not strictly necessary for the project's execution. However, to specify the Python interpreter version, one could include:

```
python>=3.8  # Or the specific Python version used, e.g., python==3.10
```


#### Instructions
1.  **Data Preparation**: Run the code cell that defines `simulated_object_input` and `knowledge_base`. This sets up the input data and the knowledge base for instruction retrieval.
2.  **Object Detection and Classification**: Execute the code cell that defines the `detect_and_classify_objects` function, `classification_keywords`, and then calls `detect_and_classify_objects` with `simulated_object_input`. This simulates the detection and classification of objects from the input.
3.  **Instruction Retrieval**: Run the code cell that defines the `retrieve_instructions` function and then calls it with the `detected_objects` from the previous step and the `knowledge_base`. This retrieves relevant handling instructions.
4.  **RAG Logic Implementation**: Execute the code cell that defines `user_question`, the `generate_rag_answer` function, and then calls it with the `handling_instructions` and `user_question`. This generates a coherent RAG response.
5.  **End-to-End Workflow Demonstration**: Run the code cell that combines all the previous steps (detection, classification, retrieval, and RAG response generation) into a single sequence, using the already defined variables and functions. This demonstrates the full RAG pipeline.
6.  **Performance Evaluation - Classification**: Execute the code cell that defines `simulated_object_input_test` and `expected_classification_test`, then calls `detect_and_classify_objects` for this test input, and finally compares the `detected_objects_test` with `expected_classification_test` to evaluate classification correctness.
7.  **Performance Evaluation - RAG Response Relevance**: Run the code cell that defines `user_question_test`, derives `expected_rag_response_keywords`, generates `actual_rag_response_test` using the RAG logic, and checks its relevance based on the keywords from `detected_objects_test`'s instructions.

### Challenges and Solutions in the RAG System

#### 1. Object Classification Challenge
*   **Challenge:** The simulated object classification component exhibited a critical limitation by failing to detect all expected classifications. Specifically, for the test input "A package with a glass vase and sharp tools inside.", it correctly detected `['SHARP']` but missed `['FRAGILE']`, leading to an overall incorrect classification for that test case.
*   **Impact:** This classification inaccuracy directly limits the comprehensiveness and effectiveness of the subsequent RAG system, as the RAG can only retrieve instructions for objects it has been correctly informed about.
*   **Proposed Solutions/Insights:** To enhance classification accuracy, it is suggested to:
    *   Expand and refine the `classification_keywords` to include a broader range of synonyms and related terms (e.g., adding 'glass' or 'vase' to 'FRAGILE').
    *   Consider integrating more sophisticated object detection and NLP techniques beyond simple keyword matching, especially for ambiguous or nuanced inputs.

#### 2. RAG Response Relevance Performance
*   **Performance:** The RAG component itself demonstrated strong relevance. When provided with the *actually detected* classification (`['SHARP']` in the test case), it successfully retrieved and generated a coherent and relevant response using the corresponding handling instructions from the `knowledge_base`.
*   **Insight:** The RAG system is effective in assembling information and generating a response *given correct input from the classification stage*. Its performance is directly dependent on the accuracy of the upstream classification.

#### Overall Summary of Challenges and Solutions
The primary challenge identified in the simulated RAG system lies in the object classification stage. While the RAG retrieval and generation capabilities are robust, any inaccuracies in initial object detection and classification have a cascading negative effect on the final output. Future improvements should focus on bolstering the classification component, either through keyword refinement or advanced NLP models, to unlock the full potential of the RAG system in providing comprehensive handling instructions.
