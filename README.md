# Advancing Hematology with AI: My Journey in the CytologIA Data Challenge

In an era where artificial intelligence is transforming industries, the CytologIA Data Challenge, organized by Trustii.io in collaboration with the French Cellular Hematology Group (GFHC) and Algoscope, offered an incredible opportunity to contribute to medical AI research. This initiative is supported by the France 2030 plan and the Health Data Hub, underscoring its importance in advancing healthcare diagnostics. As a participant in this challenge, I embarked on a journey that was both educational and exciting, filled with challenges and breakthroughs.

### The Vision Behind the CytologIA Data Challenge

The CytologIA Data Challenge aimed to automate the classification and detection of leukocytes (white blood cells), a critical task in hematology. Blood smear analysis, a gold standard for identifying leukocyte abnormalities, is traditionally manual and time-consuming. This competition sought to address the limitations of existing automated systems by leveraging AI to improve diagnostic accuracy and efficiency. With approximately 69,000 images spanning 23 leukocyte classes, the dataset was a treasure trove for machine learning enthusiasts like me to explore, learn, and innovate.

## The Dataset

The CytologIA dataset is a meticulously curated collection of approximately 69,000 leukocyte images, classified into 23 categories. These images were sourced from multiple French centers, ensuring a diverse representation of samples. The dataset is divided into training and testing subsets with a 70/30 distribution.

### Dataset Components

- **Image Repository**: A collection of high-resolution images, each containing one or more leukocytes. The leukocytes in the images are annotated with bounding boxes to indicate their positions.
- **Annotations for Training Set (CSV File)**: A CSV file detailing the annotations required for model training. Each entry includes:
    - **Image Filename**: The name of the image file.
    - **Bounding Box Coordinates**: Defined by (x1, y1) for the top-left corner and (x2, y2) for the bottom-right corner.
    - **Leukocyte Class**: The corresponding class abbreviation for each leukocyte.
- **Testing Set Annotations (CSV File)**: The testing CSV file includes only the image filenames. Participants are required to generate and submit predictions for the test set, providing both bounding box coordinates and class labels in the specified format.

### Class Information

Each leukocyte class in the dataset is represented by a unique abbreviation. Here’s a snapshot of the class mapping:

| Abbreviation | Full Class Name |
| --- | --- |
| PNN | Polynucléaire Neutrophiles |
| LAM3 | Leucémie aigüe myéloïde 3 |
| Lysee | Cellules lysées |
| LyB | Lymphoblastes |
| MO | Monocytes |
| LLC | LLC |
| MBL | Myéloblastes |
| LGL | Lymphocyte à grains |
| EO | Polynucléaire Eosinophiles |
| Thromb | Plaquettes géantes |
| Er | Erythroblaste |
| B | Blastes indifférenciés |
| M | Myélocytes |
| LY | Lymphocytes |
| MM | Métamyélocytes |
| LF | Lymphome folliculaire |
| MoB | Monoblastes |
| PM | Promyélocytes |
| BA | Polynucléaire Basophiles |
| LH_LyAc | Lymphocytes hyperbasophiles / activés |
| LM | Lymphome du manteau |
| LZMG | Lymphome de la zone marginale ganglionnaire |
| SS | Cellules de Sézary |

## The Challenge Objectives

The primary objectives of the CytologIA Data Challenge were twofold:

1. **Leukocyte Detection**: Accurately predict the bounding box coordinates for each leukocyte present in the test set images.
2. **Leukocyte Classification**: Classify each detected leukocyte into one of the 23 predefined classes.

The performance of the submitted models was assessed based on two criteria:

- **Bounding Box Accuracy**: Evaluated using Generalized Intersection Over Union (GIOU), contributing 20% to the final score.
- **Classification Performance**: Assessed using the F1 Score, contributing 80% to the final score.

## My Approach: From Data Preparation to Model Training

### Data Preparation: The Backbone of Any Machine Learning Project

Before diving into model training, I spent a significant amount of time on data preparation. This phase is often overlooked, but it’s crucial for building robust models. Here’s a breakdown of the steps I took:

### 1. **Installing Required Libraries**

The first step was to set up the environment by installing the necessary libraries. I used `ultralytics`, a powerful framework based on PyTorch, which simplifies the process of training and deploying object detection models. Additionally, I installed `pandas`, `matplotlib`, and `tqdm` for data manipulation, visualization, and progress tracking, respectively.

```python
!pip install ultralytics pandas matplotlib tqdm

```

### 2. **Loading the Dataset**

The dataset was provided in a ZIP file containing images and CSV files with annotations. I used `gdown` to download the dataset from Google Drive and then extracted the images.

### 3. **Data Cleaning and Preprocessing**

The dataset required significant cleaning and preprocessing. I started by organizing the images folder, ensuring that all images were correctly placed and that non-image files were removed. I also repaired corrupted images using the `Pillow` library.

```python
def repair_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
        img.save(image_path)
        return True
    except (IOError, SyntaxError) as e:
        print(f"Skipped corrupted image: {image_path}")
        return False

```

Next, I cleaned the training CSV file, ensuring that all required columns were present and that there were no missing or invalid values. I also validated the bounding box coordinates to ensure they were within the image boundaries.

```python
# Check for invalid bounding box coordinates (x1 <= x2 and y1 <= y2)
invalid_bbox = df[(df['x1'] > df['x2']) | (df['y1'] > df['y2'])]
if not invalid_bbox.empty:
    print("Rows with invalid bounding box coordinates detected:")
    print(invalid_bbox)
    print("Removing rows with invalid bounding box coordinates...")
    df = df[(df['x1'] <= df['x2']) & (df['y1'] <= df['y2'])]
else:
    print("All bounding box coordinates are valid. Everything is good!")

```

### 4. **Class Distribution Analysis**

Understanding the distribution of classes is crucial for building a balanced model. I used `pandas` to analyze the class distribution and found that some classes were underrepresented. This insight helped me decide on strategies like data augmentation to address class imbalance.

```python
import pandas as pd

# Load the training CSV file
df = pd.read_csv("cytologia-data-train_1732098640162.csv")

# Analyze class distribution
class_distribution = df['class'].value_counts()
print(class_distribution)

```

### Model Training with Ultralytics

With the data prepared, I moved on to model training. I chose the YOLO (You Only Look Once) model from the Ultralytics framework due to its efficiency and accuracy in object detection tasks.

### 1. **Setting Up the YOLO Model**

I initialized the YOLO model and loaded the pre-trained weights to leverage transfer learning.

```python
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8l.pt')

```

### 2. **Training the Model**

I trained the model using the prepared dataset. The training process involved multiple epochs, and I used manual checkpointing to save the model at various stages, allowing me to resume training if needed.

```python
# Train the model
results = model.train(data='cytologia.yaml', epochs=50, imgsz=640)

```

### 3. **Evaluating the Model**

After training, I evaluated the model’s performance using the validation set. The evaluation metrics included the GIOU score for bounding box accuracy and the F1 score for classification performance.

```python
# Evaluate the model
metrics = model.val()
print(metrics)

```

### Lessons Learned

This competition was a profound learning experience. Here are some key takeaways:

1. **Data Preparation is Crucial**: I learned that data preparation is the backbone of any machine learning project. Cleaning, preprocessing, and understanding the data are as important as the model itself.
2. **Class Imbalance Matters**: Analyzing the class distribution helped me understand the importance of addressing class imbalance, which can significantly impact model performance.
3. **Frameworks Simplify the Process**: Using the Ultralytics framework made the model training process much more manageable. It abstracts away many complexities, allowing me to focus on the problem at hand.
4. **Patience and Persistence**: Machine learning projects require patience and persistence. Iterating through different models, tuning hyperparameters, and debugging issues are all part of the journey.

## Conclusion

The CytologIA Data Challenge was an enriching experience that allowed me to apply my machine learning skills to a real-world problem in healthcare. From data preparation to model training, every step taught me something new and reinforced the importance of a thorough and methodical approach.

I’m excited to see how the models developed in this competition will contribute to advancing hematological diagnostics. This challenge has not only enhanced my technical skills but also deepened my appreciation for the potential of AI in transforming healthcare.

If you’re interested in exploring the code and the detailed steps I took, feel free to check out my GitHub repository. You can also find the project on my Kaggle profile for further insights. Let’s continue to push the boundaries of what AI can achieve in the medical field!

---
