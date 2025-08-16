🩻 Bone Fracture Detection using Deep Learning
<img src="xray_samples.gif" width="500"/>
📌 Overview

هذا المشروع يهدف إلى بناء نموذج ذكاء اصطناعي باستخدام Convolutional Neural Networks (CNNs) لتصنيف صور الأشعة السينية إلى:

✅ سليم (Normal)

❌ مصاب بكسر (Fracture)

🎯 الهدف الأساسي هو مساعدة الأطباء في التشخيص بشكل أسرع وأكثر دقة.

📂 Dataset

تم استخدام صور للأشعة السينية مقسمة إلى:

Train

Validation

Test

كل صورة مصنفة إما Fracture أو Normal.

🛠️ Tech Stack

Python 🐍

TensorFlow / Keras 🤖

NumPy & Pandas 📊

Matplotlib & Seaborn 📈

Scikit-learn

🚀 Model Architecture

الموديل مبني باستخدام CNN كالتالي:

Conv2D + MaxPooling

BatchNormalization

Dropout (لتقليل الـ Overfitting)

Dense Layers

Output Layer (Sigmoid للتصنيف الثنائي)

📊 Results

Training Accuracy: ~XX%

Validation Accuracy: ~XX%

Test Accuracy: ~XX%

📈 تم استخدام:

Confusion Matrix

Classification Report

ROC Curve + AUC

🖼️ Sample Predictions
Input Image	Prediction	Result
X-ray #1	Fracture	❌
X-ray #2	Normal	✅
📦 Installation
git clone https://github.com/your-username/bone-fracture-detection.git
cd bone-fracture-detection
pip install -r requirements.txt

▶️ Usage
python train.py
python evaluate.py


لو عايز تعمل اختبار على صورة معينة:

python predict.py --image test_sample.jpg

🔮 Future Improvements

استخدام Transfer Learning (ResNet, EfficientNet).

تحسين Augmentation للصور.

بناء واجهة باستخدام Streamlit لتجربة النموذج بشكل مباشر.

🙌 Acknowledgements

Dataset from [link/source].

Inspired by Medical AI research projects
