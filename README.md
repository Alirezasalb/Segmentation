#  SAM Universal Segmenter


A **universal image segmentation tool** powered by Meta's **Segment Anything Model (SAM)**. Works on **any image** — medical scans, agricultural photos, everyday objects, and more!

- ✅ Point-based interactive segmentation
- ✅ Runs on CPU or GPU
- ✅ Streamlit web interface
- ✅ Docker support


---

##  Quick Start

### Option 1: Local Setup (Recommended for first run)

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/sam-universal-segmenter.git
   cd sam-universal-segmenter 
   ```
   
2. **Download SAM model** 

    - Go to SAM Model Checkpoints 
    - Download `sam_vit_b_01ec64.pth` `(370 MB)`
    - Place it in your project root:
```
sam-universal-segmenter/
├── sam_vit_b_01ec64.pth   ← HERE
├── app.py
└── ...
   
```
3. **Install dependencies**

`pip install -r requirements.txt`

4. **Run the app**

`streamlit run app.py`

5. **Open `http://localhost:8501` in your browser!**