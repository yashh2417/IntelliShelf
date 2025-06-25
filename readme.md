# 🛍️ IntelliShelf – AI-Powered Smart Instrument Store 🧠🎸

### Entire Pipeline
![IntelliShelf Pipeline](https://github.com/yashh2417/faltu/blob/main/pipeline.png?raw=true) 

Welcome to **IntelliShelf**, a full-stack AI-powered demo platform that combines **Computer Vision**, **LLMs**, and **RAG** to deliver a futuristic musical instrument e-commerce experience. From **automated defect detection** to **LLM-generated descriptions** and **FAQ-based chatbot assistance**, everything is built with modularity and scalability in mind.

---

## 🔧 Key Features

| Capability                        | Powered By                               |
|----------------------------------|-------------------------------------------|
| 🖼️ Product Type Classification   | PyTorch (ResNet-based CNN)                |
| 🔍 Defect Detection              | YOLOv8 (Ultralytics)                      |
| 📝 Auto Description Generation   | Google Gemini API                         |
| 💬 Buyer Chatbot (Policies/FAQs) | LangChain + Chroma + Gemini (RAG)         |
| 📦 Image Upload + Storage        | FastAPI + PIL                             |
| 🛒 Return Eligibility Detection  | Vision + Business Rule Logic              |
| 🖥️ Web Interface                 | FastAPI + Jinja2 Templates + HTML/CSS     |
| ☁️ Container                     | Docker                                    |

---

## 🗂️ Project Structure

````

intellishelf/
├── store/
│   ├── main.py                # FastAPI app
│   ├── templates/             # HTML templates (Jinja2)
│   ├── static/                # CSS and static assets
│   └── ...
├── product\_db/
│   ├── product\_data.json      # Stores product catalog
│   └── uploaded\_images/       # Image storage
├── chromadb/                  # Vector DB for RAG
├── model/
│   ├── classify.pth           # PyTorch classifier
│   └── YOLOv8 weights         # YOLOv8 defect detector
└── runs/                      # YOLO training artifacts

````

---

## 🚀 Live Demo Walkthrough

### 1. Upload Product Image
![Upload Screenshot](https://github.com/yashh2417/faltu/blob/main/add-product.png?raw=true)

- Upload any musical instrument image.
- Fill in product **specifications**.
- System detects:
  - 🎵 Instrument type
  - 🧯 Whether it is **defective**

---

### 2. Description & Feature Generation

#### Clean Image
![Generated Description](https://github.com/yashh2417/faltu/blob/main/added-product.png?raw=true)

#### Defected Image
![Found Defect](https://github.com/yashh2417/faltu/blob/main/defected-product.png?raw=true)

- Based on the uploaded specs, **Gemini LLM** creates:
  - 📝 Human-readable product **description**
  - ✅ A structured list of **key features**

---

### 3. Product Catalog Page
![Product Grid](https://github.com/yashh2417/faltu/blob/main/product.png?raw=true)

- View uploaded products.
- Each item has:
  - Thumbnail
  - Class, Features, Description

---

### 4. Chatbot – Return & Policy Help
![Chatbot UI](https://github.com/yashh2417/faltu/blob/main/chatbot.png?raw=true)

- Powered by:
  - 🔍 Chroma (vector DB)
  - 🤖 Gemini + LangChain (RAG)
- Ask questions like:
  - “Can I return a damaged tabla?”
  - “How long is the warranty?”

---

### 5. Return Eligibility Checker

#### Return Eligibility Checker
![Return Status](https://github.com/yashh2417/faltu/blob/main/return.png?raw=true)

#### Return Rejected-clean product
![Return Rejected-clean product](https://github.com/yashh2417/faltu/blob/main/return-response1.png?raw=true)

#### Return Rejected-defected product
![Return Rejected-defected product](https://github.com/yashh2417/faltu/blob/main/return-response2.png?raw=true)

- Upload product photo + purchase age (in days)
- Uses:
  - CV + business logic
  - To verify if return is valid
  - In case of defect → prompts inspection/replacement

---

## 🧠 How It Works

### 🔍 Computer Vision

- **Instrument Classification**: ResNet‑based model trained on 5 categories:
  `['flute', 'guitar', 'tabla', 'violin', 'drum']`

- **Defect Detection**: YOLOv8 trained on custom dataset:
  - Detects surface damage, broken parts, or wear

### ✨ Description Generation

- **Input**: Specs like `"9 inches, steel body, 6 holes"`
- **Output**: 
```json
{
  "description": "This stylish steel flute features a sleek black finish...",
  "features": ["9 inches", "6 holes", "black color", "steel body"]
}
````

* Powered by **Google Gemini LLM API**

### 🔗 RAG for Customer Chat

* Embedded:

  * 📝 Store Policies (PDF)
  * ❓ FAQs (JSON)
* Vectorized using **Google GenerativeAI Embeddings**
* Indexed with **Chroma**
* Retrieved via **LangChain ConversationalRetrievalChain**

---

## 🛠️ Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/yourname/intellishelf.git
cd intellishelf/store
pip install -r requirements.txt
```

### 2. Environment

Create a `.env` file:

```
GOOGLE_API_KEY=your_google_gemini_key
```

### 3. Run FastAPI

```bash
uvicorn main:app --reload
```

Visit 👉 `http://localhost:8000`

---

## 🐳 Docker Usage

This app is fully containerized and ready to deploy via Docker.

### 🧱 Build & Push Docker Image (for Developers)

```bash
# Build image
docker build -t yashh2417/intellishelf-store:latest .

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push yashh2417/intellishelf-store:latest

```

### 🏃 Run via Docker (Local)
```bash
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_gemini_api_key \
  yashh2417/intellishelf-store:latest
Visit 👉 http://localhost:8000
```

---

### ☁️ Cloud Deployment (Next Step)
Once image is pushed to Docker Hub, deploy using:

#### **✅ Option 1:** AWS Lightsail (Recommended for ARM64)

* Create Container Service

* **Use image:** yashh2417/intellishelf-store:latest

* Expose port 8000

* Add GOOGLE_API_KEY as secret environment variable

#### **✅ Option 2:** Oracle Cloud (Always Free ARM VM + CapRover)

* Launch Ampere VM

* Install Docker + CapRover

* Use your DockerHub image to deploy

#### **✅ Option 3:** Azure Container Apps

* Supports private/public registry

* Auto‑restart, logging, HTTPS support

> 💡 You must ensure platform = linux/arm64 for all deployments (check Docker image architecture)

---

## 📈 Future Roadmap

### 🔁 Feedback Loop

Let users rate:

* 🤖 Description quality
* 📷 Detection accuracy
* 🤝 RAG answers

Use feedback to:

* Fine-tune prompts
* Improve data augmentation
* Retrain models periodically

---

### 💳 Monetization / Real Usage

If planning a real launch:

* 💸 Add Stripe/PayPal checkout
* 📊 Build seller dashboard:

  * Upload stock
  * Track defects
* 📬 Enable email alerts for:

  * Returns
  * Claim status

---

## 🧪 Tech Stack

* **Frontend**: HTML + Jinja2 + CSS
* **Backend**: FastAPI
* **LLM**: Gemini 1.5 Flash
* **RAG**: LangChain + Chroma
* **Embedding**: GoogleGenerativeAIEmbeddings
* **Database**: JSON + ChromaDB
* **CV Models**: YOLOv8 + PyTorch CNN
* **Container**: Docker (ARM64)
* **Deployment**: Docker Hub → AWS/Oracle/Railway

---

## 🙏 Acknowledgements

* [Google Gemini API](https://makersuite.google.com/)
* [Ultralytics YOLOv8](https://docs.ultralytics.com)
* [LangChain](https://docs.langchain.com/)
* [Docker](https://hub.docker.com/)

---

## 📬 Contact

* 📧 Email: [yashh2417@gmail.com](mailto:yashh2417@gmail.com)
* 🔗 LinedIn: [@yashh2417](https://www.linkedin.com/in/yashh2417?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BHlQpB0ovQvuXk8LrDjdYTA%3D%3D)
* 🐦 Twitter: [@yashh2417](https://twitter.com/yashh2417)
* 💼 Portfolio: [@yashh2417](https://www.datascienceportfol.io/yashh2417)
* 🌐 Docker-Image: [Docker-Image](https://hub.docker.com/repository/docker/yashh2417/intellishelf-store)

---

> ⚠️ **Note:** This is a **demo project**. All models are trained on small sample datasets and not production-hardened. Gemini API usage is rate-limited under personal key.

---


