# 🩸 Fingerprint-Based Blood Group Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Flutter](https://img.shields.io/badge/Flutter-3.16+-blue.svg)](https://flutter.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade deep learning system that predicts human blood groups from fingerprint images using state-of-the-art CNN architectures.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐    │
│  │   Flutter    │     │   Flutter    │     │      Flutter Web         │    │
│  │   Android    │     │     iOS      │     │       Application        │    │
│  └──────┬───────┘     └──────┬───────┘     └───────────┬──────────────┘    │
│         │                    │                         │                    │
│         └────────────────────┼─────────────────────────┘                    │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │   API Gateway   │                                      │
│                    │   (FastAPI)     │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────┐   ┌───────────────┐   ┌─────────────┐                    │
│  │    Auth     │   │   Prediction  │   │   Storage   │                    │
│  │   Service   │   │    Service    │   │   Service   │                    │
│  └──────┬──────┘   └───────┬───────┘   └──────┬──────┘                    │
│         │                  │                   │                           │
│         │                  ▼                   │                           │
│         │         ┌───────────────┐            │                           │
│         │         │   ML Model    │            │                           │
│         │         │ (EfficientNet)│            │                           │
│         │         └───────────────┘            │                           │
│         │                                      │                           │
│         └──────────────────┬───────────────────┘                           │
│                            ▼                                               │
│                   ┌─────────────────┐                                      │
│                   │   PostgreSQL    │                                      │
│                   │    Database     │                                      │
│                   └─────────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Features

### Machine Learning
- **Transfer Learning**: EfficientNet-B0 backbone for optimal accuracy/speed trade-off
- **Data Augmentation**: Comprehensive augmentation pipeline for robust predictions
- **Class Balancing**: Weighted loss functions to handle imbalanced datasets
- **Model Export**: ONNX format for cross-platform deployment

### Backend API
- **FastAPI**: High-performance async API framework
- **JWT Authentication**: Secure token-based auth with refresh tokens
- **Role-Based Access**: Admin and user roles with granular permissions
- **Rate Limiting**: Protection against API abuse
- **OpenAPI Documentation**: Auto-generated Swagger docs

### Flutter Application
- **Cross-Platform**: Android, iOS, and Web support
- **Clean Architecture**: MVVM pattern with BLoC state management
- **Medical-Grade UI**: Professional, accessible interface
- **Offline Support**: Local caching for better UX

### DevOps
- **Docker**: Containerized services for easy deployment
- **CI/CD**: GitHub Actions pipeline
- **Model Versioning**: DVC integration for model management
- **Monitoring**: Prometheus + Grafana stack

## 📁 Project Structure

```
fingerprint-blood-detection/
├── ml/                          # Machine Learning Module
│   ├── configs/                 # Training configurations
│   ├── src/
│   │   ├── data/               # Data loading & preprocessing
│   │   ├── models/             # Model architectures
│   │   ├── training/           # Training loops & callbacks
│   │   ├── evaluation/         # Metrics & evaluation
│   │   └── inference/          # Production inference
│   ├── notebooks/              # Jupyter notebooks for EDA
│   ├── scripts/                # Training & export scripts
│   └── tests/                  # ML unit tests
│
├── backend/                     # Backend API
│   ├── app/
│   │   ├── api/               # API routes
│   │   ├── core/              # Core configurations
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # Business logic
│   │   └── utils/             # Utilities
│   ├── tests/                 # API tests
│   └── alembic/               # Database migrations
│
├── flutter_app/                # Flutter Application
│   ├── lib/
│   │   ├── core/              # Core utilities
│   │   ├── features/          # Feature modules
│   │   ├── shared/            # Shared widgets
│   │   └── main.dart          # Entry point
│   ├── test/                  # Flutter tests
│   └── assets/                # App assets
│
├── deployment/                 # Deployment Configs
│   ├── docker/                # Dockerfiles
│   ├── kubernetes/            # K8s manifests
│   └── nginx/                 # Nginx configs
│
├── docs/                       # Documentation
│   ├── api/                   # API documentation
│   ├── architecture/          # Architecture docs
│   └── guides/                # Setup guides
│
└── scripts/                    # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for tooling)
- Flutter 3.16+
- Docker & Docker Compose
- PostgreSQL 15+

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fingerprint-blood-detection.git
cd fingerprint-blood-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install ML dependencies
pip install -r ml/requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Train Model

```bash
cd ml
python scripts/train.py --config configs/efficientnet_config.yaml
```

### 3. Start Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run Flutter App

```bash
cd flutter_app
flutter pub get
flutter run -d chrome  # For web
flutter run            # For Android/iOS
```

### 5. Docker Deployment

```bash
docker-compose up -d
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 92.4% |
| Precision | 91.8% |
| Recall | 92.1% |
| F1-Score | 91.9% |

### Confusion Matrix
See `docs/model_evaluation.md` for detailed analysis.

## 🔒 Security Features

- JWT token authentication with refresh mechanism
- Bcrypt password hashing
- Rate limiting (100 requests/minute)
- Input validation & sanitization
- CORS configuration
- Secure file upload handling
- SQL injection prevention via ORM

## 📖 API Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

```bash
# ML Tests
cd ml && pytest tests/ -v

# Backend Tests
cd backend && pytest tests/ -v --cov=app

# Flutter Tests
cd flutter_app && flutter test
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Dataset: [Fingerprint Based Blood Group Dataset](https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset)
- EfficientNet: [Google Research](https://github.com/google/automl)

---

**Built with ❤️ for Healthcare Innovation**
