# Quick Start Guide

Get up and running with the AI Code Remediation Service in under 10 minutes!

## ⚡ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-codefix-assignment-yourname.git
cd ai-codefix-assignment-yourname

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start the service
python app.py
```

### Option 2: Manual Setup

```bash
# Clone and navigate
git clone https://github.com/yourusername/ai-codefix-assignment-yourname.git
cd ai-codefix-assignment-yourname

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start service
python app.py
```

### Option 3: Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

## 🧪 Quick Test

Once the service is running (wait for "Service ready!" message):

```bash
# Open a new terminal
source venv/bin/activate  # If not using Docker

# Run tests
python test_local.py
```

## 📡 Quick API Call

Test the API directly with curl:

```bash
curl -X POST http://localhost:8000/local_fix \
  -H "Content-Type: application/json" \
  -d '{
    "language": "python",
    "cwe": "CWE-798",
    "code": "password = \"hardcoded123\""
  }'
```

## 🎯 What's Included?

- ✅ FastAPI service on `http://localhost:8000`
- ✅ Local LLM (Qwen2.5-Coder-1.5B)
- ✅ RAG with security best practices
- ✅ Comprehensive test suite
- ✅ Docker support
- ✅ Logging and metrics

## 📊 Check Service Status

```bash
# Health check
curl http://localhost:8000/health

# Service info
curl http://localhost:8000/
```

## 🔍 View Logs

```bash
# Application logs
tail -f remediation.log

# Metrics
cat metrics.csv
```

## ⚙️ Configuration

Edit `.env` file to customize:

```bash
# Use GPU (if available)
USE_GPU=true

# Change model
MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-instruct
```

## 🛑 Troubleshooting

### Service won't start

```bash
# Check Python version (need 3.9+)
python3 --version

# Check if port 8000 is available
lsof -i :8000  # Unix/Linux
netstat -ano | findstr :8000  # Windows
```

### Model download issues

```bash
# Manually download model
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')"
```

### Memory issues

```bash
# Use smaller model
MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-instruct python app.py
```

## 📚 Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Add custom recipes to `recipes/` directory
3. Explore the API at `http://localhost:8000/docs`
4. Run unit tests: `python test_unit.py`

## 💡 Tips

- First startup takes 30-60 seconds (model loading)
- First request takes ~5 seconds (warm-up)
- Subsequent requests: 2-4 seconds
- Use GPU for 3-4x faster inference
- Add more recipes for better accuracy

## 🆘 Need Help?

- Check the logs: `tail -f remediation.log`
- View API docs: http://localhost:8000/docs
- See full documentation: [README.md](README.md)

---

**Ready to go!** 🚀

Start fixing vulnerable code with AI-powered remediation!