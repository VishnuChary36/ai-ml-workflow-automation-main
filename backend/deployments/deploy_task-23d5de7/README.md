# Model Deployment Package

## Quick Start

### Option 1: Run Directly
```bash
pip install -r requirements.txt
python app.py
```

### Option 2: Docker
```bash
docker build -t ml-api .
docker run -p 8080:8080 ml-api
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /info` - Model info
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

## Example Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature1": "value1", "feature2": 123}}'
```

## Response Format

```json
{
  "prediction": 1,
  "confidence": 0.92,
  "probabilities": {"class_a": 0.92, "class_b": 0.08},
  "label": "class_a"
}
```
