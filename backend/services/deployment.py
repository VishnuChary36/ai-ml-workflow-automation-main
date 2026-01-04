"""Model Deployment Service - Creates production-ready deployment packages."""
import os
import json
import shutil
import zipfile
import joblib
import hashlib
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


class InferenceImageBuilder:
    """
    Builds and publishes Docker images for the inference service.
    Separate from the training/management backend image.
    """
    
    def __init__(self, task_id: str, emit_log_func=None):
        self.task_id = task_id
        self.emit_log = emit_log_func
    
    async def _log(self, level: str, message: str, source: str = "image_builder"):
        """Emit log message."""
        if self.emit_log:
            await self.emit_log(self.task_id, level, message, source=source)
        print(f"[{level}] {message}")
    
    def generate_image_tag(
        self,
        model_id: str,
        version: str = "1.0.0",
        include_sha: bool = True
    ) -> str:
        """
        Generate a unique image tag for the model.
        Format: model:vX.Y.Z+sha
        """
        # Generate short SHA from model ID and timestamp
        sha_input = f"{model_id}:{datetime.utcnow().isoformat()}"
        sha = hashlib.sha256(sha_input.encode()).hexdigest()[:8]
        
        if include_sha:
            return f"model:v{version}+{sha}"
        return f"model:v{version}"
    
    async def build_inference_image(
        self,
        package_dir: str,
        image_name: str,
        image_tag: str,
        registry: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a Docker image for the inference service.
        
        Args:
            package_dir: Path to deployment package with model artifacts
            image_name: Base image name
            image_tag: Image tag (e.g., "model:v1.0.0+abc123")
            registry: Optional registry URL (e.g., "ghcr.io/org")
        
        Returns:
            Build result with image details
        """
        await self._log("INFO", f"🐳 Building inference image: {image_name}:{image_tag}", "build.start")
        
        # Full image name with optional registry
        if registry:
            full_image = f"{registry}/{image_name}:{image_tag}"
        else:
            full_image = f"{image_name}:{image_tag}"
        
        try:
            # Check if Docker is available
            docker_check = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True
            )
            if docker_check.returncode != 0:
                raise RuntimeError("Docker is not available or not running")
            
            await self._log("INFO", "   ✅ Docker daemon is running", "build.check")
            
            # Build the image
            await self._log("INFO", f"   📦 Building image from {package_dir}...", "build.docker")
            
            build_cmd = [
                "docker", "build",
                "-t", full_image,
                "-f", os.path.join(package_dir, "Dockerfile"),
                package_dir
            ]
            
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if build_result.returncode != 0:
                await self._log("ERROR", f"   ❌ Build failed: {build_result.stderr}", "build.error")
                return {
                    "success": False,
                    "error": build_result.stderr,
                    "image": None
                }
            
            await self._log("INFO", f"   ✅ Image built successfully: {full_image}", "build.success")
            
            # Get image digest
            inspect_cmd = ["docker", "inspect", "--format", "{{.Id}}", full_image]
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
            image_id = inspect_result.stdout.strip() if inspect_result.returncode == 0 else None
            
            return {
                "success": True,
                "image": full_image,
                "image_id": image_id,
                "tag": image_tag,
                "registry": registry,
            }
            
        except subprocess.TimeoutExpired:
            await self._log("ERROR", "   ❌ Build timed out", "build.timeout")
            return {"success": False, "error": "Build timed out", "image": None}
        except Exception as e:
            await self._log("ERROR", f"   ❌ Build error: {str(e)}", "build.error")
            return {"success": False, "error": str(e), "image": None}
    
    async def push_inference_image(
        self,
        image: str,
        registry: str,
    ) -> Dict[str, Any]:
        """
        Push the inference image to a container registry.
        
        Args:
            image: Full image name with tag
            registry: Registry URL
        
        Returns:
            Push result
        """
        await self._log("INFO", f"📤 Pushing image to {registry}...", "push.start")
        
        try:
            # Login to registry if credentials are available
            # This assumes docker login was done previously or credentials are configured
            
            push_cmd = ["docker", "push", image]
            push_result = subprocess.run(
                push_cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if push_result.returncode != 0:
                await self._log("ERROR", f"   ❌ Push failed: {push_result.stderr}", "push.error")
                return {
                    "success": False,
                    "error": push_result.stderr,
                }
            
            await self._log("INFO", f"   ✅ Image pushed successfully", "push.success")
            
            return {
                "success": True,
                "image": image,
                "registry": registry,
            }
            
        except subprocess.TimeoutExpired:
            await self._log("ERROR", "   ❌ Push timed out", "push.timeout")
            return {"success": False, "error": "Push timed out"}
        except Exception as e:
            await self._log("ERROR", f"   ❌ Push error: {str(e)}", "push.error")
            return {"success": False, "error": str(e)}
    
    async def build_and_publish(
        self,
        package_dir: str,
        model_id: str,
        version: str = "1.0.0",
        registry: Optional[str] = None,
        image_name: str = "ml-inference",
    ) -> Dict[str, Any]:
        """
        Build and publish an inference image in one operation.
        
        Args:
            package_dir: Deployment package directory
            model_id: Model ID for tagging
            version: Semantic version
            registry: Optional registry URL
            image_name: Base image name
        
        Returns:
            Combined build and push result
        """
        # Generate tag
        image_tag = self.generate_image_tag(model_id, version)
        
        # Build image
        build_result = await self.build_inference_image(
            package_dir=package_dir,
            image_name=image_name,
            image_tag=image_tag,
            registry=registry,
        )
        
        if not build_result["success"]:
            return build_result
        
        # Push if registry is specified
        if registry:
            push_result = await self.push_inference_image(
                image=build_result["image"],
                registry=registry,
            )
            
            if not push_result["success"]:
                return {
                    **build_result,
                    "pushed": False,
                    "push_error": push_result.get("error"),
                }
            
            return {
                **build_result,
                "pushed": True,
            }
        
        return {
            **build_result,
            "pushed": False,
        }


class DeploymentService:
    """
    Creates production-ready deployment packages for real-world predictions.
    Includes trained model, preprocessing, inference API, and Docker config.
    """
    
    def __init__(self, task_id: str, emit_log_func=None):
        self.task_id = task_id
        self.emit_log = emit_log_func
        self.image_builder = InferenceImageBuilder(task_id, emit_log_func)
        
    async def _log(self, level: str, message: str, source: str = "deployment"):
        """Emit log message."""
        if self.emit_log:
            await self.emit_log(self.task_id, level, message, source=source)
        print(f"[{level}] {message}")
    
    async def package_model_async(self, 
                                   model_path: str,
                                   df: pd.DataFrame,
                                   target_column: str,
                                   model_name: str,
                                   model_type: str,
                                   metrics: Dict[str, Any],
                                   output_dir: str,
                                   build_image: bool = False,
                                   push_image: bool = False,
                                   image_registry: Optional[str] = None,
                                   model_id: Optional[str] = None,
                                   version: str = "1.0.0") -> Dict[str, Any]:
        """
        Create a complete deployment package with real-time progress updates.
        
        Optionally builds and pushes a Docker image for the inference service.
        """
        await self._log("INFO", f"🚀 Starting deployment package creation for {model_name}", "deployment.start")
        
        # Create package directory
        package_dir = os.path.join(output_dir, f"deploy_{self.task_id[:12]}")
        if os.path.exists(package_dir):
            shutil.rmtree(package_dir)
        os.makedirs(package_dir, exist_ok=True)
        
        try:
            # Step 1: Copy trained model (fast)
            await self._log("INFO", "📦 Step 1/6: Copying trained model...", "deployment.model")
            model_dest = os.path.join(package_dir, "model.joblib")
            shutil.copy2(model_path, model_dest)
            model = joblib.load(model_path)
            await self._log("INFO", "   ✅ Model packaged successfully", "deployment.model")
            
            # Step 2: Create preprocessing artifacts (fast)
            await self._log("INFO", "🔧 Step 2/6: Creating preprocessing pipeline...", "deployment.preprocess")
            preprocess_info = self._create_preprocessing_artifacts(df, target_column, package_dir)
            await self._log("INFO", f"   ✅ Preprocessors saved ({len(preprocess_info['encoders'])} encoders)", "deployment.preprocess")
            
            # Step 3: Save metadata (fast)
            await self._log("INFO", "📋 Step 3/6: Saving feature metadata...", "deployment.metadata")
            metadata = self._create_metadata(df, target_column, model_name, model_type, metrics, preprocess_info)
            with open(os.path.join(package_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            await self._log("INFO", f"   ✅ Metadata saved ({metadata['n_features']} features)", "deployment.metadata")
            
            # Step 4: Generate inference API (fast)
            await self._log("INFO", "🌐 Step 4/6: Generating production inference API...", "deployment.api")
            self._generate_inference_api(metadata, package_dir)
            await self._log("INFO", "   ✅ FastAPI inference code generated", "deployment.api")
            
            # Step 5: Generate Docker files (fast)
            await self._log("INFO", "🐳 Step 5/6: Creating Docker configuration...", "deployment.docker")
            self._generate_docker_files(package_dir)
            await self._log("INFO", "   ✅ Dockerfile and docker-compose.yml created", "deployment.docker")
            
            # Step 6: Create zip archive (fast)
            await self._log("INFO", "📦 Step 6/6: Creating deployment archive...", "deployment.archive")
            zip_path = f"{package_dir}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir)
                        zipf.write(file_path, arcname)
            await self._log("INFO", f"   ✅ Archive created: {os.path.basename(zip_path)}", "deployment.archive")
            
            result = {
                "package_path": zip_path,
                "package_dir": package_dir,
                "metadata": metadata,
                "files": os.listdir(package_dir),
                "status": "success"
            }
            
            # Optional: Build Docker image
            if build_image and model_id:
                await self._log("INFO", "🐳 Building inference Docker image...", "deployment.image")
                
                image_result = await self.image_builder.build_and_publish(
                    package_dir=package_dir,
                    model_id=model_id,
                    version=version,
                    registry=image_registry if push_image else None,
                    image_name="ml-inference",
                )
                
                result["image"] = image_result
                
                if image_result.get("success"):
                    await self._log("INFO", f"   ✅ Image ready: {image_result.get('image')}", "deployment.image")
                else:
                    await self._log("WARN", f"   ⚠️ Image build failed: {image_result.get('error')}", "deployment.image")
            
            await self._log("INFO", "🎉 Deployment package ready!", "deployment.complete")
            
            return result
            
        except Exception as e:
            await self._log("ERROR", f"❌ Deployment failed: {str(e)}", "deployment.error")
            raise
    
    def _create_preprocessing_artifacts(self, df: pd.DataFrame, target_column: str, 
                                          output_dir: str) -> Dict[str, Any]:
        """Create and save preprocessing artifacts."""
        X = df.drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create encoders for categorical columns
        encoders = {}
        encoder_classes = {}
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].fillna('_MISSING_').astype(str))
            encoders[col] = le
            encoder_classes[col] = le.classes_.tolist()
        
        # Save encoders
        if encoders:
            joblib.dump(encoders, os.path.join(output_dir, "encoders.joblib"))
        
        # Calculate fill values for numeric columns
        fill_values = {}
        for col in numeric_cols:
            fill_values[col] = float(X[col].median()) if not X[col].isnull().all() else 0.0
        
        # Target encoder if categorical
        target_encoder = None
        target_classes = None
        if y is not None and y.dtype == 'object':
            target_encoder = LabelEncoder()
            target_encoder.fit(y.fillna('_MISSING_').astype(str))
            target_classes = target_encoder.classes_.tolist()
            joblib.dump(target_encoder, os.path.join(output_dir, "target_encoder.joblib"))
        
        # Save preprocessing config
        config = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "encoder_classes": encoder_classes,
            "fill_values": fill_values,
            "target_classes": target_classes,
            "feature_order": X.columns.tolist()
        }
        
        with open(os.path.join(output_dir, "preprocessing.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            "encoders": encoder_classes,
            "fill_values": fill_values,
            "target_classes": target_classes,
            "feature_order": X.columns.tolist()
        }
    
    def _create_metadata(self, df: pd.DataFrame, target_column: str, model_name: str,
                          model_type: str, metrics: Dict, preprocess_info: Dict) -> Dict[str, Any]:
        """Create comprehensive metadata."""
        X = df.drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None
        
        features = []
        for col in X.columns:
            feat = {
                "name": col,
                "dtype": str(X[col].dtype),
                "is_numeric": np.issubdtype(X[col].dtype, np.number),
                "n_unique": int(X[col].nunique()),
            }
            if feat["is_numeric"]:
                feat["min"] = float(X[col].min()) if not X[col].isnull().all() else None
                feat["max"] = float(X[col].max()) if not X[col].isnull().all() else None
                feat["mean"] = float(X[col].mean()) if not X[col].isnull().all() else None
            else:
                feat["categories"] = X[col].unique().tolist()[:10]
            features.append(feat)
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "target_column": target_column,
            "target_classes": preprocess_info.get("target_classes"),
            "n_features": len(features),
            "n_samples_trained": len(df),
            "feature_order": preprocess_info["feature_order"],
            "features": features,
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "api_version": "v1"
        }
    
    def _generate_inference_api(self, metadata: Dict, output_dir: str):
        """Generate production-ready FastAPI inference code."""
        model_name = metadata["model_name"]
        model_type = metadata["model_type"]
        target_column = metadata["target_column"]
        feature_order = metadata["feature_order"]
        target_classes = metadata.get("target_classes")
        
        api_code = f'''"""
Production Inference API - {model_name}
Generated: {datetime.utcnow().isoformat()}

Run with: uvicorn app:app --host 0.0.0.0 --port 8080
Test: curl http://localhost:8080/health
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="{model_name} Prediction API",
    description="Production ML model serving API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
print("Model loaded successfully!")

# Load preprocessing config
with open(os.path.join(BASE_DIR, "preprocessing.json"), "r") as f:
    preprocess_config = json.load(f)

with open(os.path.join(BASE_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

# Load encoders if available
encoders = {{}}
encoders_path = os.path.join(BASE_DIR, "encoders.joblib")
if os.path.exists(encoders_path):
    encoders = joblib.load(encoders_path)
    print(f"Loaded {{len(encoders)}} encoders")

# Load target encoder if available
target_encoder = None
target_encoder_path = os.path.join(BASE_DIR, "target_encoder.joblib")
if os.path.exists(target_encoder_path):
    target_encoder = joblib.load(target_encoder_path)
    print("Target encoder loaded")

# Constants
FEATURE_ORDER = {feature_order}
MODEL_TYPE = "{model_type}"
TARGET_CLASSES = {target_classes}


class PredictRequest(BaseModel):
    """Single prediction request."""
    data: Dict[str, Any]
    
    class Config:
        json_schema_extra = {{
            "example": {{
                "data": {{{", ".join([f'"{f}": "value"' for f in feature_order[:3]])}}}
            }}
        }}


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    data: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    """Prediction response."""
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    label: Optional[str] = None


def preprocess_input(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    # Create DataFrame with correct column order
    df = pd.DataFrame([raw_data])
    
    # Ensure all required features exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = None
    
    # Select only required features in correct order
    df = df[FEATURE_ORDER]
    
    # Handle numeric columns - fill missing
    for col in preprocess_config.get("numeric_columns", []):
        if col in df.columns:
            fill_val = preprocess_config.get("fill_values", {{}}).get(col, 0)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_val)
    
    # Handle categorical columns - encode
    for col in preprocess_config.get("categorical_columns", []):
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].fillna("_MISSING_").astype(str)
            # Handle unseen categories
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df


def decode_prediction(pred, proba=None):
    """Decode prediction to original label if applicable."""
    # Ensure prediction is Python native type
    if hasattr(pred, 'item'):
        pred = pred.item()  # Convert numpy to Python
    
    result = {{
        "prediction": pred,
        "confidence": None,
        "probabilities": None,
        "label": None
    }}
    
    if MODEL_TYPE == "classification":
        # Decode to original class label
        if target_encoder is not None:
            try:
                label = target_encoder.inverse_transform([int(pred)])[0]
                result["label"] = str(label)
            except Exception as e:
                result["label"] = str(pred)
        elif TARGET_CLASSES is not None:
            try:
                result["label"] = str(TARGET_CLASSES[int(pred)])
            except Exception as e:
                result["label"] = str(pred)
        else:
            result["label"] = str(pred)
        
        # Add probabilities
        if proba is not None:
            result["confidence"] = float(np.max(proba))
            if TARGET_CLASSES:
                # Limit to top 10 classes for efficiency
                top_indices = np.argsort(proba)[-10:][::-1]
                result["probabilities"] = {{
                    str(TARGET_CLASSES[i]): float(proba[i]) for i in top_indices
                }}
            else:
                result["probabilities"] = {{
                    str(i): float(p) for i, p in enumerate(proba[:10])
                }}
    else:
        # Regression
        result["prediction"] = float(pred)
        result["label"] = f"{{float(pred):.4f}}"
    
    return result


@app.get("/")
async def root():
    """API root - returns basic info."""
    return {{
        "name": "{model_name} Prediction API",
        "version": "1.0.0",
        "model_type": MODEL_TYPE,
        "status": "running",
        "endpoints": ["/health", "/predict", "/predict/batch", "/info"]
    }}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {{
        "status": "healthy",
        "model": "{model_name}",
        "timestamp": datetime.utcnow().isoformat()
    }}


@app.get("/info")
async def info():
    """Get model information."""
    return {{
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "target": metadata.get("target_column"),
        "n_features": metadata.get("n_features"),
        "features": FEATURE_ORDER,
        "target_classes": TARGET_CLASSES,
        "metrics": metadata.get("metrics"),
        "created_at": metadata.get("created_at")
    }}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make a single prediction.
    
    Send your feature values in the `data` field.
    """
    try:
        # Preprocess
        X = preprocess_input(request.data)
        
        # Predict
        pred = model.predict(X.values)[0]
        
        # Get probabilities if classification
        proba = None
        if MODEL_TYPE == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.values)[0]
        
        # Decode result
        result = decode_prediction(pred, proba)
        
        return PredictResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{str(e)}}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """Make batch predictions."""
    try:
        results = []
        for item in request.data:
            X = preprocess_input(item)
            pred = model.predict(X.values)[0]
            
            proba = None
            if MODEL_TYPE == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X.values)[0]
            
            results.append(decode_prediction(pred, proba))
        
        return {{"predictions": results, "count": len(results)}}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {{str(e)}}")


if __name__ == "__main__":
    import uvicorn
    print("Starting prediction server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
        
        with open(os.path.join(output_dir, "app.py"), 'w') as f:
            f.write(api_code)
        
        # Also create requirements.txt
        requirements = '''fastapi==0.109.0
uvicorn[standard]==0.27.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
pydantic>=2.0.0
'''
        with open(os.path.join(output_dir, "requirements.txt"), 'w') as f:
            f.write(requirements)
    
    def _generate_docker_files(self, output_dir: str):
        """Generate Docker configuration files."""
        # Dockerfile
        dockerfile = '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
'''
        with open(os.path.join(output_dir, "Dockerfile"), 'w') as f:
            f.write(dockerfile)
        
        # docker-compose.yml
        compose = '''version: '3.8'
services:
  prediction-api:
    build: .
    ports:
      - "8080:8080"
    restart: unless-stopped
'''
        with open(os.path.join(output_dir, "docker-compose.yml"), 'w') as f:
            f.write(compose)
        
        # README
        readme = '''# Model Deployment Package

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
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
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
'''
        with open(os.path.join(output_dir, "README.md"), 'w') as f:
            f.write(readme)

    # ========================================================================
    # Inference Image Building (Separate from Training)
    # ========================================================================
    
    async def build_inference_image(
        self,
        package_dir: str,
        image_name: str,
        image_tag: str,
        registry: str = None,
        push: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a Docker image for the inference service.
        
        This creates a separate image optimized for serving predictions,
        distinct from the training/management backend.
        
        Args:
            package_dir: Directory containing the deployment package
            image_name: Name for the Docker image
            image_tag: Tag for the image (e.g., "v1.0.0+abc123")
            registry: Optional registry to push to (e.g., "ghcr.io/org")
            push: Whether to push the image to the registry
            
        Returns:
            Dict with image details and build status
        """
        import subprocess
        import hashlib
        
        await self._log("INFO", f"🐳 Building inference image: {image_name}:{image_tag}", "inference.build")
        
        try:
            # Verify package directory exists
            if not os.path.exists(package_dir):
                raise FileNotFoundError(f"Package directory not found: {package_dir}")
            
            # Verify Dockerfile exists
            dockerfile_path = os.path.join(package_dir, "Dockerfile")
            if not os.path.exists(dockerfile_path):
                await self._log("INFO", "Generating Dockerfile for inference...", "inference.dockerfile")
                self._generate_docker_files(package_dir)
            
            # Build full image name
            full_image_name = f"{image_name}:{image_tag}"
            if registry:
                full_image_name = f"{registry}/{full_image_name}"
            
            await self._log("INFO", f"Building image: {full_image_name}", "inference.build")
            
            # Build the Docker image
            build_cmd = [
                "docker", "build",
                "-t", full_image_name,
                "-f", dockerfile_path,
                "--label", f"ml.model.version={image_tag}",
                "--label", f"ml.build.timestamp={datetime.utcnow().isoformat()}",
                package_dir
            ]
            
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                await self._log("ERROR", f"Docker build failed: {result.stderr}", "inference.error")
                raise Exception(f"Docker build failed: {result.stderr}")
            
            await self._log("INFO", f"✅ Image built successfully: {full_image_name}", "inference.build")
            
            # Get image digest
            inspect_cmd = ["docker", "inspect", "--format", "{{.Id}}", full_image_name]
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
            image_digest = inspect_result.stdout.strip() if inspect_result.returncode == 0 else None
            
            # Push to registry if requested
            push_result = None
            if push and registry:
                await self._log("INFO", f"📤 Pushing image to {registry}...", "inference.push")
                
                push_cmd = ["docker", "push", full_image_name]
                push_result = subprocess.run(push_cmd, capture_output=True, text=True, timeout=600)
                
                if push_result.returncode != 0:
                    await self._log("WARN", f"Push failed: {push_result.stderr}", "inference.push")
                else:
                    await self._log("INFO", f"✅ Image pushed successfully", "inference.push")
            
            return {
                "status": "success",
                "image_name": image_name,
                "image_tag": image_tag,
                "full_image_name": full_image_name,
                "image_digest": image_digest,
                "registry": registry,
                "pushed": push and push_result and push_result.returncode == 0,
                "build_timestamp": datetime.utcnow().isoformat(),
            }
            
        except subprocess.TimeoutExpired:
            await self._log("ERROR", "Docker build timed out", "inference.error")
            raise Exception("Docker build timed out after 10 minutes")
        except FileNotFoundError as e:
            await self._log("ERROR", f"File not found: {e}", "inference.error")
            raise
        except Exception as e:
            await self._log("ERROR", f"Failed to build inference image: {e}", "inference.error")
            raise
    
    async def publish_inference_image(
        self,
        image_name: str,
        source_tag: str,
        target_registry: str,
        target_tag: str = None,
    ) -> Dict[str, Any]:
        """
        Tag and push an existing image to a different registry.
        
        Args:
            image_name: Local image name
            source_tag: Current tag
            target_registry: Target registry (e.g., "ghcr.io/org", "123456789.dkr.ecr.us-east-1.amazonaws.com")
            target_tag: Tag for target (defaults to source_tag)
            
        Returns:
            Dict with push status and details
        """
        import subprocess
        
        target_tag = target_tag or source_tag
        source_full = f"{image_name}:{source_tag}"
        target_full = f"{target_registry}/{image_name}:{target_tag}"
        
        await self._log("INFO", f"📤 Publishing {source_full} → {target_full}", "inference.publish")
        
        try:
            # Tag the image
            tag_cmd = ["docker", "tag", source_full, target_full]
            tag_result = subprocess.run(tag_cmd, capture_output=True, text=True)
            
            if tag_result.returncode != 0:
                raise Exception(f"Failed to tag image: {tag_result.stderr}")
            
            # Push to registry
            push_cmd = ["docker", "push", target_full]
            push_result = subprocess.run(push_cmd, capture_output=True, text=True, timeout=600)
            
            if push_result.returncode != 0:
                raise Exception(f"Failed to push image: {push_result.stderr}")
            
            await self._log("INFO", f"✅ Published to {target_full}", "inference.publish")
            
            return {
                "status": "success",
                "source_image": source_full,
                "target_image": target_full,
                "registry": target_registry,
                "published_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            await self._log("ERROR", f"Failed to publish image: {e}", "inference.error")
            raise
    
    @staticmethod
    def generate_image_tag(model_id: str, version: str = "1.0.0") -> str:
        """
        Generate a standardized image tag for model versioning.
        
        Format: model:vX.Y.Z+<short_sha>
        Example: model:v1.0.0+abc123
        """
        import hashlib
        
        # Generate short SHA from model_id
        sha = hashlib.sha256(f"{model_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:7]
        
        return f"v{version}+{sha}"
