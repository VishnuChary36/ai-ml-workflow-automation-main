"""File storage service for managing datasets, models, and artifacts."""
import os
import json
import joblib
from enum import Enum
from typing import Optional, Any, Dict
from datetime import datetime
import pandas as pd

from config import settings


class FileType(str, Enum):
    """Types of files that can be stored."""
    DATASET = "dataset"
    MODEL = "model"
    VISUALIZATION = "visualization"
    EXPLAINABILITY = "explainability"
    PROCESSED_DATA = "processed_data"
    ARTIFACT = "artifact"


class FileStorageRecord:
    """Represents a stored file record."""
    
    def __init__(self, id: str, file_type: FileType, path: str, metadata: Dict[str, Any] = None):
        self.id = id
        self.file_type = file_type
        self.path = path
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()


class FileStorageService:
    """Service for managing file storage operations.
    
    Provides methods for loading and storing datasets, models, 
    visualizations, and other artifacts.
    """
    
    @staticmethod
    def get_artifact_path() -> str:
        """Get the artifact storage path."""
        return settings.artifact_storage_path
    
    @staticmethod
    def get_model_path() -> str:
        """Get the model storage path."""
        return settings.model_storage_path
    
    @staticmethod
    def load_processed_dataset(db, task_id: str) -> Optional[pd.DataFrame]:
        """Load a processed dataset by task ID.
        
        Args:
            db: Database session
            task_id: The task ID associated with the processed dataset
            
        Returns:
            DataFrame if found, None otherwise
        """
        # Try to find the processed CSV file in artifacts
        artifact_path = settings.artifact_storage_path
        processed_file = os.path.join(artifact_path, f"task-{task_id}_processed.csv")
        
        # Also check without the "task-" prefix
        if not os.path.exists(processed_file):
            processed_file = os.path.join(artifact_path, f"{task_id}_processed.csv")
        
        if os.path.exists(processed_file):
            try:
                return pd.read_csv(processed_file)
            except Exception as e:
                print(f"Error loading processed dataset: {e}")
                return None
        
        # Try to get task info from database and find associated dataset
        try:
            from core.database import Task, Dataset
            task = db.query(Task).filter(Task.id == task_id).first()
            if task and task.dataset_id:
                return FileStorageService.load_dataset_as_dataframe(db, task.dataset_id)
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def load_dataset_as_dataframe(db, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load a dataset as a pandas DataFrame.
        
        Args:
            db: Database session
            dataset_id: The dataset ID
            
        Returns:
            DataFrame if found, None otherwise
        """
        try:
            from core.database import Dataset
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            
            if dataset and dataset.file_path:
                file_path = dataset.file_path
                
                if os.path.exists(file_path):
                    # Determine file type by extension
                    _, ext = os.path.splitext(file_path)
                    ext = ext.lower()
                    
                    if ext == '.csv':
                        return pd.read_csv(file_path)
                    elif ext in ['.xls', '.xlsx']:
                        return pd.read_excel(file_path)
                    elif ext == '.json':
                        return pd.read_json(file_path)
                    elif ext == '.parquet':
                        return pd.read_parquet(file_path)
                    else:
                        # Default to CSV
                        return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        return None
    
    @staticmethod
    def load_model(db, model_id: str) -> Optional[Any]:
        """Load a trained model by model ID.
        
        Args:
            db: Database session
            model_id: The model ID
            
        Returns:
            Model object if found, None otherwise
        """
        try:
            from core.database import Model
            model_record = db.query(Model).filter(Model.id == model_id).first()
            
            if model_record:
                # First try the artifact_path stored in the database
                if model_record.artifact_path and os.path.exists(model_record.artifact_path):
                    return joblib.load(model_record.artifact_path)
                
                # Try to find model file by task_id pattern in models directory
                if model_record.task_id:
                    model_dir = settings.model_storage_path
                    if os.path.exists(model_dir):
                        # Look for files matching the task ID
                        for filename in os.listdir(model_dir):
                            if model_record.task_id in filename and filename.endswith('.joblib'):
                                model_path = os.path.join(model_dir, filename)
                                return joblib.load(model_path)
                
                # Try model ID based naming
                model_dir = settings.model_storage_path
                possible_paths = [
                    os.path.join(model_dir, f"{model_id}.joblib"),
                    os.path.join(model_dir, f"model_{model_id}.joblib"),
                    os.path.join(model_dir, f"task-{model_id}.joblib"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        return joblib.load(path)
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return None
    
    @staticmethod
    def store_visualization(
        db,
        viz_data: Dict[str, Any],
        task_id: str,
        model_id: str,
    ) -> FileStorageRecord:
        """Store visualization data.
        
        Args:
            db: Database session
            viz_data: Visualization data dictionary
            task_id: Task ID
            model_id: Model ID
            
        Returns:
            FileStorageRecord with stored file information
        """
        artifact_path = settings.artifact_storage_path
        os.makedirs(artifact_path, exist_ok=True)
        
        # Generate filename
        filename = f"task-{task_id}_mdl-{model_id}_visualizations.json"
        file_path = os.path.join(artifact_path, filename)
        
        # Write visualization data to file
        with open(file_path, 'w') as f:
            json.dump(viz_data, f, indent=2, default=str)
        
        # Create record
        record = FileStorageRecord(
            id=f"viz_{task_id}_{model_id}",
            file_type=FileType.VISUALIZATION,
            path=file_path,
            metadata={
                "task_id": task_id,
                "model_id": model_id,
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
        return record
    
    @staticmethod
    def store_explainability(
        db,
        explain_data: Dict[str, Any],
        task_id: str,
        model_id: str,
    ) -> FileStorageRecord:
        """Store explainability data.
        
        Args:
            db: Database session
            explain_data: Explainability data dictionary
            task_id: Task ID
            model_id: Model ID
            
        Returns:
            FileStorageRecord with stored file information
        """
        artifact_path = settings.artifact_storage_path
        os.makedirs(artifact_path, exist_ok=True)
        
        # Generate filename
        filename = f"task-{task_id}_mdl-{model_id}_explainability.json"
        file_path = os.path.join(artifact_path, filename)
        
        # Write explainability data to file
        with open(file_path, 'w') as f:
            json.dump(explain_data, f, indent=2, default=str)
        
        # Create record
        record = FileStorageRecord(
            id=f"exp_{task_id}_{model_id}",
            file_type=FileType.EXPLAINABILITY,
            path=file_path,
            metadata={
                "task_id": task_id,
                "model_id": model_id,
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
        return record
    
    @staticmethod
    def store_model(
        db,
        model_obj: Any,
        task_id: str,
        model_name: str,
    ) -> FileStorageRecord:
        """Store a trained model.
        
        Args:
            db: Database session
            model_obj: The trained model object
            task_id: Task ID
            model_name: Name of the model/algorithm
            
        Returns:
            FileStorageRecord with stored file information
        """
        model_path = settings.model_storage_path
        os.makedirs(model_path, exist_ok=True)
        
        # Normalize model name for filename
        model_name_clean = model_name.lower().replace(' ', '_')
        
        # Generate filename
        filename = f"task-{task_id}_{model_name_clean}.joblib"
        file_path = os.path.join(model_path, filename)
        
        # Save model
        joblib.dump(model_obj, file_path)
        
        # Create record
        record = FileStorageRecord(
            id=f"mdl_{task_id}",
            file_type=FileType.MODEL,
            path=file_path,
            metadata={
                "task_id": task_id,
                "model_name": model_name,
                "saved_at": datetime.utcnow().isoformat()
            }
        )
        
        return record
    
    @staticmethod
    def store_processed_data(
        db,
        df: pd.DataFrame,
        task_id: str,
    ) -> FileStorageRecord:
        """Store processed dataset.
        
        Args:
            db: Database session
            df: Processed DataFrame
            task_id: Task ID
            
        Returns:
            FileStorageRecord with stored file information
        """
        artifact_path = settings.artifact_storage_path
        os.makedirs(artifact_path, exist_ok=True)
        
        # Generate filename
        filename = f"task-{task_id}_processed.csv"
        file_path = os.path.join(artifact_path, filename)
        
        # Save DataFrame
        df.to_csv(file_path, index=False)
        
        # Create record
        record = FileStorageRecord(
            id=f"proc_{task_id}",
            file_type=FileType.PROCESSED_DATA,
            path=file_path,
            metadata={
                "task_id": task_id,
                "rows": len(df),
                "columns": len(df.columns),
                "saved_at": datetime.utcnow().isoformat()
            }
        )
        
        return record
    
    @staticmethod
    def load_visualization(task_id: str = None, model_id: str = None) -> Optional[Dict[str, Any]]:
        """Load stored visualization data.
        
        Args:
            task_id: Task ID (optional)
            model_id: Model ID (optional)
            
        Returns:
            Visualization data dictionary if found, None otherwise
        """
        artifact_path = settings.artifact_storage_path
        
        if task_id and model_id:
            filename = f"task-{task_id}_mdl-{model_id}_visualizations.json"
            file_path = os.path.join(artifact_path, filename)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        
        # Search for matching files
        if os.path.exists(artifact_path):
            for filename in os.listdir(artifact_path):
                if filename.endswith('_visualizations.json'):
                    if (task_id and task_id in filename) or (model_id and model_id in filename):
                        file_path = os.path.join(artifact_path, filename)
                        with open(file_path, 'r') as f:
                            return json.load(f)
        
        return None
    
    @staticmethod
    def load_explainability(task_id: str = None, model_id: str = None) -> Optional[Dict[str, Any]]:
        """Load stored explainability data.
        
        Args:
            task_id: Task ID (optional)
            model_id: Model ID (optional)
            
        Returns:
            Explainability data dictionary if found, None otherwise
        """
        artifact_path = settings.artifact_storage_path
        
        if task_id and model_id:
            filename = f"task-{task_id}_mdl-{model_id}_explainability.json"
            file_path = os.path.join(artifact_path, filename)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        
        # Search for matching files
        if os.path.exists(artifact_path):
            for filename in os.listdir(artifact_path):
                if filename.endswith('_explainability.json'):
                    if (task_id and task_id in filename) or (model_id and model_id in filename):
                        file_path = os.path.join(artifact_path, filename)
                        with open(file_path, 'r') as f:
                            return json.load(f)
        
        return None
