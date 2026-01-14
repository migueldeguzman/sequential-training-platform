"""
FastAPI backend for Sequential Training and Testing Platform.
Orchestrates the Python training pipeline and provides real-time updates.
"""

import os
import json
import asyncio
import shutil
import subprocess
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from jose import JWTError, jwt
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Config file path
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "config.json"
LOGS_DIR = CONFIG_DIR / "training_logs"

# Default settings
DEFAULT_SETTINGS = {
    "jsonInputDir": "/Users/miguelitodeguzman/Projects/SDB/output",
    "textOutputDir": "/Users/miguelitodeguzman/Projects/SDB/datasets-from-json",
    "textDatasetsDir": "/Users/miguelitodeguzman/Projects/SDB/text-datasets",
    "modelOutputDir": "/Users/miguelitodeguzman/Projects/SDB/trained-models",
    "baseModelPath": "/Users/miguelitodeguzman/Projects/baseModels/zephyr",
    "pipelineScript": "/Users/miguelitodeguzman/Projects/SDB/instruction_tuning_pipeline.py",
}


def load_settings() -> dict:
    """Load settings from config file or return defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                # Merge with defaults (in case new settings are added)
                return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """Save settings to config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


def get_paths():
    """Get current paths from settings."""
    settings = load_settings()
    return {
        "json_input": Path(settings["jsonInputDir"]),
        "text_output": Path(settings["textOutputDir"]),
        "text_datasets": Path(settings["textDatasetsDir"]),
        "model_output": Path(settings["modelOutputDir"]),
        "base_model": Path(settings["baseModelPath"]),
        "pipeline": Path(settings["pipelineScript"]),
    }


# Simple password hashing (use bcrypt in production with compatible versions)
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password


# In-memory storage (use a database in production)
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": hash_password("admin"),
    }
}

# Training state
training_state = {
    "is_running": False,
    "process": None,
    "current_step": 0,
    "total_steps": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_dataset": "",
    "loss": 0.0,
    "start_time": None,
    "logs": [],
    "job_id": None,
}

# WebSocket connections
active_connections: list[WebSocket] = []


# Pydantic models
class Token(BaseModel):
    token: str
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class Dataset(BaseModel):
    name: str
    jsonFileCount: int
    textFileExists: bool
    textFilePath: Optional[str] = None
    pairCount: Optional[int] = None


class TextDataset(BaseModel):
    name: str
    filePath: str
    fileSize: str
    sampleCount: int
    modifiedAt: str
    datasetType: str = "text"  # Always "text" for standalone text files


class ConversionJob(BaseModel):
    datasetName: str
    pairCount: Optional[int] = None
    formatStyle: str = "chat"


class TrainingConfig(BaseModel):
    # datasets can include both JSON and text datasets
    # Prefix with "text:" for standalone text datasets (e.g., "text:alignment")
    # No prefix for JSON datasets (e.g., "SLSEdefense_version7")
    datasets: list[str]
    epochs: float = 1.0
    learningRate: float = 0.000042
    sampleSize: int = 75
    batchMultiplier: int = 2
    gradientAccumulation: int = 16
    formatStyle: str = "chat"
    trainingMode: str = "sequential"


class TrainingStatus(BaseModel):
    isRunning: bool
    currentStep: int
    totalSteps: int
    currentEpoch: int
    totalEpochs: int
    currentDataset: str
    loss: float
    progress: float
    startTime: Optional[str] = None
    estimatedTimeRemaining: Optional[str] = None


class ModelCheckpoint(BaseModel):
    name: str
    path: str
    createdAt: str
    datasetsTrained: list[str]
    config: dict
    size: str


class Settings(BaseModel):
    jsonInputDir: str
    textOutputDir: str
    textDatasetsDir: str
    modelOutputDir: str
    baseModelPath: str
    pipelineScript: str


class TrainingHistoryEntry(BaseModel):
    jobId: str
    startTime: str
    endTime: Optional[str]
    status: str  # "completed", "failed", "stopped"
    datasets: list[str]
    config: dict
    finalLoss: Optional[float]
    logFile: str


# Training history functions
def save_training_log(job_id: str, logs: list, config: dict, status: str, final_loss: float = None):
    """Save training logs to file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"{job_id}.json"
    log_data = {
        "jobId": job_id,
        "startTime": training_state.get("start_time"),
        "endTime": datetime.now().isoformat(),
        "status": status,
        "datasets": config.get("datasets", []),
        "config": config,
        "finalLoss": final_loss,
        "logs": logs,
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    return str(log_file)


def get_training_history() -> list[dict]:
    """Get all training history entries."""
    history = []

    if LOGS_DIR.exists():
        for log_file in sorted(LOGS_DIR.glob("*.json"), reverse=True):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    history.append({
                        "jobId": data.get("jobId"),
                        "startTime": data.get("startTime"),
                        "endTime": data.get("endTime"),
                        "status": data.get("status"),
                        "datasets": data.get("datasets", []),
                        "config": data.get("config", {}),
                        "finalLoss": data.get("finalLoss"),
                        "logFile": str(log_file),
                    })
            except Exception:
                continue

    return history


def get_training_log_detail(job_id: str) -> dict:
    """Get detailed training log for a specific job."""
    log_file = LOGS_DIR / f"{job_id}.json"

    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        return json.load(f)


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - create directories
    paths = get_paths()
    for key in ["json_input", "text_output", "text_datasets", "model_output"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify powermetrics access for Energy Profiler
    from profiling.utils import verify_powermetrics_on_startup
    verify_powermetrics_on_startup()

    yield
    # Shutdown
    if training_state["process"]:
        training_state["process"].terminate()


app = FastAPI(title="Sequential Training and Testing Platform API", lifespan=lifespan)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.231:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Auth helpers
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        return username
    except JWTError:
        return None


async def get_current_user(token: str = Query(None)) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username


# Auth endpoints
@app.post("/api/auth/login", response_model=Token)
async def login(request: LoginRequest):
    user = users_db.get(request.username)
    if not user or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": request.username})
    return Token(token=token, username=request.username)


@app.post("/api/auth/logout")
async def logout():
    return {"message": "Logged out"}


@app.get("/api/auth/verify")
async def verify_auth(
    token: str = Query(None),
    authorization: Optional[str] = Header(None)
):
    # Accept token from query param or Authorization header
    auth_token = token
    if not auth_token and authorization:
        # Extract from "Bearer <token>" format
        if authorization.startswith("Bearer "):
            auth_token = authorization[7:]

    if not auth_token:
        return {"valid": False, "username": None}

    username = verify_token(auth_token)
    if username:
        return {"valid": True, "username": username}
    return {"valid": False, "username": None}


# Settings endpoints
@app.get("/api/settings", response_model=Settings)
async def get_settings():
    """Get current directory settings."""
    settings = load_settings()
    return Settings(**settings)


@app.post("/api/settings", response_model=Settings)
async def update_settings(settings: Settings):
    """Update directory settings."""
    settings_dict = settings.model_dump()

    # Validate paths exist (create if they don't for output dirs)
    json_input = Path(settings_dict["jsonInputDir"])
    if not json_input.exists():
        raise HTTPException(status_code=400, detail=f"JSON input directory does not exist: {json_input}")

    base_model = Path(settings_dict["baseModelPath"])
    if not base_model.exists():
        raise HTTPException(status_code=400, detail=f"Base model path does not exist: {base_model}")

    # Create output directories if they don't exist
    Path(settings_dict["textOutputDir"]).mkdir(parents=True, exist_ok=True)
    Path(settings_dict["textDatasetsDir"]).mkdir(parents=True, exist_ok=True)
    Path(settings_dict["modelOutputDir"]).mkdir(parents=True, exist_ok=True)

    save_settings(settings_dict)
    return Settings(**settings_dict)


@app.post("/api/settings/browse")
async def browse_directory(path: str = Query(...)):
    """Browse a directory and return its contents."""
    dir_path = Path(path)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")

    if not dir_path.is_dir():
        return {
            "path": str(dir_path),
            "isDir": False,
            "parent": str(dir_path.parent),
            "contents": [],
        }

    contents = []
    try:
        for item in sorted(dir_path.iterdir()):
            contents.append({
                "name": item.name,
                "isDir": item.is_dir(),
                "path": str(item),
            })
    except PermissionError:
        pass

    return {
        "path": str(dir_path),
        "isDir": True,
        "parent": str(dir_path.parent),
        "contents": contents,
    }


# Dataset endpoints
@app.get("/api/datasets", response_model=list[Dataset])
async def list_datasets():
    paths = get_paths()
    datasets = []

    if paths["json_input"].exists():
        for dataset_dir in paths["json_input"].iterdir():
            if dataset_dir.is_dir():
                json_files = list(dataset_dir.glob("*.json"))
                text_file = paths["text_output"] / f"{dataset_dir.name}.text"

                datasets.append(Dataset(
                    name=dataset_dir.name,
                    jsonFileCount=len(json_files),
                    textFileExists=text_file.exists(),
                    textFilePath=str(text_file) if text_file.exists() else None,
                    pairCount=len(json_files) if json_files else None,
                ))

    return datasets


@app.get("/api/datasets/{name}", response_model=Dataset)
async def get_dataset(name: str):
    paths = get_paths()
    dataset_dir = paths["json_input"] / name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    json_files = list(dataset_dir.glob("*.json"))
    text_file = paths["text_output"] / f"{name}.text"

    return Dataset(
        name=name,
        jsonFileCount=len(json_files),
        textFileExists=text_file.exists(),
        textFilePath=str(text_file) if text_file.exists() else None,
        pairCount=len(json_files),
    )


@app.post("/api/datasets/convert")
async def convert_dataset(job: ConversionJob):
    paths = get_paths()
    dataset_dir = paths["json_input"] / job.datasetName
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load and convert JSON files
    json_files = sorted(dataset_dir.glob("*.json"))
    pairs = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pairs.append({
                    'question': data.get('question', ''),
                    'answer': data.get('answer', '')
                })
        except Exception:
            continue

    # Limit pairs if specified
    if job.pairCount:
        pairs = pairs[:job.pairCount]

    # Format and write
    output_name = f"{job.datasetName}.text"
    if job.pairCount:
        output_name = f"{job.datasetName}_{job.pairCount}pairs.text"

    output_path = paths["text_output"] / output_name

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(f"{pair['question']}\n {pair['answer']}\n\n")

    return {"outputPath": str(output_path)}


@app.get("/api/datasets/{name}/preview")
async def preview_dataset(name: str, limit: int = 5):
    """Preview converted dataset contents."""
    paths = get_paths()
    text_file = paths["text_output"] / f"{name}.text"

    if not text_file.exists():
        raise HTTPException(status_code=404, detail="Converted file not found")

    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into Q&A pairs
    pairs = [p.strip() for p in content.split('\n\n') if p.strip()]

    # Get file stats
    file_size = text_file.stat().st_size
    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1e6 else f"{file_size / (1024*1024):.2f} MB"

    return {
        "fileName": text_file.name,
        "filePath": str(text_file),
        "fileSize": size_str,
        "totalPairs": len(pairs),
        "previewPairs": pairs[:limit],
        "format": "question\\n answer\\n\\n"
    }


@app.get("/api/datasets/{name}/format")
async def get_dataset_format(name: str):
    """Get conversion format information for a dataset."""
    paths = get_paths()
    dataset_dir = paths["json_input"] / name

    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get a sample JSON file
    json_files = list(dataset_dir.glob("*.json"))
    sample_json = None
    sample_converted = None

    if json_files:
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                sample_json = json.load(f)
                sample_converted = f"{sample_json.get('question', '')}\n {sample_json.get('answer', '')}\n\n"
        except Exception:
            pass

    return {
        "datasetName": name,
        "jsonFileCount": len(json_files),
        "formatTemplate": "{question}\\n {answer}\\n\\n",
        "sampleJson": sample_json,
        "sampleConverted": sample_converted,
        "formats": {
            "chat": "{question}\\n {answer}\\n\\n",
            "simple": "Q: {question}\\nA: {answer}\\n\\n",
            "instruction": "### Instruction:\\n{question}\\n### Response:\\n{answer}\\n\\n",
            "plain": "{question}\\n{answer}\\n\\n"
        }
    }


@app.get("/api/files/converted")
async def list_converted_files():
    """List all converted .text files."""
    paths = get_paths()
    files = []

    if paths["text_output"].exists():
        for text_file in paths["text_output"].glob("*.text"):
            try:
                stat = text_file.stat()
                size = stat.st_size
                size_str = f"{size / 1024:.1f} KB" if size < 1e6 else f"{size / (1024*1024):.2f} MB"

                # Count pairs by counting double newlines
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                pair_count = len([p for p in content.split('\n\n') if p.strip()])

                files.append({
                    "name": text_file.name,
                    "path": str(text_file),
                    "size": size_str,
                    "pairCount": pair_count,
                    "modifiedAt": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception:
                continue

    return {"files": sorted(files, key=lambda x: x["modifiedAt"], reverse=True)}


@app.get("/api/files/content")
async def get_file_content(path: str = Query(...), offset: int = 0, limit: int = 10):
    """Get paginated content of a converted file."""
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security check - only allow files in text_output directory
    paths = get_paths()
    if not str(file_path).startswith(str(paths["text_output"])):
        raise HTTPException(status_code=403, detail="Access denied")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = [p.strip() for p in content.split('\n\n') if p.strip()]

    return {
        "totalPairs": len(pairs),
        "offset": offset,
        "limit": limit,
        "pairs": pairs[offset:offset + limit],
        "hasMore": offset + limit < len(pairs)
    }


# Text datasets endpoints (standalone .text files)
@app.get("/api/text-datasets", response_model=list[TextDataset])
async def list_text_datasets():
    """List all standalone text datasets."""
    paths = get_paths()
    datasets = []

    if paths["text_datasets"].exists():
        for text_file in paths["text_datasets"].glob("*.text"):
            try:
                stat = text_file.stat()
                size = stat.st_size
                size_str = f"{size / 1024:.1f} KB" if size < 1e6 else f"{size / (1024*1024):.2f} MB"

                # Detect separator and count samples
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if '<|endoftext|>' in content:
                    sample_count = len([s for s in content.split('<|endoftext|>') if s.strip()])
                elif '===END_OF_STORY===' in content:
                    sample_count = len([s for s in content.split('===END_OF_STORY===') if s.strip()])
                elif 'END_OF_STORY' in content:
                    sample_count = len([s for s in content.split('END_OF_STORY') if s.strip()])
                elif '---' in content and content.count('---') > 3:
                    sample_count = len([s for s in content.split('---') if s.strip()])
                else:
                    sample_count = 1  # Treat as continuous text

                datasets.append(TextDataset(
                    name=text_file.stem,  # filename without extension
                    filePath=str(text_file),
                    fileSize=size_str,
                    sampleCount=sample_count,
                    modifiedAt=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                ))
            except Exception:
                continue

    return sorted(datasets, key=lambda x: x.name)


@app.get("/api/text-datasets/{name}")
async def get_text_dataset(name: str):
    """Get details of a specific text dataset."""
    paths = get_paths()
    text_file = paths["text_datasets"] / f"{name}.text"

    if not text_file.exists():
        raise HTTPException(status_code=404, detail="Text dataset not found")

    stat = text_file.stat()
    size = stat.st_size
    size_str = f"{size / 1024:.1f} KB" if size < 1e6 else f"{size / (1024*1024):.2f} MB"

    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Detect separator and count samples
    if '<|endoftext|>' in content:
        sample_count = len([s for s in content.split('<|endoftext|>') if s.strip()])
    elif '===END_OF_STORY===' in content:
        sample_count = len([s for s in content.split('===END_OF_STORY===') if s.strip()])
    elif 'END_OF_STORY' in content:
        sample_count = len([s for s in content.split('END_OF_STORY') if s.strip()])
    elif '---' in content and content.count('---') > 3:
        sample_count = len([s for s in content.split('---') if s.strip()])
    else:
        sample_count = 1  # Treat as continuous text

    return TextDataset(
        name=name,
        filePath=str(text_file),
        fileSize=size_str,
        sampleCount=sample_count,
        modifiedAt=datetime.fromtimestamp(stat.st_mtime).isoformat(),
    )


@app.get("/api/text-datasets/{name}/preview")
async def preview_text_dataset(name: str, limit: int = 3):
    """Preview samples from a text dataset."""
    paths = get_paths()
    text_file = paths["text_datasets"] / f"{name}.text"

    if not text_file.exists():
        raise HTTPException(status_code=404, detail="Text dataset not found")

    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Detect separator and split
    if '<|endoftext|>' in content:
        samples = [s.strip() for s in content.split('<|endoftext|>') if s.strip()]
        format_str = "<|endoftext|> separated"
    elif '===END_OF_STORY===' in content:
        samples = [s.strip() for s in content.split('===END_OF_STORY===') if s.strip()]
        format_str = "===END_OF_STORY=== separated"
    elif 'END_OF_STORY' in content:
        samples = [s.strip() for s in content.split('END_OF_STORY') if s.strip()]
        format_str = "END_OF_STORY separated"
    elif '---' in content and content.count('---') > 3:
        samples = [s.strip() for s in content.split('---') if s.strip()]
        format_str = "--- separated"
    else:
        samples = [content.strip()]
        format_str = "continuous text"

    stat = text_file.stat()
    size = stat.st_size
    size_str = f"{size / 1024:.1f} KB" if size < 1e6 else f"{size / (1024*1024):.2f} MB"

    return {
        "fileName": text_file.name,
        "filePath": str(text_file),
        "fileSize": size_str,
        "totalSamples": len(samples),
        "previewSamples": samples[:limit],
        "format": format_str
    }


# Training endpoints
@app.get("/api/training/status", response_model=TrainingStatus)
async def get_training_status():
    progress = 0
    if training_state["total_steps"] > 0:
        progress = (training_state["current_step"] / training_state["total_steps"]) * 100

    return TrainingStatus(
        isRunning=training_state["is_running"],
        currentStep=training_state["current_step"],
        totalSteps=training_state["total_steps"],
        currentEpoch=training_state["current_epoch"],
        totalEpochs=training_state["total_epochs"],
        currentDataset=training_state["current_dataset"],
        loss=training_state["loss"],
        progress=progress,
        startTime=training_state["start_time"],
    )


@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    if training_state["is_running"]:
        raise HTTPException(status_code=400, detail="Training already in progress")

    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Reset state
    training_state["is_running"] = True
    training_state["current_step"] = 0
    training_state["total_steps"] = 100
    training_state["current_epoch"] = 0
    training_state["total_epochs"] = int(config.epochs)
    training_state["current_dataset"] = config.datasets[0] if config.datasets else ""
    training_state["loss"] = 0.0
    training_state["start_time"] = datetime.now().isoformat()
    training_state["logs"] = []
    training_state["job_id"] = job_id

    # Start training in background
    asyncio.create_task(run_training(config, job_id))

    return {"jobId": job_id}


async def run_training(config: TrainingConfig, job_id: str):
    """Run the actual training pipeline via subprocess."""
    import sys
    final_loss = 0.0
    status = "completed"
    paths = get_paths()

    try:
        await broadcast_log("info", f"Starting training with datasets: {', '.join(config.datasets)}")
        await broadcast_log("info", f"Training mode: {config.trainingMode}")

        # Calculate batch size (per-device)
        batch_size = config.sampleSize * config.batchMultiplier
        effective_batch = batch_size * config.gradientAccumulation
        await broadcast_log("info", f"Per-device batch size: {batch_size}, Gradient accumulation: {config.gradientAccumulation}")
        await broadcast_log("info", f"Effective batch size: {effective_batch}")

        # Estimate total steps based on datasets
        total_steps = len(config.datasets) * 100
        training_state["total_steps"] = total_steps
        training_state["total_epochs"] = int(config.epochs)

        # Track current model path for sequential training (model chaining)
        current_model_path = str(paths["base_model"])

        # Get base model name for checkpoint naming
        base_model_name = paths["base_model"].name

        # Process each dataset
        for i, dataset_entry in enumerate(config.datasets):
            if not training_state["is_running"]:
                status = "stopped"
                break

            # Determine dataset type and name
            # Prefix "text:" indicates standalone text dataset
            is_text_dataset = dataset_entry.startswith("text:")
            dataset_name = dataset_entry[5:] if is_text_dataset else dataset_entry

            training_state["current_dataset"] = dataset_name
            dataset_type_label = "[Text]" if is_text_dataset else "[JSON]"
            await broadcast_log("info", f"Processing dataset {i+1}/{len(config.datasets)}: {dataset_name} {dataset_type_label}")
            await broadcast_status()

            if is_text_dataset:
                # Standalone text dataset - use as-is from text_datasets directory
                text_file = paths["text_datasets"] / f"{dataset_name}.text"
                if not text_file.exists():
                    await broadcast_log("error", f"Text dataset not found: {text_file}")
                    continue
                await broadcast_log("info", f"Using standalone text dataset: {text_file.name}")
                use_endoftext_split = True
            else:
                # JSON dataset - check if text file exists, if not convert it
                text_file = paths["text_output"] / f"{dataset_name}.text"
                if not text_file.exists():
                    await broadcast_log("info", f"Converting {dataset_name} to text format...")
                    dataset_dir = paths["json_input"] / dataset_name
                    if dataset_dir.exists():
                        json_files = sorted(dataset_dir.glob("*.json"))
                        pairs = []
                        for json_file in json_files:
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    pairs.append({
                                        'question': data.get('question', ''),
                                        'answer': data.get('answer', '')
                                    })
                            except Exception:
                                continue

                        with open(text_file, 'w', encoding='utf-8') as f:
                            for pair in pairs:
                                f.write(f"{pair['question']}\n {pair['answer']}\n\n")

                        await broadcast_log("success", f"Converted {len(pairs)} pairs to {text_file.name}")
                use_endoftext_split = False

            # Step output directory - includes base model name
            step_output_dir = f"{paths['model_output']}/step_{i+1}_{dataset_name}-{base_model_name}"

            # Log model source for this step
            if i == 0:
                await broadcast_log("info", f"Loading BASE model from: {current_model_path}")
            else:
                await broadcast_log("info", f"Loading PREVIOUS checkpoint from: {current_model_path}")

            await broadcast_log("info", f"Output will be saved to: {step_output_dir}")

            # Create training script - matches instruction_tuning_pipeline.py exactly
            # with added progress logging
            script_content = f'''
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, "{paths["pipeline"].parent}")

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback, get_linear_schedule_with_warmup
from datasets import Dataset

# Custom callback to print progress (added for dashboard)
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            total = state.max_steps
            loss = logs.get("loss", 0)
            print(f"Step {{step}}/{{total}} | loss: {{loss:.4f}}")
            sys.stdout.flush()

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {{int(state.epoch)}} complete")
        sys.stdout.flush()

print("Initializing training assistant...")

# Detect device (same as pipeline)
device = torch.device("mps" if torch.backends.mps.is_available() else
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {{device}}")

# Load model (same as pipeline)
model_path = "{current_model_path}"
base_model_path = "{paths['base_model']}"  # Always use base model for tokenizer
print(f"Loading model from {{model_path}}...")

# Always load tokenizer from BASE model to avoid regex pattern issues
# The tokenizer doesn't change during fine-tuning, only model weights do
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
print(f"Loading tokenizer from {{base_model_path}} (base model)...")

# torch_dtype: float16 for CUDA, float32 otherwise (same as pipeline)
dtype = torch.float16 if device.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset (same as pipeline)
answer_file_path = "{text_file}"
output_dir = "{step_output_dir}"
is_text_dataset = {use_endoftext_split}
print(f"Training on {dataset_name}...")
print(f"Output will be saved to: {{output_dir}}")

print(f"Preparing dataset: {{answer_file_path}}...")
with open(answer_file_path, 'r', encoding='utf-8') as f:
    text_content = f.read()

if is_text_dataset:
    # Text dataset - detect and split by separator
    if '<|endoftext|>' in text_content:
        samples = [s.strip() for s in text_content.split('<|endoftext|>') if s.strip()]
        print(f"  - Loaded {{len(samples)}} samples (split by <|endoftext|>)")
    elif '===END_OF_STORY===' in text_content:
        samples = [s.strip() for s in text_content.split('===END_OF_STORY===') if s.strip()]
        print(f"  - Loaded {{len(samples)}} samples (split by ===END_OF_STORY===)")
    elif 'END_OF_STORY' in text_content:
        samples = [s.strip() for s in text_content.split('END_OF_STORY') if s.strip()]
        print(f"  - Loaded {{len(samples)}} samples (split by END_OF_STORY)")
    elif '---' in text_content and text_content.count('---') > 3:
        samples = [s.strip() for s in text_content.split('---') if s.strip()]
        print(f"  - Loaded {{len(samples)}} samples (split by ---)")
    else:
        # If no recognized separator, use entire content as single sample
        samples = [text_content.strip()]
        print(f"  - Loaded as continuous text (1 sample)")
else:
    # JSON-converted dataset - split by double newlines
    samples = [pair.strip() for pair in text_content.split('\\n\\n') if pair.strip()]
    print(f"  - Loaded {{len(samples)}} Q&A pairs")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

dataset_dict = {{'text': samples}}
train_dataset = Dataset.from_dict(dataset_dict)
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

print(f"  - Dataset size: {{len(train_dataset)}} samples")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

batch_size = {batch_size}
gradient_accumulation_steps = {config.gradientAccumulation}
epochs = {config.epochs}
learning_rate = {config.learningRate}

# Calculate total steps for scheduler (same as pipeline)
total_steps = len(train_dataset) * epochs
warmup_steps = 500  # Same as pipeline

# Custom optimizer with weight decay (same as pipeline)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)

# Linear warmup scheduler (same as pipeline)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print("Setting up training arguments...")
print(f"  - Batch size: {{batch_size}}")
print(f"  - Gradient accumulation steps: {{gradient_accumulation_steps}}")
print(f"  - Effective batch size: {{batch_size * gradient_accumulation_steps}}")
print(f"  - Expected training steps: {{len(train_dataset) // batch_size}}")
print(f"  - Warmup steps: {{warmup_steps}}")
print(f"  - Optimizer: Adam with weight_decay=0.005")

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=device.type == "cuda",
    lr_scheduler_type='cosine',  # Will be overridden by custom scheduler
    warmup_steps=warmup_steps,
    logging_steps=10,
    logging_first_step=True,
    report_to="none",
)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers=(optimizer, scheduler),  # Custom optimizer/scheduler (same as pipeline)
    callbacks=[ProgressCallback()],
)

print(f"Starting training on {dataset_name}...")
sys.stdout.flush()
trainer.train()

print(f"Saving model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model saved successfully!")
print("TRAINING_STEP_COMPLETE")
'''

            # Write script to temp file
            script_file = paths["model_output"] / f"train_script_{job_id}_{i}.py"
            script_file.parent.mkdir(parents=True, exist_ok=True)
            with open(script_file, 'w') as f:
                f.write(script_content)

            await broadcast_log("info", f"Starting model training on {dataset_name}...")

            # Execute training via subprocess using script file (safer than -c)
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(paths["pipeline"].parent)
            )
            training_state["process"] = process

            step_count = 0
            step_failed = False
            logged_lines = set()  # Track logged lines to prevent duplicates

            while True:
                if not training_state["is_running"]:
                    process.terminate()
                    status = "stopped"
                    break

                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                # Skip duplicate lines (some outputs get duplicated)
                line_hash = hash(line_str)
                if line_hash in logged_lines:
                    continue
                logged_lines.add(line_hash)

                # Log all output
                await broadcast_log("info", line_str)

                # Parse training progress
                if "loss" in line_str.lower():
                    loss_match = re.search(r"loss[:\s]+([0-9.]+)", line_str.lower())
                    if loss_match:
                        training_state["loss"] = float(loss_match.group(1))
                        final_loss = training_state["loss"]

                if "step" in line_str.lower():
                    step_count += 1
                    training_state["current_step"] = i * 100 + min(step_count, 99)

                if "TRAINING_STEP_COMPLETE" in line_str:
                    await broadcast_log("success", f"Completed training on {dataset_name}")

                # Check for errors
                if "error" in line_str.lower() or "exception" in line_str.lower():
                    if "RuntimeError" in line_str or "out of memory" in line_str.lower():
                        step_failed = True

                await broadcast_status()

            await process.wait()

            # Clean up script file
            try:
                script_file.unlink()
            except Exception:
                pass

            # Handle training failure
            if process.returncode != 0 and status != "stopped":
                await broadcast_log("error", f"Training process exited with code {process.returncode}")
                if config.trainingMode == "sequential":
                    await broadcast_log("error", f"Sequential training aborted - step {i+1} failed")
                    status = "failed"
                    break  # Stop sequential training on failure

            # Update current model path for next step (model chaining)
            if process.returncode == 0:
                current_model_path = step_output_dir
                await broadcast_log("success", f"Step {i+1} complete. Model saved to: {step_output_dir}")

            training_state["current_step"] = (i + 1) * 100
            await broadcast_status()

            if status == "stopped":
                break

        if status == "completed":
            await broadcast_log("success", "All training completed successfully!")
            await broadcast_log("info", f"Final model: {current_model_path}")
            await broadcast_log("info", f"Training chain: BASE -> {' -> '.join(config.datasets)}")

    except Exception as e:
        status = "failed"
        await broadcast_log("error", f"Training failed: {str(e)}")
        import traceback
        await broadcast_log("error", traceback.format_exc())
    finally:
        training_state["is_running"] = False
        training_state["process"] = None

        save_training_log(
            job_id=job_id,
            logs=training_state["logs"].copy(),
            config=config.model_dump(),
            status=status,
            final_loss=final_loss,
        )

        await broadcast_status()


@app.post("/api/training/stop")
async def stop_training():
    if not training_state["is_running"]:
        raise HTTPException(status_code=400, detail="No training in progress")

    await broadcast_log("warning", "Stop requested - terminating training process...")

    training_state["is_running"] = False
    if training_state["process"]:
        training_state["process"].terminate()
        training_state["process"] = None
        await broadcast_log("warning", "Training process terminated")

    await broadcast_log("error", "TRAINING STOPPED BY USER")

    return {"message": "Training stopped"}


@app.get("/api/training/logs")
async def get_training_logs(limit: int = 100):
    return {"logs": training_state["logs"][-limit:]}


# Training history endpoints
@app.get("/api/training/history")
async def get_history():
    """Get all training history."""
    return {"history": get_training_history()}


@app.get("/api/training/history/{job_id}")
async def get_history_detail(job_id: str):
    """Get detailed logs for a specific training job."""
    detail = get_training_log_detail(job_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Training log not found")
    return detail


@app.delete("/api/training/history/{job_id}")
async def delete_history(job_id: str):
    """Delete a training history entry."""
    log_file = LOGS_DIR / f"{job_id}.json"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Training log not found")

    log_file.unlink()
    return {"message": f"Training log {job_id} deleted"}


# Model endpoints
def discover_model_directories() -> list[Path]:
    """Discover all model directories from multiple locations."""
    paths = get_paths()
    model_dirs = []

    # Scan trained-models directory
    if paths["model_output"].exists():
        for item in paths["model_output"].iterdir():
            if item.is_dir():
                # Only include directories with config.json (valid model directories)
                if (item / "config.json").exists():
                    model_dirs.append(item)

    # Scan step_* directories in project root
    project_root = paths["pipeline"].parent
    for item in project_root.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            # Check if it contains model files
            if (item / "config.json").exists() or list(item.glob("*.safetensors")):
                model_dirs.append(item)

    return model_dirs


def get_model_info(model_dir: Path) -> ModelCheckpoint:
    """Get model checkpoint info from a directory."""
    # Get directory size
    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_str = f"{size / (1024*1024*1024):.2f} GB" if size > 1e9 else f"{size / (1024*1024):.2f} MB"

    # Try to load training config
    config = {}
    config_file = model_dir / "training_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except Exception:
            pass

    # Try to load model config
    model_config_file = model_dir / "config.json"
    if model_config_file.exists() and not config:
        try:
            with open(model_config_file) as f:
                model_config = json.load(f)
                config["model_type"] = model_config.get("model_type", "unknown")
                config["architectures"] = model_config.get("architectures", [])
        except Exception:
            pass

    # Parse datasets from name
    datasets = []
    if "step_" in model_dir.name:
        parts = model_dir.name.split("_")
        if len(parts) > 2:
            datasets = ["_".join(parts[2:])]

    return ModelCheckpoint(
        name=model_dir.name,
        path=str(model_dir),
        createdAt=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
        datasetsTrained=datasets,
        config=config,
        size=size_str,
    )


@app.get("/api/models", response_model=list[ModelCheckpoint])
async def list_models():
    """List all trained model checkpoints from all locations."""
    models = []

    for model_dir in discover_model_directories():
        try:
            models.append(get_model_info(model_dir))
        except Exception:
            continue

    return sorted(models, key=lambda x: x.createdAt, reverse=True)


@app.get("/api/models/{name}")
async def get_model(name: str):
    paths = get_paths()
    model_dir = paths["model_output"] / name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_str = f"{size / (1024*1024*1024):.2f} GB" if size > 1e9 else f"{size / (1024*1024):.2f} MB"

    config = {}
    config_file = model_dir / "training_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return ModelCheckpoint(
        name=name,
        path=str(model_dir),
        createdAt=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
        datasetsTrained=[],
        config=config,
        size=size_str,
    )


@app.delete("/api/models/{name}")
async def delete_model(name: str):
    paths = get_paths()
    model_dir = paths["model_output"] / name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    shutil.rmtree(model_dir)

    return {"message": f"Model {name} deleted"}


@app.get("/api/models/{name}/download")
async def download_model(name: str):
    paths = get_paths()
    model_dir = paths["model_output"] / name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # Create a zip file
    zip_path = paths["model_output"] / f"{name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', model_dir)

    return FileResponse(
        path=str(zip_path),
        filename=f"{name}.zip",
        media_type="application/zip"
    )


# WebSocket endpoint
@app.websocket("/ws/training")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(None)):
    # Verify token
    if token:
        username = verify_token(token)
        if not username:
            await websocket.close(code=4001)
            return

    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "payload": {
                "isRunning": training_state["is_running"],
                "currentStep": training_state["current_step"],
                "totalSteps": training_state["total_steps"],
                "currentEpoch": training_state["current_epoch"],
                "totalEpochs": training_state["total_epochs"],
                "currentDataset": training_state["current_dataset"],
                "loss": training_state["loss"],
                "progress": 0,
            }
        })

        while True:
            # Keep connection alive
            await websocket.receive_text()

    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_status():
    """Broadcast training status to all connected clients."""
    progress = 0
    if training_state["total_steps"] > 0:
        progress = (training_state["current_step"] / training_state["total_steps"]) * 100

    message = {
        "type": "status",
        "payload": {
            "isRunning": training_state["is_running"],
            "currentStep": training_state["current_step"],
            "totalSteps": training_state["total_steps"],
            "currentEpoch": training_state["current_epoch"],
            "totalEpochs": training_state["total_epochs"],
            "currentDataset": training_state["current_dataset"],
            "loss": training_state["loss"],
            "progress": progress,
        }
    }

    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            pass


async def broadcast_log(level: str, message: str):
    """Broadcast log message to all connected clients."""
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message,
    }

    training_state["logs"].append(log_entry)

    ws_message = {
        "type": "log",
        "payload": log_entry,
    }

    for connection in active_connections:
        try:
            await connection.send_json(ws_message)
        except Exception:
            pass


# ==================== INFERENCE ENDPOINTS ====================

# Pydantic models for inference
class InferenceConfig(BaseModel):
    temperature: float = 1.0
    topK: int = 50
    topP: float = 0.95
    maxLength: int = 1000
    noRepeatNgramSize: int = 2
    doSample: bool = True


class GenerateRequest(BaseModel):
    prompt: str
    config: InferenceConfig = InferenceConfig()


class GenerateLoopRequest(BaseModel):
    prompt: str
    repeatCount: int = 1
    config: InferenceConfig = InferenceConfig()


class GenerateBatchRequest(BaseModel):
    prompts: list[str]
    config: InferenceConfig = InferenceConfig()


class InferenceResult(BaseModel):
    id: str
    prompt: str
    response: str
    generationIndex: int
    timestamp: str
    config: dict


class ExportRequest(BaseModel):
    results: list[InferenceResult]
    format: str = "json"


# Inference state
inference_state = {
    "loaded": False,
    "model_path": None,
    "model": None,
    "tokenizer": None,
    "device": None,
}


@app.get("/api/inference/status")
async def get_inference_status():
    """Get current inference model status."""
    device_info = "N/A"
    if inference_state["device"]:
        device_info = str(inference_state["device"])

    return {
        "loaded": inference_state["loaded"],
        "modelPath": inference_state["model_path"],
        "deviceInfo": device_info,
    }


@app.post("/api/inference/load")
async def load_inference_model(request: dict):
    """Load a model for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = request.get("modelPath")
    if not model_path:
        raise HTTPException(status_code=400, detail="modelPath is required")

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        # Unload existing model if any
        if inference_state["loaded"]:
            inference_state["model"] = None
            inference_state["tokenizer"] = None
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Detect device
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
        except TypeError:
            # fix_mistral_regex not supported for this tokenizer type
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.to(device)

        # Store in state
        inference_state["loaded"] = True
        inference_state["model_path"] = model_path
        inference_state["model"] = model
        inference_state["tokenizer"] = tokenizer
        inference_state["device"] = device

        return {"message": "Model loaded successfully", "modelPath": model_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/inference/unload")
async def unload_inference_model():
    """Unload the current inference model."""
    import torch
    import gc

    if not inference_state["loaded"]:
        return {"message": "No model loaded"}

    inference_state["model"] = None
    inference_state["tokenizer"] = None
    inference_state["loaded"] = False
    inference_state["model_path"] = None

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"message": "Model unloaded"}


def generate_text(prompt: str, config: InferenceConfig) -> str:
    """Generate text from a prompt using the loaded model."""
    import torch

    if not inference_state["loaded"]:
        raise HTTPException(status_code=400, detail="No model loaded")

    model = inference_state["model"]
    tokenizer = inference_state["tokenizer"]

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(inference_state["device"]) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config.maxLength,
            num_return_sequences=1,
            no_repeat_ngram_size=config.noRepeatNgramSize,
            do_sample=config.doSample,
            top_k=config.topK,
            top_p=config.topP,
            temperature=config.temperature,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and remove prompt from output
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        response = full_text[len(prompt):].strip()
    else:
        response = full_text.strip()

    return response


@app.post("/api/inference/generate")
async def generate_single(request: GenerateRequest):
    """Generate a single response."""
    import uuid

    if not inference_state["loaded"]:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        response = generate_text(request.prompt, request.config)

        return InferenceResult(
            id=str(uuid.uuid4()),
            prompt=request.prompt,
            response=response,
            generationIndex=0,
            timestamp=datetime.now().isoformat(),
            config=request.config.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/inference/generate-loop")
async def generate_loop(request: GenerateLoopRequest):
    """Generate multiple responses from the same prompt."""
    import uuid

    if not inference_state["loaded"]:
        raise HTTPException(status_code=400, detail="No model loaded")

    results = []
    try:
        for i in range(request.repeatCount):
            response = generate_text(request.prompt, request.config)

            results.append(InferenceResult(
                id=str(uuid.uuid4()),
                prompt=request.prompt,
                response=response,
                generationIndex=i + 1,
                timestamp=datetime.now().isoformat(),
                config=request.config.model_dump(),
            ))

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/inference/generate-batch")
async def generate_batch(request: GenerateBatchRequest):
    """Generate responses for multiple prompts."""
    import uuid

    if not inference_state["loaded"]:
        raise HTTPException(status_code=400, detail="No model loaded")

    results = []
    try:
        for i, prompt in enumerate(request.prompts):
            response = generate_text(prompt, request.config)

            results.append(InferenceResult(
                id=str(uuid.uuid4()),
                prompt=prompt,
                response=response,
                generationIndex=i + 1,
                timestamp=datetime.now().isoformat(),
                config=request.config.model_dump(),
            ))

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/inference/export")
async def export_results(request: ExportRequest):
    """Export inference results as training data."""
    if not request.results:
        raise HTTPException(status_code=400, detail="No results to export")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if request.format == "json":
        # Export as JSON array of Q&A pairs
        qa_pairs = [
            {"question": r.prompt, "answer": r.response}
            for r in request.results
        ]
        return {
            "data": qa_pairs,
            "filename": f"inference_export_{timestamp}.json"
        }
    else:
        # Export as text format
        lines = []
        for r in request.results:
            lines.append(f"{r.prompt}\n {r.response}\n")

        return {
            "data": "\n".join(lines),
            "filename": f"inference_export_{timestamp}.text"
        }


# ==================== PROFILING SYSTEM ENDPOINTS ====================

# Pydantic models for profiling
class ProfiledGenerateRequest(BaseModel):
    prompt: str
    model_path: str
    profiling_depth: str = "module"  # "module" or "deep"
    tags: Optional[list[str]] = None
    experiment_name: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_length: int = 100
    batch_size: int = 1  # Batch size for throughput analysis


@app.post("/api/profiling/generate")
async def profiled_generate(request: ProfiledGenerateRequest):
    """
    Generate text with full energy profiling.

    This endpoint performs inference with comprehensive profiling of:
    - Power consumption (CPU, GPU, ANE, DRAM)
    - Layer and component timing
    - Activation statistics
    - Deep operation metrics (if profilingDepth='deep')

    Returns the run_id for querying profiling data.
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from profiling.power_monitor import PowerMonitor
    from profiling.layer_profiler import LayerProfiler
    from profiling.deep_profiler import DeepAttentionProfiler
    from profiling.database import ProfileDatabase
    from profiling.pipeline_profiler import InferencePipelineProfiler
    from profiling.model_detector import is_streaming_compatible
    from profiling.model_features import extract_model_features

    # Capture the main event loop for thread-safe async operations
    main_loop = asyncio.get_running_loop()

    model_dir = Path(request.model_path)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")

    # Track which profiling components are available
    warnings = []

    try:
        # Initialize profiling components with graceful fallbacks
        power_monitor = None
        try:
            power_monitor = PowerMonitor(sample_interval_ms=100)
            if not power_monitor.is_available():
                logger.warning("powermetrics not available - power profiling disabled")
                warnings.append("Power profiling unavailable: powermetrics not configured")
                power_monitor = None
        except Exception as e:
            logger.warning(f"Failed to initialize PowerMonitor: {e}")
            warnings.append(f"Power profiling unavailable: {str(e)}")
            power_monitor = None

        # Detect device
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Send model loading start event
        model_name = Path(request.model_path).name
        await profiling_manager.broadcast({
            "type": ProfilingMessageType.MODEL_LOADING,
            "timestamp": time.time() * 1000,
            "data": {
                "status": "loading",
                "model_name": model_name,
                "model_path": request.model_path,
                "message": f"Loading tokenizer for {model_name}..."
            }
        })

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(request.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        await profiling_manager.broadcast({
            "type": ProfilingMessageType.MODEL_LOADING,
            "timestamp": time.time() * 1000,
            "data": {
                "status": "loading",
                "model_name": model_name,
                "model_path": request.model_path,
                "message": f"Loading model weights for {model_name}..."
            }
        })

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            request.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        await profiling_manager.broadcast({
            "type": ProfilingMessageType.MODEL_LOADING,
            "timestamp": time.time() * 1000,
            "data": {
                "status": "loading",
                "model_name": model_name,
                "model_path": request.model_path,
                "message": f"Moving model to {device}..."
            }
        })

        model.to(device)

        await profiling_manager.broadcast({
            "type": ProfilingMessageType.MODEL_LOADING,
            "timestamp": time.time() * 1000,
            "data": {
                "status": "complete",
                "model_name": model_name,
                "model_path": request.model_path,
                "message": f"Model {model_name} loaded successfully"
            }
        })

        # Extract model features for database storage (BUG-033)
        try:
            model_features = extract_model_features(model, model_name)
            logger.info(f"Extracted model features: {model_features.architecture_type} with {model_features.total_params:,} parameters")
        except Exception as e:
            logger.warning(f"Failed to extract model features: {e}")
            model_features = None

        # Initialize profilers with graceful fallbacks
        layer_profiler = None
        try:
            layer_profiler = LayerProfiler(model)
            logger.info("LayerProfiler initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LayerProfiler: {e}")
            warnings.append(f"Layer profiling unavailable: {str(e)}")
            layer_profiler = None

        deep_profiler = None
        if request.profiling_depth == "deep":
            try:
                deep_profiler = DeepAttentionProfiler(model)
                deep_profiler.patch()
                logger.info("DeepAttentionProfiler initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepAttentionProfiler: {e}")
                warnings.append(f"Deep profiling unavailable: {str(e)}")
                deep_profiler = None

        database = ProfileDatabase()

        # Define power sample callback for WebSocket streaming
        def stream_power_sample(sample):
            """Callback to stream power samples via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.POWER_SAMPLE,
                    "timestamp": sample.relative_time_ms,  # Use relative time for frontend charts
                    "data": {
                        "timestamp": sample.relative_time_ms,  # Time since profiling start (ms)
                        "cpu_power_mw": sample.cpu_power_mw,
                        "gpu_power_mw": sample.gpu_power_mw,
                        "ane_power_mw": sample.ane_power_mw,
                        "dram_power_mw": sample.dram_power_mw,
                        "total_power_mw": sample.total_power_mw,
                        "phase": sample.phase
                    }
                }
                # Broadcast to all connected WebSocket clients (thread-safe)
                asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(message), main_loop)
            except Exception as e:
                logger.error(f"Failed to stream power sample: {e}")

        # Define section event callback for WebSocket streaming
        def stream_section_event(event_type, phase, section_name, timestamp, data):
            """Callback to stream section start/end events via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.SECTION_START if event_type == "section_start" else ProfilingMessageType.SECTION_END,
                    "timestamp": timestamp,
                    "data": {
                        "phase": phase,
                        "section_name": section_name,
                        **data
                    }
                }
                # Broadcast to all connected WebSocket clients (thread-safe)
                asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(message), main_loop)
            except Exception as e:
                logger.error(f"Failed to stream section event: {e}")

        # Define token complete callback for WebSocket streaming
        def stream_token_complete(token_data):
            """Callback to stream token completion events via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.TOKEN_COMPLETE,
                    "timestamp": token_data.get("timestamp"),
                    "data": {
                        "token_position": token_data.get("token_index"),  # Frontend expects token_position
                        "token_text": token_data.get("token_text"),
                        "duration_ms": token_data.get("duration_ms"),
                        "energy_mj": token_data.get("energy_mj"),
                        "avg_power_mw": token_data.get("avg_power_mw"),
                        "power_snapshot": token_data.get("power_snapshot"),
                        "layer_metrics_summary": token_data.get("layer_metrics_summary"),
                        "layer_metrics": token_data.get("layer_metrics", [])  # Add full layer-by-layer data
                    }
                }
                # Broadcast to all connected WebSocket clients (thread-safe)
                asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(message), main_loop)
            except Exception as e:
                logger.error(f"Failed to stream token complete event: {e}")

        # Define layer metrics callback for WebSocket streaming
        def stream_layer_metrics(layer_metrics_data):
            """Callback to stream layer metrics events via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.LAYER_METRICS,
                    "timestamp": layer_metrics_data.get("timestamp"),
                    "data": {
                        "token_position": layer_metrics_data.get("token_index"),  # Frontend expects token_position
                        "layer_index": layer_metrics_data.get("layer_index"),
                        "layer_name": layer_metrics_data.get("layer_name"),
                        "total_duration_ms": layer_metrics_data.get("total_duration_ms"),
                        "num_components": layer_metrics_data.get("num_components"),
                        "activation_stats": layer_metrics_data.get("activation_stats"),
                        "components": layer_metrics_data.get("components")
                    }
                }
                # Broadcast to all connected WebSocket clients (thread-safe)
                asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(message), main_loop)
            except Exception as e:
                logger.error(f"Failed to stream layer metrics event: {e}")

        # Define component metrics callback for WebSocket streaming
        def stream_component_metrics(component_metrics_data):
            """Callback to stream component metrics events via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.COMPONENT_METRICS,
                    "timestamp": component_metrics_data.get("timestamp"),
                    "data": {
                        "token_position": component_metrics_data.get("token_index"),  # Frontend expects token_position
                        "layer_index": component_metrics_data.get("layer_index"),
                        "component_name": component_metrics_data.get("component_name"),
                        "module_path": component_metrics_data.get("module_path"),
                        "duration_ms": component_metrics_data.get("duration_ms"),
                        "activation_stats": component_metrics_data.get("activation_stats")
                    }
                }
                # Broadcast to all connected WebSocket clients (thread-safe)
                asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(message), main_loop)
            except Exception as e:
                logger.error(f"Failed to stream component metrics event: {e}")

        # Define inference complete callback for WebSocket streaming
        def stream_inference_complete(inference_complete_data):
            """Callback to stream inference complete event via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.INFERENCE_COMPLETE,
                    "timestamp": inference_complete_data.get("timestamp"),
                    "data": {
                        "run_id": inference_complete_data.get("run_id"),
                        "total_duration_ms": inference_complete_data.get("total_duration_ms"),
                        "total_energy_mj": inference_complete_data.get("total_energy_mj"),
                        "token_count": inference_complete_data.get("token_count"),
                        "tokens_per_second": inference_complete_data.get("tokens_per_second"),
                        "summary_statistics": inference_complete_data.get("summary_statistics")
                    }
                }
                # Broadcast to all connected WebSocket clients (thread-safe)
                asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(message), main_loop)
            except Exception as e:
                logger.error(f"Failed to stream inference complete event: {e}")

        # Create pipeline profiler with streaming callbacks
        profiler = InferencePipelineProfiler(
            power_monitor=power_monitor,
            layer_profiler=layer_profiler,
            deep_profiler=deep_profiler,
            database=database,
            power_sample_callback=stream_power_sample,
            section_event_callback=stream_section_event,
            token_complete_callback=stream_token_complete,
            layer_metrics_callback=stream_layer_metrics,
            component_metrics_callback=stream_component_metrics,
            inference_complete_callback=stream_inference_complete
        )

        # Start profiling session
        with profiler.run(
            prompt=request.prompt,
            model_name=model_dir.name,
            profiling_depth=request.profiling_depth,
            experiment_name=request.experiment_name,
            tags=request.tags,
            batch_size=request.batch_size,
            model=model
        ) as session:
            # Store model features in session for database (BUG-033)
            if model_features:
                session.num_layers = model_features.num_layers
                session.hidden_size = model_features.hidden_size
                session.intermediate_size = model_features.intermediate_size
                session.num_attention_heads = model_features.num_attention_heads
                session.num_key_value_heads = model_features.num_key_value_heads
                session.total_params = model_features.total_params
                session.attention_mechanism = model_features.attention_mechanism
                session.is_moe = model_features.is_moe
                session.num_experts = model_features.num_experts
                session.num_active_experts = model_features.num_active_experts
                session.architecture_type = model_features.architecture_type

            # Pre-inference phase
            with session.section("tokenization", phase="pre_inference"):
                # Duplicate prompt for batch processing
                prompts = [request.prompt] * request.batch_size
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                # Track input token count for accurate per-token energy analysis (per sample)
                session.input_token_count = len(inputs['input_ids'][0])

            with session.section("tensor_transfer", phase="pre_inference"):
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Check if model supports streaming generation
            supports_streaming = is_streaming_compatible(model)

            if supports_streaming:
                # Use streaming generation for per-token profiling
                from transformers import TextIteratorStreamer
                import threading

                # Note: TextIteratorStreamer works with batch_size=1 per stream
                # For batch_size > 1, we process the first sequence in the batch
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": request.max_length,
                    "num_return_sequences": 1,  # Per input in batch
                    "no_repeat_ngram_size": 2,
                    "do_sample": request.temperature > 0,
                    "top_k": 50,
                    "top_p": request.top_p,
                    "temperature": max(request.temperature, 0.01),
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "streamer": streamer,
                }

                # Run generation in a separate thread
                generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)

                # Prefill phase
                with session.section("prefill", phase="prefill"):
                    generation_thread.start()

                # Decode phase - stream tokens one by one
                generated_tokens = []
                token_index = 0
                last_token_time = None  # Track time of last token for proper duration measurement

                with session.section("decode", phase="decode"):
                    for token_text in streamer:
                        if token_text:
                            current_token_time = time.time()

                            # Calculate actual token generation duration
                            # Duration is time between consecutive token arrivals
                            if last_token_time is None:
                                # First token: duration from start of decode section
                                token_duration_ms = (current_token_time - time.time()) * 1000
                                # Better estimate: assume decode section started just before first token
                                token_duration_ms = 50.0  # Reasonable default for first token
                            else:
                                # Subsequent tokens: time since last token
                                token_duration_ms = (current_token_time - last_token_time) * 1000

                            generated_tokens.append(token_text)

                            # Get current power for energy estimation
                            current_power_mw = 0.0
                            if profiler.power_monitor:
                                current_sample = profiler.power_monitor.get_current()
                                if current_sample:
                                    current_power_mw = current_sample.total_power_mw

                            # Estimate energy for this token (power * time)
                            token_energy_mj = current_power_mw * token_duration_ms / 1000.0

                            # Emit token event via WebSocket
                            profiler.emit_token_complete_event(
                                session=session,
                                token_index=token_index,
                                token_text=token_text,
                                duration_ms=token_duration_ms,
                                energy_mj=token_energy_mj,
                                avg_power_mw=current_power_mw
                            )

                            last_token_time = current_token_time
                            token_index += 1

                # Wait for generation to complete
                generation_thread.join()

                # Track output token count for accurate per-token energy analysis
                # For batch processing, track total tokens across all sequences
                session.output_token_count = len(generated_tokens) * request.batch_size
            else:
                # Use non-streaming generation for incompatible models (e.g., StableLM)
                logger.info("Using non-streaming generation for model compatibility")
                warnings.append("Using non-streaming mode due to model architecture limitations")

                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": request.max_length,
                    "num_return_sequences": 1,  # Per input in batch
                    "no_repeat_ngram_size": 2,
                    "do_sample": request.temperature > 0,
                    "top_k": 50,
                    "top_p": request.top_p,
                    "temperature": max(request.temperature, 0.01),
                    "use_cache": False,  # Disable cache for problematic models
                    "pad_token_id": tokenizer.pad_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }

                generated_tokens = []

                # Prefill + Decode phase combined (non-streaming)
                with session.section("prefill", phase="prefill"):
                    generation_start = time.time()

                with session.section("decode", phase="decode"):
                    outputs = model.generate(**generation_kwargs)
                    generation_end = time.time()

                    # Extract generated token IDs (excluding input) for all sequences in batch
                    input_length = inputs["input_ids"].shape[1]

                    # Process all sequences in the batch
                    all_generated_texts = []
                    total_tokens_in_batch = 0

                    for batch_idx in range(request.batch_size):
                        generated_ids = outputs.sequences[batch_idx][input_length:]
                        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        all_generated_texts.append(response_text)
                        total_tokens_in_batch += len(generated_ids)

                    generated_tokens = all_generated_texts

                    # Calculate average time per token for reporting (across batch)
                    total_duration_ms = (generation_end - generation_start) * 1000
                    avg_token_duration_ms = total_duration_ms / max(total_tokens_in_batch, 1)

                    # Get average power
                    avg_power_mw = 0.0
                    if profiler.power_monitor:
                        current_sample = profiler.power_monitor.get_current()
                        if current_sample:
                            avg_power_mw = current_sample.total_power_mw

                    # Emit a single aggregated token event
                    profiler.emit_token_complete_event(
                        session=session,
                        token_index=0,
                        token_text="\n---\n".join(all_generated_texts),  # Combine batch outputs
                        duration_ms=total_duration_ms,
                        energy_mj=avg_power_mw * total_duration_ms / 1000.0,
                        avg_power_mw=avg_power_mw
                    )

                    # Track output token count for accurate per-token energy analysis (total across batch)
                    session.output_token_count = total_tokens_in_batch

            # Post-inference phase
            # For batch processing, join outputs with separators
            if request.batch_size > 1:
                response = "\n---\n".join(generated_tokens)
            else:
                response = "".join(generated_tokens)

            with session.section("detokenization", phase="post_inference"):
                session.response = response

            # Calculate KV cache size after generation (BUG-032)
            # KV cache stores key and value tensors for each layer and token
            try:
                if hasattr(model.config, 'num_hidden_layers') and hasattr(model.config, 'hidden_size'):
                    num_layers = model.config.num_hidden_layers
                    hidden_size = model.config.hidden_size
                    num_heads = getattr(model.config, 'num_attention_heads', 0)
                    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)  # For GQA models

                    # Calculate head dimension
                    if num_heads > 0:
                        head_dim = hidden_size // num_heads
                    else:
                        head_dim = 0

                    # Total sequence length (input + output tokens)
                    total_seq_len = (session.input_token_count or 0) + (session.output_token_count or 0)

                    # KV cache size calculation:
                    # - num_layers: number of transformer layers
                    # - 2: separate cache for keys and values
                    # - total_seq_len: cache grows with sequence length
                    # - num_kv_heads: number of key-value heads (may differ from query heads in GQA)
                    # - head_dim: dimension of each head
                    # - dtype_size: bytes per element (2 for FP16, 4 for FP32)

                    # Determine dtype size from model
                    dtype_size = 2  # Default to FP16
                    if hasattr(model, 'dtype'):
                        if model.dtype == torch.float32:
                            dtype_size = 4
                        elif model.dtype == torch.float16 or model.dtype == torch.bfloat16:
                            dtype_size = 2

                    # Calculate total KV cache size in bytes (multiplied by batch_size)
                    kv_cache_size_bytes = num_layers * 2 * total_seq_len * num_kv_heads * head_dim * dtype_size * request.batch_size

                    # Convert to megabytes
                    kv_cache_size_mb = kv_cache_size_bytes / (1024 * 1024)

                    # Store in session for database
                    session.kv_cache_size_mb = kv_cache_size_mb
                    session.context_length = total_seq_len

                    logger.info(f"Calculated KV cache size: {kv_cache_size_mb:.2f} MB for {total_seq_len} tokens across {num_layers} layers")
                else:
                    logger.warning("Could not calculate KV cache size: missing model config attributes")
                    session.kv_cache_size_mb = None
                    session.context_length = None
            except Exception as e:
                logger.warning(f"Failed to calculate KV cache size: {e}")
                session.kv_cache_size_mb = None
                session.context_length = None

        # Data is automatically saved to database via profiler.run context manager
        # Cleanup is handled automatically by the context manager's finally block
        result = {
            "runId": session.run_id,
            "response": response,
            "message": "Profiled inference completed successfully"
        }

        # Include warnings if any profiling components failed
        if warnings:
            result["warnings"] = warnings
            result["message"] = "Inference completed with limited profiling data"

        return result

    except Exception as e:
        logger.error(f"Profiled generation failed: {str(e)}")

        # Notify frontend of profiling error via WebSocket
        try:
            error_message = {
                "type": ProfilingMessageType.ERROR,
                "timestamp": time.time() * 1000,
                "data": {
                    "error": str(e),
                    "message": "Profiling failed"
                }
            }
            asyncio.run_coroutine_threadsafe(profiling_manager.broadcast(error_message), main_loop)
        except Exception as ws_error:
            logger.error(f"Failed to send error notification via WebSocket: {ws_error}")

        raise HTTPException(status_code=500, detail=f"Profiled generation failed: {str(e)}")


@app.get("/api/profiling/powermetrics/status")
async def get_powermetrics_status():
    """Get powermetrics availability status."""
    from profiling.utils import get_powermetrics_status
    return get_powermetrics_status()


@app.get("/api/profiling/runs")
async def get_profiling_runs(
    model: Optional[str] = Query(None, description="Filter by model name"),
    date_from: Optional[str] = Query(None, description="Filter runs from this timestamp (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter runs up to this timestamp (ISO format)"),
    tags: Optional[str] = Query(None, description="Filter by comma-separated tags"),
    experiment: Optional[str] = Query(None, description="Filter by experiment name"),
    inference_engine: Optional[str] = Query(None, description="Filter by inference engine/backend"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    sort_by: str = Query("joules_per_token", description="Sort by: date, duration, energy, joules_per_token (default)"),
):
    """
    List profiling runs with optional filtering and pagination.

    Query parameters:
    - model: Filter by model name
    - date_from: Filter runs from this timestamp (ISO format)
    - date_to: Filter runs up to this timestamp (ISO format)
    - tags: Filter by comma-separated tags
    - experiment: Filter by experiment name
    - inference_engine: Filter by inference engine/backend (transformers, mlx, vllm, etc.)
    - limit: Maximum number of results (1-1000, default 100)
    - offset: Number of results to skip for pagination (default 0)
    - sort_by: Sort by date, duration, energy, or joules_per_token (default)

    Returns list with summary metrics per run including J/t (Joules per token) as primary efficiency metric.
    """
    from profiling.database import ProfileDatabase

    try:
        database = ProfileDatabase()

        # Get runs with filters
        runs = database.get_runs(
            model=model,
            date_from=date_from,
            date_to=date_to,
            tags=tags,
            experiment=experiment,
            inference_engine=inference_engine,
            limit=limit,
            offset=offset,
        )

        # Add summary metrics for each run
        runs_with_summary = []
        for run in runs:
            run_id = run["run_id"]

            # Get basic summary stats from database
            summary = database.get_run_summary(run_id)

            if summary:
                # Calculate total metrics from phase breakdown
                total_duration_ms = sum(
                    phase.get("total_duration_ms", 0) or 0
                    for phase in summary.get("phase_breakdown", [])
                )
                total_energy_mj = sum(
                    phase.get("total_energy_mj", 0) or 0
                    for phase in summary.get("phase_breakdown", [])
                )

                # Extract J/t metrics from efficiency_metrics
                efficiency = summary.get("efficiency_metrics", {})
                joules_per_token = efficiency.get("joules_per_token", 0)
                joules_per_input_token = efficiency.get("joules_per_input_token", 0)
                joules_per_output_token = efficiency.get("joules_per_output_token", 0)
                tokens_per_joule = efficiency.get("tokens_per_joule", 0)

                runs_with_summary.append({
                    "id": run["run_id"],  # Frontend expects 'id' field
                    "run_id": run["run_id"],  # Keep for backward compatibility
                    "timestamp": run["timestamp"],
                    "model_name": run["model_name"],
                    "prompt": run["prompt"],
                    "response": run.get("response"),
                    "experiment_name": run.get("experiment_name"),
                    "tags": run.get("tags"),
                    "profiling_depth": run.get("profiling_depth"),
                    "inference_engine": run.get("inference_engine"),
                    "status": run.get("status"),
                    "total_duration_ms": total_duration_ms,
                    "total_energy_mj": total_energy_mj,
                    # Frontend expects both naming conventions
                    "input_tokens": run.get("input_token_count"),
                    "output_tokens": run.get("output_token_count"),
                    "input_token_count": run.get("input_token_count"),
                    "output_token_count": run.get("output_token_count"),
                    "token_count": run.get("token_count"),
                    "tokens_per_second": run.get("tokens_per_second"),
                    # J/t metrics (primary efficiency metrics)
                    "joules_per_token": joules_per_token,
                    "joules_per_input_token": joules_per_input_token,
                    "joules_per_output_token": joules_per_output_token,
                    "tokens_per_joule": tokens_per_joule,
                })
            else:
                # Fallback if summary is not available
                runs_with_summary.append({
                    "id": run["run_id"],  # Frontend expects 'id' field
                    "run_id": run["run_id"],  # Keep for backward compatibility
                    "timestamp": run["timestamp"],
                    "model_name": run["model_name"],
                    "prompt": run["prompt"],
                    "response": run.get("response"),
                    "experiment_name": run.get("experiment_name"),
                    "tags": run.get("tags"),
                    "profiling_depth": run.get("profiling_depth"),
                    "inference_engine": run.get("inference_engine"),
                    "status": run.get("status"),
                    "total_duration_ms": None,
                    "total_energy_mj": None,
                    # Frontend expects both naming conventions
                    "input_tokens": None,
                    "output_tokens": None,
                    "input_token_count": None,
                    "output_token_count": None,
                    "token_count": None,
                    "tokens_per_second": None,
                    "joules_per_token": None,
                    "joules_per_input_token": None,
                    "joules_per_output_token": None,
                    "tokens_per_joule": None,
                })

        # Sort results if requested
        if sort_by == "duration" and runs_with_summary:
            runs_with_summary.sort(
                key=lambda x: x.get("total_duration_ms") or 0,
                reverse=True
            )
        elif sort_by == "energy" and runs_with_summary:
            runs_with_summary.sort(
                key=lambda x: x.get("total_energy_mj") or 0,
                reverse=True
            )
        elif sort_by == "joules_per_token" and runs_with_summary:
            # Sort by J/t (lower is better = more efficient)
            runs_with_summary.sort(
                key=lambda x: x.get("joules_per_token") or float('inf'),
                reverse=False
            )
        elif sort_by == "date" and runs_with_summary:
            # Sort by date (already handled by database, but apply here for consistency)
            runs_with_summary.sort(
                key=lambda x: x.get("timestamp") or "",
                reverse=True
            )
        # Default: sort by joules_per_token (lower is better)
        elif runs_with_summary:
            runs_with_summary.sort(
                key=lambda x: x.get("joules_per_token") or float('inf'),
                reverse=False
            )

        return {
            "runs": runs_with_summary,
            "total": len(runs_with_summary),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Failed to retrieve profiling runs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profiling runs: {str(e)}")


@app.get("/api/profiling/run/{run_id}")
async def get_profiling_run(run_id: str):
    """
    Get full profiling run data including all nested metrics.

    This endpoint returns complete profiling data for a single run:
    - Run metadata (model, prompt, response, timestamps)
    - All power samples with full timeline
    - All pipeline sections with timing and energy
    - All tokens with per-token metrics
    - Layer metrics nested under each token
    - Component metrics nested under each layer
    - Deep operation metrics (if profiling_depth='deep')

    Args:
        run_id: Unique identifier for the profiling run

    Returns:
        Complete nested profiling data structure
    """
    from profiling.database import ProfileDatabase

    try:
        database = ProfileDatabase()
        database.connect()

        # Get basic run data
        run = database.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Profiling run {run_id} not found")

        # Get all power samples
        power_samples = database.get_power_timeline(run_id)

        # Get all pipeline sections
        cursor = database.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM pipeline_sections
            WHERE run_id = ?
            ORDER BY start_time_ms
            """,
            (run_id,)
        )
        pipeline_sections = [dict(row) for row in cursor.fetchall()]

        # Get all tokens with nested layer and component metrics
        tokens = database.get_tokens(run_id)

        # For each token, get layer metrics
        for token in tokens:
            token_id = token["id"]
            layer_metrics = database.get_layer_metrics(token_id)

            # For each layer, get component metrics
            for layer in layer_metrics:
                layer_metric_id = layer["id"]
                component_metrics = database.get_component_metrics(layer_metric_id)

                # For each component, get deep operation metrics if they exist
                for component in component_metrics:
                    component_metric_id = component["id"]
                    cursor.execute(
                        """
                        SELECT * FROM deep_operation_metrics
                        WHERE component_metric_id = ?
                        ORDER BY operation_name
                        """,
                        (component_metric_id,)
                    )
                    deep_operations = [dict(row) for row in cursor.fetchall()]
                    component["deep_operations"] = deep_operations

                layer["components"] = component_metrics

            token["layers"] = layer_metrics

        database.close()

        # Build complete response structure
        return {
            "run": run,
            "power_samples": power_samples,
            "pipeline_sections": pipeline_sections,
            "tokens": tokens,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve profiling run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profiling run: {str(e)}")


@app.get("/api/profiling/run/{run_id}/summary")
async def get_profiling_run_summary(run_id: str):
    """
    Get aggregated summary statistics for a profiling run.

    This endpoint returns summary metrics including:
    - Total duration and energy
    - Per-phase breakdown (pre_inference, prefill, decode, post_inference)
    - Average metrics per layer (across all tokens)
    - Average metrics per component (across all layers and tokens)
    - Hottest components (top 10 by energy consumption)

    Args:
        run_id: Unique identifier for the profiling run

    Returns:
        Summary statistics dictionary with aggregated metrics
    """
    from profiling.database import ProfileDatabase

    try:
        database = ProfileDatabase()
        database.connect()

        # Get summary from database (which calculates all required metrics)
        summary = database.get_run_summary(run_id)

        database.close()

        if not summary:
            raise HTTPException(status_code=404, detail=f"Profiling run {run_id} not found")

        # Calculate total duration and energy from phase breakdown
        total_duration_ms = 0
        total_energy_mj = 0
        if "phase_breakdown" in summary:
            for phase in summary["phase_breakdown"]:
                total_duration_ms += phase.get("total_duration_ms", 0) or 0
                total_energy_mj += phase.get("total_energy_mj", 0) or 0

        # Add calculated totals to summary
        summary["total_duration_ms"] = total_duration_ms
        summary["total_energy_mj"] = total_energy_mj

        # Calculate derived metrics
        if total_duration_ms > 0:
            summary["avg_power_mw"] = (total_energy_mj / total_duration_ms) * 1000 if total_duration_ms > 0 else 0

        # Add token counts if available
        token_count = summary.get("token_count", 0)
        if token_count and total_duration_ms > 0:
            summary["tokens_per_second"] = (token_count / total_duration_ms) * 1000

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve profiling run summary {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profiling run summary: {str(e)}")


@app.get("/api/profiling/run/{run_id}/pipeline")
async def get_profiling_pipeline_breakdown(run_id: str):
    """
    Get hierarchical pipeline section breakdown for a profiling run.

    This endpoint returns detailed phase > section breakdown with timing and energy data.
    Sections are organized hierarchically by phase (pre_inference, prefill, decode, post_inference).

    Each section includes:
    - Timing (duration_ms, start_time_ms, end_time_ms)
    - Energy (energy_mj)
    - Power (avg_power_mw)
    - Percentage of total duration and energy

    Args:
        run_id: Unique identifier for the profiling run

    Returns:
        Hierarchical breakdown with phases and sections
    """
    from profiling.database import ProfileDatabase

    try:
        database = ProfileDatabase()
        database.connect()

        # Get all pipeline sections
        sections = database.get_pipeline_sections(run_id)

        database.close()

        if not sections:
            raise HTTPException(status_code=404, detail=f"No pipeline data found for run {run_id}")

        # Calculate totals for percentage calculations
        total_duration_ms = sum(s.get("duration_ms", 0) or 0 for s in sections)
        total_energy_mj = sum(s.get("energy_mj", 0) or 0 for s in sections)

        # Group sections by phase
        phases = {}
        for section in sections:
            phase = section["phase"]
            if phase not in phases:
                phases[phase] = {
                    "phase": phase,
                    "sections": [],
                    "total_duration_ms": 0,
                    "total_energy_mj": 0,
                    "total_avg_power_mw": 0,
                    "section_count": 0,
                }

            # Calculate percentages for this section
            duration_pct = (section.get("duration_ms", 0) / total_duration_ms * 100) if total_duration_ms > 0 else 0
            energy_pct = (section.get("energy_mj", 0) / total_energy_mj * 100) if total_energy_mj > 0 else 0

            # Add section with percentages
            section_with_pct = {
                **section,
                "duration_percentage": round(duration_pct, 2),
                "energy_percentage": round(energy_pct, 2),
            }
            phases[phase]["sections"].append(section_with_pct)

            # Accumulate phase totals
            phases[phase]["total_duration_ms"] += section.get("duration_ms", 0) or 0
            phases[phase]["total_energy_mj"] += section.get("energy_mj", 0) or 0
            phases[phase]["section_count"] += 1

        # Calculate phase percentages and averages
        for phase_data in phases.values():
            phase_data["duration_percentage"] = round(
                (phase_data["total_duration_ms"] / total_duration_ms * 100) if total_duration_ms > 0 else 0,
                2
            )
            phase_data["energy_percentage"] = round(
                (phase_data["total_energy_mj"] / total_energy_mj * 100) if total_energy_mj > 0 else 0,
                2
            )
            if phase_data["total_duration_ms"] > 0:
                phase_data["avg_power_mw"] = (phase_data["total_energy_mj"] / phase_data["total_duration_ms"]) * 1000

        # Return structured response
        return {
            "run_id": run_id,
            "total_duration_ms": total_duration_ms,
            "total_energy_mj": total_energy_mj,
            "phases": list(phases.values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve pipeline breakdown for run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pipeline breakdown: {str(e)}")


@app.get("/api/profiling/export/{run_id}")
async def export_profiling_run(
    run_id: str,
    format: str = Query("json", regex="^(json|csv)$")
):
    """
    Export profiling run data in JSON or CSV format.

    Args:
        run_id: Unique identifier of the profiling run
        format: Export format - "json" (default) or "csv"

    Returns:
        - JSON format: Complete nested data structure with all metrics
        - CSV format: Flattened tables (run metadata + power samples + sections)

    Response headers:
        - Content-Type: application/json or text/csv
        - Content-Disposition: attachment; filename="profiling_run_{run_id}.{format}"
    """
    import csv
    import io
    from fastapi.responses import StreamingResponse

    db = None
    try:
        # Initialize database connection
        db = ProfileDatabase()

        # Verify run exists
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Profiling run {run_id} not found")

        if format == "json":
            # JSON format: Full nested structure (same as detail endpoint)
            power_samples = db.get_power_timeline(run_id)

            # Get pipeline sections
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT * FROM pipeline_sections
                WHERE run_id = ?
                ORDER BY start_time_ms
            """, (run_id,))
            pipeline_sections = [dict(row) for row in cursor.fetchall()]

            # Get tokens with nested layer and component metrics
            tokens = db.get_tokens(run_id)
            for token in tokens:
                token_id = token['id']

                # Get layer metrics for this token
                layer_metrics = db.get_layer_metrics(token_id)
                for layer in layer_metrics:
                    layer_metric_id = layer['id']

                    # Get component metrics for this layer
                    component_metrics = db.get_component_metrics(layer_metric_id)
                    for component in component_metrics:
                        component_id = component['id']

                        # Get deep operation metrics for this component
                        cursor.execute("""
                            SELECT * FROM deep_operation_metrics
                            WHERE component_metric_id = ?
                            ORDER BY operation_name
                        """, (component_id,))
                        deep_operations = [dict(row) for row in cursor.fetchall()]
                        component['deep_operations'] = deep_operations

                    layer['components'] = component_metrics

                token['layers'] = layer_metrics

            # Build complete export data
            export_data = {
                "run": run,
                "power_samples": power_samples,
                "pipeline_sections": pipeline_sections,
                "tokens": tokens
            }

            # Convert to JSON string
            import json
            json_content = json.dumps(export_data, indent=2, default=str)

            # Create streaming response
            return StreamingResponse(
                io.BytesIO(json_content.encode('utf-8')),
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="profiling_run_{run_id}.json"'
                }
            )

        else:  # format == "csv"
            # CSV format: Flattened tables in a single CSV file
            # Structure: Run metadata, then power samples, then sections

            output = io.StringIO()
            writer = csv.writer(output)

            # Section 1: Run Metadata
            writer.writerow(["### RUN METADATA ###"])
            writer.writerow(["Field", "Value"])
            for key, value in run.items():
                writer.writerow([key, value])
            writer.writerow([])  # Blank line

            # Section 2: Power Samples
            writer.writerow(["### POWER SAMPLES ###"])
            power_samples = db.get_power_timeline(run_id)
            if power_samples:
                # Header row
                writer.writerow(power_samples[0].keys())
                # Data rows
                for sample in power_samples:
                    writer.writerow(sample.values())
            writer.writerow([])  # Blank line

            # Section 3: Pipeline Sections
            writer.writerow(["### PIPELINE SECTIONS ###"])
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT * FROM pipeline_sections
                WHERE run_id = ?
                ORDER BY start_time_ms
            """, (run_id,))
            pipeline_sections = [dict(row) for row in cursor.fetchall()]
            if pipeline_sections:
                # Header row
                writer.writerow(pipeline_sections[0].keys())
                # Data rows
                for section in pipeline_sections:
                    writer.writerow(section.values())
            writer.writerow([])  # Blank line

            # Section 4: Tokens (if any)
            writer.writerow(["### TOKENS ###"])
            tokens = db.get_tokens(run_id)
            if tokens:
                # Header row
                writer.writerow(tokens[0].keys())
                # Data rows
                for token in tokens:
                    writer.writerow(token.values())
            writer.writerow([])  # Blank line

            # Section 5: Layer Metrics Summary (flattened)
            writer.writerow(["### LAYER METRICS SUMMARY ###"])
            cursor.execute("""
                SELECT lm.*, t.token_index, t.token_text
                FROM layer_metrics lm
                LEFT JOIN tokens t ON lm.token_id = t.id
                WHERE lm.run_id = ?
                ORDER BY t.token_index, lm.layer_index
            """, (run_id,))
            layer_metrics = [dict(row) for row in cursor.fetchall()]
            if layer_metrics:
                # Header row
                writer.writerow(layer_metrics[0].keys())
                # Data rows
                for metric in layer_metrics:
                    writer.writerow(metric.values())
            writer.writerow([])  # Blank line

            # Section 6: Component Metrics Summary (flattened)
            writer.writerow(["### COMPONENT METRICS SUMMARY ###"])
            cursor.execute("""
                SELECT cm.*, lm.layer_index, t.token_index
                FROM component_metrics cm
                LEFT JOIN layer_metrics lm ON cm.layer_metric_id = lm.id
                LEFT JOIN tokens t ON lm.token_id = t.id
                WHERE lm.run_id = ?
                ORDER BY t.token_index, lm.layer_index, cm.component_name
            """, (run_id,))
            component_metrics = [dict(row) for row in cursor.fetchall()]
            if component_metrics:
                # Header row
                writer.writerow(component_metrics[0].keys())
                # Data rows
                for metric in component_metrics:
                    writer.writerow(metric.values())

            # Get CSV content
            csv_content = output.getvalue()

            # Create streaming response
            return StreamingResponse(
                io.BytesIO(csv_content.encode('utf-8')),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="profiling_run_{run_id}.csv"'
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export profiling run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export profiling run: {str(e)}")
    finally:
        if db:
            db.close()


@app.delete("/api/profiling/run/{run_id}")
async def delete_profiling_run(run_id: str):
    """
    Delete a profiling run and all related records.

    Cascade deletes all related records:
    - power_samples
    - pipeline_sections
    - tokens
    - layer_metrics
    - component_metrics
    - deep_operation_metrics

    Args:
        run_id: Unique identifier for the profiling run

    Returns:
        Success confirmation message

    Raises:
        HTTPException: 404 if run not found, 500 for deletion failures
    """
    db = None
    try:
        # Connect to database
        db = ProfileDatabase()

        # Check if run exists
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=404,
                detail=f"Profiling run {run_id} not found"
            )

        # Delete run (cascade delete handles all related records)
        db.delete_run(run_id)
        logger.info(f"Successfully deleted profiling run {run_id}")

        return {
            "success": True,
            "message": f"Profiling run {run_id} deleted successfully",
            "run_id": run_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete profiling run {run_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete profiling run: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.post("/api/profiling/compare")
async def compare_profiling_runs(run_ids: List[str]):
    """
    Compare multiple profiling runs with normalized metrics.

    Accepts a list of run IDs and returns comparison data with metrics normalized
    by prompt length for fair comparison across different inputs.

    Args:
        run_ids: List of profiling run IDs to compare (2-10 runs)

    Returns:
        {
            "runs": [
                {
                    "run_id": str,
                    "model_name": str,
                    "prompt": str,
                    "input_tokens": int,
                    "output_tokens": int,
                    "total_tokens": int,
                    "duration_ms": float,
                    "total_energy_mj": float,
                    "peak_power_mw": float,
                    "avg_power_mw": float,
                    # Efficiency metrics
                    "energy_per_token_mj": float,
                    "energy_per_input_token_mj": float,
                    "energy_per_output_token_mj": float,
                    "tokens_per_second": float,
                    "tokens_per_joule": float,
                    # Model features
                    "total_params": int,
                    "num_layers": int,
                    "hidden_size": int,
                    "attention_mechanism": str,
                    # Phase breakdown
                    "prefill_energy_mj": float,
                    "decode_energy_mj": float,
                    "prefill_duration_ms": float,
                    "decode_duration_ms": float
                },
                ...
            ],
            "comparison": {
                "most_efficient": {run_id, tokens_per_joule},
                "fastest": {run_id, tokens_per_second},
                "lowest_energy": {run_id, total_energy_mj},
                "energy_range": {min, max, spread_factor}
            }
        }

    Raises:
        HTTPException: 400 if invalid number of runs, 404 if run not found
    """
    db = None
    try:
        # Validate input
        if not run_ids or len(run_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 run IDs required for comparison"
            )
        if len(run_ids) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 runs allowed for comparison"
            )

        # Connect to database
        db = ProfileDatabase()

        # Fetch all runs
        runs_data = []
        for run_id in run_ids:
            run = db.get_run(run_id)
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Profiling run {run_id} not found"
                )

            # Get summary for efficiency metrics
            summary = db.get_run_summary(run_id)

            # Calculate efficiency metrics
            total_tokens = run.get('input_tokens', 0) + run.get('output_tokens', 0)
            energy_per_token = (run['total_energy_mj'] / total_tokens) if total_tokens > 0 else 0

            prefill_energy = summary.get('prefill', {}).get('energy_mj', 0)
            decode_energy = summary.get('decode', {}).get('energy_mj', 0)

            input_tokens = run.get('input_tokens', 0)
            output_tokens = run.get('output_tokens', 0)

            energy_per_input_token = (prefill_energy / input_tokens) if input_tokens > 0 else 0
            energy_per_output_token = (decode_energy / output_tokens) if output_tokens > 0 else 0

            duration_s = run['duration_ms'] / 1000.0 if run['duration_ms'] else 0
            tokens_per_second = (total_tokens / duration_s) if duration_s > 0 else 0

            energy_joules = run['total_energy_mj'] / 1000.0
            tokens_per_joule = (total_tokens / energy_joules) if energy_joules > 0 else 0

            # Build comparison entry
            runs_data.append({
                "run_id": run['run_id'],
                "model_name": run['model_name'],
                "prompt": run['prompt'],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "duration_ms": run['duration_ms'],
                "total_energy_mj": run['total_energy_mj'],
                "peak_power_mw": run.get('peak_power_mw', 0),
                "avg_power_mw": run.get('avg_power_mw', 0),
                # Efficiency metrics
                "energy_per_token_mj": energy_per_token,
                "energy_per_input_token_mj": energy_per_input_token,
                "energy_per_output_token_mj": energy_per_output_token,
                "tokens_per_second": tokens_per_second,
                "tokens_per_joule": tokens_per_joule,
                # Model features
                "total_params": run.get('total_params', 0),
                "num_layers": run.get('num_layers', 0),
                "hidden_size": run.get('hidden_size', 0),
                "attention_mechanism": run.get('attention_mechanism', 'unknown'),
                # Phase breakdown
                "prefill_energy_mj": prefill_energy,
                "decode_energy_mj": decode_energy,
                "prefill_duration_ms": summary.get('prefill', {}).get('duration_ms', 0),
                "decode_duration_ms": summary.get('decode', {}).get('duration_ms', 0)
            })

        # Calculate comparison metrics
        most_efficient = max(runs_data, key=lambda x: x['tokens_per_joule'])
        fastest = max(runs_data, key=lambda x: x['tokens_per_second'])
        lowest_energy = min(runs_data, key=lambda x: x['total_energy_mj'])

        energy_values = [r['total_energy_mj'] for r in runs_data]
        energy_min = min(energy_values)
        energy_max = max(energy_values)
        energy_spread = (energy_max / energy_min) if energy_min > 0 else 0

        return {
            "runs": runs_data,
            "comparison": {
                "most_efficient": {
                    "run_id": most_efficient['run_id'],
                    "tokens_per_joule": most_efficient['tokens_per_joule']
                },
                "fastest": {
                    "run_id": fastest['run_id'],
                    "tokens_per_second": fastest['tokens_per_second']
                },
                "lowest_energy": {
                    "run_id": lowest_energy['run_id'],
                    "total_energy_mj": lowest_energy['total_energy_mj']
                },
                "energy_range": {
                    "min": energy_min,
                    "max": energy_max,
                    "spread_factor": energy_spread
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare profiling runs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare profiling runs: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/batch-size-analysis")
async def get_batch_size_analysis(
    model_name: Optional[str] = None,
    min_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None
):
    """
    Analyze how batch size affects energy per token for throughput optimization.

    Based on TokenPowerBench research: 2-3× energy spread between batch 32-1024,
    steepest drop at 32-256. This endpoint helps identify the optimal batch size
    for energy efficiency.

    Args:
        model_name: Optional filter by model name
        min_batch_size: Optional minimum batch size to include
        max_batch_size: Optional maximum batch size to include

    Returns:
        {
            "data_points": [
                {
                    "run_id": str,
                    "batch_size": int,
                    "model_name": str,
                    "energy_per_token_mj": float,
                    "joules_per_token": float,
                    "total_energy_mj": float,
                    "total_tokens": int,
                    "tokens_per_second": float,
                    "duration_ms": float,
                    "timestamp": str
                },
                ...
            ],
            "analysis": {
                "optimal_batch_size": {
                    "for_energy_efficiency": int,  # batch size with lowest J/t
                    "for_throughput": int,  # batch size with highest t/s
                    "for_edp": int  # batch size with lowest energy-delay product
                },
                "energy_vs_batch_curve": [
                    {"batch_size": int, "avg_energy_per_token": float},
                    ...
                ],
                "throughput_vs_energy_tradeoff": [
                    {"batch_size": int, "tokens_per_second": float, "energy_per_token": float},
                    ...
                ]
            },
            "statistics": {
                "total_runs": int,
                "batch_sizes_tested": [int, ...],
                "energy_spread_factor": float  # max/min energy ratio
            }
        }

    Raises:
        HTTPException: 500 if database error

    Example:
        GET /api/profiling/batch-size-analysis
        GET /api/profiling/batch-size-analysis?model_name=llama-7b
        GET /api/profiling/batch-size-analysis?min_batch_size=16&max_batch_size=256
    """
    db = None
    try:
        db = ProfileDatabase()

        # Build query with filters
        query = """
            SELECT
                run_id,
                batch_size,
                model_name,
                total_energy_mj,
                token_count,
                tokens_per_second,
                total_duration_ms,
                timestamp
            FROM profiling_runs
            WHERE status = 'completed'
                AND batch_size IS NOT NULL
                AND total_energy_mj IS NOT NULL
                AND token_count > 0
        """
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if min_batch_size is not None:
            query += " AND batch_size >= ?"
            params.append(min_batch_size)

        if max_batch_size is not None:
            query += " AND batch_size <= ?"
            params.append(max_batch_size)

        query += " ORDER BY batch_size ASC, timestamp DESC"

        cursor = db.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return {
                "data_points": [],
                "analysis": {
                    "optimal_batch_size": {},
                    "energy_vs_batch_curve": [],
                    "throughput_vs_energy_tradeoff": []
                },
                "statistics": {
                    "total_runs": 0,
                    "batch_sizes_tested": [],
                    "energy_spread_factor": 0
                }
            }

        # Process data points
        data_points = []
        batch_size_groups = {}  # Group runs by batch size for averaging

        for row in rows:
            total_tokens = row['token_count'] or 0
            energy_per_token_mj = (row['total_energy_mj'] / total_tokens) if total_tokens > 0 else 0
            joules_per_token = energy_per_token_mj / 1000.0

            data_point = {
                "run_id": row['run_id'],
                "batch_size": row['batch_size'],
                "model_name": row['model_name'],
                "energy_per_token_mj": energy_per_token_mj,
                "joules_per_token": joules_per_token,
                "total_energy_mj": row['total_energy_mj'],
                "total_tokens": total_tokens,
                "tokens_per_second": row['tokens_per_second'] or 0,
                "duration_ms": row['total_duration_ms'] or 0,
                "timestamp": row['timestamp']
            }
            data_points.append(data_point)

            # Group by batch size for analysis
            batch_size = row['batch_size']
            if batch_size not in batch_size_groups:
                batch_size_groups[batch_size] = []
            batch_size_groups[batch_size].append(data_point)

        # Calculate averaged metrics per batch size
        energy_vs_batch_curve = []
        throughput_vs_energy = []

        for batch_size in sorted(batch_size_groups.keys()):
            group = batch_size_groups[batch_size]

            avg_energy = sum(p['energy_per_token_mj'] for p in group) / len(group)
            avg_throughput = sum(p['tokens_per_second'] for p in group) / len(group)

            energy_vs_batch_curve.append({
                "batch_size": batch_size,
                "avg_energy_per_token": avg_energy,
                "sample_count": len(group)
            })

            throughput_vs_energy.append({
                "batch_size": batch_size,
                "tokens_per_second": avg_throughput,
                "energy_per_token": avg_energy
            })

        # Find optimal batch sizes
        optimal_for_energy = min(energy_vs_batch_curve, key=lambda x: x['avg_energy_per_token'])
        optimal_for_throughput = max(throughput_vs_energy, key=lambda x: x['tokens_per_second'])

        # Calculate Energy-Delay Product (EDP) for each batch size
        edp_scores = []
        for te in throughput_vs_energy:
            # EDP = energy × latency (lower is better)
            # latency ≈ 1 / throughput
            if te['tokens_per_second'] > 0:
                latency_per_token = 1000.0 / te['tokens_per_second']  # ms per token
                edp = te['energy_per_token'] * latency_per_token
                edp_scores.append({
                    "batch_size": te['batch_size'],
                    "edp": edp
                })

        optimal_for_edp = min(edp_scores, key=lambda x: x['edp']) if edp_scores else None

        # Calculate statistics
        all_energies = [p['energy_per_token_mj'] for p in data_points if p['energy_per_token_mj'] > 0]
        energy_spread = (max(all_energies) / min(all_energies)) if all_energies else 0

        batch_sizes_tested = sorted(set(batch_size_groups.keys()))

        return {
            "data_points": data_points,
            "analysis": {
                "optimal_batch_size": {
                    "for_energy_efficiency": optimal_for_energy['batch_size'],
                    "for_throughput": optimal_for_throughput['batch_size'],
                    "for_edp": optimal_for_edp['batch_size'] if optimal_for_edp else None
                },
                "energy_vs_batch_curve": energy_vs_batch_curve,
                "throughput_vs_energy_tradeoff": throughput_vs_energy
            },
            "statistics": {
                "total_runs": len(data_points),
                "batch_sizes_tested": batch_sizes_tested,
                "energy_spread_factor": round(energy_spread, 2)
            }
        }

    except Exception as e:
        logger.error(f"Batch size analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze batch size data: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/architectural-analysis")
async def get_architectural_analysis(
    model_filter: Optional[str] = None,
    min_params: Optional[int] = None,
    max_params: Optional[int] = None
):
    """
    Analyze how model architecture affects energy consumption.

    Returns correlation analysis showing:
    - Energy vs num_layers (expect linear relationship)
    - Energy vs hidden_size (expect quadratic relationship)
    - Energy vs intermediate_size
    - Comparison of MHA vs GQA vs MQA energy profiles
    - Correlation coefficients for each architectural feature

    Inspired by Caravaca et al. 2025 "From Prompts to Power" finding that
    layers scale linearly while dimensionality scales quadratically with energy.

    Query Parameters:
        model_filter: Optional substring to filter models by name
        min_params: Minimum total parameters (filter)
        max_params: Maximum total parameters (filter)

    Returns:
        {
            "data_points": [
                {
                    "run_id": str,
                    "model_name": str,
                    "num_layers": int,
                    "hidden_size": int,
                    "intermediate_size": int,
                    "num_attention_heads": int,
                    "attention_mechanism": str,
                    "total_params": int,
                    "total_energy_mj": float,
                    "energy_per_token_mj": float,
                    "tokens_per_joule": float
                },
                ...
            ],
            "correlations": {
                "energy_vs_layers": {
                    "coefficient": float,
                    "p_value": float,
                    "interpretation": str
                },
                "energy_vs_hidden_size": {
                    "coefficient": float,
                    "p_value": float,
                    "interpretation": str
                },
                "energy_vs_intermediate_size": {
                    "coefficient": float,
                    "p_value": float,
                    "interpretation": str
                },
                "energy_vs_total_params": {
                    "coefficient": float,
                    "p_value": float,
                    "interpretation": str
                }
            },
            "attention_mechanism_comparison": {
                "MHA": {
                    "count": int,
                    "avg_energy_per_token": float,
                    "avg_tokens_per_joule": float
                },
                "GQA": {...},
                "MQA": {...}
            },
            "regression_models": {
                "linear_layers": {
                    "slope": float,
                    "intercept": float,
                    "r_squared": float
                },
                "quadratic_hidden_size": {
                    "coefficient": float,
                    "intercept": float,
                    "r_squared": float
                }
            }
        }

    Raises:
        HTTPException: 500 if analysis fails
    """
    db = None
    try:
        # Import scipy for correlation analysis
        try:
            from scipy.stats import pearsonr
            import numpy as np
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="scipy is required for correlation analysis. Install with: pip install scipy"
            )

        # Connect to database
        db = ProfileDatabase()

        # Get all profiling runs with filters
        all_runs = db.get_runs(
            model_filter=model_filter,
            limit=1000  # Get up to 1000 runs for analysis
        )

        if not all_runs:
            return {
                "data_points": [],
                "correlations": {},
                "attention_mechanism_comparison": {},
                "regression_models": {},
                "message": "No profiling runs found matching criteria"
            }

        # Extract data points with architectural features
        data_points = []
        for run in all_runs:
            # Filter by parameter count if specified
            total_params = run.get('total_params', 0)
            if min_params and total_params < min_params:
                continue
            if max_params and total_params > max_params:
                continue

            # Only include runs with architectural features and valid energy data
            if (run.get('num_layers') and run.get('hidden_size') and
                run.get('total_energy_mj') and run.get('token_count')):

                total_tokens = run.get('token_count', 0)
                energy_per_token = (run['total_energy_mj'] / total_tokens) if total_tokens > 0 else 0
                tokens_per_joule = (total_tokens * 1000 / run['total_energy_mj']) if run['total_energy_mj'] > 0 else 0

                data_points.append({
                    "run_id": run['run_id'],
                    "model_name": run['model_name'],
                    "num_layers": run.get('num_layers', 0),
                    "hidden_size": run.get('hidden_size', 0),
                    "intermediate_size": run.get('intermediate_size', 0),
                    "num_attention_heads": run.get('num_attention_heads', 0),
                    "attention_mechanism": run.get('attention_mechanism', 'unknown'),
                    "total_params": total_params,
                    "total_energy_mj": run['total_energy_mj'],
                    "energy_per_token_mj": energy_per_token,
                    "tokens_per_joule": tokens_per_joule
                })

        if len(data_points) < 2:
            return {
                "data_points": data_points,
                "correlations": {},
                "attention_mechanism_comparison": {},
                "regression_models": {},
                "message": "Insufficient data points for correlation analysis (minimum 2 required)"
            }

        # Extract arrays for correlation analysis
        num_layers_arr = np.array([d['num_layers'] for d in data_points])
        hidden_size_arr = np.array([d['hidden_size'] for d in data_points])
        intermediate_size_arr = np.array([d['intermediate_size'] for d in data_points])
        total_params_arr = np.array([d['total_params'] for d in data_points])
        energy_per_token_arr = np.array([d['energy_per_token_mj'] for d in data_points])

        # Calculate correlations
        correlations = {}

        # Energy vs num_layers (expect linear relationship)
        if len(set(num_layers_arr)) > 1:
            corr_layers, p_layers = pearsonr(num_layers_arr, energy_per_token_arr)
            correlations['energy_vs_layers'] = {
                "coefficient": float(corr_layers),
                "p_value": float(p_layers),
                "interpretation": "strong positive" if corr_layers > 0.7 else "moderate positive" if corr_layers > 0.4 else "weak"
            }

            # Linear regression for layers
            if len(num_layers_arr) >= 2:
                poly_layers = np.polyfit(num_layers_arr, energy_per_token_arr, 1)
                predictions_layers = np.polyval(poly_layers, num_layers_arr)
                ss_tot_layers = np.sum((energy_per_token_arr - np.mean(energy_per_token_arr)) ** 2)
                ss_res_layers = np.sum((energy_per_token_arr - predictions_layers) ** 2)
                r_squared_layers = 1 - (ss_res_layers / ss_tot_layers) if ss_tot_layers > 0 else 0
        else:
            correlations['energy_vs_layers'] = {
                "coefficient": None,
                "p_value": None,
                "interpretation": "insufficient variance in num_layers"
            }
            r_squared_layers = 0
            poly_layers = [0, 0]

        # Energy vs hidden_size (expect quadratic relationship)
        if len(set(hidden_size_arr)) > 1:
            # Use hidden_size^2 for correlation (testing quadratic hypothesis)
            hidden_size_squared = hidden_size_arr ** 2
            corr_hidden, p_hidden = pearsonr(hidden_size_squared, energy_per_token_arr)
            correlations['energy_vs_hidden_size'] = {
                "coefficient": float(corr_hidden),
                "p_value": float(p_hidden),
                "interpretation": "strong positive" if corr_hidden > 0.7 else "moderate positive" if corr_hidden > 0.4 else "weak"
            }

            # Quadratic regression for hidden_size
            if len(hidden_size_arr) >= 2:
                # Fit: energy = a * hidden_size^2 + b
                A = np.vstack([hidden_size_squared, np.ones(len(hidden_size_squared))]).T
                coeffs_hidden = np.linalg.lstsq(A, energy_per_token_arr, rcond=None)[0]
                predictions_hidden = coeffs_hidden[0] * hidden_size_squared + coeffs_hidden[1]
                ss_tot_hidden = np.sum((energy_per_token_arr - np.mean(energy_per_token_arr)) ** 2)
                ss_res_hidden = np.sum((energy_per_token_arr - predictions_hidden) ** 2)
                r_squared_hidden = 1 - (ss_res_hidden / ss_tot_hidden) if ss_tot_hidden > 0 else 0
        else:
            correlations['energy_vs_hidden_size'] = {
                "coefficient": None,
                "p_value": None,
                "interpretation": "insufficient variance in hidden_size"
            }
            r_squared_hidden = 0
            coeffs_hidden = [0, 0]

        # Energy vs intermediate_size
        if len(set(intermediate_size_arr)) > 1:
            corr_intermediate, p_intermediate = pearsonr(intermediate_size_arr, energy_per_token_arr)
            correlations['energy_vs_intermediate_size'] = {
                "coefficient": float(corr_intermediate),
                "p_value": float(p_intermediate),
                "interpretation": "strong positive" if corr_intermediate > 0.7 else "moderate positive" if corr_intermediate > 0.4 else "weak"
            }
        else:
            correlations['energy_vs_intermediate_size'] = {
                "coefficient": None,
                "p_value": None,
                "interpretation": "insufficient variance in intermediate_size"
            }

        # Energy vs total_params
        if len(set(total_params_arr)) > 1:
            corr_params, p_params = pearsonr(total_params_arr, energy_per_token_arr)
            correlations['energy_vs_total_params'] = {
                "coefficient": float(corr_params),
                "p_value": float(p_params),
                "interpretation": "strong positive" if corr_params > 0.7 else "moderate positive" if corr_params > 0.4 else "weak"
            }
        else:
            correlations['energy_vs_total_params'] = {
                "coefficient": None,
                "p_value": None,
                "interpretation": "insufficient variance in total_params"
            }

        # Compare attention mechanisms
        attention_comparison = {}
        mechanisms = set(d['attention_mechanism'] for d in data_points)
        for mechanism in mechanisms:
            mechanism_data = [d for d in data_points if d['attention_mechanism'] == mechanism]
            if mechanism_data:
                avg_energy = np.mean([d['energy_per_token_mj'] for d in mechanism_data])
                avg_efficiency = np.mean([d['tokens_per_joule'] for d in mechanism_data])
                attention_comparison[mechanism] = {
                    "count": len(mechanism_data),
                    "avg_energy_per_token": float(avg_energy),
                    "avg_tokens_per_joule": float(avg_efficiency)
                }

        # Build regression models
        regression_models = {
            "linear_layers": {
                "slope": float(poly_layers[0]) if len(poly_layers) > 0 else 0,
                "intercept": float(poly_layers[1]) if len(poly_layers) > 1 else 0,
                "r_squared": float(r_squared_layers),
                "description": "energy_per_token = slope × num_layers + intercept"
            },
            "quadratic_hidden_size": {
                "coefficient": float(coeffs_hidden[0]) if len(coeffs_hidden) > 0 else 0,
                "intercept": float(coeffs_hidden[1]) if len(coeffs_hidden) > 1 else 0,
                "r_squared": float(r_squared_hidden),
                "description": "energy_per_token = coefficient × hidden_size² + intercept"
            }
        }

        return {
            "data_points": data_points,
            "correlations": correlations,
            "attention_mechanism_comparison": attention_comparison,
            "regression_models": regression_models
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform architectural analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform architectural analysis: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/retention/stats")
async def get_retention_stats():
    """
    Get database retention statistics.

    Returns statistics about the profiling database including:
    - Total number of runs
    - Oldest and newest run dates
    - Database size in bytes and MB

    Returns:
        {
            "total_runs": int,
            "oldest_run_date": str (ISO format),
            "newest_run_date": str (ISO format),
            "db_size_bytes": int,
            "db_size_mb": float
        }
    """
    db = None
    try:
        db = ProfileDatabase()
        db.connect()

        stats = db.get_retention_stats()

        return stats

    except Exception as e:
        logger.error(f"Failed to get retention stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get retention stats: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.post("/api/profiling/retention/cleanup")
async def cleanup_profiling_runs(
    max_runs: int = 0,
    max_age_days: int = 0,
    dry_run: bool = True
):
    """
    Clean up old profiling runs based on retention policies.

    Query Parameters:
        max_runs: Keep only this many most recent runs (0 = unlimited)
        max_age_days: Delete runs older than this many days (0 = unlimited)
        dry_run: If true, return what would be deleted without actually deleting (default: true)

    Returns:
        {
            "runs_to_delete": int,
            "run_ids": list[str],
            "db_size_before": int (bytes),
            "db_size_after": int (bytes, null if dry_run),
            "space_freed_bytes": int (only if not dry_run)
        }

    Example:
        POST /api/profiling/retention/cleanup?max_runs=100&dry_run=true
        - Preview deletion of runs beyond the 100 most recent

        POST /api/profiling/retention/cleanup?max_age_days=30&dry_run=false
        - Actually delete runs older than 30 days
    """
    db = None
    try:
        # Validate parameters
        if max_runs < 0:
            raise HTTPException(
                status_code=400,
                detail="max_runs must be >= 0"
            )
        if max_age_days < 0:
            raise HTTPException(
                status_code=400,
                detail="max_age_days must be >= 0"
            )
        if max_runs == 0 and max_age_days == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one retention policy (max_runs or max_age_days) must be specified"
            )

        db = ProfileDatabase()
        db.connect()

        result = db.cleanup_old_runs(
            max_runs=max_runs,
            max_age_days=max_age_days,
            dry_run=dry_run
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup profiling runs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup profiling runs: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.post("/api/profiling/predict")
async def predict_energy(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int = 1
):
    """
    Predict energy consumption before running inference.

    Uses a machine learning model trained on historical profiling data to estimate
    energy consumption based on model architecture and prompt characteristics.

    Body Parameters:
        model_name: Name of the model to predict for
        input_tokens: Number of input tokens (prompt length)
        output_tokens: Number of output tokens to generate
        batch_size: Batch size for inference (default: 1)

    Returns:
        {
            "predicted_total_energy_mj": float,
            "predicted_prefill_energy_mj": float,
            "predicted_decode_energy_mj": float,
            "predicted_energy_per_token_mj": float,
            "confidence_interval_95_pct": [lower, upper],
            "model_accuracy_r2": float,
            "features_used": list[str],
            "prediction_notes": str (optional)
        }

    Example:
        POST /api/profiling/predict
        {
            "model_name": "llama-3.2-1b",
            "input_tokens": 100,
            "output_tokens": 50,
            "batch_size": 1
        }
    """
    from .profiling.energy_predictor import EnergyPredictor, prepare_training_data_from_database
    from .profiling.model_features import extract_model_features

    db = None
    try:
        # Validate parameters
        if input_tokens < 1:
            raise HTTPException(status_code=400, detail="input_tokens must be >= 1")
        if output_tokens < 1:
            raise HTTPException(status_code=400, detail="output_tokens must be >= 1")
        if batch_size < 1:
            raise HTTPException(status_code=400, detail="batch_size must be >= 1")

        # Initialize predictor
        predictor = EnergyPredictor()

        # If model not trained, try to train on existing data
        if not predictor.is_trained:
            db = ProfileDatabase()
            db.connect()

            training_data = prepare_training_data_from_database(db)

            if len(training_data) < 5:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough training data. Need at least 5 completed profiling runs, found {len(training_data)}. Run more profiling sessions first."
                )

            # Train the model
            metrics = predictor.train(training_data)
            predictor.save_model()

            logger.info(f"Trained energy predictor: R²={metrics['r2']:.4f}")

        # Get model features
        # For simplicity, we'll need to load the model or use cached features
        # In production, you'd cache this or require it as input
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)

            # Create a minimal model features dict from config
            model_features = {
                "num_layers": getattr(config, "num_hidden_layers", 0),
                "hidden_size": getattr(config, "hidden_size", 0),
                "intermediate_size": getattr(config, "intermediate_size", 0),
                "num_attention_heads": getattr(config, "num_attention_heads", 0),
                "num_key_value_heads": getattr(config, "num_key_value_heads", None),
                "total_params": getattr(config, "num_parameters", 0),
                "attention_mechanism": "MHA",  # Default, could be detected
                "is_moe": False,  # Default, could be detected
            }
        except Exception as e:
            # If model not available, try to find features from previous runs
            if db is None:
                db = ProfileDatabase()
                db.connect()

            runs = db.get_runs(filters={"model_name": model_name, "status": "completed"})
            if not runs:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found. Either load the model or run a profiling session with it first."
                )

            # Use features from most recent run
            latest_run = runs[0]
            model_features = {
                "num_layers": latest_run.get("num_layers", 0),
                "hidden_size": latest_run.get("hidden_size", 0),
                "intermediate_size": latest_run.get("intermediate_size", 0),
                "num_attention_heads": latest_run.get("num_attention_heads", 0),
                "num_key_value_heads": latest_run.get("num_key_value_heads"),
                "total_params": latest_run.get("total_params", 0),
                "attention_mechanism": latest_run.get("attention_mechanism", "MHA"),
                "is_moe": latest_run.get("is_moe", False),
            }

        # Make prediction
        prediction = predictor.predict(
            model_features=model_features,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            batch_size=batch_size,
        )

        return prediction.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to predict energy: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict energy: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.post("/api/profiling/train-predictor")
async def train_energy_predictor():
    """
    Train the energy prediction model on all completed profiling runs.

    This endpoint manually triggers training of the prediction model.
    The model is automatically trained when first prediction is requested,
    but this can be used to retrain after accumulating more data.

    Returns:
        {
            "success": bool,
            "metrics": {
                "r2": float,
                "mae": float,
                "rmse": float,
                "n_samples": int
            },
            "message": str
        }
    """
    from .profiling.energy_predictor import EnergyPredictor, prepare_training_data_from_database

    db = None
    try:
        db = ProfileDatabase()
        db.connect()

        # Prepare training data
        training_data = prepare_training_data_from_database(db)

        if len(training_data) < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough training data. Need at least 5 completed profiling runs, found {len(training_data)}."
            )

        # Train model
        predictor = EnergyPredictor()
        metrics = predictor.train(training_data)
        predictor.save_model()

        return {
            "success": True,
            "metrics": metrics,
            "message": f"Energy predictor trained successfully on {metrics['n_samples']} samples with R²={metrics['r2']:.4f}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to train energy predictor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train energy predictor: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/long-context-analysis")
def get_long_context_analysis(
    run_id: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Analyze how context length affects energy consumption and KV cache pressure.

    Based on TokenPowerBench findings that long context significantly increases energy.
    KV cache is a major bottleneck for long-context inference.

    Query Parameters:
        run_id: Optional specific run to analyze
        model_name: Optional filter by model name

    Returns:
        {
            "context_length_vs_energy": [
                {
                    "run_id": "abc123",
                    "context_length": 1024,
                    "energy_mj": 150.5,
                    "energy_per_token_mj": 0.147,
                    "duration_ms": 1250.3,
                    "kv_cache_size_mb": 85.2,
                    "kv_cache_utilization_pct": 42.5
                },
                ...
            ],
            "kv_cache_stats": {
                "avg_utilization_pct": 35.2,
                "max_utilization_pct": 78.5,
                "min_utilization_pct": 12.1
            },
            "saturation_point": {
                "context_length": 8192,
                "energy_increase_pct": 25.3,
                "message": "Energy per token increased by 25.3% at context length 8192"
            },
            "warnings": [
                {
                    "run_id": "xyz789",
                    "context_length": 16384,
                    "utilization_pct": 87.5,
                    "message": "KV cache utilization at 87.5% - approaching memory limit"
                }
            ]
        }

    Example:
        GET /api/profiling/long-context-analysis
        - Analyze all runs

        GET /api/profiling/long-context-analysis?model_name=llama-7b
        - Analyze only llama-7b runs

        GET /api/profiling/long-context-analysis?run_id=abc123
        - Analyze specific run
    """
    db = None
    try:
        db = ProfileDatabase()
        db.connect()

        analysis = db.get_long_context_analysis(
            run_id=run_id,
            model_name=model_name
        )

        return analysis

    except Exception as e:
        logger.error(f"Failed to get long context analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get long context analysis: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/energy-scaling-analysis")
def get_energy_scaling_analysis(
    model_name: Optional[str] = None,
    min_params: Optional[float] = None,
    max_params: Optional[float] = None
):
    """
    Analyze super-linear energy scaling with model parameters across different model sizes.

    Based on TokenPowerBench finding: Energy scaling is super-linear but sublinear to parameter count.
    For example, 1B→70B shows 7.3× energy increase (not 70×) due to memory/cache overhead.

    This endpoint:
    1. Collects energy data across models of different sizes
    2. Plots energy vs total_params scatter
    3. Fits power-law curve: energy = a × params^b
    4. Calculates scaling exponent b (expect b < 1 for sub-linear, but can be super-linear in practice)
    5. Shows energy per million parameters metric
    6. Identifies scaling efficiency

    Query Parameters:
        model_name: Optional filter by model name pattern
        min_params: Optional minimum parameter count (in millions)
        max_params: Optional maximum parameter count (in millions)

    Returns:
        {
            "scaling_data": [
                {
                    "run_id": "abc123",
                    "model_name": "llama-7b",
                    "total_params": 7000000000,
                    "total_params_millions": 7000.0,
                    "total_energy_mj": 450.2,
                    "energy_per_million_params": 0.0643,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "joules_per_token": 0.003
                },
                ...
            ],
            "power_law_fit": {
                "coefficient_a": 2.5,
                "exponent_b": 0.85,
                "formula": "energy_mj = 2.5 × (params_millions ^ 0.85)",
                "r_squared": 0.92,
                "interpretation": "Sub-linear scaling: larger models are more energy-efficient per parameter"
            },
            "scaling_efficiency": {
                "smallest_model": {
                    "name": "llama-1b",
                    "params_millions": 1000.0,
                    "energy_per_million_params": 0.0850
                },
                "largest_model": {
                    "name": "llama-70b",
                    "params_millions": 70000.0,
                    "energy_per_million_params": 0.0621
                },
                "efficiency_gain_pct": 26.9,
                "conclusion": "Larger models are 26.9% more efficient per parameter"
            },
            "statistics": {
                "model_count": 5,
                "total_runs": 25,
                "param_range_millions": [1000.0, 70000.0],
                "energy_range_mj": [50.2, 2456.8]
            }
        }

    Example:
        GET /api/profiling/energy-scaling-analysis
        - Analyze all models

        GET /api/profiling/energy-scaling-analysis?model_name=llama
        - Analyze only models with "llama" in the name

        GET /api/profiling/energy-scaling-analysis?min_params=1000&max_params=10000
        - Analyze models between 1B and 10B parameters
    """
    db = None
    try:
        db = ProfileDatabase()
        db.connect()

        # Get runs with model parameters
        cursor = db.conn.cursor()

        # Build query with filters
        query = """
            SELECT
                run_id,
                model_name,
                total_params,
                total_energy_mj,
                input_token_count,
                output_token_count,
                token_count,
                embedding_params,
                attention_params_per_layer,
                ffn_params_per_layer,
                num_layers
            FROM profiling_runs
            WHERE status = 'completed'
                AND total_params IS NOT NULL
                AND total_energy_mj IS NOT NULL
                AND total_energy_mj > 0
        """

        params = []

        if model_name:
            query += " AND model_name LIKE ?"
            params.append(f"%{model_name}%")

        if min_params is not None:
            query += " AND total_params >= ?"
            params.append(min_params * 1_000_000)  # Convert millions to actual count

        if max_params is not None:
            query += " AND total_params <= ?"
            params.append(max_params * 1_000_000)  # Convert millions to actual count

        query += " ORDER BY total_params ASC"

        cursor.execute(query, params)
        runs = cursor.fetchall()

        if not runs:
            return {
                "scaling_data": [],
                "power_law_fit": None,
                "scaling_efficiency": None,
                "statistics": {
                    "model_count": 0,
                    "total_runs": 0,
                    "message": "No profiling runs found with parameter count data. Run profiled inference first."
                }
            }

        # Build scaling data
        scaling_data = []
        for run in runs:
            total_params = run["total_params"]
            total_params_millions = total_params / 1_000_000
            total_energy_mj = run["total_energy_mj"]
            token_count = run["token_count"] or ((run["input_token_count"] or 0) + (run["output_token_count"] or 0))

            energy_per_million_params = total_energy_mj / total_params_millions if total_params_millions > 0 else 0
            joules_per_token = (total_energy_mj / 1000) / token_count if token_count > 0 else 0

            scaling_data.append({
                "run_id": run["run_id"],
                "model_name": run["model_name"],
                "total_params": total_params,
                "total_params_millions": round(total_params_millions, 2),
                "total_energy_mj": round(total_energy_mj, 4),
                "energy_per_million_params": round(energy_per_million_params, 6),
                "input_tokens": run["input_token_count"],
                "output_tokens": run["output_token_count"],
                "joules_per_token": round(joules_per_token, 6)
            })

        # Calculate power-law fit: energy = a × params^b
        # Use numpy for curve fitting if available, otherwise use simple log-log linear regression
        power_law_fit = None
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            # Extract data for fitting
            params_millions_array = np.array([d["total_params_millions"] for d in scaling_data])
            energy_array = np.array([d["total_energy_mj"] for d in scaling_data])

            # Define power law function
            def power_law(x, a, b):
                return a * np.power(x, b)

            # Fit the curve
            popt, _ = curve_fit(power_law, params_millions_array, energy_array, p0=[1.0, 0.8])
            a, b = popt

            # Calculate R²
            residuals = energy_array - power_law(params_millions_array, a, b)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((energy_array - np.mean(energy_array))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Interpret the exponent
            if b < 0.95:
                interpretation = "Sub-linear scaling: larger models are more energy-efficient per parameter"
            elif b > 1.05:
                interpretation = "Super-linear scaling: larger models are less energy-efficient per parameter"
            else:
                interpretation = "Near-linear scaling: energy scales proportionally with parameters"

            power_law_fit = {
                "coefficient_a": round(a, 4),
                "exponent_b": round(b, 4),
                "formula": f"energy_mj = {round(a, 4)} × (params_millions ^ {round(b, 4)})",
                "r_squared": round(r_squared, 4),
                "interpretation": interpretation
            }

        except ImportError:
            # Fallback: simple log-log linear regression
            import math

            # Take log of both params and energy
            log_params = [math.log(d["total_params_millions"]) for d in scaling_data]
            log_energy = [math.log(d["total_energy_mj"]) for d in scaling_data]

            # Simple linear regression on log-log space
            n = len(log_params)
            sum_x = sum(log_params)
            sum_y = sum(log_energy)
            sum_xy = sum(x * y for x, y in zip(log_params, log_energy))
            sum_x2 = sum(x * x for x in log_params)

            # Calculate slope (b) and intercept (log_a)
            b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            log_a = (sum_y - b * sum_x) / n
            a = math.exp(log_a)

            # Interpret the exponent
            if b < 0.95:
                interpretation = "Sub-linear scaling: larger models are more energy-efficient per parameter"
            elif b > 1.05:
                interpretation = "Super-linear scaling: larger models are less energy-efficient per parameter"
            else:
                interpretation = "Near-linear scaling: energy scales proportionally with parameters"

            power_law_fit = {
                "coefficient_a": round(a, 4),
                "exponent_b": round(b, 4),
                "formula": f"energy_mj = {round(a, 4)} × (params_millions ^ {round(b, 4)})",
                "r_squared": None,  # Not calculated in fallback
                "interpretation": interpretation,
                "note": "Computed using simple log-log regression (scipy not available)"
            }
        except Exception as e:
            logger.warning(f"Failed to compute power law fit: {str(e)}")
            power_law_fit = {
                "error": "Failed to compute power law fit",
                "message": str(e)
            }

        # Calculate scaling efficiency (smallest vs largest)
        scaling_efficiency = None
        if len(scaling_data) >= 2:
            smallest = min(scaling_data, key=lambda x: x["total_params_millions"])
            largest = max(scaling_data, key=lambda x: x["total_params_millions"])

            efficiency_gain_pct = ((smallest["energy_per_million_params"] - largest["energy_per_million_params"])
                                   / smallest["energy_per_million_params"] * 100)

            if efficiency_gain_pct > 0:
                conclusion = f"Larger models are {abs(round(efficiency_gain_pct, 1))}% more efficient per parameter"
            elif efficiency_gain_pct < 0:
                conclusion = f"Smaller models are {abs(round(efficiency_gain_pct, 1))}% more efficient per parameter"
            else:
                conclusion = "Models show similar efficiency per parameter"

            scaling_efficiency = {
                "smallest_model": {
                    "name": smallest["model_name"],
                    "params_millions": smallest["total_params_millions"],
                    "energy_per_million_params": smallest["energy_per_million_params"]
                },
                "largest_model": {
                    "name": largest["model_name"],
                    "params_millions": largest["total_params_millions"],
                    "energy_per_million_params": largest["energy_per_million_params"]
                },
                "efficiency_gain_pct": round(efficiency_gain_pct, 1),
                "conclusion": conclusion
            }

        # Calculate statistics
        unique_models = set(d["model_name"] for d in scaling_data)
        param_values = [d["total_params_millions"] for d in scaling_data]
        energy_values = [d["total_energy_mj"] for d in scaling_data]

        statistics = {
            "model_count": len(unique_models),
            "total_runs": len(scaling_data),
            "param_range_millions": [round(min(param_values), 2), round(max(param_values), 2)],
            "energy_range_mj": [round(min(energy_values), 4), round(max(energy_values), 4)]
        }

        return {
            "scaling_data": scaling_data,
            "power_law_fit": power_law_fit,
            "scaling_efficiency": scaling_efficiency,
            "statistics": statistics
        }

    except Exception as e:
        logger.error(f"Failed to get energy scaling analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get energy scaling analysis: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/throughput-energy-tradeoff")
async def get_throughput_energy_tradeoff(
    model_name: Optional[str] = None,
    min_throughput: Optional[float] = None,
    max_throughput: Optional[float] = None,
    min_energy_per_token: Optional[float] = None,
    max_energy_per_token: Optional[float] = None
):
    """
    Analyze the relationship between throughput (tokens/s) and energy efficiency.

    Based on TokenPowerBench research: Shows batching improves both throughput and
    efficiency up to a point. This endpoint helps identify Pareto-optimal configurations
    and the knee of the curve (best tradeoff point).

    Args:
        model_name: Optional filter by model name
        min_throughput: Optional minimum throughput (tokens/s)
        max_throughput: Optional maximum throughput (tokens/s)
        min_energy_per_token: Optional minimum energy per token (mJ)
        max_energy_per_token: Optional maximum energy per token (mJ)

    Returns:
        {
            "data_points": [
                {
                    "run_id": str,
                    "model_name": str,
                    "throughput_tokens_per_second": float,
                    "energy_per_token_mj": float,
                    "tokens_per_joule": float,
                    "total_energy_mj": float,
                    "total_tokens": int,
                    "duration_ms": float,
                    "batch_size": int | None,
                    "is_pareto_optimal": bool
                }
            ],
            "pareto_frontier": [
                {
                    "run_id": str,
                    "throughput_tokens_per_second": float,
                    "energy_per_token_mj": float,
                    "tokens_per_joule": float
                }
            ],
            "knee_point": {
                "run_id": str,
                "throughput_tokens_per_second": float,
                "energy_per_token_mj": float,
                "tokens_per_joule": float,
                "interpretation": str
            } | None,
            "statistics": {
                "total_runs": int,
                "unique_models": int,
                "throughput_range": [float, float],
                "energy_per_token_range": [float, float],
                "best_throughput": {...},
                "best_efficiency": {...}
            }
        }

    Example usage:
        GET /api/profiling/throughput-energy-tradeoff
        GET /api/profiling/throughput-energy-tradeoff?model_name=llama-7b
        GET /api/profiling/throughput-energy-tradeoff?min_throughput=10&max_throughput=100
    """
    db = None
    try:
        db = ProfileDatabase()
        db.connect()

        # Query runs with throughput and energy data
        cursor = db.conn.cursor()

        query = """
            SELECT
                run_id,
                model_name,
                tokens_per_second,
                total_energy_mj,
                token_count,
                total_duration_ms,
                batch_size
            FROM profiling_runs
            WHERE
                tokens_per_second IS NOT NULL
                AND total_energy_mj IS NOT NULL
                AND token_count IS NOT NULL
                AND token_count > 0
                AND total_energy_mj > 0
        """

        params = []

        if model_name:
            query += " AND model_name LIKE ?"
            params.append(f"%{model_name}%")

        query += " ORDER BY tokens_per_second"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return {
                "data_points": [],
                "pareto_frontier": [],
                "knee_point": None,
                "statistics": {
                    "total_runs": 0,
                    "unique_models": 0,
                    "throughput_range": [0, 0],
                    "energy_per_token_range": [0, 0],
                    "best_throughput": None,
                    "best_efficiency": None
                }
            }

        # Process data points
        data_points = []
        for row in rows:
            throughput = row["tokens_per_second"]
            energy_per_token = row["total_energy_mj"] / row["token_count"]
            tokens_per_joule = 1000.0 / energy_per_token if energy_per_token > 0 else 0

            # Apply filters
            if min_throughput and throughput < min_throughput:
                continue
            if max_throughput and throughput > max_throughput:
                continue
            if min_energy_per_token and energy_per_token < min_energy_per_token:
                continue
            if max_energy_per_token and energy_per_token > max_energy_per_token:
                continue

            data_points.append({
                "run_id": row["run_id"],
                "model_name": row["model_name"],
                "throughput_tokens_per_second": round(throughput, 2),
                "energy_per_token_mj": round(energy_per_token, 4),
                "tokens_per_joule": round(tokens_per_joule, 2),
                "total_energy_mj": round(row["total_energy_mj"], 4),
                "total_tokens": row["token_count"],
                "duration_ms": round(row["total_duration_ms"], 2),
                "batch_size": row["batch_size"],
                "is_pareto_optimal": False  # Will be updated below
            })

        if not data_points:
            return {
                "data_points": [],
                "pareto_frontier": [],
                "knee_point": None,
                "statistics": {
                    "total_runs": 0,
                    "unique_models": 0,
                    "throughput_range": [0, 0],
                    "energy_per_token_range": [0, 0],
                    "best_throughput": None,
                    "best_efficiency": None
                }
            }

        # Calculate Pareto frontier
        # A point is Pareto optimal if no other point has both higher throughput AND higher efficiency
        pareto_frontier = []
        for i, point in enumerate(data_points):
            is_dominated = False
            for other in data_points:
                if (other["throughput_tokens_per_second"] > point["throughput_tokens_per_second"] and
                    other["tokens_per_joule"] > point["tokens_per_joule"]):
                    is_dominated = True
                    break

            if not is_dominated:
                data_points[i]["is_pareto_optimal"] = True
                pareto_frontier.append({
                    "run_id": point["run_id"],
                    "throughput_tokens_per_second": point["throughput_tokens_per_second"],
                    "energy_per_token_mj": point["energy_per_token_mj"],
                    "tokens_per_joule": point["tokens_per_joule"]
                })

        # Sort pareto frontier by throughput for display
        pareto_frontier.sort(key=lambda x: x["throughput_tokens_per_second"])

        # Find knee point (using distance from origin method)
        knee_point = None
        if len(pareto_frontier) >= 3:
            # Normalize values to 0-1 range for fair comparison
            throughputs = [p["throughput_tokens_per_second"] for p in pareto_frontier]
            efficiencies = [p["tokens_per_joule"] for p in pareto_frontier]

            min_throughput_val = min(throughputs)
            max_throughput_val = max(throughputs)
            min_efficiency_val = min(efficiencies)
            max_efficiency_val = max(efficiencies)

            if max_throughput_val > min_throughput_val and max_efficiency_val > min_efficiency_val:
                max_distance = 0
                knee_idx = 0

                for i, point in enumerate(pareto_frontier):
                    # Normalize to 0-1
                    norm_throughput = (point["throughput_tokens_per_second"] - min_throughput_val) / (max_throughput_val - min_throughput_val)
                    norm_efficiency = (point["tokens_per_joule"] - min_efficiency_val) / (max_efficiency_val - min_efficiency_val)

                    # Distance from origin (0,0) - knee is the point farthest from origin
                    distance = math.sqrt(norm_throughput ** 2 + norm_efficiency ** 2)

                    if distance > max_distance:
                        max_distance = distance
                        knee_idx = i

                knee = pareto_frontier[knee_idx]
                knee_point = {
                    "run_id": knee["run_id"],
                    "throughput_tokens_per_second": knee["throughput_tokens_per_second"],
                    "energy_per_token_mj": knee["energy_per_token_mj"],
                    "tokens_per_joule": knee["tokens_per_joule"],
                    "interpretation": f"Best tradeoff point: {round(knee['throughput_tokens_per_second'], 1)} tokens/s with {round(knee['tokens_per_joule'], 1)} tokens/joule efficiency"
                }

        # Calculate statistics
        unique_models = set(d["model_name"] for d in data_points)
        throughput_values = [d["throughput_tokens_per_second"] for d in data_points]
        energy_per_token_values = [d["energy_per_token_mj"] for d in data_points]

        best_throughput_point = max(data_points, key=lambda x: x["throughput_tokens_per_second"])
        best_efficiency_point = max(data_points, key=lambda x: x["tokens_per_joule"])

        statistics = {
            "total_runs": len(data_points),
            "unique_models": len(unique_models),
            "throughput_range": [round(min(throughput_values), 2), round(max(throughput_values), 2)],
            "energy_per_token_range": [round(min(energy_per_token_values), 4), round(max(energy_per_token_values), 4)],
            "best_throughput": {
                "run_id": best_throughput_point["run_id"],
                "model_name": best_throughput_point["model_name"],
                "throughput_tokens_per_second": best_throughput_point["throughput_tokens_per_second"],
                "energy_per_token_mj": best_throughput_point["energy_per_token_mj"]
            },
            "best_efficiency": {
                "run_id": best_efficiency_point["run_id"],
                "model_name": best_efficiency_point["model_name"],
                "throughput_tokens_per_second": best_efficiency_point["throughput_tokens_per_second"],
                "tokens_per_joule": best_efficiency_point["tokens_per_joule"]
            }
        }

        return {
            "data_points": data_points,
            "pareto_frontier": pareto_frontier,
            "knee_point": knee_point,
            "statistics": statistics
        }

    except Exception as e:
        logger.error(f"Failed to get throughput-energy tradeoff analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get throughput-energy tradeoff analysis: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/quantization-comparison")
async def get_quantization_comparison(
    model_name_filter: Optional[str] = None,
    base_model_name: Optional[str] = None
):
    """
    Compare energy consumption across different quantization levels.

    Based on TokenPowerBench research: Quantization significantly reduces energy
    in memory-constrained scenarios, especially for large models.

    Note: Apple Silicon (M4 Max) uses unified memory architecture, which may show
    different quantization benefits compared to NVIDIA GPUs.

    Args:
        model_name_filter: Optional filter to include only models matching this name pattern
        base_model_name: Optional base model name to compare different quantization levels of the same model

    Returns:
        {
            "precision_levels": ["FP32", "FP16", "BF16", "FP8", "INT8", "INT4"],
            "runs_by_precision": {
                "FP16": [
                    {
                        "run_id": str,
                        "model_name": str,
                        "precision": str,
                        "quantization_method": str | None,
                        "energy_per_token_mj": float,
                        "tokens_per_second": float,
                        "total_energy_mj": float,
                        "token_count": int
                    }
                ],
                ...
            },
            "average_energy_per_token": {
                "FP16": float,
                "INT8": float,
                ...
            },
            "energy_savings": {
                "INT8_vs_FP16": {
                    "absolute_mj": float,
                    "percent": float
                },
                ...
            },
            "throughput": {
                "FP16": float,
                "INT8": float,
                ...
            },
            "notes": [str]
        }

    Example usage:
        GET /api/profiling/quantization-comparison
        GET /api/profiling/quantization-comparison?base_model_name=llama-7b
        GET /api/profiling/quantization-comparison?model_name_filter=llama
    """
    db = None
    try:
        from profiling.model_features import compare_quantization_levels

        db = ProfileDatabase()
        db.connect()

        cursor = db.conn.cursor()

        # Query runs with precision and quantization data
        query = """
            SELECT
                run_id,
                model_name,
                precision,
                quantization_method,
                total_energy_mj,
                token_count,
                tokens_per_second
            FROM profiling_runs
            WHERE
                precision IS NOT NULL
                AND total_energy_mj IS NOT NULL
                AND token_count IS NOT NULL
                AND token_count > 0
        """

        params = []

        if model_name_filter:
            query += " AND model_name LIKE ?"
            params.append(f"%{model_name_filter}%")

        if base_model_name:
            query += " AND model_name LIKE ?"
            params.append(f"%{base_model_name}%")

        query += " ORDER BY precision, model_name"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return {
                "precision_levels": [],
                "runs_by_precision": {},
                "average_energy_per_token": {},
                "energy_savings": {},
                "throughput": {},
                "notes": [
                    "No profiling runs with quantization data found.",
                    "Run models with different quantization levels to compare energy consumption."
                ]
            }

        # Group runs by precision
        runs_by_precision = {}

        for row in rows:
            run_id, model_name, precision, quantization_method, total_energy_mj, token_count, tokens_per_second = row

            energy_per_token_mj = total_energy_mj / token_count if token_count > 0 else 0

            run_data = {
                "run_id": run_id,
                "model_name": model_name,
                "precision": precision,
                "quantization_method": quantization_method,
                "energy_per_token_mj": round(energy_per_token_mj, 6),
                "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else 0,
                "total_energy_mj": round(total_energy_mj, 2),
                "token_count": token_count
            }

            if precision not in runs_by_precision:
                runs_by_precision[precision] = []

            runs_by_precision[precision].append(run_data)

        # Use the compare_quantization_levels function from model_features
        comparison = compare_quantization_levels(runs_by_precision)

        # Add runs_by_precision to the output
        comparison["runs_by_precision"] = runs_by_precision

        return comparison

    except Exception as e:
        logger.error(f"Failed to get quantization comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quantization comparison: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.get("/api/profiling/moe-analysis/{run_id}")
async def get_moe_analysis(run_id: str):
    """
    Get MoE (Mixture of Experts) analysis for a profiling run.

    Based on TokenPowerBench research: MoE models use 2-3× less energy than
    dense models with similar quality by activating only a subset of experts per token.

    Args:
        run_id: The profiling run ID to analyze

    Returns:
        {
            "run_id": str,
            "model_name": str,
            "is_moe": bool,
            "num_experts": int | None,
            "num_active_experts": int | None,
            "total_params": int,
            "effective_params_per_token": int | None,
            "param_efficiency": float,  # effective / total
            "expert_activations": [
                {
                    "token_index": int,
                    "token_text": str,
                    "layer_index": int,
                    "active_expert_ids": str,
                    "num_active_experts": int,
                    "expert_weights": str | None,
                    "routing_entropy": float | None
                }
            ],
            "load_balance": {
                "expert_utilization": {
                    "expert_0": {
                        "activation_count": int,
                        "utilization_percent": float
                    },
                    ...
                },
                "load_balance_score": float,  # 1.0 = perfect balance
                "total_activations": int,
                "num_experts_used": int,
                "notes": [str]
            },
            "notes": [str]
        }

    Example usage:
        GET /api/profiling/moe-analysis/run_abc123
    """
    db = None
    try:
        from profiling.model_features import calculate_moe_expert_balance

        db = ProfileDatabase()
        db.connect()

        cursor = db.conn.cursor()

        # Get run metadata
        cursor.execute("""
            SELECT
                run_id,
                model_name,
                is_moe,
                num_experts,
                num_active_experts,
                total_params,
                architecture_type
            FROM profiling_runs
            WHERE run_id = ?
        """, (run_id,))

        run_row = cursor.fetchone()

        if not run_row:
            raise HTTPException(
                status_code=404,
                detail=f"Profiling run {run_id} not found"
            )

        run_id_db, model_name, is_moe, num_experts, num_active_experts, total_params, architecture_type = run_row

        if not is_moe:
            return {
                "run_id": run_id,
                "model_name": model_name,
                "is_moe": False,
                "notes": [
                    f"Model {model_name} is not a Mixture of Experts (MoE) model.",
                    "MoE analysis is only applicable to models with expert routing mechanisms."
                ]
            }

        # Get expert activations for this run
        cursor.execute("""
            SELECT
                t.token_index,
                t.token_text,
                mea.layer_index,
                mea.active_expert_ids,
                mea.num_active_experts,
                mea.expert_weights,
                mea.routing_entropy,
                mea.load_balance_loss
            FROM tokens t
            JOIN moe_expert_activations mea ON mea.token_id = t.id
            WHERE t.run_id = ?
            ORDER BY t.token_index, mea.layer_index
        """, (run_id,))

        expert_activation_rows = cursor.fetchall()

        # Format expert activations
        expert_activations = []
        expert_activation_dicts = []

        for row in expert_activation_rows:
            token_index, token_text, layer_index, active_expert_ids, num_active, expert_weights, routing_entropy, load_balance_loss = row

            activation_data = {
                "token_index": token_index,
                "token_text": token_text,
                "layer_index": layer_index,
                "active_expert_ids": active_expert_ids,
                "num_active_experts": num_active,
                "expert_weights": expert_weights,
                "routing_entropy": round(routing_entropy, 4) if routing_entropy else None,
                "load_balance_loss": round(load_balance_loss, 4) if load_balance_loss else None
            }

            expert_activations.append(activation_data)
            expert_activation_dicts.append(activation_data)

        # Calculate load balance metrics
        load_balance = calculate_moe_expert_balance(expert_activation_dicts)

        # Calculate parameter efficiency
        effective_params_per_token = None
        param_efficiency = None

        if num_active_experts and num_experts and total_params:
            # Rough estimate: assume experts are in FFN layers
            # This is a simplification; actual calculation would need layer details
            effective_params_per_token = int(total_params * (num_active_experts / num_experts))
            param_efficiency = effective_params_per_token / total_params if total_params > 0 else 0.0

        response = {
            "run_id": run_id,
            "model_name": model_name,
            "is_moe": True,
            "architecture_type": architecture_type,
            "num_experts": num_experts,
            "num_active_experts": num_active_experts,
            "total_params": total_params,
            "effective_params_per_token": effective_params_per_token,
            "param_efficiency": round(param_efficiency, 4) if param_efficiency else None,
            "expert_activations": expert_activations,
            "load_balance": load_balance,
            "notes": [
                f"TokenPowerBench finding: MoE models typically use 2-3× less energy than dense models with similar quality.",
                f"This model activates {num_active_experts} of {num_experts} experts per token.",
                f"Effective parameters per token: {effective_params_per_token:,} vs {total_params:,} total ({param_efficiency*100:.1f}% utilization)" if effective_params_per_token else "Parameter efficiency data unavailable."
            ]
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MoE analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get MoE analysis: {str(e)}"
        )
    finally:
        if db:
            db.close()


@app.post("/api/profiling/moe-comparison")
async def compare_moe_to_dense(
    moe_run_id: str,
    dense_run_id: str
):
    """
    Compare energy efficiency of MoE model vs dense model.

    Based on TokenPowerBench research: MoE models use 2-3× less energy than
    dense models with similar quality.

    Args:
        moe_run_id: Run ID for the MoE model
        dense_run_id: Run ID for the dense model to compare against

    Returns:
        {
            "moe_model": str,
            "dense_model": str,
            "metrics": {
                "moe_energy_mj": float,
                "dense_energy_mj": float,
                "moe_total_params": int,
                "moe_effective_params": int,
                "dense_params": int,
                "moe_param_efficiency": float,
                "moe_energy_per_effective_param": float,
                "dense_energy_per_param": float,
                "moe_throughput_tokens_per_sec": float,
                "dense_throughput_tokens_per_sec": float
            },
            "efficiency_gains": {
                "energy_reduction_factor": float,  # e.g., 2.5 means MoE uses 2.5× less energy
                "energy_savings_percent": float,
                "throughput_improvement_factor": float
            },
            "notes": [str]
        }

    Example usage:
        POST /api/profiling/moe-comparison
        Body: {"moe_run_id": "run_abc123", "dense_run_id": "run_def456"}
    """
    db = None
    try:
        from profiling.model_features import analyze_moe_efficiency

        db = ProfileDatabase()
        db.connect()

        cursor = db.conn.cursor()

        # Get MoE run data
        cursor.execute("""
            SELECT
                run_id,
                model_name,
                is_moe,
                num_experts,
                num_active_experts,
                total_params,
                total_energy_mj,
                tokens_per_second,
                token_count
            FROM profiling_runs
            WHERE run_id = ?
        """, (moe_run_id,))

        moe_row = cursor.fetchone()

        if not moe_row:
            raise HTTPException(
                status_code=404,
                detail=f"MoE run {moe_run_id} not found"
            )

        moe_run_id_db, moe_model_name, is_moe, num_experts, num_active_experts, moe_total_params, moe_energy, moe_throughput, moe_tokens = moe_row

        if not is_moe:
            raise HTTPException(
                status_code=400,
                detail=f"Run {moe_run_id} is not a MoE model. MoE comparison requires a MoE model."
            )

        # Calculate effective params for MoE
        moe_effective_params = moe_total_params
        if num_active_experts and num_experts and num_experts > 0:
            moe_effective_params = int(moe_total_params * (num_active_experts / num_experts))

        moe_run = {
            "model_name": moe_model_name,
            "num_experts": num_experts,
            "num_active_experts": num_active_experts,
            "total_params": moe_total_params,
            "effective_params_per_token": moe_effective_params,
            "total_energy_mj": moe_energy,
            "tokens_per_second": moe_throughput,
            "token_count": moe_tokens
        }

        # Get dense run data
        cursor.execute("""
            SELECT
                run_id,
                model_name,
                is_moe,
                total_params,
                total_energy_mj,
                tokens_per_second,
                token_count
            FROM profiling_runs
            WHERE run_id = ?
        """, (dense_run_id,))

        dense_row = cursor.fetchone()

        if not dense_row:
            raise HTTPException(
                status_code=404,
                detail=f"Dense run {dense_run_id} not found"
            )

        dense_run_id_db, dense_model_name, dense_is_moe, dense_total_params, dense_energy, dense_throughput, dense_tokens = dense_row

        if dense_is_moe:
            logger.warning(f"Run {dense_run_id} is marked as MoE model but being used as dense comparison")

        dense_run = {
            "model_name": dense_model_name,
            "total_params": dense_total_params,
            "total_energy_mj": dense_energy,
            "tokens_per_second": dense_throughput,
            "token_count": dense_tokens
        }

        # Perform comparison analysis
        comparison = analyze_moe_efficiency(moe_run, dense_run)

        return comparison

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare MoE to dense: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare MoE to dense: {str(e)}"
        )
    finally:
        if db:
            db.close()


# WebSocket connection manager for profiling streams
class ProfilingConnectionManager:
    """Manages WebSocket connections for real-time profiling data streaming."""

    def __init__(self):
        """Initialize connection manager with active connections list."""
        self.active_connections: List[WebSocket] = []
        self.message_queues: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to register
            client_id: Unique identifier for this client connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.message_queues[client_id] = asyncio.Queue()
        logger.info(f"WebSocket client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str):
        """
        Remove a WebSocket connection from active connections.

        Args:
            websocket: The WebSocket connection to remove
            client_id: Unique identifier for this client connection
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if client_id in self.message_queues:
            del self.message_queues[client_id]
        logger.info(f"WebSocket client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, message: dict, client_id: str = None):
        """
        Send a message to a specific client or broadcast to all clients.

        Args:
            message: Dictionary message to send (will be JSON-encoded)
            client_id: Optional client ID. If None, broadcasts to all clients
        """
        if client_id:
            # Send to specific client
            if client_id in self.message_queues:
                await self.message_queues[client_id].put(message)
        else:
            # Broadcast to all clients
            for queue_id in self.message_queues:
                await self.message_queues[queue_id].put(message)

    async def broadcast(self, message: dict):
        """
        Broadcast a message to all connected clients.

        Args:
            message: Dictionary message to broadcast (will be JSON-encoded)
        """
        await self.send_message(message, client_id=None)


# Message type definitions for WebSocket profiling events
class ProfilingMessageType:
    """Enumeration of message types for profiling WebSocket events."""
    POWER_SAMPLE = "power_sample"
    SECTION_START = "section_start"
    SECTION_END = "section_end"
    TOKEN_COMPLETE = "token_complete"
    LAYER_METRICS = "layer_metrics"
    COMPONENT_METRICS = "component_metrics"
    INFERENCE_COMPLETE = "inference_complete"
    MODEL_LOADING = "model_loading"
    ERROR = "error"


# Global connection manager instance
profiling_manager = ProfilingConnectionManager()


@app.websocket("/ws/profiling")
async def websocket_profiling_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time profiling data streaming.

    Streams profiling events during inference including:
    - Power samples (100ms intervals)
    - Section start/end events
    - Token generation events
    - Layer and component metrics
    - Inference completion summary

    Message format:
    {
        "type": "message_type",
        "timestamp": float,
        "data": {...}
    }

    Usage:
        Connect to ws://localhost:8000/ws/profiling
        Receive JSON messages with profiling events in real-time
    """
    # Generate unique client ID
    import uuid
    client_id = str(uuid.uuid4())

    await profiling_manager.connect(websocket, client_id)

    try:
        # Start message sender task
        sender_task = asyncio.create_task(
            _send_messages_to_client(websocket, client_id)
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive messages from client (heartbeat, config, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle client messages (e.g., config updates, start/stop requests)
                logger.info(f"Received message from client {client_id}: {message.get('type', 'unknown')}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected")
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {client_id}")
                await profiling_manager.send_message(
                    {
                        "type": ProfilingMessageType.ERROR,
                        "timestamp": datetime.now().timestamp(),
                        "data": {"error": "Invalid JSON format"}
                    },
                    client_id=client_id
                )

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")

    finally:
        # Clean up
        profiling_manager.disconnect(websocket, client_id)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


async def _send_messages_to_client(websocket: WebSocket, client_id: str):
    """
    Background task to send queued messages to a specific WebSocket client.

    Args:
        websocket: The WebSocket connection to send messages to
        client_id: Unique identifier for this client
    """
    try:
        queue = profiling_manager.message_queues.get(client_id)
        if not queue:
            return

        while True:
            # Wait for message from queue
            message = await queue.get()

            # Send message as JSON
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to client {client_id}: {str(e)}")
                break

    except asyncio.CancelledError:
        logger.info(f"Message sender task cancelled for client {client_id}")
    except Exception as e:
        logger.error(f"Error in message sender for client {client_id}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
