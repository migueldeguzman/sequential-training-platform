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


class ConversionJob(BaseModel):
    datasetName: str
    pairCount: Optional[int] = None
    formatStyle: str = "chat"


class TrainingConfig(BaseModel):
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
    for key in ["json_input", "text_output", "model_output"]:
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

        # Process each dataset
        for i, dataset_name in enumerate(config.datasets):
            if not training_state["is_running"]:
                status = "stopped"
                break

            training_state["current_dataset"] = dataset_name
            await broadcast_log("info", f"Processing dataset {i+1}/{len(config.datasets)}: {dataset_name}")
            await broadcast_status()

            # Check if text file exists, if not convert it
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

            # Step output directory
            step_output_dir = f"{paths['model_output']}/step_{i+1}_{dataset_name}"

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
print(f"Loading model from {{model_path}}...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Loading tokenizer from {{model_path}}...")

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
print(f"Training on {dataset_name}...")
print(f"Output will be saved to: {{output_dir}}")

print(f"Preparing dataset: {{answer_file_path}}...")
with open(answer_file_path, 'r', encoding='utf-8') as f:
    text_content = f.read()

qa_pairs = [pair.strip() for pair in text_content.split('\\n\\n') if pair.strip()]
print(f"  - Loaded {{len(qa_pairs)}} Q&A pairs")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

dataset_dict = {{'text': qa_pairs}}
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
    modelPath: str
    profilingDepth: str = "module"  # "module" or "deep"
    tags: Optional[str] = None
    experimentName: Optional[str] = None
    config: InferenceConfig = InferenceConfig()


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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from profiling.power_monitor import PowerMonitor
    from profiling.layer_profiler import LayerProfiler
    from profiling.deep_profiler import DeepAttentionProfiler
    from profiling.database import ProfileDatabase
    from profiling.pipeline_profiler import InferencePipelineProfiler

    model_dir = Path(request.modelPath)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {request.modelPath}")

    try:
        # Initialize profiling components
        power_monitor = PowerMonitor(sample_interval_ms=100)
        if not power_monitor.is_available():
            raise HTTPException(
                status_code=503,
                detail="powermetrics not available. Run setup_powermetrics.sh to configure sudo access."
            )

        # Detect device
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(request.modelPath, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            request.modelPath,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.to(device)

        # Initialize profilers
        layer_profiler = LayerProfiler(model)
        deep_profiler = None
        if request.profilingDepth == "deep":
            deep_profiler = DeepAttentionProfiler(model)
            deep_profiler.patch()

        database = ProfileDatabase()

        # Define power sample callback for WebSocket streaming
        def stream_power_sample(sample):
            """Callback to stream power samples via WebSocket"""
            try:
                message = {
                    "type": ProfilingMessageType.POWER_SAMPLE,
                    "timestamp": sample.timestamp,
                    "data": {
                        "relative_time_ms": sample.relative_time_ms,
                        "cpu_power_mw": sample.cpu_power_mw,
                        "gpu_power_mw": sample.gpu_power_mw,
                        "ane_power_mw": sample.ane_power_mw,
                        "dram_power_mw": sample.dram_power_mw,
                        "total_power_mw": sample.total_power_mw
                    }
                }
                # Broadcast to all connected WebSocket clients
                asyncio.create_task(profiling_manager.broadcast(message))
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
                # Broadcast to all connected WebSocket clients
                asyncio.create_task(profiling_manager.broadcast(message))
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
                        "token_index": token_data.get("token_index"),
                        "token_text": token_data.get("token_text"),
                        "duration_ms": token_data.get("duration_ms"),
                        "energy_mj": token_data.get("energy_mj"),
                        "avg_power_mw": token_data.get("avg_power_mw"),
                        "power_snapshot": token_data.get("power_snapshot"),
                        "layer_metrics_summary": token_data.get("layer_metrics_summary")
                    }
                }
                # Broadcast to all connected WebSocket clients
                asyncio.create_task(profiling_manager.broadcast(message))
            except Exception as e:
                logger.error(f"Failed to stream token complete event: {e}")

        # Create pipeline profiler with streaming callbacks
        profiler = InferencePipelineProfiler(
            power_monitor=power_monitor,
            layer_profiler=layer_profiler,
            deep_profiler=deep_profiler,
            database=database,
            power_sample_callback=stream_power_sample,
            section_event_callback=stream_section_event,
            token_complete_callback=stream_token_complete
        )

        # Start profiling session
        with profiler.run(
            prompt=request.prompt,
            model_name=model_dir.name,
            profiling_depth=request.profilingDepth,
            experiment_name=request.experimentName,
            tags=request.tags
        ) as session:
            # Pre-inference phase
            with session.section("tokenization", phase="pre_inference"):
                inputs = tokenizer(request.prompt, return_tensors="pt", padding=True)

            with session.section("tensor_transfer", phase="pre_inference"):
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Prefill phase (single forward pass for entire prompt)
            with session.section("prefill", phase="prefill"):
                # Generate will handle both prefill and decode internally
                # We profile it as one operation here for simplicity
                with torch.no_grad():
                    output = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=request.config.maxLength,
                        num_return_sequences=1,
                        no_repeat_ngram_size=request.config.noRepeatNgramSize,
                        do_sample=request.config.doSample,
                        top_k=request.config.topK,
                        top_p=request.config.topP,
                        temperature=request.config.temperature,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

            # Post-inference phase
            with session.section("detokenization", phase="post_inference"):
                full_text = tokenizer.decode(output[0], skip_special_tokens=True)
                if full_text.startswith(request.prompt):
                    response = full_text[len(request.prompt):].strip()
                else:
                    response = full_text.strip()

                session.response = response

        # Cleanup
        if deep_profiler:
            deep_profiler.unpatch()
        layer_profiler.detach()

        # Data is automatically saved to database via profiler.run context manager
        return {
            "runId": session.run_id,
            "response": response,
            "message": "Profiled inference completed successfully"
        }

    except Exception as e:
        logger.error(f"Profiled generation failed: {str(e)}")
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
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    sort_by: str = Query("date", description="Sort by: date, duration, energy"),
):
    """
    List profiling runs with optional filtering and pagination.

    Query parameters:
    - model: Filter by model name
    - date_from: Filter runs from this timestamp (ISO format)
    - date_to: Filter runs up to this timestamp (ISO format)
    - tags: Filter by comma-separated tags
    - experiment: Filter by experiment name
    - limit: Maximum number of results (1-1000, default 100)
    - offset: Number of results to skip for pagination (default 0)
    - sort_by: Sort by date (default), duration, or energy

    Returns list with summary metrics per run.
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

                runs_with_summary.append({
                    "run_id": run["run_id"],
                    "timestamp": run["timestamp"],
                    "model_name": run["model_name"],
                    "prompt": run["prompt"],
                    "response": run.get("response"),
                    "experiment_name": run.get("experiment_name"),
                    "tags": run.get("tags"),
                    "profiling_depth": run.get("profiling_depth"),
                    "status": run.get("status"),
                    "total_duration_ms": total_duration_ms,
                    "total_energy_mj": total_energy_mj,
                    "input_tokens": run.get("input_tokens"),
                    "output_tokens": run.get("output_tokens"),
                })
            else:
                # Fallback if summary is not available
                runs_with_summary.append({
                    "run_id": run["run_id"],
                    "timestamp": run["timestamp"],
                    "model_name": run["model_name"],
                    "prompt": run["prompt"],
                    "response": run.get("response"),
                    "experiment_name": run.get("experiment_name"),
                    "tags": run.get("tags"),
                    "profiling_depth": run.get("profiling_depth"),
                    "status": run.get("status"),
                    "total_duration_ms": None,
                    "total_energy_mj": None,
                    "input_tokens": None,
                    "output_tokens": None,
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
        # Default sort by date is already handled by database query

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
