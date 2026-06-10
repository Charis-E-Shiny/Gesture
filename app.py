from fastapi import FastAPI, WebSocket, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import asyncio
import tempfile
import base64
import torch
from pathlib import Path

from inference import RealtimeInferencePipeline

app = FastAPI(title="ASL Recognition API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline
inference_pipeline = None


class FrameData(BaseModel):
    """Frame data for inference"""
    frame_base64: str
    frame_id: int


class InferenceResult(BaseModel):
    """Inference result"""
    gloss: Optional[str] = None
    english: Optional[str] = None
    latency_ms: float
    frame_count: int


@app.on_event("startup")
async def startup_event():
    """Initialize inference pipeline on startup"""
    global inference_pipeline
    try:
        inference_pipeline = RealtimeInferencePipeline()
        print("Inference pipeline initialized")
    except Exception as e:
        print(f"Error initializing inference pipeline: {e}")
        print("Pipeline will be created on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "ASL Recognition Pipeline",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "ASL Recognition & Translation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "infer_frame": "POST /infer_frame",
            "video_stream": "WebSocket /ws",
            "upload_video": "POST /upload_video"
        }
    }


@app.post("/infer_frame", response_model=InferenceResult)
async def infer_frame(frame_data: FrameData):
    """
    Infer ASL recognition from a single frame
    
    Args:
        frame_data: Base64 encoded frame
    
    Returns:
        InferenceResult with gloss and English translation
    """
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.frame_base64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode frame")
        
        # Process frame
        results = inference_pipeline.process_frame(frame)
        
        return InferenceResult(
            gloss=results.get('gloss'),
            english=results.get('english'),
            latency_ms=results['latency_ms'],
            frame_count=results['frame_count']
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    global inference_pipeline
    
    if inference_pipeline is None:
        await websocket.close(code=4003, reason="Pipeline not initialized")
        return
    
    await websocket.accept()
    
    try:
        while True:
            # Receive frame
            data = await websocket.receive_json()
            frame_base64 = data.get('frame')
            
            if not frame_base64:
                continue
            
            # Decode frame
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Process frame
            results = inference_pipeline.process_frame(frame)
            
            # Send results back
            await websocket.send_json({
                "gloss": results.get('gloss'),
                "english": results.get('english'),
                "latency_ms": results['latency_ms'],
                "frame_count": results['frame_count'],
                "timestamp": data.get('timestamp')
            })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=4000, reason=str(e))


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload and process a video file
    
    Args:
        file: Video file
    
    Returns:
        Path to processed video with captions
    """
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name
        
        # Process video
        output_path = input_path.replace('.mp4', '_output.mp4')
        results = inference_pipeline.process_video_file(input_path, output_path)
        
        return {
            "output_video": output_path,
            "frames_processed": len(results),
            "average_latency_ms": np.mean([r['latency_ms'] for r in results]),
            "results": [
                {
                    "frame": i,
                    "gloss": r.get('gloss'),
                    "english": r.get('english'),
                    "latency_ms": r['latency_ms']
                }
                for i, r in enumerate(results)
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/download_video/{video_id}")
async def download_video(video_id: str):
    """Download processed video"""
    video_path = Path(f"/tmp/{video_id}_output.mp4")
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(video_path, media_type="video/mp4",
                       filename=f"{video_id}_output.mp4")


@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "ASL Recognition & Translation Pipeline",
        "architecture": {
            "encoders": [
                "Hand Encoder (21 landmarks × 3 coords)",
                "Face Encoder (468 landmarks × 3 coords)",
                "Pose Encoder (33 landmarks × 3 coords)"
            ],
            "fusion": "Cross-modal Fusion Transformer (6 heads)",
            "streaming": "Conformer Streaming Encoder",
            "recognition": "RNNT Transducer",
            "translation": "Seq2Seq Transformer"
        },
        "vocab_size": 1000,
        "english_vocab_size": 5000,
        "max_latency_ms": 500,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
