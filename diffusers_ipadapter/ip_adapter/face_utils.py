"""
Face detection and processing utilities for IPAdapter FaceID models.
Provides InsightFace integration and face processing pipeline.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import warnings

def get_insightface_model(model_name: str = "buffalo_l", providers: Optional[List[str]] = None):
    """
    Load and initialize InsightFace model for face detection and embedding extraction.
    
    Args:
        model_name: InsightFace model name ('buffalo_l' or 'antelopev2')
        providers: ONNX runtime providers (auto-detected if None)
        
    Returns:
        Initialized InsightFace FaceAnalysis model
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "InsightFace is required for FaceID models. Install with: pip install insightface"
        )
    
    # Auto-detect providers if not specified
    if providers is None:
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        except ImportError:
            providers = ['CPUExecutionProvider']
    
    # Create models directory if it doesn't exist
    models_dir = os.path.expanduser("~/.insightface/models")
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Initialize FaceAnalysis model
        model = FaceAnalysis(name=model_name, root=models_dir, providers=providers)
        model.prepare(ctx_id=0, det_size=(640, 640))
        return model
    except Exception as e:
        raise RuntimeError(f"get_insightface_model: Failed to initialize InsightFace model: {e}")

def detect_faces_multires(insightface_model, image: np.ndarray, min_size: int = 256) -> List:
    """
    Detect faces with multi-resolution fallback for better detection rates.
    
    Args:
        insightface_model: Initialized InsightFace model
        image: Input image as numpy array (H, W, C)
        min_size: Minimum detection size to try
        
    Returns:
        List of detected faces (empty if no faces found)
    """
    # Try different detection sizes from large to small
    for size in range(640, min_size - 1, -64):
        insightface_model.det_model.input_size = (size, size)
        faces = insightface_model.get(image)
        
        if faces:
            if size != 640:
                print(f"detect_faces_multires: InsightFace detection resolution lowered to {size}x{size}")
            return faces
    
    return []

def extract_face_embeddings(
    insightface_model, 
    images: Union[Image.Image, List[Image.Image]], 
    normalize: bool = True
) -> Tuple[torch.Tensor, List[Image.Image]]:
    """
    Extract face embeddings and aligned face crops from input images.
    
    Args:
        insightface_model: Initialized InsightFace model
        images: Single image or list of images
        normalize: Whether to use normalized embeddings
        
    Returns:
        Tuple of (face_embeddings_tensor, cropped_face_images)
        
    Raises:
        ValueError: If no face is detected in any image
    """
    if isinstance(images, Image.Image):
        images = [images]
    
    face_embeddings = []
    cropped_faces = []
    
    try:
        from insightface.utils import face_align
    except ImportError:
        raise ImportError("InsightFace face_align utility is required")
    
    for i, image in enumerate(images):
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Detect faces with multi-resolution fallback
        faces = detect_faces_multires(insightface_model, image_np)
        
        if not faces:
            raise ValueError(f"extract_face_embeddings: No face detected in image {i}")
        
        # Use the first detected face
        face = faces[0]
        
        # Extract face embedding
        if normalize:
            embedding = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        else:
            embedding = torch.from_numpy(face.embedding).unsqueeze(0)
        
        face_embeddings.append(embedding)
        
        # Extract and align face crop
        # Use 224 for SD1.5, 256 for SDXL (will be determined by model)
        cropped_face = face_align.norm_crop(image_np, landmark=face.kps, image_size=224)
        cropped_faces.append(Image.fromarray(cropped_face))
    
    return torch.cat(face_embeddings, dim=0), cropped_faces

def get_face_crop_size(is_sdxl: bool = False, is_kolors: bool = False) -> int:
    """
    Get appropriate face crop size based on model type.
    
    Args:
        is_sdxl: Whether this is an SDXL model
        is_kolors: Whether this is a Kolors model
        
    Returns:
        Face crop size in pixels
    """
    if is_kolors:
        return 336
    elif is_sdxl:
        return 256
    else:
        return 224

def prepare_face_conditioning(
    insightface_model,
    images: Union[Image.Image, List[Image.Image]],
    is_sdxl: bool = False,
    is_kolors: bool = False,
    normalize_embeddings: bool = True
) -> Tuple[torch.Tensor, List[Image.Image]]:
    """
    Complete face conditioning preparation pipeline.
    
    Args:
        insightface_model: Initialized InsightFace model
        images: Input images containing faces
        is_sdxl: Whether this is an SDXL model
        is_kolors: Whether this is a Kolors model  
        normalize_embeddings: Whether to normalize face embeddings
        
    Returns:
        Tuple of (face_embeddings, aligned_face_crops)
    """
    # Extract face embeddings and initial crops
    face_embeddings, cropped_faces = extract_face_embeddings(
        insightface_model, images, normalize=normalize_embeddings
    )
    
    # Re-crop faces to appropriate size for the model
    crop_size = get_face_crop_size(is_sdxl, is_kolors)
    
    if crop_size != 224:  # Need to re-crop
        try:
            from insightface.utils import face_align
            
            final_crops = []
            if isinstance(images, Image.Image):
                images = [images]
                
            for i, image in enumerate(images):
                image_np = np.array(image)
                faces = detect_faces_multires(insightface_model, image_np)
                if faces:
                    cropped_face = face_align.norm_crop(
                        image_np, landmark=faces[0].kps, image_size=crop_size
                    )
                    final_crops.append(Image.fromarray(cropped_face))
                else:
                    # Fallback to resizing existing crop
                    final_crops.append(cropped_faces[i].resize((crop_size, crop_size)))
            
            cropped_faces = final_crops
            
        except Exception as e:
            print(f"prepare_face_conditioning: Warning - could not re-crop faces: {e}")
            # Fallback to resizing existing crops
            cropped_faces = [face.resize((crop_size, crop_size)) for face in cropped_faces]
    
    return face_embeddings, cropped_faces

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor to PIL images.
    
    Args:
        tensor: Image tensor of shape (B, C, H, W) or (B, H, W, C)
        
    Returns:
        List of PIL Images
    """
    if tensor.dim() == 4:
        if tensor.shape[1] == 3:  # (B, C, H, W)
            tensor = tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # Ensure values are in [0, 255] range
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
            if tensor.max() <= 1.0:
                tensor = tensor * 255
            tensor = tensor.clamp(0, 255).byte()
        
        images = []
        for i in range(tensor.shape[0]):
            img_array = tensor[i].cpu().numpy()
            images.append(Image.fromarray(img_array))
        
        return images
    else:
        raise ValueError("tensor_to_pil: Expected 4D tensor")

def pil_to_tensor(images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Convert PIL images to tensor.
    
    Args:
        images: Single PIL image or list of PIL images
        
    Returns:
        Tensor of shape (B, C, H, W) with values in [0, 1]
    """
    if isinstance(images, Image.Image):
        images = [images]
    
    tensors = []
    for img in images:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        tensors.append(tensor)
    
    return torch.stack(tensors)