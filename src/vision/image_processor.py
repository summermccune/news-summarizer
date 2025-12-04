"""Image processing and captioning using vision-language models."""
from typing import Optional, List, Union
import torch
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from ..config import load_config, get_device


class ImageProcessor:
    """
    Image processing and understanding using vision-language models.
    
    Uses BLIP (Bootstrapped Language-Image Pre-training) for:
    - Image captioning
    - Visual question answering
    - Image-text matching
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the image processor.
        
        Args:
            model_name: HuggingFace vision model name
            device: Device to use ('cuda', 'cpu', 'mps')
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set model and device
        self.model_name = model_name or self.config['models']['vision']['model_name']
        self.device = device or get_device(self.config)
        
        # Load processor and model
        print(f"Loading vision model: {self.model_name}")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get generation parameters from config
        self.gen_params = self.config['models']['vision']
        
        print(f"Vision model loaded on device: {self.device}")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image
        """
        return Image.open(image_path).convert('RGB')
    
    def generate_caption(
        self,
        image: Union[str, Path, Image.Image],
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: Image path or PIL Image
            max_length: Maximum caption length
            
        Returns:
            Generated caption
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Set max length
        max_length = max_length or self.gen_params['caption_max_length']
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3
            )
        
        # Decode caption
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption
    
    def generate_captions(
        self,
        images: Union[List[str], List[Path], List[Image.Image]],
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            max_length: Maximum caption length
            
        Returns:
            List of captions
        """
        captions = []
        for image in images:
            caption = self.generate_caption(image, max_length)
            captions.append(caption)
        return captions
    
    def answer_visual_question(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Answer a question about an image.
        
        Args:
            image: Image path or PIL Image
            question: Question about the image
            max_length: Maximum answer length
            
        Returns:
            Answer to the question
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Set max length
        max_length = max_length or self.gen_params['caption_max_length']
        
        # Process image and question
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3
            )
        
        # Decode answer
        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return answer
    
    def extract_visual_features(
        self,
        image: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        """
        Extract visual features from an image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Feature tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            features = outputs.last_hidden_state
        
        return features
    
    def batch_process_images(
        self,
        images: List[Union[str, Path, Image.Image]],
        task: str = 'caption',
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Process multiple images in batches.
        
        Args:
            images: List of images
            task: Task to perform ('caption' or 'feature')
            batch_size: Number of images per batch
            **kwargs: Additional arguments
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            if task == 'caption':
                batch_results = self.generate_captions(batch, **kwargs)
                results.extend(batch_results)
            else:
                for image in batch:
                    result = self.extract_visual_features(image)
                    results.append(result)
        
        return results
