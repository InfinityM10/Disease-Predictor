import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import time
import concurrent.futures

class EnhancedHyperComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EnhancedHyperComplexConv2d, self).__init__()
        # Separate convolutions for real and imaginary components
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()  # Smooth activation function
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.attention = ChannelAttentionModule(out_channels * 2)

    def forward(self, x):
        real = self.conv_r(x)
        imag = self.conv_i(x)
        out = torch.cat([real, imag], dim=1)
        return self.attention(out)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class EnhancedHyperComplexCNN(nn.Module):
    def __init__(self, input_channels=2, num_classes=5):
        super(EnhancedHyperComplexCNN, self).__init__()
        
        # Enhanced Convolutional Layers with Residual Connections
        self.conv1 = EnhancedHyperComplexConv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.residual1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.conv2 = EnhancedHyperComplexConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.residual2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.conv3 = EnhancedHyperComplexConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling and Fully Connected Layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # More Complex Fully Connected Layers with Dropout and Layer Normalization
        self.fc_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x_residual = self.residual1(x)
        x = x + x_residual
        
        # Second Convolutional Block
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x_residual = self.residual2(x)
        x = x + x_residual
        
        # Third Convolutional Block
        x = self.conv3(x)
        
        # Global Pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc_block(x)
        
        # Classification
        return self.classifier(x)

class HCNNPredictor:
    def __init__(self, model_path, openslide_path=None):
        """
        Initialize the HCNN predictor with model and optional OpenSlide path
        
        Args:
            model_path (str): Path to the trained model weights
            openslide_path (str, optional): Path to OpenSlide binaries
        """
        # Set OpenSlide path if provided
        if openslide_path:
            os.environ['PATH'] = openslide_path + ';' + os.environ['PATH']
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        self.model = self._create_model()
        self._load_model_weights(model_path)
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x], dim=0))
        ])
    
    def _create_model(self):
        """Create the Enhanced HyperComplexCNN model architecture"""
        model = EnhancedHyperComplexCNN(input_channels=2, num_classes=5)
        return model.to(self.device)
    
    def _load_model_weights(self, model_path):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle potential state_dict wrapping
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Error loading model weights: {e}")
    
    def _tile_whole_slide_image(self, image_path, tile_size=2048, overlap=0):
        """
        Tile a whole slide image for comprehensive analysis with optimized performance
        
        Args:
            image_path (str): Path to the whole slide image
            tile_size (int): Size of each tile
            overlap (int): Overlap between tiles
        
        Returns:
            list: Paths to generated tile images
        """
        start_time = time.time()
        
        try:
            slide = open_slide(image_path)
            tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
            cols, rows = tiles.level_tiles[tiles.level_count - 1]
            
            temp_dir = os.path.join(os.path.dirname(image_path), 'temp_tiles')
            os.makedirs(temp_dir, exist_ok=True)
            
            def process_tile(row, col):
                """Process a single tile"""
                tile_name = os.path.join(temp_dir, f'{col}_{row}.png')
                # Only save if tile doesn't already exist
                if not os.path.exists(tile_name):
                    tile = tiles.get_tile(tiles.level_count - 1, (col, row))
                    tile_RGB = tile.convert('RGB')
                    tile_RGB.save(tile_name)
                return tile_name
            
            # Use concurrent processing to speed up tiling
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(process_tile, row, col) 
                    for row in range(rows) 
                    for col in range(cols)
                ]
                
                tile_paths = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            print(f"Tiling completed in {end_time - start_time:.2f} seconds")
            
            return tile_paths
        
        except Exception as e:
            raise ValueError(f"Error tiling whole slide image: {e}")
    
    def predict(self, image_path):
        """
        Predict disease type for a given image
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            dict: Prediction results
        """
        try:
            categories = ["empty", "connective", "okc_epithelial", "dc_epithelial", "df_epithelial"]
            disease_names = ["No Specific Tissue", "Connective Tissue", 
                           "Odontogenic Keratocyst", "Dentigerous Cyst", "Dental Follicle"]
            
            # Check if it's a WSI or regular image
            is_wsi = image_path.lower().endswith(('.tiff', '.tif'))
            
            if is_wsi:
                # Tile WSI and analyze tiles
                tile_paths = self._tile_whole_slide_image(image_path)
                predictions = []
                probabilities = []
                
                # Use concurrent processing for prediction
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    def predict_tile(tile_path):
                        image = Image.open(tile_path).convert("RGB")
                        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(image_tensor)
                            probs = F.softmax(outputs, dim=1)
                            pred = outputs.argmax(dim=1).item()
                            # Clean up tile
                            os.remove(tile_path)
                            return pred, probs[0].cpu().numpy()
                    
                    futures = [executor.submit(predict_tile, tile_path) for tile_path in tile_paths]
                    
                    for future in concurrent.futures.as_completed(futures):
                        pred, prob = future.result()
                        predictions.append(pred)
                        probabilities.append(prob)
                
                # Determine primary disease based on tile counts and probabilities
                tile_counts = {cat: predictions.count(idx) for idx, cat in enumerate(categories)}
                class_probabilities = {cat: [] for cat in categories}
                for idx, prob in zip(predictions, probabilities):
                    class_probabilities[categories[idx]].append(prob)
                
                # Find the disease with the highest weighted score
                disease_scores = {}
                for idx, category in enumerate(categories[2:], start=2):
                    if tile_counts[category] > 0:
                        avg_prob = np.mean(class_probabilities[category], axis=0)[idx]
                        disease_scores[category] = tile_counts[category] * avg_prob
                
                if disease_scores:
                    primary_disease = max(disease_scores, key=disease_scores.get)
                    primary_idx = categories.index(primary_disease)
                    confidence = np.mean(class_probabilities[primary_disease], axis=0)[primary_idx] * 100
                    overall_prediction = disease_names[primary_idx]
                else:
                    overall_prediction = "No Specific Tissue Detected"
                    confidence = 0
            
            else:
                # Regular image prediction
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probs = F.softmax(outputs, dim=1)
                    pred = outputs.argmax(dim=1).item()
                    confidence = probs[0][pred].item() * 100
                    overall_prediction = disease_names[pred]
            
            return {
                "prediction_successful": True,
                "disease": overall_prediction,
                "confidence": confidence
            }
        
        except Exception as e:
            return {
                "prediction_successful": False,
                "error": str(e)
            }

def save_upload_file(file):
    """
    Save uploaded file to a temporary location
    
    Args:
        file (FileStorage): Uploaded file object
    
    Returns:
        str: Path to saved file
    """
    upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    return file_path