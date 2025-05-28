# Face Landmarks Detection with Spatial Configuration Network (SCN)

A PyTorch implementation of Spatial Configuration Network for facial landmark detection using the WFLW dataset. This project trains a deep learning model to detect 98 facial landmarks with high accuracy.

## Features

- **WFLW Dataset Support**: Automatic download and preprocessing of the Wider Facial Landmarks in the Wild (WFLW) dataset
- **98 Facial Landmarks**: Detects comprehensive facial features including face contour, eyebrows, eyes, nose, and mouth
- **MPS Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)
- **ResNet-style Architecture**: Deep CNN with residual blocks for robust feature extraction
- **Spatial Configuration Module**: Specialized architecture for spatial landmark relationships
- **Real-time Visualization**: Interactive plotting of predicted vs. ground truth landmarks
- **Automatic Fallback**: Creates dummy dataset if download fails for testing purposes

## Architecture

The Spatial Configuration Network (SCN) consists of:

1. **Feature Extraction**: ResNet-style CNN with multiple residual blocks
   - Initial 7x7 convolution with batch normalization
   - Three residual blocks with increasing channels (64→128→256→512)
   - MaxPooling and ReLU activations

2. **Spatial Configuration Module**: 
   - Global Average Pooling for spatial information aggregation
   - Fully connected layers with dropout for landmark coordinate regression
   - Output: 98 landmarks × 2 coordinates (x, y)

## Dataset

The project uses the **WFLW (Wider Facial Landmarks in the Wild)** dataset:
- **98 facial landmarks** per image
- **10,000 training images** and **2,500 test images**
- Challenging conditions: pose variations, expressions, illumination, occlusions
- Automatic download via Google Drive integration

### Landmark Regions:
- Face contour: 33 points (0-32)
- Eyebrows: 18 points (33-50) 
- Eyes: 16 points (51-66)
- Nose: 12 points (67-78)
- Mouth outer: 12 points (79-90)
- Mouth inner: 8 points (91-97)

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
Pillow>=8.0.0
tqdm>=4.60.0
gdown>=4.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bemoregt/Train_SCN_FaceLandmarks.git
cd Train_SCN_FaceLandmarks
```

2. Install dependencies:
```bash
pip install torch torchvision numpy opencv-python matplotlib Pillow tqdm gdown
```

3. Run the training script:
```bash
python train_scn.py
```

## Usage

### Training

The main script automatically handles:
- Dataset download and preparation
- Model initialization and training
- Validation and testing
- Results visualization

```python
python train_scn.py
```

### Key Training Parameters

- **Batch Size**: 32
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Epochs**: 30
- **Loss Function**: MSE Loss
- **Optimizer**: Adam with weight decay (1e-5)
- **Image Size**: 224×224 pixels

### Model Architecture Details

```python
# Feature extraction layers
- Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
- 3 Residual blocks: 64→128→256→512 channels
- Global Average Pooling
- FC layers: 512→1024→512→196 (98×2 coordinates)
```

## Results

The model outputs:
- **Training/Validation Loss curves**
- **Test MSE score**
- **Visual comparison** of predicted vs. ground truth landmarks
- **Color-coded landmarks** by facial regions

### Visualization

The project provides comprehensive visualization:
- Different colors for each facial region
- Side-by-side comparison of predictions and ground truth
- Interactive matplotlib plots for result analysis

## File Structure

```
Train_SCN_FaceLandmarks/
├── train_scn.py              # Main training script
├── data/                     # Dataset directory
│   └── wflw/                 # WFLW dataset
│       ├── WFLW_images/      # Training/test images
│       └── WFLW_annotations/ # Landmark annotations
├── best_scn_model.pth        # Best model checkpoint
├── scn_model_wflw_complete.pth # Final model with metadata
└── README.md                 # This file
```

## Model Performance

The SCN model achieves competitive performance on facial landmark detection:
- **Robust feature extraction** with ResNet-style architecture
- **Spatial relationship modeling** through configuration network
- **Multi-scale processing** with progressive downsampling
- **Regularization** through dropout and batch normalization

## Hardware Requirements

- **GPU**: CUDA-compatible GPU or Apple Silicon (MPS)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB for dataset and model files

## Troubleshooting

### Dataset Download Issues
If automatic download fails, the script creates a dummy dataset for testing:
- 100 synthetic face images with 98 landmarks
- Maintains WFLW format compatibility
- Allows code testing without full dataset

### Memory Issues
- Reduce batch size if encountering OOM errors
- Use smaller image resolution (adjust transforms)
- Enable gradient checkpointing for deeper models

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **WFLW Dataset**: Wu, Wayne, et al. "Look at boundary: A boundary-aware face alignment algorithm." CVPR 2018.
- **Spatial Configuration Networks**: For hierarchical facial landmark detection
- **PyTorch Team**: For the excellent deep learning framework

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{scn_face_landmarks_2025,
  title={Face Landmarks Detection with Spatial Configuration Network},
  author={bemoregt},
  year={2025},
  url={https://github.com/bemoregt/Train_SCN_FaceLandmarks}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub or contact the repository owner.

---

**Note**: This implementation is optimized for Apple Silicon (MPS) but also supports CUDA and CPU execution. The model automatically detects the best available device for training.