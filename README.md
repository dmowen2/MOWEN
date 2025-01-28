# MOWEN (Multi-Objective Wavelength Encoder Network)

MOWEN is a state-of-the-art pre-trained AI model designed for thermal imagery applications. Combining the strengths of Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Masked Autoencoders (MAEs), MOWEN offers a robust, scalable, and versatile solution for tasks such as anomaly detection, segmentation, and classification.

---

## Features
- **Pre-Trained Hybrid Model**: Leverages CNNs for localized features, ViTs for global spatial understanding, and MAEs for robustness to incomplete data.
- **User-Friendly API**: Fine-tune or evaluate MOWEN with minimal configuration.
- **Versatile Applications**: Designed for domains such as medical diagnostics, industrial monitoring, environmental tracking, and surveillance.
- **Scalable and Efficient**: Pre-trained on 1.5 million thermal images, with optimized architecture for high performance.

---

## Installation

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/dmowen2/MOWEN.git
cd MOWEN
pip install -r requirements.txt
```

---

## Quick Start

### Loading the Pre-Trained Model
```python
from mowen import MOWEN

# Load the pre-trained MOWEN model
model = MOWEN(pretrained=True)
```

### Fine-Tuning
Fine-tune MOWEN on a custom dataset:

```python
model.fine_tune(dataset="path/to/dataset", task="classification")
```

### Evaluation
Evaluate the model on a test dataset:

```python
results = model.evaluate(dataset="path/to/test_data")
print("Evaluation Results:", results)
```

---

## Project Structure

```plaintext
mowen/
├── mowen/                         # Core Python package
│   ├── __init__.py                # Package initialization
│   ├── model.py                   # MOWEN hybrid architecture
│   ├── pretrain.py                # Pretraining pipeline for MAE (internal use)
│   ├── fine_tune.py               # Fine-tuning pipeline
│   ├── utils.py                   # Helper functions
│   └── weights/                   # Pre-trained model weights
│       └── mowen_pretrained.pth   # Example pre-trained weights file
├── examples/                      # Example scripts
│   ├── fine_tune_example.py       # Fine-tuning example
│   ├── evaluate.py                # Evaluation example
│   └── config.yaml                # Configuration for experiments
├── tests/                         # Unit and integration tests
├── scripts/                       # Utility scripts
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and instructions
├── LICENSE                        # Licensing information
└── setup.py                       # Python package setup
```

---

## Applications

MOWEN is designed to address a wide range of real-world challenges in thermal imaging:

1. **Medical Diagnostics**:
   - Identify inflammation, tumors, or anomalies in thermal scans.
2. **Industrial Monitoring**:
   - Detect overheating machinery, pipeline leaks, or thermal inefficiencies.
3. **Environmental Tracking**:
   - Monitor wildlife, detect wildfires, or analyze environmental patterns.
4. **Surveillance and Security**:
   - Recognize individuals, vehicles, or other objects in thermal imagery.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push your branch and open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

MOWEN leverages state-of-the-art AI methodologies and open-source tools like:
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/)
- [timm](https://github.com/rwightman/pytorch-image-models)

Special thanks to UTSA's ARC resources for providing computational power for pretraining.

---

## Contact
For questions, suggestions, or issues, please reach out via the [GitHub Issues](https://github.com/dmowen2/MOWEN/issues) page.
