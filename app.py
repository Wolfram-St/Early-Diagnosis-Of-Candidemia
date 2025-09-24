#!/usr/bin/env python3
"""
Candida Infection Classification Web Application
A comprehensive web interface for patch-based Candida infection classification
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime
import base64
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Candida Infection Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PatchBasedCandidaClassifier:
    """Patch-based Candida classifier for web deployment"""
    
    def __init__(self, model_path, model_type='resnet50', patch_size=128, stride=64, 
                 quality_threshold=0.3, device=None):
        self.model_type = model_type
        self.patch_size = patch_size
        self.stride = stride
        self.quality_threshold = quality_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.class_names = ['CA (Candida albicans)', 'CG (Candida glabrata)', 'Mock (Control)']
        self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """Load trained model"""
        if self.model_type == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 3)
        elif self.model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        return model

    def calculate_patch_quality(self, patch):
        """Calculate patch quality score"""
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = patch
        
        # Calculate variance (texture measure)
        variance = np.var(gray)
        
        # Calculate gradient magnitude (edge measure)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Combine metrics
        quality_score = (variance / 10000) + (np.mean(gradient) / 100)
        return min(quality_score, 1.0)

    def extract_patches(self, image):
        """Extract high-quality patches from image"""
        h, w = image.shape[:2]
        patches = []
        coordinates = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                # Calculate quality
                quality = self.calculate_patch_quality(patch)
                
                if quality >= self.quality_threshold:
                    patches.append(patch)
                    coordinates.append((x, y, quality))
        
        return patches, coordinates

    def classify_patches(self, patches):
        """Classify extracted patches"""
        if not patches:
            return [], []
        
        predictions = []
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for patch in patches:
                # Convert to PIL and apply transforms
                if isinstance(patch, np.ndarray):
                    patch_pil = Image.fromarray(patch.astype('uint8'))
                else:
                    patch_pil = patch
                
                # Transform and predict
                patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
                
                output = self.model(patch_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                confidence, predicted = torch.max(probabilities, 1)
                
                predictions.append(predicted.item())
                confidences.append(confidence.item())
        
        return predictions, confidences

    def aggregate_predictions(self, predictions, confidences):
        """Aggregate patch predictions into final result"""
        if not predictions:
            return None, 0.0, {}
        
        # Count predictions by class
        class_counts = {0: 0, 1: 0, 2: 0}
        confidence_sums = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for pred, conf in zip(predictions, confidences):
            class_counts[pred] += 1
            confidence_sums[pred] += conf
        
        # Weighted by confidence scores
        total_conf = sum(confidence_sums.values())
        if total_conf > 0:
            class_scores = {k: v/total_conf for k, v in confidence_sums.items()}
        else:
            class_scores = {k: v/len(predictions) for k, v in class_counts.items()}
        
        # Get final prediction
        final_class = max(class_scores, key=class_scores.get)
        final_confidence = class_scores[final_class]
        
        return final_class, final_confidence, class_scores

    def classify_image(self, image):
        """Classify a single image using patch-based approach"""
        # Extract patches
        patches, coordinates = self.extract_patches(image)
        
        if not patches:
            return {
                'error': 'No quality patches found',
                'total_patches': 0,
                'prediction': None,
                'class_probabilities': {},
                'patch_analysis': {}
            }
        
        # Classify patches
        predictions, confidences = self.classify_patches(patches)
        
        # Aggregate results
        final_class, final_confidence, class_scores = self.aggregate_predictions(
            predictions, confidences)
        
        return {
            'prediction': {
                'class_id': final_class,
                'class_name': self.class_names[final_class],
                'confidence': final_confidence
            },
            'class_probabilities': {
                self.class_names[i]: score for i, score in class_scores.items()
            },
            'patch_analysis': {
                'total_patches': len(patches),
                'patch_predictions': predictions,
                'patch_confidences': confidences,
                'coordinates': coordinates
            },
            'timestamp': datetime.now().isoformat()
        }

def load_available_models():
    """Load available trained models"""
    models_dir = Path("models")
    if not models_dir.exists():
        return {}
    
    available_models = {}
    for model_file in models_dir.glob("*.pth"):
        model_name = model_file.stem
        if 'resnet50' in model_name.lower():
            model_type = 'resnet50'
        elif 'efficientnet' in model_name.lower():
            model_type = 'efficientnet_b0'
        else:
            continue
        
        available_models[model_name] = {
            'path': str(model_file),
            'type': model_type,
            'size': model_file.stat().st_size / (1024*1024),  # MB
            'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
        }
    
    return available_models

def create_patch_visualization(image, result):
    """Create patch analysis visualization"""
    if 'error' in result:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original Image with Patches', 'Class Probabilities', 
                       'Patch Confidence Distribution', 'Patch Predictions'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Original image (placeholder - would need proper image handling)
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode='markers', showlegend=False),
        row=1, col=1
    )
    
    # Class probabilities
    class_names = ['CA', 'CG', 'Mock']
    probs = [result['class_probabilities'][name] for name in result['class_probabilities']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig.add_trace(
        go.Bar(x=class_names, y=probs, marker_color=colors, name='Probabilities'),
        row=1, col=2
    )
    
    # Patch confidence distribution
    confidences = result['patch_analysis']['patch_confidences']
    fig.add_trace(
        go.Histogram(x=confidences, nbinsx=20, name='Confidence Distribution'),
        row=2, col=1
    )
    
    # Patch predictions count
    predictions = result['patch_analysis']['patch_predictions']
    pred_counts = [predictions.count(i) for i in range(3)]
    
    fig.add_trace(
        go.Bar(x=class_names, y=pred_counts, marker_color=colors, name='Patch Counts'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Patch Analysis Results")
    return fig

def main():
    """Main Streamlit application"""
    st.title("🔬 Candida Infection Classification System")
    st.markdown("### Patch-based Deep Learning Analysis for Microscopy Images")
    
    # Sidebar for configuration
    st.sidebar.header("🛠️ Configuration")
    
    # Load available models
    available_models = load_available_models()
    
    if not available_models:
        st.error("No trained models found in the 'models' directory. Please train models first.")
        st.info("Expected model files: resnet50_*.pth or efficientnet_b0_*.pth")
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(available_models.keys()),
        format_func=lambda x: f"{x} ({available_models[x]['type']})"
    )
    
    # Model info
    model_info = available_models[selected_model]
    st.sidebar.info(f"""
    **Model Information:**
    - Type: {model_info['type']}
    - Size: {model_info['size']:.1f} MB
    - Modified: {model_info['modified'].strftime('%Y-%m-%d %H:%M')}
    """)
    
    # Patch parameters
    st.sidebar.subheader("Patch Parameters")
    patch_size = st.sidebar.slider("Patch Size", 64, 256, 128, 32)
    stride = st.sidebar.slider("Stride", 16, 128, 64, 16)
    quality_threshold = st.sidebar.slider("Quality Threshold", 0.1, 1.0, 0.3, 0.1)
    
    # Initialize classifier
    @st.cache_resource
    def load_classifier(model_path, model_type, patch_size, stride, quality_threshold):
        return PatchBasedCandidaClassifier(
            model_path=model_path,
            model_type=model_type,
            patch_size=patch_size,
            stride=stride,
            quality_threshold=quality_threshold
        )
    
    try:
        classifier = load_classifier(
            model_info['path'], 
            model_info['type'],
            patch_size,
            stride,
            quality_threshold
        )
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Batch Processing", "📊 Results History"])
    
    with tab1:
        st.header("Single Image Classification")
        
        uploaded_file = st.file_uploader(
            "Upload microscopy image",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            help="Supported formats: TIF, TIFF, PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                st.info(f"""
                **Image Information:**
                - Size: {image.size[0]} × {image.size[1]} pixels
                - Mode: {image.mode}
                - Format: {uploaded_file.type}
                """)
            
            with col2:
                st.subheader("Classification Results")
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Extracting patches and analyzing..."):
                        # Ensure RGB format
                        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                            image_np = image_np[:, :, :3]
                        elif len(image_np.shape) == 2:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                        
                        # Classify image
                        result = classifier.classify_image(image_np)
                        
                        if 'error' in result:
                            st.error(f"Classification failed: {result['error']}")
                        else:
                            # Display prediction
                            pred = result['prediction']
                            st.success(f"""
                            **Prediction: {pred['class_name']}**
                            
                            Confidence: {pred['confidence']:.3f}
                            """)
                            
                            # Class probabilities
                            st.subheader("Class Probabilities")
                            prob_data = pd.DataFrame([
                                {'Class': k, 'Probability': v, 'Color': c} 
                                for (k, v), c in zip(result['class_probabilities'].items(), 
                                                   ['#FF6B6B', '#4ECDC4', '#45B7D1'])
                            ])
                            
                            fig_prob = px.bar(
                                prob_data, 
                                x='Class', 
                                y='Probability',
                                color='Class',
                                color_discrete_map={
                                    'CA (Candida albicans)': '#FF6B6B',
                                    'CG (Candida glabrata)': '#4ECDC4', 
                                    'Mock (Control)': '#45B7D1'
                                }
                            )
                            fig_prob.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig_prob, use_container_width=True)
                            
                            # Patch analysis
                            st.subheader("Patch Analysis")
                            patch_info = result['patch_analysis']
                            
                            col3, col4, col5 = st.columns(3)
                            with col3:
                                st.metric("Total Patches", patch_info['total_patches'])
                            with col4:
                                st.metric("Avg Confidence", 
                                         f"{np.mean(patch_info['patch_confidences']):.3f}")
                            with col5:
                                st.metric("Quality Patches", 
                                         f"{len([c for c in patch_info['patch_confidences'] if c > 0.5])}")
                            
                            # Confidence distribution
                            fig_conf = px.histogram(
                                x=patch_info['patch_confidences'],
                                nbins=20,
                                title="Patch Confidence Distribution"
                            )
                            fig_conf.update_layout(height=300)
                            st.plotly_chart(fig_conf, use_container_width=True)
                            
                            # Download results
                            result_json = json.dumps(result, indent=2, default=str)
                            st.download_button(
                                "📥 Download Results (JSON)",
                                result_json,
                                file_name=f"classification_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
    
    with tab2:
        st.header("Batch Processing")
        st.info("Upload multiple images for batch classification")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files for processing")
            
            if st.button("🚀 Process Batch", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            image = Image.open(uploaded_file)
                            image_np = np.array(image)
                            
                            # Ensure RGB format
                            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                                image_np = image_np[:, :, :3]
                            elif len(image_np.shape) == 2:
                                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                            
                            result = classifier.classify_image(image_np)
                            result['filename'] = uploaded_file.name
                            results.append(result)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                if results:
                    st.success(f"Processed {len(results)} images successfully!")
                    
                    # Summary table
                    summary_data = []
                    for result in results:
                        if 'error' not in result:
                            pred = result['prediction']
                            summary_data.append({
                                'Filename': result['filename'],
                                'Prediction': pred['class_name'],
                                'Confidence': pred['confidence'],
                                'Patches': result['patch_analysis']['total_patches']
                            })
                    
                    if summary_data:
                        df = pd.DataFrame(summary_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CA Predictions", len(df[df['Prediction'].str.contains('albicans')]))
                        with col2:
                            st.metric("CG Predictions", len(df[df['Prediction'].str.contains('glabrata')]))
                        with col3:
                            st.metric("Mock Predictions", len(df[df['Prediction'].str.contains('Control')]))
                        
                        # Download batch results
                        results_json = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            "📥 Download Batch Results",
                            results_json,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
    
    with tab3:
        st.header("Results History")
        st.info("Feature coming soon: View and manage previous classification results")
        
        # Placeholder for results history
        st.write("This tab will show:")
        st.write("- Previous classification results")
        st.write("- Performance statistics")
        st.write("- Export capabilities")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Candida Infection Classification System**
    
    Powered by patch-based deep learning for accurate microscopy image analysis.
    Models trained on ResNet50 and EfficientNet architectures.
    """)

if __name__ == "__main__":
    main()