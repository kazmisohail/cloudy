# libraries import kr rh h
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io
import base64
from datetime import datetime
import time

# main page ka setup
st.set_page_config(
    page_title="Cloud Detection & Classification System",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS se styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .uploadedFile {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# configuration
IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = 256, 256, 5
CLASS_NAMES = ['Fill', 'Clear', 'Shadow', 'Thin Cloud', 'Thick Cloud'] # ya 5 labels h
CLASS_COLORS = np.array([
    [255, 0, 0],      # Fill - Red
    [0, 255, 0],      # Clear - Green  
    [0, 0, 255],      # Shadow - Blue
    [255, 165, 0],    # Thin Cloud - Orange
    [128, 0, 128]     # Thick Cloud - Purple
], dtype=np.uint8)

# model load krne k functions
@st.cache_resource
def load_model():
    try:
        # custom loss functions jo hm ne train krte hue use kyaa
        def weighted_categorical_crossentropy(class_weights):
            def loss(y_true, y_pred):
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
                weights = tf.reduce_sum(class_weights * y_true, axis=-1)
                return loss * weights
            return loss

        def dice_coefficient(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
            y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
            intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
            union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return tf.reduce_mean(dice)

        def dice_loss(y_true, y_pred):
            return 1.0 - dice_coefficient(y_true, y_pred)

        def combined_loss(y_true, y_pred):
            class_weights = tf.constant([3.523, 0.412, 15.000, 1.442, 0.656])
            ce_loss = weighted_categorical_crossentropy(class_weights)(y_true, y_pred)
            d_loss = dice_loss(y_true, y_pred)
            return 0.1 * tf.reduce_mean(ce_loss) + 0.9 * d_loss

        def shadow_iou(y_true, y_pred):
            shadow_true = y_true[:, :, :, 2]
            shadow_pred = y_pred[:, :, :, 2]
            shadow_pred_binary = tf.cast(shadow_pred > 0.5, tf.float32)
            intersection = tf.reduce_sum(shadow_true * shadow_pred_binary)
            union = tf.reduce_sum(shadow_true + shadow_pred_binary) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            return iou

        custom_objects = {
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient,
            'shadow_iou': shadow_iou,
            'dice_loss': dice_loss
        }

        # model ki location
        model_path = "attention_unet_1.keras"
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image_array):
    # input size ko model k req size m convert kr rh h
    if len(image_array.shape) == 3:
        resized = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    else:
        resized = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # batch dimensions add kr rh
    return np.expand_dims(normalized, axis=0)

# mask img m convert kr rh
def postprocess_prediction(prediction):
    return np.argmax(prediction[0], axis=-1)

# mask ko RGB k acc set kr rh
def mask_to_rgb(mask):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        rgb_mask[mask == class_id] = CLASS_COLORS[class_id]
    return rgb_mask

# metrics calc kr rh
def calculate_metrics(true_mask, pred_mask):
    metrics = {}
    
    # accuracy
    metrics['pixel_accuracy'] = np.mean(true_mask == pred_mask)
    
    # per-class metrics
    class_metrics = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_class = (true_mask == class_id)
        pred_class = (pred_mask == class_id)
        
        # pixel counts
        true_count = np.sum(true_class)
        pred_count = np.sum(pred_class)
        
        # IoU calc
        if true_count == 0:
            iou = 1.0 if pred_count == 0 else 0.0
        else:
            intersection = np.sum(true_class & pred_class)
            union = np.sum(true_class | pred_class)
            iou = intersection / union if union > 0 else 0.0
        
        # precision or recall
        if pred_count > 0:
            precision = np.sum(true_class & pred_class) / pred_count
        else:
            precision = 1.0 if true_count == 0 else 0.0
            
        if true_count > 0:
            recall = np.sum(true_class & pred_class) / true_count
        else:
            recall = 1.0
        
        class_metrics[class_name] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'true_pixels': true_count,
            'pred_pixels': pred_count,
            'percentage': (pred_count / (IMG_HEIGHT * IMG_WIDTH)) * 100
        }
    
    metrics['per_class'] = class_metrics
    metrics['mean_iou'] = np.mean([class_metrics[name]['iou'] for name in CLASS_NAMES])
    
    return metrics

# metrics if ground truth available nh
def create_prediction_only_metrics(pred_mask):
    metrics = {}
    total_pixels = IMG_HEIGHT * IMG_WIDTH
    
    class_metrics = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        pred_class = (pred_mask == class_id)
        pred_count = np.sum(pred_class)
        percentage = (pred_count / total_pixels) * 100
        
        class_metrics[class_name] = {
            'pred_pixels': pred_count,
            'percentage': percentage
        }
    
    metrics['per_class'] = class_metrics
    return metrics

# main app
def main():
    # Header
    st.markdown('<h1 class="main-header">☁️ Satellite Cloud Detection & Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced cloud detection using Attention U-Net...!!</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🛠️ Model Information")
    st.sidebar.info("""
    **Model**: Attention U-Net
    **Classes**: Fill, Clear, Shadow, Thin Cloud, Thick Cloud
    **Input Size**: 256×256 pixels
    **Framework**: TensorFlow/Keras
    """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, error = load_model()
    
    if model is None:
        st.error(f"❌ Failed to load model: {error}")
        st.warning("Model sahi location pr nh hai...!!")
        st.stop()
    
    st.success("✅ Model load ho gyaa...!!")
    
    # File upload section
    st.header("📁 Satellite/Cloud Image Upload Kryy")
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        help="Upload a satellite image for cloud detection and classification"
    )
    
    # Demo option
    # col1, col2 = st.columns([3, 1])
    # with col1:
    #     st.write("Don't have a satellite image? Try our demo:")
    # with col2:
    #     use_demo = st.button("🎯 Use Demo Image", type="secondary")
    
    # if use_demo:
    #     # Create a demo image (you can replace this with an actual demo image)
    #     demo_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    #     # Add some cloud-like patterns
    #     center = (128, 128)
    #     cv2.circle(demo_image, center, 60, (255, 255, 255), -1)
    #     cv2.circle(demo_image, (180, 80), 40, (200, 200, 200), -1)
    #     uploaded_file = "demo"
    #     image_array = demo_image
    #     st.info("🎯 Demo image loaded! This is a synthetic image for demonstration.")
    
    if uploaded_file is not None:
        # Process uploaded file
        if uploaded_file != "demo":
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Display original image info
            st.subheader("Original Image")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Satellite Image Upload ho gya", use_container_width=True)
            
            with col2:
                st.markdown("**Image Information:**")
                st.write(f"**Dimensions:** {image_array.shape[1]} × {image_array.shape[0]} pixels")
                st.write(f"**Channels:** {image_array.shape[2] if len(image_array.shape) > 2 else 1}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**Format:** {image.format}")
        
        # Process with model
        st.header("AI Processing ho rh h...!!")
        
        with st.spinner("Image ko analyze kr rh with Attention U-Net...!!"):
            # Preprocess
            processed_image = preprocess_image(image_array)
            
            # Predict
            start_time = time.time()
            prediction = model.predict(processed_image, verbose=0)
            processing_time = time.time() - start_time
            
            # Postprocess
            pred_mask = postprocess_prediction(prediction)
            confidence_scores = prediction[0]
            
        st.success(f"✅ Processing completed in {processing_time:.2f} seconds!")
        
        # Results Section
        st.header("Detection Results")
        
        # Main visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            resized_original = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
            st.image(resized_original, caption="Input Image (256×256)", use_container_width=True)
        
        with col2:
            st.subheader("Segmentation Result")
            rgb_mask = mask_to_rgb(pred_mask)
            st.image(rgb_mask, caption="Detected Classes", use_container_width=True)
        
        with col3:
            st.subheader("Class Legend")
            legend_image = np.zeros((200, 50, 3), dtype=np.uint8)
            for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
                y_start = i * 40
                y_end = (i + 1) * 40
                legend_image[y_start:y_end, :] = color
            
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.imshow(legend_image)
            ax.set_yticks([20, 60, 100, 140, 180])
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xticks([])
            ax.set_title("Class Colors")
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)
            plt.close()
        
        # Detailed Metrics
        st.header("Detailed Analysis")

        # Calculate metrics
        pred_metrics = create_prediction_only_metrics(pred_mask)
        
        # Class distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            
            # Create pie chart
            percentages = [pred_metrics['per_class'][name]['percentage'] for name in CLASS_NAMES]
            colors_hex = ['#%02x%02x%02x' % tuple(color) for color in CLASS_COLORS]
            
            fig = go.Figure(data=[go.Pie(
                labels=CLASS_NAMES,
                values=percentages,
                marker_colors=colors_hex,
                textinfo='label+percent',
                hovertemplate='%{label}<br>%{percent}<br>%{value:.1f}%<extra></extra>'
            )])
            
            fig.update_layout(
                title="Land Cover Distribution",
                font=dict(size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Heatmaps")
            
            # Create confidence subplot
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=CLASS_NAMES,
                specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
            )
            
            for i, class_name in enumerate(CLASS_NAMES):
                row = i // 3 + 1
                col = i % 3 + 1
                
                confidence_map = confidence_scores[:, :, i]
                
                fig.add_trace(
                    go.Heatmap(
                        z=confidence_map,
                        colorscale='Blues',
                        showscale=False,
                        hovertemplate=f'{class_name}<br>Confidence: %{{z:.3f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=500, title_text="Model Confidence per Class")
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics table
        st.subheader("📋 Detailed Statistics")
        
        stats_data = []
        for class_name in CLASS_NAMES:
            class_data = pred_metrics['per_class'][class_name]
            stats_data.append({
                'Class': class_name,
                'Pixels': f"{class_data['pred_pixels']:,}",
                'Percentage': f"{class_data['percentage']:.2f}%",
                'Max Confidence': f"{np.max(confidence_scores[:, :, CLASS_NAMES.index(class_name)]):.3f}",
                'Mean Confidence': f"{np.mean(confidence_scores[:, :, CLASS_NAMES.index(class_name)]):.3f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Advanced Analysis
        st.header("🔬 Advanced Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cloud Coverage Analysis")
            
            # Calculate cloud metrics
            total_cloud_pixels = (pred_metrics['per_class']['Thin Cloud']['pred_pixels'] + 
                                pred_metrics['per_class']['Thick Cloud']['pred_pixels'])
            total_pixels = IMG_HEIGHT * IMG_WIDTH
            cloud_coverage = (total_cloud_pixels / total_pixels) * 100
            
            shadow_coverage = pred_metrics['per_class']['Shadow']['percentage']
            clear_coverage = pred_metrics['per_class']['Clear']['percentage']
            
            # Cloud coverage gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = cloud_coverage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Total Cloud Coverage (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightblue"},
                        {'range': [75, 100], 'color': "blue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            st.metric("Shadow Coverage", f"{shadow_coverage:.2f}%")
            st.metric("Clear Sky", f"{clear_coverage:.2f}%")
            
        with col2:
            st.subheader("Atmospheric Conditions")
            
            # Interpret results
            if cloud_coverage > 75:
                weather_condition = "Heavy Cloud Cover"
                weather_emoji = "☁️☁️☁️"
                weather_color = "red"
            elif cloud_coverage > 50:
                weather_condition = "Moderate Cloud Cover"
                weather_emoji = "⛅⛅"
                weather_color = "orange"
            elif cloud_coverage > 25:
                weather_condition = "Light Cloud Cover"
                weather_emoji = "🌤️"
                weather_color = "yellow"
            else:
                weather_condition = "Clear Sky"
                weather_emoji = "☀️"
                weather_color = "green"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {weather_color}20; border-radius: 10px;">
                <h2 style="color: {weather_color};">{weather_emoji}</h2>
                <h3>{weather_condition}</h3>
                <p><strong>Visibility:</strong> {'Poor' if shadow_coverage > 5 else 'Good'}</p>
                <p><strong>Suitable for:</strong> {'Ground observations' if cloud_coverage < 30 else 'Weather monitoring'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Shadow analysis
            if shadow_coverage > 1:
                st.warning(f"⚠️ Significant shadow presence detected ({shadow_coverage:.1f}%)")
            
            # Confidence analysis
            max_confidence = np.max(confidence_scores)
            mean_confidence = np.mean(confidence_scores)
            
            st.metric("Model Confidence", f"{mean_confidence:.3f}", f"{max_confidence:.3f} max")
        
        # Export Results
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Analysis Report", type="primary"):
                # Create analysis report
                report = f"""
Cloud Detection Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Weather Condition: {weather_condition}
- Total Cloud Coverage: {cloud_coverage:.2f}%
- Shadow Coverage: {shadow_coverage:.2f}%
- Clear Sky: {clear_coverage:.2f}%

DETAILED BREAKDOWN:
"""
                for class_name in CLASS_NAMES:
                    class_data = pred_metrics['per_class'][class_name]
                    report += f"- {class_name}: {class_data['percentage']:.2f}% ({class_data['pred_pixels']:,} pixels)\n"
                
                report += f"\nMODEL PERFORMANCE:\n- Mean Confidence: {mean_confidence:.3f}\n- Processing Time: {processing_time:.2f} seconds"
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"cloud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("🖼️ Download Segmentation Mask"):
                # Convert mask to image
                mask_image = Image.fromarray(rgb_mask)
                buf = io.BytesIO()
                mask_image.save(buf, format='PNG')
                buf.seek(0)
                
                st.download_button(
                    label="💾 Download Mask",
                    data=buf.getvalue(),
                    file_name=f"segmentation_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        
        with col3:
            if st.button("📈 Download Metrics Data"):
                # Create CSV with metrics
                metrics_df = pd.DataFrame([
                    {
                        'Class': name,
                        'Pixels': data['pred_pixels'],
                        'Percentage': data['percentage'],
                        'Max_Confidence': np.max(confidence_scores[:, :, i]),
                        'Mean_Confidence': np.mean(confidence_scores[:, :, i])
                    }
                    for i, (name, data) in enumerate(pred_metrics['per_class'].items())
                ])
                
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# metrics calc kr rh h for no ground truth
def create_prediction_only_metrics(pred_mask):
    metrics = {}
    total_pixels = IMG_HEIGHT * IMG_WIDTH
    
    class_metrics = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        pred_class = (pred_mask == class_id)
        pred_count = np.sum(pred_class)
        percentage = (pred_count / total_pixels) * 100
        
        class_metrics[class_name] = {
            'pred_pixels': pred_count,
            'percentage': percentage
        }
    
    metrics['per_class'] = class_metrics
    return metrics

if __name__ == "__main__":
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:** CLOUDY TEAM")
    st.sidebar.markdown("**University:** BAHRIA UNIVERSITY KARACHI CAMPUS")
    st.sidebar.markdown("**Year:** 2025-2026")

    main()
