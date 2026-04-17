import gradio as gr
import numpy as np
import cv2
from PIL import Image
from app.services.detection_service import get_detection_service
from app.services.comparison_service import ComparisonService
from app.core.config import get_settings

settings = get_settings()
detector = get_detection_service()
comparator = ComparisonService(detector)

def process_detection(image, conf_threshold):
    if image is None:
        return None, "يرجى رفع صورة أولاً."
    
    # Convert RGB (Gradio) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run detection
    response = detector.detect(img_bgr, confidence_threshold=conf_threshold, return_annotated=True)
    
    # Decode annotated image
    import base64
    from io import BytesIO
    annotated_data = base64.b64decode(response.annotated_image_base64)
    annotated_img = Image.open(BytesIO(annotated_data))
    
    # Format results table
    results = []
    for det in response.detections:
        status = "❌ مخالفة" if det.is_violation else "✅ طبيعي"
        results.append([det.class_name_ar, f"{det.confidence:.2%}", det.severity_ar, status])
    
    return annotated_img, results

def process_comparison(before_img, after_img, conf_threshold):
    if before_img is None or after_img is None:
        return "يرجى رفع الصورتين للمقارنة."
    
    # Convert to BGR
    img_before = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2BGR)
    img_after = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2BGR)
    
    # Run comparison
    resp = comparator.compare(img_before, img_after, confidence_threshold=conf_threshold)
    
    summary = f"""
    ### ملخص المقارنة:
    - **نسبة التشابه الهيكلي (SSIM):** {resp.ssim_score:.4f}
    - **نسبة التغيير المكتشفة:** {resp.change_percentage}%
    - **عدد المخالفات الجدیدة:** {len(resp.new_violations)}
    - **عدد المخالفات التي تم حلها:** {len(resp.resolved_violations)}
    """
    
    delta_results = []
    for v in resp.new_violations:
        delta_results.append(["مخالفة جديدة", v.class_name_ar, v.severity_ar])
    for v in resp.resolved_violations:
        delta_results.append(["تمت معالجتها", v.class_name_ar, v.severity_ar])
        
    return summary, delta_results

# Create Gradio Blocks
with gr.Blocks(title="Smart Inspection Vision System") as demo:
    gr.Markdown("# 🔍 نظام الرؤية الذكي للتفتيش والامتثال")
    gr.Markdown("قم برفع الصور للكشف عن المخالفات أو مقارنة حالة الموقع قبل وبعد.")
    
    with gr.Tabs():
        # --- Tab 1: Detection ---
        with gr.TabItem("الكشف عن المخالفات"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="ارفع صورة المعاينة")
                    conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, label="عتبة الثقة (Confidence)")
                    det_btn = gr.Button("بدء الكشف", variant="primary")
                with gr.Column():
                    output_annotated = gr.Image(label="النتيجة البصرية")
                    output_table = gr.Dataframe(headers=["العنصر", "الثقة", "الخطورة", "الحالة"], label="تفاصيل الكشف")
            
            det_btn.click(process_detection, inputs=[input_img, conf_slider], outputs=[output_annotated, output_table])

        # --- Tab 2: Comparison ---
        with gr.TabItem("مقارنة قبل وبعد"):
            with gr.Row():
                before_input = gr.Image(type="pil", label="الصورة قبل (Before)")
                after_input = gr.Image(type="pil", label="الصورة بعد (After)")
            
            comp_btn = gr.Button("مقارنة وتحليل التغيير", variant="primary")
            
            with gr.Row():
                comp_summary = gr.Markdown("نتائج المقارنة ستظهر هنا...")
                comp_delta_table = gr.Dataframe(headers=["نوع التغيير", "العنصر", "الخطورة"], label="تغييرات الامتثال")
            
            comp_btn.click(process_comparison, inputs=[before_input, after_input, conf_slider], outputs=[comp_summary, comp_delta_table])

    gr.Markdown("---")
    gr.Markdown("### نظام الرؤية الذكي للتفتيش - تطوير قسورة المحمدي")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
