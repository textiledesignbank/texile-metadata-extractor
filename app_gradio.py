"""
Textile Image Metadata Extractor MVP - Gradio Version
CTO Demo - 2026.01.12

Usage:
    python app_gradio.py
"""

import gradio as gr
import google.generativeai as genai
from PIL import Image
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ============================================
# 설정
# ============================================

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# 모델 설정
MODEL_OPTIONS = {
    "Gemini 2.5 Flash-Lite (최저가 - 1200개 $0.31)": {
        "id": "gemini-2.5-flash-lite",
        "input_cost": 0.10 / 1_000_000,
        "output_cost": 0.40 / 1_000_000,
        "tokens_per_image": 560,
    },
    "Gemini 3 Flash (고품질 - 1200개 $2.14)": {
        "id": "gemini-3-flash",
        "input_cost": 0.50 / 1_000_000,
        "output_cost": 3.00 / 1_000_000,
        "tokens_per_image": 560,
    },
}

TOKENS_PER_OUTPUT = 500

# 전역 비용 추적
total_cost = 0.0
image_count = 0

# ============================================
# 분석 프롬프트
# ============================================

ANALYSIS_PROMPT = """You are an expert textile design analyst. Analyze this textile/pattern design image.

Return ONLY valid JSON with this structure:
{
  "category": {
    "primary": "floral/geometric/ethnic/animal/nature/abstract/novelty",
    "secondary": ["subcategories"],
    "confidence": 0.0-1.0
  },
  "colors": {
    "dominant": ["#hex1", "#hex2", "#hex3"],
    "palette_name": "descriptive name",
    "mood": "warm/cool/neutral/vibrant/muted"
  },
  "style": {
    "type": "style name",
    "era": "time period if applicable",
    "technique": "apparent technique"
  },
  "pattern": {
    "scale": "small/medium/large",
    "repeat_type": "block/brick/half-drop/mirror/random",
    "density": "sparse/moderate/dense"
  },
  "mood": {
    "primary": "main mood",
    "secondary": ["other moods"]
  },
  "keywords": {
    "search_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "description": "One sentence description"
  },
  "usage_suggestion": {
    "products": ["product1", "product2"],
    "season": ["season1"],
    "target_market": ["market1"]
  }
}

Return ONLY the JSON, no other text."""


# ============================================
# 분석 함수
# ============================================

def analyze_image(image, model_name):
    """이미지 분석"""
    global total_cost, image_count

    if image is None:
        return "이미지를 업로드해주세요.", "", "", "", get_cost_summary()

    if not API_KEY:
        return "❌ GEMINI_API_KEY가 설정되지 않았습니다.", "", "", "", get_cost_summary()

    model_config = MODEL_OPTIONS[model_name]
    model = genai.GenerativeModel(model_config["id"])

    try:
        # PIL Image로 변환
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # API 호출
        response = model.generate_content([ANALYSIS_PROMPT, image])
        result_text = response.text.strip()

        # JSON 블록 추출
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])

        metadata = json.loads(result_text)

        # 비용 계산
        input_cost = model_config["tokens_per_image"] * model_config["input_cost"]
        output_cost = TOKENS_PER_OUTPUT * model_config["output_cost"]
        this_cost = input_cost + output_cost
        total_cost += this_cost
        image_count += 1

        # 결과 포맷팅
        category_text = f"""## 📂 카테고리
- **Primary**: {metadata.get('category', {}).get('primary', 'N/A')}
- **Secondary**: {', '.join(metadata.get('category', {}).get('secondary', []))}
- **Confidence**: {metadata.get('category', {}).get('confidence', 0):.0%}"""

        colors = metadata.get('colors', {})
        colors_text = f"""## 🎨 색상
- **Dominant**: {' '.join(colors.get('dominant', []))}
- **Palette**: {colors.get('palette_name', 'N/A')}
- **Mood**: {colors.get('mood', 'N/A')}"""

        style = metadata.get('style', {})
        mood = metadata.get('mood', {})
        style_text = f"""## ✨ 스타일 & 무드
- **Style**: {style.get('type', 'N/A')}
- **Era**: {style.get('era', 'N/A')}
- **Technique**: {style.get('technique', 'N/A')}
- **Mood**: {mood.get('primary', 'N/A')} ({', '.join(mood.get('secondary', []))})"""

        keywords = metadata.get('keywords', {})
        usage = metadata.get('usage_suggestion', {})
        keywords_text = f"""## 🏷️ 키워드 & 용도
- **Tags**: {', '.join(keywords.get('search_tags', []))}
- **Description**: {keywords.get('description', 'N/A')}
- **Products**: {', '.join(usage.get('products', []))}
- **Season**: {', '.join(usage.get('season', []))}
- **Target**: {', '.join(usage.get('target_market', []))}"""

        return category_text, colors_text, style_text, keywords_text, get_cost_summary()

    except json.JSONDecodeError as e:
        return f"❌ JSON 파싱 오류: {e}", "", "", "", get_cost_summary()
    except Exception as e:
        return f"❌ 오류: {e}", "", "", "", get_cost_summary()


def get_cost_summary():
    """비용 요약"""
    global total_cost, image_count

    if image_count > 0:
        avg_cost = total_cost / image_count
        projected = avg_cost * 1200
    else:
        # 기본 모델 기준 이론값
        projected = 0.31

    return f"""## 💰 비용 대시보드

| 항목 | 값 |
|-----|-----|
| 분석된 이미지 | **{image_count}**개 |
| 현재 총 비용 | **${total_cost:.4f}** |
| 1200개 예상 비용 | **${projected:.2f}** (약 {int(projected * 1300)}원) |"""


def reset_stats():
    """통계 초기화"""
    global total_cost, image_count
    total_cost = 0.0
    image_count = 0
    return get_cost_summary()


def batch_analyze(files, model_name, progress=gr.Progress()):
    """배치 분석"""
    global total_cost, image_count

    if not files:
        return "파일을 업로드해주세요.", get_cost_summary()

    results = []

    for i, file in enumerate(progress.tqdm(files, desc="분석 중...")):
        image = Image.open(file.name)
        cat, col, style, kw, _ = analyze_image(image, model_name)

        results.append(f"""
### 📁 {os.path.basename(file.name)}
{cat}
{col}
{style}
{kw}
---
""")

    return "\n".join(results), get_cost_summary()


# ============================================
# Gradio UI
# ============================================

with gr.Blocks(
    title="Textile Metadata Extractor",
    theme=gr.themes.Soft(),
    css="""
    .main-title { text-align: center; margin-bottom: 20px; }
    .cost-box { background: #f0f9ff; padding: 15px; border-radius: 10px; }
    """
) as demo:

    gr.Markdown(
        """
        # 🎨 텍스타일 이미지 메타데이터 추출기
        ### CTO Demo - LLM Vision API를 활용한 이미지 분류 및 메타데이터 추출
        """,
        elem_classes="main-title"
    )

    with gr.Row():
        # 왼쪽: 입력
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=list(MODEL_OPTIONS.keys())[0],
                label="🤖 Vision 모델 선택"
            )

            gr.Markdown("---")

            with gr.Tab("단일 이미지"):
                image_input = gr.Image(
                    label="이미지 업로드",
                    type="pil",
                    height=300
                )
                analyze_btn = gr.Button("🚀 분석하기", variant="primary")

            with gr.Tab("배치 분석"):
                files_input = gr.File(
                    label="여러 이미지 업로드",
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_btn = gr.Button("🚀 일괄 분석", variant="primary")

            gr.Markdown("---")

            cost_display = gr.Markdown(
                value=get_cost_summary(),
                elem_classes="cost-box"
            )

            reset_btn = gr.Button("🔄 통계 초기화")

        # 오른쪽: 결과
        with gr.Column(scale=2):
            gr.Markdown("## 📊 분석 결과")

            with gr.Row():
                category_output = gr.Markdown(label="카테고리")
                colors_output = gr.Markdown(label="색상")

            with gr.Row():
                style_output = gr.Markdown(label="스타일")
                keywords_output = gr.Markdown(label="키워드")

            gr.Markdown("---")

            batch_output = gr.Markdown(label="배치 결과", visible=True)

    # 이벤트 연결
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, model_dropdown],
        outputs=[category_output, colors_output, style_output, keywords_output, cost_display]
    )

    batch_btn.click(
        fn=batch_analyze,
        inputs=[files_input, model_dropdown],
        outputs=[batch_output, cost_display]
    )

    reset_btn.click(
        fn=reset_stats,
        outputs=[cost_display]
    )

    # 하단 정보
    gr.Markdown(
        """
        ---
        ### 💡 비용 정보 (2026년 1월 기준)

        | 모델 | 1200개 비용 | 특징 |
        |-----|------------|------|
        | **Gemini 2.5 Flash-Lite** | **$0.31 (400원)** | 최저가, 기본 분석 |
        | **Gemini 3 Flash** | **$2.14 (2,800원)** | 고품질, 상세 분석 |
        """
    )


if __name__ == "__main__":
    if not API_KEY:
        print("⚠️  GEMINI_API_KEY가 설정되지 않았습니다!")
        print("    .env 파일에 API 키를 설정해주세요.")
        print("    예: GEMINI_API_KEY=your_api_key_here")
        print()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
