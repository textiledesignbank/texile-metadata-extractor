"""
Textile Image Metadata Extractor MVP
5개 모델 비교 테스트 버전 - 2026.01.13

Usage:
    streamlit run app.py
"""

import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import json
import os
import io
import base64
import time
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# 환경변수 로드
load_dotenv()

# ============================================
# API 설정 (로컬: .env, 배포: Streamlit Secrets)
# ============================================

def get_api_key(key_name: str) -> str:
    """API 키 가져오기 (Streamlit Secrets 또는 환경변수)"""
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)

GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")

# 어드민 계정 (로컬: .env, 배포: Streamlit Secrets)
ADMIN_USERNAME = get_api_key("ADMIN_USERNAME") or "admin"
ADMIN_PASSWORD = get_api_key("ADMIN_PASSWORD") or "admin123"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None


# ============================================
# 로그인 기능
# ============================================

def show_login_page():
    """로그인 페이지 표시"""
    st.title("🔐 로그인")
    st.caption("텍스타일 이미지 메타데이터 추출기")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("아이디")
            password = st.text_input("비밀번호", type="password")
            submit = st.form_submit_button("로그인", use_container_width=True)

            if submit:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다.")


def check_login():
    """로그인 상태 확인"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in


def logout():
    """로그아웃"""
    st.session_state.logged_in = False
    st.rerun()

# 환율
EXCHANGE_RATE = 1470  # ₩1,470/$1

# ============================================
# SQLite 데이터베이스 설정
# ============================================

DB_PATH = "results.db"

def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            model TEXT NOT NULL,
            resolution TEXT NOT NULL,
            success INTEGER NOT NULL,
            metadata JSON,
            cost_usd REAL,
            cost_krw REAL,
            elapsed_time REAL,
            error TEXT,
            image_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 기존 테이블에 image_data 컬럼이 없으면 추가
    cursor.execute("PRAGMA table_info(analysis_results)")
    columns = [col[1] for col in cursor.fetchall()]
    if "image_data" not in columns:
        cursor.execute("ALTER TABLE analysis_results ADD COLUMN image_data TEXT")

    conn.commit()
    conn.close()

def save_result_to_db(result_data: dict):
    """분석 결과를 DB에 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 이미지를 base64로 인코딩
    image_data = None
    if "image" in result_data and result_data["image"] is not None:
        image_data = image_to_base64(result_data["image"])

    cursor.execute("""
        INSERT INTO analysis_results
        (filename, model, resolution, success, metadata, cost_usd, cost_krw, elapsed_time, error, image_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result_data.get("filename"),
        result_data.get("model"),
        result_data.get("resolution"),
        1 if result_data.get("result", {}).get("success") else 0,
        json.dumps(result_data.get("result", {}).get("metadata")) if result_data.get("result", {}).get("success") else None,
        result_data.get("result", {}).get("cost", {}).get("total", 0),
        result_data.get("result", {}).get("cost", {}).get("krw", 0),
        result_data.get("result", {}).get("elapsed_time", 0),
        result_data.get("result", {}).get("error"),
        image_data
    ))

    conn.commit()
    conn.close()

def load_results_from_db(limit: int = 100, offset: int = 0, model_filter: str = None, resolution_filter: str = None, success_filter: str = None):
    """DB에서 분석 결과 불러오기 (페이지네이션 + 필터 지원)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 동적 WHERE 절 구성
    conditions = []
    params = []

    if model_filter and model_filter != "전체":
        conditions.append("model = ?")
        params.append(model_filter)

    if resolution_filter and resolution_filter != "전체":
        conditions.append("resolution = ?")
        params.append(resolution_filter)

    if success_filter == "성공만":
        conditions.append("success = 1")
    elif success_filter == "실패만":
        conditions.append("success = 0")

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT * FROM analysis_results
        {where_clause}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cursor.execute(query, params)

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "filename": row["filename"],
            "model": row["model"],
            "resolution": row["resolution"],
            "success": bool(row["success"]),
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "cost_usd": row["cost_usd"],
            "cost_krw": row["cost_krw"],
            "elapsed_time": row["elapsed_time"],
            "error": row["error"],
            "image_data": row["image_data"] if "image_data" in row.keys() else None,
            "created_at": row["created_at"]
        })

    return results


def get_filtered_count(model_filter: str = None, resolution_filter: str = None, success_filter: str = None) -> int:
    """필터 적용된 결과 개수 조회"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    conditions = []
    params = []

    if model_filter and model_filter != "전체":
        conditions.append("model = ?")
        params.append(model_filter)

    if resolution_filter and resolution_filter != "전체":
        conditions.append("resolution = ?")
        params.append(resolution_filter)

    if success_filter == "성공만":
        conditions.append("success = 1")
    elif success_filter == "실패만":
        conditions.append("success = 0")

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    cursor.execute(f"SELECT COUNT(*) FROM analysis_results {where_clause}", params)
    count = cursor.fetchone()[0]
    conn.close()

    return count

def get_db_stats():
    """DB 통계 조회"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(cost_usd) FROM analysis_results WHERE success = 1")
    total_cost = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT model, COUNT(*) as count, SUM(cost_usd) as cost
        FROM analysis_results
        WHERE success = 1
        GROUP BY model
    """)
    model_stats = cursor.fetchall()

    conn.close()

    return {
        "total_count": total_count,
        "total_cost_usd": total_cost,
        "total_cost_krw": total_cost * EXCHANGE_RATE,
        "model_stats": model_stats
    }

# DB 초기화
init_db()

# ============================================
# 모델 설정 (비용 기준 상위 5개)
# ============================================

MODEL_OPTIONS = {
    "gemini-2.0-flash-lite": {
        "name": "1. Gemini 2.0 Flash-Lite (최저가)",
        "provider": "gemini",
        "input_cost": 0.075 / 1_000_000,
        "output_cost": 0.30 / 1_000_000,
        "tokens_per_image": {
            "low": 280,
            "medium": 560,
            "high": 1120,
        },
        "supports_resolution": True,
    },
    "gemini-2.5-flash-lite": {
        "name": "2. Gemini 2.5 Flash-Lite",
        "provider": "gemini",
        "input_cost": 0.10 / 1_000_000,
        "output_cost": 0.40 / 1_000_000,
        "tokens_per_image": {
            "low": 280,
            "medium": 560,
            "high": 1120,
        },
        "supports_resolution": True,
    },
    "gpt-5-mini": {
        "name": "3. GPT-5-mini (OpenAI)",
        "provider": "openai",
        "input_cost": 0.25 / 1_000_000,
        "output_cost": 2.00 / 1_000_000,
        "tokens_per_image": {
            "low": 85,
            "medium": 85,  # OpenAI low는 고정
            "high": 765,   # 1024x1024 기준
        },
        "supports_resolution": True,
    },
    "gemini-2.5-flash": {
        "name": "4. Gemini 2.5 Flash",
        "provider": "gemini",
        "input_cost": 0.30 / 1_000_000,
        "output_cost": 2.50 / 1_000_000,
        "tokens_per_image": {
            "low": 280,
            "medium": 560,
            "high": 1120,
        },
        "supports_resolution": True,
    },
    "gemini-3-flash-preview": {
        "name": "5. Gemini 3 Flash (최신)",
        "provider": "gemini",
        "input_cost": 0.50 / 1_000_000,
        "output_cost": 3.00 / 1_000_000,
        "tokens_per_image": {
            "low": 280,
            "medium": 560,
            "high": 1120,
        },
        "supports_resolution": True,
    },
}

TOKENS_PER_OUTPUT = 500  # 예상 출력 토큰
TOKENS_PER_PROMPT = 200  # 프롬프트 토큰

# ============================================
# 분석 프롬프트
# ============================================

SYSTEM_PROMPT = """You are an expert textile design analyst. Analyze the uploaded pattern/textile design image and extract structured metadata.

Your analysis must be:
1. Accurate - Based on visual evidence in the image
2. Specific - Use precise terminology for textile/fashion industry
3. Searchable - Generate keywords that designers would use to find this pattern

Output your analysis as valid JSON only, no additional text."""

# 카테고리 목록 (영어)
CATEGORY_OPTIONS = [
    "Natural", "Traditional", "Floral", "Ethnic", "Abstract", "Stripe",
    "Tropical", "Camouflage", "Geometric", "Animal", "Conversational",
    "Check", "Paisley", "Tie-dye", "Animal Skins", "Dot", "Heart",
    "Star", "Ditsy", "Patchwork"
]

ANALYSIS_PROMPT = f"""Analyze this textile/pattern design image and provide metadata in the following JSON structure:

{{
  "category": {{
    "matches": ["top 3 categories in order of relevance - first is most relevant (MUST be from: {', '.join(CATEGORY_OPTIONS)})"],
    "confidence": 0.0-1.0
  }},
  "colors": {{
    "dominant": ["#hex1", "#hex2", "#hex3"],
    "palette_name": "descriptive name",
    "mood": "warm/cool/neutral/vibrant/muted"
  }},
  "style": {{
    "type": "style name",
    "era": "time period if applicable",
    "technique": "apparent technique"
  }},
  "pattern": {{
    "scale": "small/medium/large",
    "repeat_type": "block/brick/half-drop/mirror/random",
    "density": "sparse/moderate/dense"
  }},
  "mood": {{
    "primary": "main mood",
    "secondary": ["other moods"]
  }},
  "keywords": {{
    "search_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "description": "One sentence description for search"
  }},
  "usage_suggestion": {{
    "products": ["product1", "product2"],
    "season": ["season1"],
    "target_market": ["market1"]
  }}
}}

Return ONLY the JSON, no other text."""


# ============================================
# 이미지 변환 유틸리티
# ============================================

def image_to_base64(image: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """이미지 전처리 (리사이즈)"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 비율 유지하면서 리사이즈
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return image


# ============================================
# 분석 함수
# ============================================

def analyze_with_gemini(image: Image.Image, model_id: str, resolution: str) -> dict:
    """Gemini API로 이미지 분석"""
    model_config = MODEL_OPTIONS[model_id]

    # 해상도 설정
    resolution_map = {
        "low": "media_resolution_low",
        "medium": "media_resolution_medium",
        "high": "media_resolution_high",
    }

    try:
        model = genai.GenerativeModel(model_id)

        start_time = time.time()

        # API 호출
        response = model.generate_content(
            [SYSTEM_PROMPT, ANALYSIS_PROMPT, image],
            generation_config={
                "response_mime_type": "application/json",
            }
        )

        elapsed_time = time.time() - start_time

        # JSON 파싱
        result_text = response.text.strip()

        # JSON 블록 추출
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])

        metadata = json.loads(result_text)

        # 비용 계산
        tokens_image = model_config["tokens_per_image"][resolution]
        input_cost = (tokens_image + TOKENS_PER_PROMPT) * model_config["input_cost"]
        output_cost = TOKENS_PER_OUTPUT * model_config["output_cost"]
        total_cost = input_cost + output_cost

        return {
            "success": True,
            "metadata": metadata,
            "cost": {
                "input": input_cost,
                "output": output_cost,
                "total": total_cost,
                "krw": total_cost * EXCHANGE_RATE,
            },
            "elapsed_time": elapsed_time,
            "model": model_id,
            "resolution": resolution,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON 파싱 오류: {str(e)}",
            "raw_response": response.text if 'response' in locals() else None,
            "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
            "model": model_id,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
            "model": model_id,
        }


def analyze_with_openai(image: Image.Image, model_id: str, resolution: str) -> dict:
    """OpenAI API로 이미지 분석 (GPT-4o, GPT-5 시리즈 지원)"""
    if not openai_client:
        return {
            "success": False,
            "error": "OPENAI_API_KEY가 설정되지 않았습니다.",
            "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
            "model": model_id,
        }

    model_config = MODEL_OPTIONS[model_id]
    detail = "low" if resolution in ["low", "medium"] else "high"
    is_gpt5 = model_id.startswith("gpt-5")

    try:
        base64_image = image_to_base64(image)
        start_time = time.time()

        # 메시지 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": detail
                        }
                    }
                ]
            }
        ]

        # GPT-5 vs 이전 모델 파라미터 분기
        if is_gpt5:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_completion_tokens=2000,
                response_format={"type": "json_object"}
            )
        else:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

        elapsed_time = time.time() - start_time

        # 응답 파싱
        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason

        # 응답 검증
        if not content or not content.strip():
            return {
                "success": False,
                "error": f"빈 응답 | finish_reason: {finish_reason}",
                "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
                "model": model_id,
            }

        # JSON 파싱
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])

        metadata = json.loads(content)

        # 비용 계산
        tokens_image = model_config["tokens_per_image"][resolution]
        input_cost = (tokens_image + TOKENS_PER_PROMPT) * model_config["input_cost"]
        output_cost = TOKENS_PER_OUTPUT * model_config["output_cost"]
        total_cost = input_cost + output_cost

        return {
            "success": True,
            "metadata": metadata,
            "cost": {
                "input": input_cost,
                "output": output_cost,
                "total": total_cost,
                "krw": total_cost * EXCHANGE_RATE,
            },
            "elapsed_time": elapsed_time,
            "model": model_id,
            "resolution": resolution,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON 파싱 오류: {str(e)[:100]}",
            "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
            "model": model_id,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
            "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
            "model": model_id,
        }


def analyze_image(image: Image.Image, model_id: str, resolution: str = "medium") -> dict:
    """이미지 분석 (모델에 따라 적절한 API 호출)"""
    model_config = MODEL_OPTIONS[model_id]
    provider = model_config["provider"]

    if provider == "gemini":
        return analyze_with_gemini(image, model_id, resolution)
    elif provider == "openai":
        return analyze_with_openai(image, model_id, resolution)
    else:
        return {
            "success": False,
            "error": f"Unknown provider: {provider}",
            "cost": {"input": 0, "output": 0, "total": 0, "krw": 0},
            "model": model_id,
        }


# ============================================
# Streamlit UI
# ============================================

def show_detail_inline(result: dict):
    """분석 결과 상세 정보 인라인 표시"""
    st.subheader(f"📋 상세 정보: #{result['id']} - {result['filename']}")

    col1, col2 = st.columns([1, 2])

    with col1:
        # 이미지 표시
        if result.get("image_data"):
            st.image(
                f"data:image/png;base64,{result['image_data']}",
                caption=result['filename'],
                use_container_width=True
            )
        else:
            st.warning("이미지 없음")

        # 기본 정보
        st.markdown("**기본 정보**")
        st.caption(f"모델: {result['model']}")
        st.caption(f"해상도: {result['resolution']}")
        st.caption(f"비용: ₩{result['cost_krw']:.2f}")
        st.caption(f"시간: {result['elapsed_time']:.2f}s")
        st.caption(f"일시: {result['created_at']}")

    with col2:
        metadata = result["metadata"]

        # 카테고리
        category = metadata.get("category", {})
        matches = category.get("matches", [])
        if matches:
            st.markdown(f"### 카테고리: {', '.join(matches)}")
        else:
            # 기존 형식 호환
            primary = category.get('primary', 'N/A')
            secondary = category.get('secondary', [])
            all_cats = [primary] + secondary if secondary else [primary]
            st.markdown(f"### 카테고리: {', '.join(all_cats)}")
        if category.get("confidence"):
            st.caption(f"신뢰도: {category.get('confidence', 0):.0%}")

        # 스타일 & 무드
        col_a, col_b = st.columns(2)
        with col_a:
            style = metadata.get("style", {})
            st.markdown(f"**스타일:** {style.get('type', 'N/A')}")
            if style.get("era"):
                st.caption(f"시대: {style.get('era')}")
            if style.get("technique"):
                st.caption(f"기법: {style.get('technique')}")
        with col_b:
            mood = metadata.get("mood", {})
            st.markdown(f"**무드:** {mood.get('primary', 'N/A')}")
            if mood.get("secondary"):
                st.caption(f"부가: {', '.join(mood.get('secondary', []))}")

        # 색상
        colors = metadata.get("colors", {})
        dominant = colors.get("dominant", [])
        if dominant:
            color_html = " ".join([
                f'<span style="background-color:{c};padding:8px 16px;border-radius:4px;margin:2px;border:1px solid #ccc;">{c}</span>'
                for c in dominant
            ])
            st.markdown(f"**색상:** {color_html}", unsafe_allow_html=True)
            if colors.get("palette_name"):
                st.caption(f"팔레트: {colors.get('palette_name')} | 무드: {colors.get('mood', 'N/A')}")

        # 패턴
        pattern = metadata.get("pattern", {})
        if pattern:
            st.markdown(f"**패턴:** {pattern.get('scale', 'N/A')} / {pattern.get('repeat_type', 'N/A')} / {pattern.get('density', 'N/A')}")

        # 키워드
        keywords = metadata.get("keywords", {})
        if keywords.get("search_tags"):
            st.markdown(f"**키워드:** `{'`, `'.join(keywords.get('search_tags', []))}`")
        if keywords.get("description"):
            st.info(keywords.get("description"))

        # 용도 제안
        usage = metadata.get("usage_suggestion", {})
        if usage:
            st.markdown("**용도 제안:**")
            usage_text = []
            if usage.get("products"):
                usage_text.append(f"제품: {', '.join(usage.get('products', []))}")
            if usage.get("season"):
                usage_text.append(f"시즌: {', '.join(usage.get('season', []))}")
            if usage.get("target_market"):
                usage_text.append(f"타겟: {', '.join(usage.get('target_market', []))}")
            st.caption(" | ".join(usage_text))

    # 전체 JSON
    with st.expander("📄 전체 JSON 데이터"):
        st.json(metadata)


@st.dialog("📋 상세 정보", width="large")
def show_detail_dialog(result: dict):
    """분석 결과 상세 정보 다이얼로그"""
    col1, col2 = st.columns([1, 2])

    with col1:
        # 이미지 표시
        if result.get("image_data"):
            st.image(
                f"data:image/png;base64,{result['image_data']}",
                caption=result['filename'],
                use_container_width=True
            )
        else:
            st.warning("이미지 없음")

        # 기본 정보
        st.markdown("**기본 정보**")
        st.caption(f"ID: #{result['id']}")
        st.caption(f"파일: {result['filename']}")
        st.caption(f"모델: {result['model']}")
        st.caption(f"해상도: {result['resolution']}")
        st.caption(f"비용: ₩{result['cost_krw']:.2f}")
        st.caption(f"시간: {result['elapsed_time']:.2f}s")
        st.caption(f"일시: {result['created_at']}")

    with col2:
        metadata = result["metadata"]

        # 카테고리
        category = metadata.get("category", {})
        matches = category.get("matches", [])
        if matches:
            st.markdown(f"### 카테고리: {', '.join(matches)}")
        else:
            # 기존 형식 호환
            primary = category.get('primary', 'N/A')
            secondary = category.get('secondary', [])
            all_cats = [primary] + secondary if secondary else [primary]
            st.markdown(f"### 카테고리: {', '.join(all_cats)}")
        if category.get("confidence"):
            st.caption(f"신뢰도: {category.get('confidence', 0):.0%}")

        # 스타일
        style = metadata.get("style", {})
        st.markdown(f"**스타일:** {style.get('type', 'N/A')}")
        if style.get("era"):
            st.caption(f"시대: {style.get('era')}")
        if style.get("technique"):
            st.caption(f"기법: {style.get('technique')}")

        # 무드
        mood = metadata.get("mood", {})
        st.markdown(f"**무드:** {mood.get('primary', 'N/A')}")
        if mood.get("secondary"):
            st.caption(f"부가: {', '.join(mood.get('secondary', []))}")

        # 색상
        colors = metadata.get("colors", {})
        dominant = colors.get("dominant", [])
        if dominant:
            color_html = " ".join([
                f'<span style="background-color:{c};padding:8px 16px;border-radius:4px;margin:2px;border:1px solid #ccc;">{c}</span>'
                for c in dominant
            ])
            st.markdown(f"**색상:** {color_html}", unsafe_allow_html=True)
            if colors.get("palette_name"):
                st.caption(f"팔레트: {colors.get('palette_name')}")
            if colors.get("mood"):
                st.caption(f"색상 무드: {colors.get('mood')}")

        # 패턴
        pattern = metadata.get("pattern", {})
        if pattern:
            st.markdown(f"**패턴:** {pattern.get('scale', 'N/A')} / {pattern.get('repeat_type', 'N/A')} / {pattern.get('density', 'N/A')}")

        # 키워드
        keywords = metadata.get("keywords", {})
        if keywords.get("search_tags"):
            st.markdown(f"**키워드:** `{'`, `'.join(keywords.get('search_tags', []))}`")
        if keywords.get("description"):
            st.info(keywords.get("description"))

        # 용도 제안
        usage = metadata.get("usage_suggestion", {})
        if usage:
            st.markdown("**용도 제안:**")
            if usage.get("products"):
                st.caption(f"제품: {', '.join(usage.get('products', []))}")
            if usage.get("season"):
                st.caption(f"시즌: {', '.join(usage.get('season', []))}")
            if usage.get("target_market"):
                st.caption(f"타겟: {', '.join(usage.get('target_market', []))}")

    # 전체 JSON
    with st.expander("📄 전체 JSON 데이터"):
        st.json(metadata)


def main():
    st.set_page_config(
        page_title="Textile Metadata Extractor",
        page_icon="🎨",
        layout="wide"
    )

    # 로그인 체크
    if not check_login():
        show_login_page()
        st.stop()

    st.title("🎨 텍스타일 이미지 메타데이터 추출기")

    # API 키 확인
    api_status = []
    if GEMINI_API_KEY:
        api_status.append("✅ Gemini API")
    else:
        api_status.append("❌ Gemini API")

    if OPENAI_API_KEY:
        api_status.append("✅ OpenAI API")
    else:
        api_status.append("❌ OpenAI API")

    st.caption(" | ".join(api_status))

    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        st.error("⚠️ API 키가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
        st.code("GEMINI_API_KEY=your_gemini_key\nOPENAI_API_KEY=your_openai_key", language="bash")
        st.stop()

    # 세션 상태 초기화
    if "results" not in st.session_state:
        st.session_state.results = []
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []

    # ============================================
    # 사이드바
    # ============================================

    with st.sidebar:
        # 로그아웃 버튼
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"👤 {ADMIN_USERNAME}")
        with col2:
            if st.button("🚪", help="로그아웃"):
                logout()

        st.divider()
        st.header("⚙️ 설정")

        # 테스트 모드 선택
        test_mode = st.radio(
            "테스트 모드",
            ["단일 모델", "모델 비교"],
            help="단일 모델: 선택한 모델로만 분석\n모델 비교: 여러 모델로 동일 이미지 분석"
        )

        st.divider()

        if test_mode == "단일 모델":
            # 모델 선택
            available_models = []
            for model_id, config in MODEL_OPTIONS.items():
                if config["provider"] == "gemini" and GEMINI_API_KEY:
                    available_models.append(model_id)
                elif config["provider"] == "openai" and OPENAI_API_KEY:
                    available_models.append(model_id)

            selected_model = st.selectbox(
                "Vision 모델",
                options=available_models,
                format_func=lambda x: MODEL_OPTIONS[x]["name"],
                index=0
            )

            model_config = MODEL_OPTIONS[selected_model]
            st.caption(f"Input: ${model_config['input_cost']*1_000_000:.3f}/1M tokens")
            st.caption(f"Output: ${model_config['output_cost']*1_000_000:.2f}/1M tokens")

        else:
            # 비교할 모델 선택
            st.subheader("비교할 모델 선택")
            selected_models = []

            for model_id, config in MODEL_OPTIONS.items():
                disabled = False
                if config["provider"] == "gemini" and not GEMINI_API_KEY:
                    disabled = True
                elif config["provider"] == "openai" and not OPENAI_API_KEY:
                    disabled = True

                if st.checkbox(
                    config["name"],
                    value=not disabled,
                    disabled=disabled,
                    key=f"model_{model_id}"
                ):
                    selected_models.append(model_id)

        st.divider()

        # 해상도 설정
        resolution = st.select_slider(
            "이미지 해상도",
            options=["low", "medium", "high"],
            value="low",
            help="low: 최저 비용 (280 tokens)\nmedium: 기본 (560 tokens)\nhigh: 고품질 (1120 tokens)"
        )

        # 해상도별 토큰 수 표시
        st.caption(f"Gemini: {MODEL_OPTIONS['gemini-2.0-flash-lite']['tokens_per_image'][resolution]} tokens")
        st.caption(f"OpenAI: {MODEL_OPTIONS['gpt-5-mini']['tokens_per_image'][resolution]} tokens")

        st.divider()

        # 비용 대시보드
        st.header("💰 비용 대시보드")

        total_cost = sum(r["result"]["cost"]["total"] for r in st.session_state.results if r["result"]["success"])
        total_krw = total_cost * EXCHANGE_RATE
        image_count = len([r for r in st.session_state.results if r["result"]["success"]])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("분석 수", image_count)
        with col2:
            st.metric("총 비용", f"${total_cost:.4f}")

        st.metric("원화", f"₩{total_krw:.1f}")

        if image_count > 0:
            avg_cost = total_cost / image_count
            st.metric(
                "1200개 예상",
                f"₩{avg_cost * 1200 * EXCHANGE_RATE:.0f}"
            )

        st.divider()

        if st.button("🔄 결과 초기화", use_container_width=True):
            st.session_state.results = []
            st.session_state.comparison_results = []
            st.rerun()

        # DB 통계
        st.divider()
        st.header("💾 DB 저장 현황")

        db_stats = get_db_stats()
        st.metric("총 분석 수", db_stats["total_count"])
        st.metric("누적 비용", f"₩{db_stats['total_cost_krw']:.1f}")

        if db_stats["model_stats"]:
            st.caption("모델별 분석 수:")
            for model, count, cost in db_stats["model_stats"]:
                model_name = MODEL_OPTIONS.get(model, {}).get("name", model)
                st.caption(f"  • {model_name.split('. ')[-1]}: {count}건")

    # ============================================
    # 메인 영역
    # ============================================

    # 이미지 업로드
    uploaded_files = st.file_uploader(
        "텍스타일 이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="여러 이미지를 한 번에 업로드할 수 있습니다."
    )

    if uploaded_files:
        if test_mode == "단일 모델":
            # 단일 모델 테스트
            if st.button("🚀 분석 시작", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"분석 중... {idx + 1}/{len(uploaded_files)}: {file.name}")

                    image = Image.open(file)
                    image = preprocess_image(image)

                    result = analyze_image(image, selected_model, resolution)

                    result_data = {
                        "filename": file.name,
                        "image": image,
                        "result": result,
                        "model": selected_model,
                        "resolution": resolution,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.results.append(result_data)

                    # DB에 저장
                    save_result_to_db(result_data)

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.text("✅ 분석 완료! (DB 저장됨)")
                st.rerun()

        else:
            # 모델 비교 테스트 (병렬 처리)
            if not selected_models:
                st.warning("비교할 모델을 선택해주세요.")
            else:
                if st.button("🔬 모델 비교 테스트 (병렬)", type="primary", use_container_width=True):
                    for file in uploaded_files:
                        image = Image.open(file)
                        image = preprocess_image(image)

                        st.subheader(f"📁 {file.name}")

                        # 이미지 썸네일 표시
                        img_col, info_col = st.columns([1, 3])
                        with img_col:
                            st.image(image, caption=file.name, width=150)

                        # 병렬 API 호출
                        with st.spinner(f"🚀 {len(selected_models)}개 모델 병렬 분석 중..."):
                            results_map = {}

                            def analyze_model(model_id):
                                return model_id, analyze_image(image, model_id, resolution)

                            with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
                                futures = {executor.submit(analyze_model, m): m for m in selected_models}
                                for future in as_completed(futures):
                                    model_id, result = future.result()
                                    results_map[model_id] = result

                        # 결과 표시 (선택한 순서대로)
                        cols = st.columns(len(selected_models))
                        comparison = {"filename": file.name, "image": image, "results": {}}

                        for idx, model_id in enumerate(selected_models):
                            result = results_map[model_id]
                            comparison["results"][model_id] = result

                            # DB에 저장
                            save_result_to_db({
                                "filename": file.name,
                                "model": model_id,
                                "resolution": resolution,
                                "result": result,
                                "image": image
                            })

                            with cols[idx]:
                                model_name = MODEL_OPTIONS[model_id]["name"].split(". ")[1]
                                st.caption(f"**{model_name}**")

                                if result["success"]:
                                    st.success(f"✅ {result['elapsed_time']:.2f}s | ₩{result['cost']['krw']:.2f}")

                                    metadata = result["metadata"]
                                    cat_matches = metadata.get('category', {}).get('matches', [])
                                    cat_display = ', '.join(cat_matches) if cat_matches else metadata.get('category', {}).get('primary', 'N/A')
                                    st.markdown(f"**카테고리:** {cat_display}")
                                    st.markdown(f"**스타일:** {metadata.get('style', {}).get('type', 'N/A')}")
                                    st.markdown(f"**무드:** {metadata.get('mood', {}).get('primary', 'N/A')}")

                                    colors = metadata.get("colors", {}).get("dominant", [])
                                    if colors:
                                        color_html = " ".join([
                                            f'<span style="background-color:{c};padding:4px 10px;border-radius:3px;margin:1px;">&nbsp;</span>'
                                            for c in colors[:5]
                                        ])
                                        st.markdown(f"**색상:** {color_html}", unsafe_allow_html=True)

                                    keywords = metadata.get("keywords", {}).get("search_tags", [])
                                    if keywords:
                                        st.markdown(f"**키워드:** {', '.join(keywords[:5])}")

                                    with st.expander("전체 JSON"):
                                        st.json(metadata)
                                else:
                                    st.error(f"❌ {result.get('error', 'Error')[:50]}")

                        st.session_state.comparison_results.append(comparison)
                        st.divider()

    # ============================================
    # 결과 표시
    # ============================================

    if st.session_state.results:
        st.divider()
        st.subheader("📊 분석 결과")

        for item in reversed(st.session_state.results[-10:]):  # 최근 10개만 표시
            model_name = MODEL_OPTIONS[item['model']]['name']
            with st.expander(f"📁 {item['filename']} | {model_name}", expanded=False):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(item["image"], caption=item["filename"], width=200)
                    st.caption(f"모델: {model_name}")
                    st.caption(f"해상도: {item['resolution']}")
                    st.caption(f"비용: ${item['result']['cost']['total']:.5f} (₩{item['result']['cost']['krw']:.2f})")
                    if item['result'].get('elapsed_time'):
                        st.caption(f"시간: {item['result']['elapsed_time']:.2f}s")

                with col2:
                    if item["result"]["success"]:
                        metadata = item["result"]["metadata"]

                        cat_matches = metadata.get('category', {}).get('matches', [])
                        cat_display = ', '.join(cat_matches) if cat_matches else metadata.get('category', {}).get('primary', 'N/A')
                        st.markdown(f"**카테고리:** {cat_display}")
                        st.markdown(f"**스타일:** {metadata.get('style', {}).get('type', 'N/A')}")
                        st.markdown(f"**무드:** {metadata.get('mood', {}).get('primary', 'N/A')}")

                        colors = metadata.get("colors", {}).get("dominant", [])
                        if colors:
                            color_html = " ".join([
                                f'<span style="background-color:{c};padding:5px 15px;border-radius:3px;margin-right:5px;">&nbsp;</span>'
                                for c in colors[:5]
                            ])
                            st.markdown(f"**색상:** {color_html}", unsafe_allow_html=True)

                        keywords = metadata.get("keywords", {}).get("search_tags", [])
                        if keywords:
                            st.markdown(f"**키워드:** {', '.join(keywords)}")

                        with st.expander("전체 JSON"):
                            st.json(metadata)
                    else:
                        st.error(f"분석 실패: {item['result'].get('error', 'Unknown error')}")

        # CSV 다운로드
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 결과 CSV 다운로드", use_container_width=True):
                import pandas as pd

                rows = []
                for item in st.session_state.results:
                    if item["result"]["success"]:
                        m = item["result"]["metadata"]
                        cat_matches = m.get("category", {}).get("matches", [])
                        rows.append({
                            "filename": item["filename"],
                            "model": item["model"],
                            "resolution": item["resolution"],
                            "category": ", ".join(cat_matches) if cat_matches else m.get("category", {}).get("primary", ""),
                            "style": m.get("style", {}).get("type", ""),
                            "mood": m.get("mood", {}).get("primary", ""),
                            "colors": ", ".join(m.get("colors", {}).get("dominant", [])),
                            "keywords": ", ".join(m.get("keywords", {}).get("search_tags", [])),
                            "description": m.get("keywords", {}).get("description", ""),
                            "cost_usd": item["result"]["cost"]["total"],
                            "cost_krw": item["result"]["cost"]["krw"],
                            "elapsed_time": item["result"].get("elapsed_time", 0),
                        })

                if rows:
                    df = pd.DataFrame(rows)
                    csv = df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        "다운로드",
                        csv,
                        f"metadata_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )

        with col2:
            # 비용 요약
            if st.button("📊 비용 요약 보기", use_container_width=True):
                import pandas as pd

                summary = {}
                for item in st.session_state.results:
                    if item["result"]["success"]:
                        model = item["model"]
                        if model not in summary:
                            summary[model] = {"count": 0, "total_cost": 0, "total_time": 0}
                        summary[model]["count"] += 1
                        summary[model]["total_cost"] += item["result"]["cost"]["total"]
                        summary[model]["total_time"] += item["result"].get("elapsed_time", 0)

                if summary:
                    rows = []
                    for model, data in summary.items():
                        rows.append({
                            "모델": MODEL_OPTIONS[model]["name"],
                            "분석 수": data["count"],
                            "총 비용 ($)": f"${data['total_cost']:.4f}",
                            "총 비용 (₩)": f"₩{data['total_cost'] * EXCHANGE_RATE:.1f}",
                            "평균 시간": f"{data['total_time']/data['count']:.2f}s",
                            "1200개 예상": f"₩{(data['total_cost']/data['count']) * 1200 * EXCHANGE_RATE:.0f}",
                        })

                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ============================================
    # DB 저장 결과 조회 (페이지네이션)
    # ============================================

    st.divider()
    st.subheader("💾 저장된 분석 결과 (DB)")

    # 필터 옵션
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    with filter_col1:
        # 모델 필터
        model_options = ["전체"] + list(MODEL_OPTIONS.keys())
        selected_model_filter = st.selectbox(
            "모델 필터",
            options=model_options,
            format_func=lambda x: "전체" if x == "전체" else MODEL_OPTIONS.get(x, {}).get("name", x).split(". ")[-1]
        )

    with filter_col2:
        # 해상도 필터
        resolution_options = ["전체", "low", "medium", "high"]
        selected_resolution_filter = st.selectbox(
            "해상도 필터",
            options=resolution_options
        )

    with filter_col3:
        # 성공/실패 필터
        success_options = ["전체", "성공만", "실패만"]
        selected_success_filter = st.selectbox(
            "결과 필터",
            options=success_options
        )

    # 필터 적용된 총 개수 조회
    filtered_count = get_filtered_count(
        model_filter=selected_model_filter,
        resolution_filter=selected_resolution_filter,
        success_filter=selected_success_filter
    )
    db_stats = get_db_stats()
    total_count = db_stats["total_count"]

    # 필터 상태 표시
    if selected_model_filter != "전체" or selected_resolution_filter != "전체" or selected_success_filter != "전체":
        st.info(f"🔍 필터 적용됨: {filtered_count}건 / 전체 {total_count}건")

    # CSV 내보내기 버튼
    col_export1, col_export2 = st.columns([1, 3])
    with col_export1:
        if st.button("📥 전체 CSV 내보내기", use_container_width=True, disabled=total_count == 0):
            # 전체 데이터 조회
            all_results = load_results_from_db(limit=10000, offset=0)

            if all_results:
                import pandas as pd

                csv_rows = []
                for r in all_results:
                    row = {
                        "ID": r["id"],
                        "파일명": r["filename"],
                        "모델": r["model"],
                        "해상도": r["resolution"],
                        "성공": "Y" if r["success"] else "N",
                        "비용_USD": r["cost_usd"],
                        "비용_KRW": r["cost_krw"],
                        "소요시간": r["elapsed_time"],
                        "일시": r["created_at"],
                    }

                    # 메타데이터 필드 추가
                    if r["success"] and r["metadata"]:
                        m = r["metadata"]
                        cat_matches = m.get("category", {}).get("matches", [])
                        row["카테고리"] = ", ".join(cat_matches) if cat_matches else m.get("category", {}).get("primary", "")
                        row["스타일"] = m.get("style", {}).get("type", "")
                        row["스타일_시대"] = m.get("style", {}).get("era", "")
                        row["스타일_기법"] = m.get("style", {}).get("technique", "")
                        row["무드"] = m.get("mood", {}).get("primary", "")
                        row["무드_부가"] = ", ".join(m.get("mood", {}).get("secondary", []))
                        row["색상"] = ", ".join(m.get("colors", {}).get("dominant", []))
                        row["팔레트"] = m.get("colors", {}).get("palette_name", "")
                        row["색상무드"] = m.get("colors", {}).get("mood", "")
                        row["패턴_크기"] = m.get("pattern", {}).get("scale", "")
                        row["패턴_반복"] = m.get("pattern", {}).get("repeat_type", "")
                        row["패턴_밀도"] = m.get("pattern", {}).get("density", "")
                        row["키워드"] = ", ".join(m.get("keywords", {}).get("search_tags", []))
                        row["설명"] = m.get("keywords", {}).get("description", "")
                        row["추천제품"] = ", ".join(m.get("usage_suggestion", {}).get("products", []))
                        row["추천시즌"] = ", ".join(m.get("usage_suggestion", {}).get("season", []))
                        row["타겟마켓"] = ", ".join(m.get("usage_suggestion", {}).get("target_market", []))

                    csv_rows.append(row)

                df = pd.DataFrame(csv_rows)
                csv_data = df.to_csv(index=False, encoding="utf-8-sig")

                st.download_button(
                    label=f"📄 다운로드 ({total_count}건)",
                    data=csv_data,
                    file_name=f"textile_analysis_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with col_export2:
        st.caption(f"총 {total_count}건의 분석 결과가 저장되어 있습니다.")

    st.divider()

    # 페이지네이션 설정
    items_per_page = st.selectbox("페이지당 항목 수", [10, 20, 50], index=0)

    # 필터 적용된 개수로 페이지네이션
    display_count = filtered_count if (selected_model_filter != "전체" or selected_resolution_filter != "전체" or selected_success_filter != "전체") else total_count

    if display_count > 0:
        total_pages = (display_count + items_per_page - 1) // items_per_page

        # 페이지 선택
        if "db_page" not in st.session_state:
            st.session_state.db_page = 1

        # 필터 변경 시 페이지 리셋
        filter_key = f"{selected_model_filter}_{selected_resolution_filter}_{selected_success_filter}"
        if "last_filter_key" not in st.session_state:
            st.session_state.last_filter_key = filter_key
        if st.session_state.last_filter_key != filter_key:
            st.session_state.db_page = 1
            st.session_state.last_filter_key = filter_key

        # 페이지 범위 조정
        if st.session_state.db_page > total_pages:
            st.session_state.db_page = max(1, total_pages)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀ 이전", disabled=st.session_state.db_page <= 1):
                st.session_state.db_page -= 1
                st.rerun()
        with col2:
            st.markdown(f"<center>페이지 {st.session_state.db_page} / {total_pages} (총 {display_count}건)</center>", unsafe_allow_html=True)
        with col3:
            if st.button("다음 ▶", disabled=st.session_state.db_page >= total_pages):
                st.session_state.db_page += 1
                st.rerun()

        # 현재 페이지 데이터 조회 (필터 적용)
        offset = (st.session_state.db_page - 1) * items_per_page
        db_results = load_results_from_db(
            limit=items_per_page,
            offset=offset,
            model_filter=selected_model_filter,
            resolution_filter=selected_resolution_filter,
            success_filter=selected_success_filter
        )

        if db_results:
            import pandas as pd

            st.caption("📌 테이블에서 행을 선택하면 상세 정보를 볼 수 있습니다.")

            # 테이블 데이터 구성
            df_data = []
            for r in db_results:
                df_data.append({
                    "ID": r["id"],
                    "파일명": r["filename"],
                    "모델": MODEL_OPTIONS.get(r["model"], {}).get("name", r["model"]).split(". ")[-1],
                    "해상도": r["resolution"],
                    "성공": "✅" if r["success"] else "❌",
                    "비용(₩)": f"₩{r['cost_krw']:.2f}" if r["cost_krw"] else "-",
                    "시간(s)": f"{r['elapsed_time']:.2f}" if r["elapsed_time"] else "-",
                    "일시": r["created_at"][:16] if r["created_at"] else "-"
                })

            df = pd.DataFrame(df_data)

            # 선택 가능한 데이터프레임
            event = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="db_table"
            )

            # 선택된 행이 있으면 상세 정보 표시
            if event.selection and event.selection.rows:
                selected_row_idx = event.selection.rows[0]
                selected_id = df_data[selected_row_idx]["ID"]
                selected_result = next((r for r in db_results if r["id"] == selected_id), None)

                if selected_result and selected_result["success"] and selected_result["metadata"]:
                    show_detail_dialog(selected_result)
    else:
        st.info("저장된 결과가 없습니다.")


if __name__ == "__main__":
    main()
