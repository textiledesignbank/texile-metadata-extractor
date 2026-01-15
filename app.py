"""
Textile Image Metadata Extractor MVP
5개 모델 비교 테스트 버전 - 2026.01.13

Usage:
    streamlit run app.py
"""

import streamlit as st
import google.generativeai as genai
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

# 어드민 계정 (로컬: .env, 배포: Streamlit Secrets)
ADMIN_USERNAME = get_api_key("ADMIN_USERNAME") or "admin"
ADMIN_PASSWORD = get_api_key("ADMIN_PASSWORD") or "admin123"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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


def delete_results_from_db(ids: list) -> int:
    """DB에서 분석 결과 삭제"""
    if not ids:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ",".join(["?" for _ in ids])
    cursor.execute(f"DELETE FROM analysis_results WHERE id IN ({placeholders})", ids)

    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()

    return deleted_count


def get_model_comparison_stats():
    """모델별 상세 비교 통계 조회 (수치형 데이터)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 기본 통계 + 상세 통계
    cursor.execute("""
        SELECT
            model,
            resolution,
            COUNT(*) as total_count,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
            AVG(CASE WHEN success = 1 THEN cost_usd ELSE NULL END) as avg_cost,
            MIN(CASE WHEN success = 1 THEN cost_usd ELSE NULL END) as min_cost,
            MAX(CASE WHEN success = 1 THEN cost_usd ELSE NULL END) as max_cost,
            AVG(CASE WHEN success = 1 THEN elapsed_time ELSE NULL END) as avg_time,
            MIN(CASE WHEN success = 1 THEN elapsed_time ELSE NULL END) as min_time,
            MAX(CASE WHEN success = 1 THEN elapsed_time ELSE NULL END) as max_time,
            SUM(CASE WHEN success = 1 THEN cost_usd ELSE 0 END) as total_cost
        FROM analysis_results
        GROUP BY model, resolution
        ORDER BY model, resolution
    """)

    rows = cursor.fetchall()

    # 표준편차 계산을 위한 추가 쿼리
    stats = []
    for row in rows:
        model, resolution, total, success, avg_cost, min_cost, max_cost, avg_time, min_time, max_time, total_cost = row

        # 표준편차 계산
        cursor.execute("""
            SELECT
                AVG((elapsed_time - ?) * (elapsed_time - ?)) as time_variance,
                AVG((cost_usd - ?) * (cost_usd - ?)) as cost_variance
            FROM analysis_results
            WHERE model = ? AND resolution = ? AND success = 1
        """, (avg_time or 0, avg_time or 0, avg_cost or 0, avg_cost or 0, model, resolution))

        variance_row = cursor.fetchone()
        time_stddev = (variance_row[0] ** 0.5) if variance_row[0] else 0
        cost_stddev = (variance_row[1] ** 0.5) if variance_row[1] else 0

        success_rate = (success / total * 100) if total > 0 else 0
        fail_count = total - success

        stats.append({
            "model": model,
            "resolution": resolution,
            "total_count": total,
            "success_count": success,
            "fail_count": fail_count,
            "success_rate": success_rate,
            # 비용 통계
            "avg_cost_usd": avg_cost or 0,
            "avg_cost_krw": (avg_cost or 0) * EXCHANGE_RATE,
            "min_cost_usd": min_cost or 0,
            "max_cost_usd": max_cost or 0,
            "cost_stddev": cost_stddev,
            "total_cost_usd": total_cost or 0,
            # 시간 통계
            "avg_time": avg_time or 0,
            "min_time": min_time or 0,
            "max_time": max_time or 0,
            "time_stddev": time_stddev,
            # 예상 비용
            "cost_per_1200": (avg_cost or 0) * 1200 * EXCHANGE_RATE,
            "cost_per_10000": (avg_cost or 0) * 10000 * EXCHANGE_RATE,
            "cost_per_100000": (avg_cost or 0) * 100000 * EXCHANGE_RATE,
        })

    conn.close()
    return stats


def get_model_categorical_stats():
    """모델별 카테고리 데이터 집계 (빈도 기반)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 모든 성공한 분석 결과 조회
    cursor.execute("""
        SELECT model, resolution, metadata
        FROM analysis_results
        WHERE success = 1
        ORDER BY model, resolution
    """)

    results = cursor.fetchall()
    conn.close()

    from collections import Counter

    # 모델+해상도별 집계
    model_stats = {}

    for r in results:
        key = f"{r['model']}|{r['resolution']}"
        if key not in model_stats:
            model_stats[key] = {
                "model": r["model"],
                "resolution": r["resolution"],
                "count": 0,
                "categories": [],
                "colors": [],
                "palettes": [],
                "styles": [],
                "moods": [],
                "keywords": [],
            }

        model_stats[key]["count"] += 1

        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        cat_data = meta.get("category", {})
        colors_data = meta.get("colors", {})
        keywords_data = meta.get("keywords", {})
        style_data = meta.get("style", {})
        mood_data = meta.get("mood", {})

        # 카테고리 수집
        categories = cat_data.get("matches", [])
        model_stats[key]["categories"].extend(categories)

        # 색상 수집
        colors = colors_data.get("dominant", [])
        model_stats[key]["colors"].extend(colors)

        # 팔레트 수집
        palette = colors_data.get("palette_name", "")
        if palette:
            model_stats[key]["palettes"].append(palette)

        # 스타일 수집
        style = style_data.get("type", "")
        if style:
            model_stats[key]["styles"].append(style)

        # 무드 수집
        mood = mood_data.get("primary", "")
        if mood:
            model_stats[key]["moods"].append(mood)

        # 키워드 수집
        keywords = keywords_data.get("search_tags", [])
        model_stats[key]["keywords"].extend(keywords)

    # 빈도 계산
    aggregated = []
    for key, stats in model_stats.items():
        cat_counter = Counter(stats["categories"])
        color_counter = Counter(stats["colors"])
        palette_counter = Counter(stats["palettes"])
        style_counter = Counter(stats["styles"])
        mood_counter = Counter(stats["moods"])
        keyword_counter = Counter(stats["keywords"])

        aggregated.append({
            "model": stats["model"],
            "resolution": stats["resolution"],
            "분석수": stats["count"],
            # Top N 빈도
            "top_categories": cat_counter.most_common(5),
            "top_colors": color_counter.most_common(5),
            "top_palettes": palette_counter.most_common(3),
            "top_styles": style_counter.most_common(3),
            "top_moods": mood_counter.most_common(3),
            "top_keywords": keyword_counter.most_common(10),
            # 고유값 수
            "unique_categories": len(cat_counter),
            "unique_colors": len(color_counter),
            "unique_keywords": len(keyword_counter),
        })

    return aggregated


def get_same_image_comparison():
    """동일 이미지에 대한 모델별 상세 비교 데이터 (썸네일 포함)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 여러 모델로 분석된 파일명 찾기
    cursor.execute("""
        SELECT filename, COUNT(DISTINCT model || '_' || resolution) as variant_count
        FROM analysis_results
        WHERE success = 1
        GROUP BY filename
        HAVING variant_count > 1
        ORDER BY variant_count DESC
    """)

    multi_model_files = cursor.fetchall()

    comparisons = []
    for file_row in multi_model_files:
        filename = file_row["filename"]

        # 해당 파일의 모든 분석 결과 (이미지 데이터 포함)
        cursor.execute("""
            SELECT model, resolution, metadata, cost_usd, elapsed_time, image_data
            FROM analysis_results
            WHERE filename = ? AND success = 1
            ORDER BY model, resolution
        """, (filename,))

        results = cursor.fetchall()

        # 첫 번째 결과에서 썸네일 이미지 가져오기
        thumbnail = None
        for r in results:
            if r["image_data"]:
                thumbnail = r["image_data"]
                break

        file_comparison = {
            "filename": filename,
            "thumbnail": thumbnail,
            "variant_count": file_row["variant_count"],
            "results": []
        }

        for r in results:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            cat_data = meta.get("category", {})
            colors_data = meta.get("colors", {})
            keywords_data = meta.get("keywords", {})
            style_data = meta.get("style", {})
            mood_data = meta.get("mood", {})
            pattern_data = meta.get("pattern", {})
            usage_data = meta.get("usage_suggestion", {})

            file_comparison["results"].append({
                "model": r["model"],
                "resolution": r["resolution"],
                "cost_usd": r["cost_usd"],
                "elapsed_time": r["elapsed_time"],
                # 카테고리
                "categories": cat_data.get("matches", []),
                "confidence": cat_data.get("confidence"),
                # 스타일
                "style_type": style_data.get("type", ""),
                "style_era": style_data.get("era", ""),
                "style_technique": style_data.get("technique", ""),
                # 무드
                "mood_primary": mood_data.get("primary", ""),
                "mood_secondary": mood_data.get("secondary", []),
                # 패턴
                "pattern_scale": pattern_data.get("scale", ""),
                "pattern_repeat": pattern_data.get("repeat_type", ""),
                "pattern_density": pattern_data.get("density", ""),
                # 색상
                "colors_dominant": colors_data.get("dominant", []),
                "colors_palette": colors_data.get("palette_name", ""),
                "colors_mood": colors_data.get("mood", ""),
                # 키워드
                "keywords": keywords_data.get("search_tags", []),
                "description": keywords_data.get("description", ""),
                # 활용 제안
                "usage_products": usage_data.get("products", []),
                "usage_season": usage_data.get("season", []),
                "usage_target": usage_data.get("target_market", []),
            })

        comparisons.append(file_comparison)

    conn.close()
    return comparisons


def get_confidence_stats():
    """모델별 신뢰도(confidence) 통계"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model, resolution, metadata
        FROM analysis_results
        WHERE success = 1 AND metadata IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    # 모델/해상도별 신뢰도 수집
    confidence_data = {}
    for row in rows:
        key = (row["model"], row["resolution"])
        if key not in confidence_data:
            confidence_data[key] = []

        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        conf = meta.get("category", {}).get("confidence")
        if conf is not None:
            confidence_data[key].append(conf)

    # 통계 계산
    stats = []
    for (model, resolution), confidences in confidence_data.items():
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
            stddev = variance ** 0.5

            stats.append({
                "model": model,
                "resolution": resolution,
                "count": len(confidences),
                "avg_confidence": avg_conf,
                "min_confidence": min_conf,
                "max_confidence": max_conf,
                "stddev_confidence": stddev,
            })

    return stats

# DB 초기화
init_db()

# ============================================
# 모델 설정 (Gemini 모델만 사용)
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
    "gemini-2.5-flash": {
        "name": "3. Gemini 2.5 Flash",
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
        "name": "4. Gemini 3 Flash (최신)",
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


def analyze_image(image: Image.Image, model_id: str, resolution: str = "medium") -> dict:
    """이미지 분석 (Gemini API 사용)"""
    return analyze_with_gemini(image, model_id, resolution)


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

    # 다이얼로그가 열려있음을 표시 (닫힐 때 감지용)
    st.session_state.dialog_was_open = True


def main():
    st.set_page_config(
        page_title="Textile Metadata Extractor",
        page_icon="🎨",
        layout="wide"
    )

    # 사이드바 토글 버튼 - 항상 보이게 (hover 없이도)
    st.markdown("""
        <style>
        /* 사이드바 접기 버튼 - 항상 표시 */
        [data-testid="stSidebarCollapseButton"] {
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapseButton"] span {
            opacity: 1 !important;
            visibility: visible !important;
            color: inherit !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 로그인 체크
    if not check_login():
        show_login_page()
        st.stop()

    st.title("🎨 텍스타일 이미지 메타데이터 추출기")

    # 사용법 가이드
    with st.expander("📖 **사용법 가이드**", expanded=True):
        st.markdown("""
        <div style="max-height: 400px; overflow-y: auto; padding-right: 10px;">

        ### 🎯 서비스 소개
        텍스타일 이미지를 AI로 분석하여 **카테고리, 색상, 스타일, 무드, 패턴, 키워드** 등의 메타데이터를 자동으로 추출하는 도구입니다.

        ---

        ### 🔬 테스트 모드 (사이드바)

        | 모드 | 설명 | 용도 |
        |------|------|------|
        | **단일 모델** | 선택한 1개 모델로 분석 | 빠른 분석, 일반 사용 |
        | **모델 비교** | 여러 모델로 동일 이미지 분석 | 모델 성능 비교, 일관성 테스트 |

        ---

        ### 📤 분석하기 탭

        **1. 이미지 해상도 설정**

        | 해상도 | 토큰 수 | 비용 | 권장 용도 |
        |--------|---------|------|-----------|
        | low | 280 | 최저 | 빠른 테스트, 대량 처리 (기본값) |
        | medium | 560 | 중간 | 일반 분석 |
        | high | 1,120 | 최고 | 정밀 분석, 복잡한 패턴 |

        **2. 이미지 업로드**
        - 지원 형식: PNG, JPG, JPEG, WEBP
        - 여러 이미지 동시 업로드 가능
        - 업로드 후 썸네일 미리보기 제공

        **3. 분석 실행**
        - `🔍 n개 이미지 분석` 버튼 클릭
        - 분석 진행률 및 예상 시간 표시
        - 완료 후 결과 자동 DB 저장

        **4. 결과 확인**
        - 각 이미지별 분석 결과 토글로 표시
        - 카테고리, 색상, 스타일, 무드, 패턴, 키워드 등 상세 정보
        - 비용 및 처리 시간 표시

        ---

        ### 💾 저장된 결과 탭

        **📊 모델 비교 분석**
        - **수치형 통계**: 모델별 성공률, 비용, 처리시간 비교
        - **신뢰도 통계**: 모델별 분류 신뢰도 분포
        - **동일 이미지 비교**: 같은 이미지의 모델별 분석 결과 비교

        **📋 데이터 조회**
        - 필터: 모델, 해상도, 성공/실패별 조회
        - 페이지네이션: 페이지당 10개씩 표시
        - 삭제: 선택한 항목 일괄 삭제 가능

        ---

        ### 💡 팁

        - **비용 절약**: `low` 해상도로 대량 처리 시 최대 75% 비용 절감
        - **모델 비교**: 동일 이미지를 여러 모델로 분석하여 일관성 확인
        - **결과 초기화**: 세션 결과만 초기화 (DB 데이터는 유지)
        - **사이드바 접기**: 좌측 상단 화살표로 사이드바 접기/펼치기

        ---

        ### ⚠️ 주의사항

        - API 호출 시 비용이 발생합니다 (Gemini API)
        - 대량 이미지 분석 시 예상 비용을 확인하세요
        - 분석 중 브라우저를 닫으면 진행 중인 분석이 중단됩니다

        </div>
        """, unsafe_allow_html=True)

    # API 키 확인
    if GEMINI_API_KEY:
        st.caption("✅ Gemini API 연결됨")
    else:
        st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
        st.code("GEMINI_API_KEY=your_gemini_key", language="bash")
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

        # 테스트 모드 선택
        test_mode = st.radio(
            "🔬 테스트 모드",
            ["단일 모델", "모델 비교"],
            help="단일 모델: 선택한 모델로만 분석\n모델 비교: 여러 모델로 동일 이미지 분석"
        )

        st.divider()

        if test_mode == "단일 모델":
            # 모델 선택
            available_models = list(MODEL_OPTIONS.keys())

            selected_model = st.selectbox(
                "Vision 모델",
                options=available_models,
                format_func=lambda x: MODEL_OPTIONS[x]["name"],
                index=0
            )

            model_config = MODEL_OPTIONS[selected_model]
            st.caption(f"💵 Input: ${model_config['input_cost']*1_000_000:.3f}/1M")
            st.caption(f"💵 Output: ${model_config['output_cost']*1_000_000:.2f}/1M")

        else:
            # 비교할 모델 선택
            st.markdown("**비교할 모델**")
            selected_models = []

            for model_id, config in MODEL_OPTIONS.items():
                if st.checkbox(
                    config["name"],
                    value=True,
                    key=f"model_{model_id}"
                ):
                    selected_models.append(model_id)

    # ============================================
    # 메인 영역 (탭으로 구분)
    # ============================================

    tab1, tab2 = st.tabs(["📤 분석하기", "💾 저장된 결과"])

    with tab1:
        # 설정 영역
        setting_col1, setting_col2, setting_col3 = st.columns([1, 1, 2])

        with setting_col1:
            # 해상도 설정
            resolution = st.select_slider(
                "🖼️ 이미지 해상도",
                options=["low", "medium", "high"],
                value="low",
                help="low: 최저 비용 (280 tokens)\nmedium: 기본 (560 tokens)\nhigh: 고품질 (1120 tokens)"
            )
            st.caption(f"토큰: {MODEL_OPTIONS['gemini-2.0-flash-lite']['tokens_per_image'][resolution]}/이미지")

        with setting_col2:
            # 현재 세션 비용
            total_cost = sum(r["result"]["cost"]["total"] for r in st.session_state.results if r["result"]["success"])
            total_krw = total_cost * EXCHANGE_RATE
            image_count = len([r for r in st.session_state.results if r["result"]["success"]])
            st.metric("💰 세션 비용", f"₩{total_krw:.1f}", delta=f"{image_count}건")

        with setting_col3:
            # 1200개 예상 비용 및 초기화
            if image_count > 0:
                avg_cost = total_cost / image_count
                st.metric("📊 1200개 예상", f"₩{avg_cost * 1200 * EXCHANGE_RATE:.0f}")
            else:
                st.metric("📊 1200개 예상", "-")

            if st.button("🔄 결과 초기화", use_container_width=True):
                st.session_state.results = []
                st.session_state.comparison_results = []
                st.rerun()

        st.divider()

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
                            "confidence": m.get("category", {}).get("confidence", ""),
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

    with tab2:
        # DB 저장 현황 요약
        db_stats = get_db_stats()
        db_col1, db_col2, db_col3, db_col4 = st.columns(4)
        with db_col1:
            st.metric("💾 총 분석 수", db_stats["total_count"])
        with db_col2:
            st.metric("💰 누적 비용", f"₩{db_stats['total_cost_krw']:.1f}")
        with db_col3:
            if db_stats["model_stats"]:
                model_count = len(db_stats["model_stats"])
                st.metric("🤖 모델 수", model_count)
            else:
                st.metric("🤖 모델 수", 0)
        with db_col4:
            if db_stats["model_stats"]:
                models_str = ", ".join([MODEL_OPTIONS.get(m, {}).get("name", m).split(". ")[-1] for m, _, _ in db_stats["model_stats"][:3]])
                st.caption(f"분석된 모델:\n{models_str}")

        st.divider()

        # 서브탭: 모델 비교 분석 / 데이터 조회
        subtab1, subtab2 = st.tabs(["📊 모델 비교 분석", "📋 데이터 조회"])

        # ============================================
        # 모델 비교 분석 탭
        # ============================================
        with subtab1:
            model_stats = get_model_comparison_stats()
            confidence_stats = get_confidence_stats()
            categorical_stats = get_model_categorical_stats()
            image_comparisons = get_same_image_comparison()

            if model_stats:
                import pandas as pd

                # 내부 서브탭: 수치형/카테고리 비교/이미지 비교
                stat_tab1, stat_tab2, stat_tab3 = st.tabs(["📈 수치형 통계", "🎯 신뢰도 통계", "🖼️ 동일 이미지 비교"])

                # ========== 수치형 통계 탭 ==========
                with stat_tab1:
                    st.subheader("📈 수치형 데이터 통계")

                    # 1. 기본 통계 테이블
                    st.markdown("**기본 통계**")
                    basic_data = []
                    for s in model_stats:
                        model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                        basic_data.append({
                            "모델": model_name.split(". ")[-1] if ". " in model_name else model_name,
                            "해상도": s["resolution"],
                            "총 분석": s["total_count"],
                            "성공": s["success_count"],
                            "실패": s["fail_count"],
                            "성공률": f"{s['success_rate']:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(basic_data), use_container_width=True, hide_index=True)

                    st.divider()

                    # 2. 비용 통계 테이블
                    st.markdown("**비용 통계 (USD)**")
                    cost_data = []
                    for s in model_stats:
                        model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                        cost_data.append({
                            "모델": model_name.split(". ")[-1] if ". " in model_name else model_name,
                            "해상도": s["resolution"],
                            "평균": f"${s['avg_cost_usd']:.6f}",
                            "최소": f"${s['min_cost_usd']:.6f}",
                            "최대": f"${s['max_cost_usd']:.6f}",
                            "표준편차": f"${s['cost_stddev']:.6f}",
                            "총비용": f"${s['total_cost_usd']:.4f}",
                        })
                    st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

                    st.divider()

                    # 3. 응답시간 통계 테이블
                    st.markdown("**응답시간 통계 (초)**")
                    time_data = []
                    for s in model_stats:
                        model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                        time_data.append({
                            "모델": model_name.split(". ")[-1] if ". " in model_name else model_name,
                            "해상도": s["resolution"],
                            "평균": f"{s['avg_time']:.3f}s",
                            "최소": f"{s['min_time']:.3f}s",
                            "최대": f"{s['max_time']:.3f}s",
                            "표준편차": f"{s['time_stddev']:.3f}s",
                        })
                    st.dataframe(pd.DataFrame(time_data), use_container_width=True, hide_index=True)

                    st.divider()

                    # 4. 예상 비용 테이블
                    st.markdown("**규모별 예상 비용 (KRW)**")
                    scale_data = []
                    for s in model_stats:
                        model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                        scale_data.append({
                            "모델": model_name.split(". ")[-1] if ". " in model_name else model_name,
                            "해상도": s["resolution"],
                            "1,200개": f"₩{s['cost_per_1200']:,.0f}",
                            "10,000개": f"₩{s['cost_per_10000']:,.0f}",
                            "100,000개": f"₩{s['cost_per_100000']:,.0f}",
                        })
                    st.dataframe(pd.DataFrame(scale_data), use_container_width=True, hide_index=True)

                    st.divider()

                    # CSV 내보내기
                    csv_full_stats = []
                    for s in model_stats:
                        model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                        csv_full_stats.append({
                            "모델ID": s["model"],
                            "모델명": model_name,
                            "해상도": s["resolution"],
                            "총분석수": s["total_count"],
                            "성공수": s["success_count"],
                            "실패수": s["fail_count"],
                            "성공률(%)": round(s["success_rate"], 2),
                            "평균비용_USD": s["avg_cost_usd"],
                            "최소비용_USD": s["min_cost_usd"],
                            "최대비용_USD": s["max_cost_usd"],
                            "비용표준편차_USD": s["cost_stddev"],
                            "총비용_USD": s["total_cost_usd"],
                            "평균시간(s)": s["avg_time"],
                            "최소시간(s)": s["min_time"],
                            "최대시간(s)": s["max_time"],
                            "시간표준편차(s)": s["time_stddev"],
                            "1200개예상_KRW": s["cost_per_1200"],
                            "10000개예상_KRW": s["cost_per_10000"],
                            "100000개예상_KRW": s["cost_per_100000"],
                        })

                    df_csv = pd.DataFrame(csv_full_stats)
                    st.download_button(
                        label="📥 수치형 통계 CSV 다운로드",
                        data=df_csv.to_csv(index=False, encoding="utf-8-sig"),
                        file_name=f"model_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # ========== 신뢰도 통계 탭 ==========
                with stat_tab2:
                    st.subheader("🎯 카테고리 신뢰도(Confidence) 통계")

                    if confidence_stats:
                        conf_data = []
                        for s in confidence_stats:
                            model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                            conf_data.append({
                                "모델": model_name.split(". ")[-1] if ". " in model_name else model_name,
                                "해상도": s["resolution"],
                                "샘플수": s["count"],
                                "평균": f"{s['avg_confidence']:.2%}",
                                "최소": f"{s['min_confidence']:.2%}",
                                "최대": f"{s['max_confidence']:.2%}",
                                "표준편차": f"{s['stddev_confidence']:.4f}",
                            })
                        st.dataframe(pd.DataFrame(conf_data), use_container_width=True, hide_index=True)

                        # CSV 내보내기
                        csv_conf = []
                        for s in confidence_stats:
                            model_name = MODEL_OPTIONS.get(s["model"], {}).get("name", s["model"])
                            csv_conf.append({
                                "모델ID": s["model"],
                                "모델명": model_name,
                                "해상도": s["resolution"],
                                "샘플수": s["count"],
                                "평균신뢰도": s["avg_confidence"],
                                "최소신뢰도": s["min_confidence"],
                                "최대신뢰도": s["max_confidence"],
                                "표준편차": s["stddev_confidence"],
                            })
                        df_conf_csv = pd.DataFrame(csv_conf)
                        st.download_button(
                            label="📥 신뢰도 통계 CSV 다운로드",
                            data=df_conf_csv.to_csv(index=False, encoding="utf-8-sig"),
                            file_name=f"confidence_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("신뢰도 데이터가 없습니다.")

                # ========== 동일 이미지 비교 탭 ==========
                with stat_tab3:
                    st.subheader("🖼️ 동일 이미지 모델별 상세 비교")
                    st.caption("같은 이미지를 여러 모델/해상도로 분석한 결과를 상세 비교합니다.")

                    if image_comparisons:
                        for comp_idx, comp in enumerate(image_comparisons[:20]):  # 최대 20개
                            with st.expander(f"📄 {comp['filename']} ({comp['variant_count']}개 시행)", expanded=(comp_idx == 0)):
                                # 썸네일
                                if comp["thumbnail"]:
                                    st.image(
                                        f"data:image/png;base64,{comp['thumbnail']}",
                                        caption=comp["filename"],
                                        width=200
                                    )

                                st.divider()

                                # 통합 비교 테이블 (행: 각 시행, 열: 상세 항목)
                                st.markdown("**📊 시행별 상세 비교** (각 행 = 모델 시행)")

                                comparison_rows = []
                                for r in comp["results"]:
                                    model_name = MODEL_OPTIONS.get(r["model"], {}).get("name", r["model"])
                                    short_name = model_name.split(". ")[-1] if ". " in model_name else model_name

                                    # 색상을 텍스트로 표시
                                    colors_str = ", ".join(r["colors_dominant"][:3]) if r["colors_dominant"] else "-"

                                    comparison_rows.append({
                                        "모델": short_name,
                                        "해상도": r["resolution"],
                                        "카테고리": ", ".join(r["categories"][:3]) if r["categories"] else "-",
                                        "신뢰도": f"{r['confidence']:.0%}" if r["confidence"] else "-",
                                        "스타일": r["style_type"] or "-",
                                        "시대": r["style_era"] or "-",
                                        "기법": r["style_technique"] or "-",
                                        "무드(주)": r["mood_primary"] or "-",
                                        "무드(부)": ", ".join(r["mood_secondary"][:2]) if r["mood_secondary"] else "-",
                                        "패턴크기": r["pattern_scale"] or "-",
                                        "패턴반복": r["pattern_repeat"] or "-",
                                        "패턴밀도": r["pattern_density"] or "-",
                                        "색상": colors_str,
                                        "팔레트": r["colors_palette"] or "-",
                                        "키워드": ", ".join(r["keywords"][:5]) if r["keywords"] else "-",
                                        "추천제품": ", ".join(r["usage_products"][:2]) if r["usage_products"] else "-",
                                        "시즌": ", ".join(r["usage_season"]) if r["usage_season"] else "-",
                                        "타겟": ", ".join(r["usage_target"][:2]) if r["usage_target"] else "-",
                                        "비용($)": f"{r['cost_usd']:.5f}",
                                        "시간(s)": f"{r['elapsed_time']:.2f}",
                                    })

                                df_comparison = pd.DataFrame(comparison_rows)
                                st.dataframe(df_comparison, use_container_width=True, hide_index=True, height=min(600, 75 + len(comparison_rows) * 52))
                    else:
                        st.info("동일 이미지를 여러 모델로 분석한 데이터가 없습니다.\n모델 비교 테스트를 실행해주세요.")

            else:
                st.info("분석 데이터가 없습니다. 먼저 이미지를 분석해주세요.")

        # ============================================
        # 데이터 조회 탭
        # ============================================
        with subtab2:
            st.subheader("📋 저장된 분석 결과")

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
                                row["신뢰도"] = m.get("category", {}).get("confidence", "")
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
                    # 삭제 확인 상태 초기화
                    if "confirm_delete" not in st.session_state:
                        st.session_state.confirm_delete = False
                    if "delete_ids" not in st.session_state:
                        st.session_state.delete_ids = []
                    if "selected_for_delete" not in st.session_state:
                        st.session_state.selected_for_delete = set()

                    # 삭제 확인 다이얼로그
                    if st.session_state.confirm_delete and st.session_state.delete_ids:
                        st.error(f"⚠️ 정말로 {len(st.session_state.delete_ids)}개 항목을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.")
                        col_confirm1, col_confirm2 = st.columns(2)
                        with col_confirm1:
                            if st.button("❌ 취소", use_container_width=True):
                                st.session_state.confirm_delete = False
                                st.session_state.delete_ids = []
                                st.session_state.selected_for_delete = set()
                                st.rerun()
                        with col_confirm2:
                            if st.button("✅ 확인 삭제", type="primary", use_container_width=True):
                                deleted = delete_results_from_db(st.session_state.delete_ids)
                                st.session_state.confirm_delete = False
                                st.session_state.delete_ids = []
                                st.session_state.selected_for_delete = set()
                                st.success(f"✅ {deleted}개 항목이 삭제되었습니다.")
                                time.sleep(0.5)
                                st.rerun()
                        st.divider()

                    # 전체 선택 / 선택 삭제 버튼
                    page_ids = [r["id"] for r in db_results]
                    all_selected = all(pid in st.session_state.selected_for_delete for pid in page_ids)

                    col_sel_all, col_sel_info, col_sel_del = st.columns([1, 2, 1])
                    with col_sel_all:
                        if all_selected:
                            if st.button("☑️ 전체 해제", use_container_width=True):
                                for pid in page_ids:
                                    st.session_state.selected_for_delete.discard(pid)
                                st.rerun()
                        else:
                            if st.button("☐ 전체 선택", use_container_width=True):
                                for pid in page_ids:
                                    st.session_state.selected_for_delete.add(pid)
                                st.rerun()

                    with col_sel_info:
                        if st.session_state.selected_for_delete:
                            st.warning(f"🗑️ {len(st.session_state.selected_for_delete)}개 항목 선택됨")

                    with col_sel_del:
                        if st.session_state.selected_for_delete and not st.session_state.confirm_delete:
                            if st.button("🗑️ 선택 삭제", type="secondary", use_container_width=True):
                                st.session_state.confirm_delete = True
                                st.session_state.delete_ids = list(st.session_state.selected_for_delete)
                                st.rerun()

                    st.divider()

                    # 토글(Expander) 방식으로 데이터 표시
                    for r in db_results:
                        meta = r.get("metadata", {}) or {}
                        cat_matches = meta.get("category", {}).get("matches", [])
                        category_str = ", ".join(cat_matches[:2]) if cat_matches else meta.get("category", {}).get("primary", "-")
                        model_name = MODEL_OPTIONS.get(r["model"], {}).get("name", r["model"]).split(". ")[-1]
                        status_icon = "✅" if r["success"] else "❌"
                        confidence = meta.get("category", {}).get("confidence")

                        # 헤더 구성
                        header = f"{status_icon} **{r['filename']}** | {model_name} | {r['resolution']} | {category_str}"

                        col_check, col_expander = st.columns([0.5, 9.5])

                        with col_check:
                            is_selected = r["id"] in st.session_state.selected_for_delete
                            if st.checkbox("", value=is_selected, key=f"del_check_{r['id']}", label_visibility="collapsed"):
                                st.session_state.selected_for_delete.add(r["id"])
                            else:
                                st.session_state.selected_for_delete.discard(r["id"])

                        with col_expander:
                            with st.expander(header, expanded=False):
                                if r["success"] and meta:
                                    # 썸네일 + 기본 정보
                                    thumb_col, info_col = st.columns([1, 3])

                                    with thumb_col:
                                        if r.get("image_data"):
                                            st.image(
                                                f"data:image/png;base64,{r['image_data']}",
                                                caption=r["filename"],
                                                use_container_width=True
                                            )
                                        else:
                                            st.info("이미지 없음")

                                    with info_col:
                                        info_col1, info_col2, info_col3 = st.columns(3)
                                        with info_col1:
                                            st.metric("비용", f"₩{r['cost_krw']:.2f}" if r["cost_krw"] else "-")
                                        with info_col2:
                                            st.metric("소요시간", f"{r['elapsed_time']:.2f}s" if r["elapsed_time"] else "-")
                                        with info_col3:
                                            st.metric("신뢰도", f"{confidence:.0%}" if confidence else "-")

                                    st.divider()

                                    # 카테고리 & 스타일
                                    detail_col1, detail_col2 = st.columns(2)
                                    with detail_col1:
                                        st.markdown("**📁 카테고리**")
                                        if cat_matches:
                                            st.write(", ".join(cat_matches))
                                        else:
                                            st.write("-")

                                        st.markdown("**🎨 스타일**")
                                        style_data = meta.get("style", {})
                                        st.write(f"유형: {style_data.get('type', '-')}")
                                        st.write(f"시대: {style_data.get('era', '-')}")
                                        st.write(f"기법: {style_data.get('technique', '-')}")

                                    with detail_col2:
                                        st.markdown("**🎭 무드**")
                                        mood_data = meta.get("mood", {})
                                        st.write(f"주요: {mood_data.get('primary', '-')}")
                                        secondary = mood_data.get("secondary", [])
                                        st.write(f"부가: {', '.join(secondary) if secondary else '-'}")

                                        st.markdown("**🔲 패턴**")
                                        pattern_data = meta.get("pattern", {})
                                        st.write(f"크기: {pattern_data.get('scale', '-')}")
                                        st.write(f"반복: {pattern_data.get('repeat_type', '-')}")
                                        st.write(f"밀도: {pattern_data.get('density', '-')}")

                                    st.divider()

                                    # 색상
                                    st.markdown("**🌈 색상**")
                                    colors_data = meta.get("colors", {})
                                    dominant = colors_data.get("dominant", [])
                                    if dominant:
                                        st.write(f"주요 색상: {', '.join(dominant)}")
                                    st.write(f"팔레트: {colors_data.get('palette_name', '-')}")
                                    st.write(f"색상 무드: {colors_data.get('mood', '-')}")

                                    # 키워드
                                    st.markdown("**🏷️ 키워드**")
                                    keywords_data = meta.get("keywords", {})
                                    tags = keywords_data.get("search_tags", [])
                                    if tags:
                                        st.write(", ".join(tags))
                                    desc = keywords_data.get("description", "")
                                    if desc:
                                        st.caption(f"설명: {desc}")

                                    # 활용 제안
                                    usage = meta.get("usage_suggestion", {})
                                    if usage:
                                        st.markdown("**💡 활용 제안**")
                                        products = usage.get("products", [])
                                        season = usage.get("season", [])
                                        target = usage.get("target_market", [])
                                        if products:
                                            st.write(f"추천 제품: {', '.join(products)}")
                                        if season:
                                            st.write(f"추천 시즌: {', '.join(season)}")
                                        if target:
                                            st.write(f"타겟 마켓: {', '.join(target)}")

                                    # 분석 일시
                                    st.caption(f"📅 분석 일시: {r['created_at']}")
                                else:
                                    st.warning("분석 실패 또는 메타데이터 없음")
                                    if r.get("error_message"):
                                        st.error(f"오류: {r['error_message']}")
            else:
                st.info("저장된 결과가 없습니다.")


if __name__ == "__main__":
    main()
