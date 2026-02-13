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
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorthief import ColorThief
import boto3
from botocore.exceptions import ClientError

# SQLAlchemy ORM
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, JSON, func, distinct
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

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
# SQLAlchemy ORM 설정 (MariaDB/MySQL - AWS RDS)
# ============================================

Base = declarative_base()

class AnalysisResult(Base):
    """분석 결과 ORM 모델"""
    __tablename__ = 'analysis_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False, index=True)
    image_hash = Column(String(64), nullable=True, index=True)  # 이미지 해시 (중복 체크용)
    image_url = Column(String(1000), nullable=True)  # S3 URL
    model = Column(String(100), nullable=False, index=True)
    resolution = Column(String(50), nullable=False)
    success = Column(Boolean, nullable=False)
    meta_data = Column('metadata', JSON, nullable=True)  # DB 컬럼명은 'metadata' 유지
    cost_usd = Column(Float, nullable=True)
    cost_krw = Column(Float, nullable=True)
    elapsed_time = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        """딕셔너리 변환"""
        return {
            "id": self.id,
            "filename": self.filename,
            "image_hash": self.image_hash,
            "image_url": self.image_url,
            "model": self.model,
            "resolution": self.resolution,
            "success": self.success,
            "metadata": self.meta_data,
            "cost_usd": float(self.cost_usd) if self.cost_usd else 0,
            "cost_krw": float(self.cost_krw) if self.cost_krw else 0,
            "elapsed_time": float(self.elapsed_time) if self.elapsed_time else 0,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# ============================================
# S3 이미지 저장 (중복 체크)
# ============================================

_s3_client = None

def get_s3_client():
    """S3 클라이언트 반환 (싱글톤)"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=get_api_key("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_api_key("AWS_SECRET_ACCESS_KEY"),
            region_name=get_api_key("AWS_REGION") or "ap-northeast-2"
        )
    return _s3_client


def calculate_image_hash(image: Image.Image) -> str:
    """이미지의 SHA256 해시 계산"""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    return hashlib.sha256(img_bytes).hexdigest()


def get_existing_image_url(image_hash: str) -> str | None:
    """동일 해시의 이미지 URL이 이미 있는지 확인 -> 이미지 픽셀 해쉬화 해서 중복 조회"""
    session = get_session()
    try:
        result = session.query(AnalysisResult.image_url).filter(
            AnalysisResult.image_hash == image_hash,
            AnalysisResult.image_url.isnot(None)
        ).first()
        return result.image_url if result else None
    finally:
        session.close()


def upload_image_to_s3(image: Image.Image, filename: str, image_hash: str) -> str:
    """
    이미지를 S3에 업로드하고 URL 반환
    - 동일 해시의 이미지가 이미 있으면 기존 URL 반환
    """
    # 이미 업로드된 이미지인지 확인
    existing_url = get_existing_image_url(image_hash)
    if existing_url:
        return existing_url

    # S3에 업로드
    s3_client = get_s3_client()
    bucket_name = get_api_key("S3_BUCKET_NAME")
    storage_path = get_api_key("S3_STORAGE_PATH") or "tdb/storage/uploads"

    # 파일 경로: {storage_path}/metadata-extractor/{hash[:8]}/{hash}.png
    s3_key = f"{storage_path}/metadata-extractor/{image_hash[:8]}/{image_hash}.png"

    # 이미지를 바이트로 변환
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    try:
        s3_client.upload_fileobj(
            img_buffer,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': 'image/png',
                'CacheControl': 'max-age=31536000'  # 1년 캐시
            }
        )
        # CloudFront URL 생성 (S3 직접 접근은 403 → CloudFront 경유 필요)
        cdn_domain = get_api_key("CDN_DOMAIN") or f"{bucket_name}.textiledesignbank.com"
        image_url = f"https://{cdn_domain}/{s3_key}"
        return image_url

    except ClientError as e:
        raise Exception(f"S3 업로드 실패: {e}")


# 데이터베이스 엔진 및 세션 (싱글톤)
_engine = None
_SessionLocal = None

def get_database_url() -> str:
    """DATABASE_URL 가져오기 (mysql → mysql+pymysql 변환)"""
    database_url = get_api_key("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

    # SQLAlchemy용 드라이버 지정
    if database_url.startswith("mysql://"):
        database_url = database_url.replace("mysql://", "mysql+pymysql://", 1)

    return database_url

def get_engine():
    """SQLAlchemy 엔진 반환 (싱글톤)"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            echo=False
        )
    return _engine

def get_session():
    """SQLAlchemy 세션 반환"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()

def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    engine = get_engine()
    Base.metadata.create_all(engine)

def save_result_to_db(result_data: dict):
    """분석 결과를 DB에 저장 (이미지는 S3에 업로드)"""
    session = get_session()
    try:
        image_hash = None
        image_url = None

        # 이미지를 S3에 업로드 (중복 체크)
        if "image" in result_data and result_data["image"] is not None:
            image = result_data["image"]
            image_hash = calculate_image_hash(image)
            image_url = upload_image_to_s3(image, result_data.get("filename", "unknown"), image_hash)

        result = AnalysisResult(
            filename=result_data.get("filename"),
            image_hash=image_hash,
            image_url=image_url,
            model=result_data.get("model"),
            resolution=result_data.get("resolution"),
            success=result_data.get("result", {}).get("success", False),
            meta_data=result_data.get("result", {}).get("metadata") if result_data.get("result", {}).get("success") else None,
            cost_usd=result_data.get("result", {}).get("cost", {}).get("total", 0),
            cost_krw=result_data.get("result", {}).get("cost", {}).get("krw", 0),
            elapsed_time=result_data.get("result", {}).get("elapsed_time", 0),
            error=result_data.get("result", {}).get("error"),
        )
        session.add(result)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def load_results_from_db(limit: int = 100, offset: int = 0, model_filter: str = None, resolution_filter: str = None, success_filter: str = None):
    """DB에서 분석 결과 불러오기 (페이지네이션 + 필터 지원)"""
    session = get_session()
    try:
        query = session.query(AnalysisResult)

        # 필터 적용
        if model_filter and model_filter != "전체":
            query = query.filter(AnalysisResult.model == model_filter)

        if resolution_filter and resolution_filter != "전체":
            query = query.filter(AnalysisResult.resolution == resolution_filter)

        if success_filter == "성공만":
            query = query.filter(AnalysisResult.success == True)
        elif success_filter == "실패만":
            query = query.filter(AnalysisResult.success == False)

        # 정렬 및 페이지네이션
        results = query.order_by(AnalysisResult.id.desc()).offset(offset).limit(limit).all()

        return [r.to_dict() for r in results]
    finally:
        session.close()


def get_filtered_count(model_filter: str = None, resolution_filter: str = None, success_filter: str = None) -> int:
    """필터 적용된 결과 개수 조회"""
    session = get_session()
    try:
        query = session.query(func.count(AnalysisResult.id))

        if model_filter and model_filter != "전체":
            query = query.filter(AnalysisResult.model == model_filter)

        if resolution_filter and resolution_filter != "전체":
            query = query.filter(AnalysisResult.resolution == resolution_filter)

        if success_filter == "성공만":
            query = query.filter(AnalysisResult.success == True)
        elif success_filter == "실패만":
            query = query.filter(AnalysisResult.success == False)

        return query.scalar()
    finally:
        session.close()

def get_db_stats():
    """DB 통계 조회"""
    session = get_session()
    try:
        total_count = session.query(func.count(AnalysisResult.id)).scalar()

        total_cost = session.query(func.coalesce(func.sum(AnalysisResult.cost_usd), 0)).filter(
            AnalysisResult.success == True
        ).scalar()
        total_cost = float(total_cost) if total_cost else 0

        model_stats_query = session.query(
            AnalysisResult.model,
            func.count(AnalysisResult.id),
            func.sum(AnalysisResult.cost_usd)
        ).filter(AnalysisResult.success == True).group_by(AnalysisResult.model).all()

        model_stats = [(row[0], row[1], float(row[2]) if row[2] else 0) for row in model_stats_query]

        return {
            "total_count": total_count,
            "total_cost_usd": total_cost,
            "total_cost_krw": total_cost * EXCHANGE_RATE,
            "model_stats": model_stats
        }
    finally:
        session.close()


def delete_results_from_db(ids: list) -> int:
    """DB에서 분석 결과 삭제"""
    if not ids:
        return 0

    session = get_session()
    try:
        deleted_count = session.query(AnalysisResult).filter(
            AnalysisResult.id.in_(ids)
        ).delete(synchronize_session=False)
        session.commit()
        return deleted_count
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_model_comparison_stats():
    """모델별 상세 비교 통계 조회 (수치형 데이터)"""
    session = get_session()
    try:
        from sqlalchemy import case

        # 기본 통계 쿼리
        query = session.query(
            AnalysisResult.model,
            AnalysisResult.resolution,
            func.count(AnalysisResult.id).label('total_count'),
            func.sum(case((AnalysisResult.success == True, 1), else_=0)).label('success_count'),
            func.avg(case((AnalysisResult.success == True, AnalysisResult.cost_usd), else_=None)).label('avg_cost'),
            func.min(case((AnalysisResult.success == True, AnalysisResult.cost_usd), else_=None)).label('min_cost'),
            func.max(case((AnalysisResult.success == True, AnalysisResult.cost_usd), else_=None)).label('max_cost'),
            func.avg(case((AnalysisResult.success == True, AnalysisResult.elapsed_time), else_=None)).label('avg_time'),
            func.min(case((AnalysisResult.success == True, AnalysisResult.elapsed_time), else_=None)).label('min_time'),
            func.max(case((AnalysisResult.success == True, AnalysisResult.elapsed_time), else_=None)).label('max_time'),
            func.sum(case((AnalysisResult.success == True, AnalysisResult.cost_usd), else_=0)).label('total_cost')
        ).group_by(AnalysisResult.model, AnalysisResult.resolution).order_by(
            AnalysisResult.model, AnalysisResult.resolution
        )

        rows = query.all()

        stats = []
        for row in rows:
            model = row.model
            resolution = row.resolution
            total = row.total_count
            success = row.success_count or 0
            avg_cost = float(row.avg_cost) if row.avg_cost else 0
            min_cost = float(row.min_cost) if row.min_cost else 0
            max_cost = float(row.max_cost) if row.max_cost else 0
            avg_time = float(row.avg_time) if row.avg_time else 0
            min_time = float(row.min_time) if row.min_time else 0
            max_time = float(row.max_time) if row.max_time else 0
            total_cost = float(row.total_cost) if row.total_cost else 0

            # 표준편차 계산
            variance_query = session.query(
                func.avg(func.pow(AnalysisResult.elapsed_time - avg_time, 2)).label('time_variance'),
                func.avg(func.pow(AnalysisResult.cost_usd - avg_cost, 2)).label('cost_variance')
            ).filter(
                AnalysisResult.model == model,
                AnalysisResult.resolution == resolution,
                AnalysisResult.success == True
            ).first()

            time_stddev = (float(variance_query.time_variance) ** 0.5) if variance_query.time_variance else 0
            cost_stddev = (float(variance_query.cost_variance) ** 0.5) if variance_query.cost_variance else 0

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
                "avg_cost_usd": avg_cost,
                "avg_cost_krw": avg_cost * EXCHANGE_RATE,
                "min_cost_usd": min_cost,
                "max_cost_usd": max_cost,
                "cost_stddev": cost_stddev,
                "total_cost_usd": total_cost,
                # 시간 통계
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "time_stddev": time_stddev,
                # 예상 비용
                "cost_per_1200": avg_cost * 1200 * EXCHANGE_RATE,
                "cost_per_10000": avg_cost * 10000 * EXCHANGE_RATE,
                "cost_per_100000": avg_cost * 100000 * EXCHANGE_RATE,
            })

        return stats
    finally:
        session.close()


def get_model_categorical_stats():
    """모델별 카테고리 데이터 집계 (빈도 기반)"""
    session = get_session()
    try:
        # 모든 성공한 분석 결과 조회
        results = session.query(
            AnalysisResult.model,
            AnalysisResult.resolution,
            AnalysisResult.meta_data
        ).filter(AnalysisResult.success == True).order_by(
            AnalysisResult.model, AnalysisResult.resolution
        ).all()

        from collections import Counter

        # 모델+해상도별 집계
        model_stats = {}

        for r in results:
            key = f"{r.model}|{r.resolution}"
            if key not in model_stats:
                model_stats[key] = {
                    "model": r.model,
                    "resolution": r.resolution,
                    "count": 0,
                    "categories": [],
                    "colors": [],
                    "palettes": [],
                    "styles": [],
                    "moods": [],
                    "keywords": [],
                }

            model_stats[key]["count"] += 1

            meta = r.meta_data if r.meta_data else {}

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
    finally:
        session.close()


def get_same_image_comparison():
    """동일 이미지에 대한 모델별 상세 비교 데이터 (썸네일 포함)"""
    session = get_session()
    try:
        # 여러 모델로 분석된 파일명 찾기
        from sqlalchemy import distinct, literal_column

        subquery = session.query(
            AnalysisResult.filename,
            func.count(distinct(func.concat(AnalysisResult.model, '_', AnalysisResult.resolution))).label('variant_count')
        ).filter(AnalysisResult.success == True).group_by(AnalysisResult.filename).having(
            func.count(distinct(func.concat(AnalysisResult.model, '_', AnalysisResult.resolution))) > 1
        ).order_by(func.count(distinct(func.concat(AnalysisResult.model, '_', AnalysisResult.resolution))).desc()).all()

        comparisons = []
        for file_row in subquery:
            filename = file_row.filename

            # 해당 파일의 모든 분석 결과
            results = session.query(AnalysisResult).filter(
                AnalysisResult.filename == filename,
                AnalysisResult.success == True
            ).order_by(AnalysisResult.model, AnalysisResult.resolution).all()

            # 첫 번째 결과에서 썸네일 이미지 가져오기
            thumbnail = None
            for r in results:
                if r.image_url:
                    thumbnail = r.image_url
                    break

            file_comparison = {
                "filename": filename,
                "thumbnail": thumbnail,
                "variant_count": file_row.variant_count,
                "results": []
            }

            for r in results:
                meta = r.meta_data if r.meta_data else {}

                cat_data = meta.get("category", {})
                colors_data = meta.get("colors", {})
                keywords_data = meta.get("keywords", {})
                style_data = meta.get("style", {})
                mood_data = meta.get("mood", {})
                pattern_data = meta.get("pattern", {})
                usage_data = meta.get("usage_suggestion", {})

                file_comparison["results"].append({
                    "model": r.model,
                    "resolution": r.resolution,
                    "cost_usd": float(r.cost_usd) if r.cost_usd else 0,
                    "elapsed_time": float(r.elapsed_time) if r.elapsed_time else 0,
                    # 제목
                    "title": meta.get("title", ""),
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
                    "usage_fabrics": usage_data.get("fabrics", []),
                })

            comparisons.append(file_comparison)

        return comparisons
    finally:
        session.close()


def get_confidence_stats():
    """모델별 신뢰도(confidence) 통계"""
    session = get_session()
    try:
        results = session.query(
            AnalysisResult.model,
            AnalysisResult.resolution,
            AnalysisResult.meta_data
        ).filter(
            AnalysisResult.success == True,
            AnalysisResult.meta_data.isnot(None)
        ).all()

        # 모델/해상도별 신뢰도 수집
        confidence_data = {}
        for row in results:
            key = (row.model, row.resolution)
            if key not in confidence_data:
                confidence_data[key] = []

            meta = row.meta_data if row.meta_data else {}
            conf = meta.get("category", {}).get("confidence")
            if conf is not None:
                confidence_data[key].append(float(conf))

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
    finally:
        session.close()

# DB 초기화 (MySQL 연결 실패 시 재시도)
try:
    init_db()
except Exception as e:
    print(f"⚠️ DB 초기화 실패: {e}")

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
  "title": "A creative, evocative design title (2-4 words in English, like a professional textile designer would name it. Examples: 'Midnight Garden', 'Coral Bloom', 'Azure Wave', 'Wild Meadow', 'Ember Glow'. Capture the mood, color, and essence of the design poetically.)",
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
    "target_market": ["market1"],
    "fabrics": ["MUST be from: cotton, silk, polyester, linen, wool, nylon, rayon, denim, velvet, satin, chiffon, leather"]
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
# 색상 추출 함수 (Python 패키지 사용)
# ============================================

def rgb_to_hex(rgb: tuple) -> str:
    """RGB 튜플을 HEX 문자열로 변환"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def extract_colors_from_image(image: Image.Image, color_count: int = 5) -> dict:
    """
    ColorThief를 사용하여 이미지에서 주요 색상 추출

    Args:
        image: PIL Image 객체
        color_count: 추출할 색상 수 (기본값: 5)

    Returns:
        dict: {
            "dominant": ["#hex1", "#hex2", ...],
            "palette_name": "자동 생성된 팔레트명",
            "mood": "warm/cool/neutral/vibrant/muted"
        }
    """
    try:
        # PIL Image를 BytesIO로 변환 (ColorThief는 파일 객체 필요)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # ColorThief로 색상 추출
        color_thief = ColorThief(img_buffer)

        # 주요 색상 팔레트 추출
        palette = color_thief.get_palette(color_count=color_count, quality=10)

        # RGB를 HEX로 변환
        hex_colors = [rgb_to_hex(color) for color in palette]

        # 색상 분석하여 mood 결정
        mood = _analyze_color_mood(palette)

        # 팔레트 이름 생성
        palette_name = _generate_palette_name(palette)

        return {
            "dominant": hex_colors,
            "palette_name": palette_name,
            "mood": mood
        }

    except Exception as e:
        # 색상 추출 실패 시 기본값 반환
        return {
            "dominant": [],
            "palette_name": "Unknown",
            "mood": "neutral",
            "error": str(e)
        }


def _analyze_color_mood(palette: list) -> str:
    """색상 팔레트의 전체적인 무드 분석"""
    if not palette:
        return "neutral"

    total_r, total_g, total_b = 0, 0, 0
    total_saturation = 0
    total_brightness = 0

    for r, g, b in palette:
        total_r += r
        total_g += g
        total_b += b

        # HSV 계산을 위한 변환
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        brightness = max_c / 255
        saturation = (max_c - min_c) / max_c if max_c > 0 else 0

        total_saturation += saturation
        total_brightness += brightness

    n = len(palette)
    avg_r, avg_g, avg_b = total_r / n, total_g / n, total_b / n
    avg_saturation = total_saturation / n
    avg_brightness = total_brightness / n

    # 무드 결정 로직
    if avg_saturation > 0.6 and avg_brightness > 0.5:
        return "vibrant"
    elif avg_saturation < 0.3:
        return "muted"
    elif avg_r > avg_b and avg_r > avg_g * 0.9:
        return "warm"
    elif avg_b > avg_r and avg_b > avg_g * 0.9:
        return "cool"
    else:
        return "neutral"


def _generate_palette_name(palette: list) -> str:
    """색상 팔레트의 특성에 기반한 이름 생성"""
    if not palette:
        return "Unknown"

    # 주요 색상(첫 번째)의 특성 분석
    r, g, b = palette[0]

    # 밝기 계산
    brightness = (r + g + b) / 3 / 255

    # 채도 계산
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    saturation = (max_c - min_c) / max_c if max_c > 0 else 0

    # 색조 결정
    if max_c == min_c:
        hue_name = "Gray"
    elif r >= g and r >= b:
        if g > b:
            hue_name = "Orange" if saturation > 0.5 else "Tan"
        else:
            hue_name = "Red" if saturation > 0.5 else "Pink"
    elif g >= r and g >= b:
        if r > b:
            hue_name = "Yellow-Green"
        else:
            hue_name = "Green" if saturation > 0.5 else "Sage"
    else:  # b is max
        if r > g:
            hue_name = "Purple" if saturation > 0.5 else "Lavender"
        else:
            hue_name = "Blue" if saturation > 0.5 else "Sky"

    # 밝기 수식어
    if brightness > 0.7:
        brightness_adj = "Light"
    elif brightness < 0.3:
        brightness_adj = "Dark"
    else:
        brightness_adj = ""

    # 팔레트 이름 조합
    if brightness_adj:
        return f"{brightness_adj} {hue_name} Tones"
    else:
        return f"{hue_name} Tones"


# ============================================
# 분석 함수
# ============================================

def _call_gemini_api(image: Image.Image, model_id: str, resolution: str) -> dict:
    """Gemini API만 호출하는 내부 함수"""
    model_config = MODEL_OPTIONS[model_id]

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


def analyze_with_gemini(image: Image.Image, model_id: str, resolution: str) -> dict:
    """
    Gemini API와 색상 추출을 병렬로 실행하여 이미지 분석

    - LLM API: 카테고리, 스타일, 무드, 패턴, 키워드 등 분석
    - ColorThief: 색상 추출 (일관된 결과 보장)
    """
    # 병렬 실행: LLM API 호출 + 색상 추출
    with ThreadPoolExecutor(max_workers=2) as executor:
        # LLM API 호출 (비동기)
        llm_future = executor.submit(_call_gemini_api, image, model_id, resolution)

        # 색상 추출 (비동기)
        color_future = executor.submit(extract_colors_from_image, image, 5)

        # 결과 수집
        llm_result = llm_future.result()
        color_result = color_future.result()

    # LLM 결과에 색상 추출 결과 병합
    if llm_result["success"]:
        # LLM이 추출한 색상 대신 ColorThief 결과로 대체
        llm_result["metadata"]["colors"] = color_result

    return llm_result


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
        if result.get("image_url"):
            st.image(
                result['image_url'],
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

        # 제목
        title = metadata.get("title")
        if title:
            st.markdown(f"## ✨ {title}")

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
            if usage.get("fabrics"):
                usage_text.append(f"추천원단: {', '.join(usage.get('fabrics', []))}")
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
        if result.get("image_url"):
            st.image(
                result['image_url'],
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

        # 제목
        title = metadata.get("title")
        if title:
            st.markdown(f"## ✨ {title}")

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
            if usage.get("fabrics"):
                st.caption(f"추천원단: {', '.join(usage.get('fabrics', []))}")

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

    # 커스텀 스타일
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
        /* 컴팩트 메트릭 */
        [data-testid="stMetric"] {
            padding: 0.5rem 0;
        }
        [data-testid="stMetric"] label {
            font-size: 0.85rem;
        }
        /* 탭 간격 조정 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        /* 버튼 간격 */
        .stButton > button {
            padding: 0.4rem 1rem;
        }
        /* selectbox 높이 조정 */
        [data-testid="stSelectbox"] {
            min-height: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # 로그인 체크
    if not check_login():
        show_login_page()
        st.stop()

    st.title("🎨 텍스타일 이미지 메타데이터 추출기")

    # 사용법 가이드 (간소화, 기본 접힘)
    with st.expander("📖 사용법 가이드", expanded=False):
        st.markdown("""
        **🎯 서비스**: 텍스타일 이미지 AI 분석 → 카테고리, 색상, 스타일, 무드, 패턴, 키워드, 추천원단 추출

        | 해상도 | 토큰 | 용도 |
        |:---:|:---:|---|
        | low | 280 | 빠른 테스트 (기본) |
        | medium | 560 | 일반 분석 |
        | high | 1,120 | 정밀 분석 |

        **💡 Tip**: 사이드바에서 단일/비교 모드 선택 • 결과는 자동 DB 저장 • Excel 내보내기 지원
        """)

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
        # 설정 + 비용 통합 영역 (컴팩트)
        col_res, col_cost1, col_cost2, col_reset = st.columns([2, 1.5, 1.5, 1])

        with col_res:
            resolution = st.select_slider(
                "해상도",
                options=["low", "medium", "high"],
                value="low",
                help="low(280) • medium(560) • high(1120) 토큰"
            )

        # 비용 계산
        total_cost = sum(r["result"]["cost"]["total"] for r in st.session_state.results if r["result"]["success"])
        total_krw = total_cost * EXCHANGE_RATE
        image_count = len([r for r in st.session_state.results if r["result"]["success"]])

        with col_cost1:
            st.metric("세션 비용", f"₩{total_krw:.0f}", delta=f"{image_count}건" if image_count else None)

        with col_cost2:
            if image_count > 0:
                avg_cost = total_cost / image_count
                st.metric("1200개 예상", f"₩{avg_cost * 1200 * EXCHANGE_RATE:,.0f}")
            else:
                st.metric("1200개 예상", "-")

        with col_reset:
            st.write("")  # 정렬용
            if st.button("🔄 초기화", use_container_width=True):
                st.session_state.results = []
                st.session_state.comparison_results = []
                st.rerun()

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
                        try:
                            save_result_to_db(result_data)
                        except Exception as db_err:
                            st.warning(f"⚠️ DB 저장 실패: {db_err}")

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

                                # PIL Image는 thread-safe하지 않으므로 각 모델별 복사본 생성
                                def analyze_model(model_id, img_copy):
                                    return model_id, analyze_image(img_copy, model_id, resolution)

                                with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
                                    futures = {executor.submit(analyze_model, m, image.copy()): m for m in selected_models}
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
                                try:
                                    save_result_to_db({
                                        "filename": file.name,
                                        "model": model_id,
                                        "resolution": resolution,
                                        "result": result,
                                        "image": image
                                    })
                                except Exception as db_err:
                                    st.warning(f"⚠️ DB 저장 실패 ({model_id}): {db_err}")

                                with cols[idx]:
                                    model_name = MODEL_OPTIONS[model_id]["name"].split(". ")[1]
                                    st.caption(f"**{model_name}**")

                                    if result["success"]:
                                        st.success(f"✅ {result['elapsed_time']:.2f}s | ₩{result['cost']['krw']:.2f}")

                                        metadata = result["metadata"]
                                        title = metadata.get("title")
                                        if title:
                                            st.markdown(f"**✨ {title}**")
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

                        title = metadata.get("title")
                        if title:
                            st.markdown(f"**✨ {title}**")
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

        # Excel 다운로드
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 결과 Excel 다운로드", use_container_width=True):
                import pandas as pd
                import io

                rows = []
                for item in st.session_state.results:
                    if item["result"]["success"]:
                        m = item["result"]["metadata"]
                        cat_matches = m.get("category", {}).get("matches", [])
                        rows.append({
                            "filename": item["filename"],
                            "title": m.get("title", ""),
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
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='분석결과', index=False)
                    excel_buffer.seek(0)
                    st.download_button(
                        "다운로드",
                        excel_buffer,
                        f"metadata_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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

                # 내부 서브탭: 모델별 통계 / 예상 비용 통계 / 신뢰도 통계
                stat_tab1, stat_tab2, stat_tab3 = st.tabs(["🖼️ 모델별 통계", "📈 예상 비용 통계", "🎯 신뢰도 통계"])

                # ========== 모델별 통계 탭 (구 동일 이미지 비교) ==========
                with stat_tab1:
                    st.subheader("🖼️ 모델별 상세 비교")
                    st.caption("같은 이미지를 여러 모델/해상도로 분석한 결과를 상세 비교합니다.")

                    if image_comparisons:
                        for comp_idx, comp in enumerate(image_comparisons[:20]):
                            with st.expander(f"📄 {comp['filename']} ({comp['variant_count']}개 시행)", expanded=(comp_idx == 0)):
                                if comp["thumbnail"]:
                                    st.image(
                                        comp['thumbnail'],  # S3 URL
                                        caption=comp["filename"],
                                        width=200
                                    )

                                st.divider()

                                st.markdown("**📊 시행별 상세 비교** (각 행 = 모델 시행)")

                                comparison_rows = []
                                for r in comp["results"]:
                                    model_name = MODEL_OPTIONS.get(r["model"], {}).get("name", r["model"])
                                    short_name = model_name.split(". ")[-1] if ". " in model_name else model_name
                                    colors_str = ", ".join(r["colors_dominant"][:3]) if r["colors_dominant"] else "-"

                                    comparison_rows.append({
                                        "모델": short_name,
                                        "해상도": r["resolution"],
                                        "제목": r.get("title", "") or "-",
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
                                        "추천원단": ", ".join(r["usage_fabrics"][:2]) if r.get("usage_fabrics") else "-",
                                        "비용($)": f"{r['cost_usd']:.5f}",
                                        "시간(s)": f"{r['elapsed_time']:.2f}",
                                    })

                                df_comparison = pd.DataFrame(comparison_rows)
                                st.dataframe(df_comparison, use_container_width=True, hide_index=True, height=min(600, 75 + len(comparison_rows) * 52))
                    else:
                        st.info("동일 이미지를 여러 모델로 분석한 데이터가 없습니다.\n모델 비교 테스트를 실행해주세요.")

                # ========== 예상 비용 통계 탭 ==========
                with stat_tab2:
                    st.subheader("📈 예상 비용 통계")

                    # 1. 규모별 예상 비용 테이블 (제일 위)
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

                    # 4. 성공률 통계 테이블 (제일 마지막)
                    st.markdown("**성공률 통계**")
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
                        label="📥 예상 비용 통계 CSV 다운로드",
                        data=df_csv.to_csv(index=False, encoding="utf-8-sig"),
                        file_name=f"model_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # ========== 신뢰도 통계 탭 ==========
                with stat_tab3:
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

            else:
                st.info("분석 데이터가 없습니다. 먼저 이미지를 분석해주세요.")

        # ============================================
        # 데이터 조회 탭
        # ============================================
        with subtab2:
            # 필터 + 내보내기 통합 영역 (한 줄)
            f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([1.5, 1, 1, 1, 1.5])

            with f_col1:
                model_options = ["전체"] + list(MODEL_OPTIONS.keys())
                selected_model_filter = st.selectbox(
                    "모델",
                    options=model_options,
                    format_func=lambda x: "전체" if x == "전체" else MODEL_OPTIONS.get(x, {}).get("name", x).split(". ")[-1],
                    label_visibility="collapsed"
                )

            with f_col2:
                resolution_options = ["전체", "low", "medium", "high"]
                selected_resolution_filter = st.selectbox(
                    "해상도",
                    options=resolution_options,
                    label_visibility="collapsed"
                )

            with f_col3:
                success_options = ["전체", "성공만", "실패만"]
                selected_success_filter = st.selectbox(
                    "결과",
                    options=success_options,
                    label_visibility="collapsed"
                )

            # 통계 계산
            filtered_count = get_filtered_count(
                model_filter=selected_model_filter,
                resolution_filter=selected_resolution_filter,
                success_filter=selected_success_filter
            )
            db_stats = get_db_stats()
            total_count = db_stats["total_count"]

            is_filtered = selected_model_filter != "전체" or selected_resolution_filter != "전체" or selected_success_filter != "전체"

            with f_col4:
                if is_filtered:
                    st.caption(f"🔍 {filtered_count}/{total_count}건")
                else:
                    st.caption(f"📊 총 {total_count}건")

            with f_col5:
                if st.button("📥 Excel", use_container_width=True, disabled=total_count == 0):
                    all_results = load_results_from_db(limit=10000, offset=0)
                    if all_results:
                        import pandas as pd
                        from io import BytesIO

                        excel_rows = []
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
                            if r["success"] and r["metadata"]:
                                m = r["metadata"]
                                row["제목"] = m.get("title", "")
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
                                row["추천원단"] = ", ".join(m.get("usage_suggestion", {}).get("fabrics", []))
                            excel_rows.append(row)

                        df = pd.DataFrame(excel_rows)
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='분석결과')
                        excel_data = output.getvalue()

                        st.download_button(
                            label=f"📄 다운로드 ({total_count}건)",
                            data=excel_data,
                            file_name=f"textile_analysis_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            # 페이지네이션 설정 (한 줄로 통합)
            pg_col1, pg_col2 = st.columns([1, 5])
            with pg_col1:
                items_per_page = st.selectbox("항목수", [10, 20, 50], index=0, label_visibility="collapsed")

            # 필터 적용된 개수로 페이지네이션
            display_count = filtered_count if is_filtered else total_count

            if display_count > 0:
                total_pages = (display_count + items_per_page - 1) // items_per_page

                if "db_page" not in st.session_state:
                    st.session_state.db_page = 1

                # 필터 변경 시 페이지 리셋
                filter_key = f"{selected_model_filter}_{selected_resolution_filter}_{selected_success_filter}"
                if "last_filter_key" not in st.session_state:
                    st.session_state.last_filter_key = filter_key
                if st.session_state.last_filter_key != filter_key:
                    st.session_state.db_page = 1
                    st.session_state.last_filter_key = filter_key

                if st.session_state.db_page > total_pages:
                    st.session_state.db_page = max(1, total_pages)

                # 페이지네이션 컨트롤 (pg_col2에 배치)
                with pg_col2:
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                    with nav_col1:
                        if st.button("◀", disabled=st.session_state.db_page <= 1, use_container_width=True):
                            st.session_state.db_page -= 1
                            st.rerun()
                    with nav_col2:
                        st.markdown(f"<center style='padding:8px;'>{st.session_state.db_page} / {total_pages}</center>", unsafe_allow_html=True)
                    with nav_col3:
                        if st.button("▶", disabled=st.session_state.db_page >= total_pages, use_container_width=True):
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
                        title_str = meta.get("title", "")
                        title_part = f" | ✨ {title_str}" if title_str else ""
                        header = f"{status_icon} **{r['filename']}**{title_part} | {model_name} | {r['resolution']} | {category_str}"

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
                                        if r.get("image_url"):
                                            st.image(
                                                r['image_url'],
                                                caption=r["filename"],
                                                use_container_width=True
                                            )
                                        else:
                                            st.info("이미지 없음")

                                    with info_col:
                                        title_val = meta.get("title")
                                        if title_val:
                                            st.markdown(f"### ✨ {title_val}")
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
                                        fabrics = usage.get("fabrics", [])
                                        if products:
                                            st.write(f"추천 제품: {', '.join(products)}")
                                        if season:
                                            st.write(f"추천 시즌: {', '.join(season)}")
                                        if target:
                                            st.write(f"타겟 마켓: {', '.join(target)}")
                                        if fabrics:
                                            st.write(f"추천 원단: {', '.join(fabrics)}")

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
