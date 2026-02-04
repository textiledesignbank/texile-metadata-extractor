"""
SQLAlchemy ORM Models
분리된 모델 파일 - Alembic 마이그레이션용
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base

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
