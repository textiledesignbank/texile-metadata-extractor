"""add image_hash and image_url columns

Revision ID: 001
Revises:
Create Date: 2026-01-26

기존 테이블에 image_hash, image_url 컬럼 추가
image_data 컬럼 삭제
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. 새 컬럼 추가
    op.add_column('analysis_results', sa.Column('image_hash', sa.String(64), nullable=True))
    op.add_column('analysis_results', sa.Column('image_url', sa.String(1000), nullable=True))

    # 2. 인덱스 추가
    op.create_index('idx_image_hash', 'analysis_results', ['image_hash'])

    # 3. 기존 image_data 컬럼 삭제 (있는 경우)
    try:
        op.drop_column('analysis_results', 'image_data')
    except Exception:
        pass  # 컬럼이 없으면 무시


def downgrade() -> None:
    # 1. 인덱스 삭제
    op.drop_index('idx_image_hash', table_name='analysis_results')

    # 2. 새 컬럼 삭제
    op.drop_column('analysis_results', 'image_url')
    op.drop_column('analysis_results', 'image_hash')

    # 3. image_data 컬럼 복원
    op.add_column('analysis_results', sa.Column('image_data', sa.Text(), nullable=True))
