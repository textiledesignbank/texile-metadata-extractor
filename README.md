# 텍스타일 이미지 메타데이터 추출기 MVP

CTO 데모용 프로토타입 - LLM Vision API를 활용한 텍스타일 디자인 이미지 분류 및 메타데이터 추출 , Streamlit 을 통해 배포되어있습니다.

## 빠른 시작

### 1. 설치

```bash
cd image-metadata-mvp

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 키는 구글드라이브 tips/김우혁/환경변수 폴더에 정리되어있습니다.

**API 키 발급**: [Google AI Studio](https://aistudio.google.com/) → Get API Key

!!!\*\* **[ 4월12일 ]** 이후에는 무료 크레딧 만료되어 비용 청구됨

### 3. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 기능

- **이미지 업로드**: 여러 이미지 동시 업로드 지원
- **메타데이터 추출**:
  - 디자인 제목 자동 생성
  - 카테고리 분류 (floral, geometric, ethnic 등)
  - 색상 분석 (Hex 코드, 팔레트 이름)
  - 스타일/무드 분석
  - 검색 키워드 생성
  - 제품 추천
- **비용 추적**: 실시간 API 비용 계산
- **1200개 예상 비용**: 대량 처리 시 예상 비용 표시
- **Excel 다운로드**: 분석 결과 내보내기
- **S3 이미지 저장**: AWS S3에 이미지 업로드 및 URL 관리
- **RDS 데이터베이스**: SQLAlchemy ORM + Alembic 마이그레이션 기반 분석 결과 영구 저장
- **이미지 중복 체크**: SHA-256 해시 기반 중복 이미지 감지
- **로그인 인증**: 어드민 계정 기반 접근 제어

## 추출되는 메타데이터

```json
{
  "title": "Midnight Garden",
  "category": {
    "primary": "floral",
    "secondary": ["botanical", "nature"],
    "confidence": 0.95
  },
  "colors": {
    "dominant": ["#E8D5B7", "#4A7C59", "#8B4513"],
    "palette_name": "Earth Tones",
    "mood": "warm"
  },
  "style": {
    "type": "vintage",
    "era": "1970s",
    "technique": "watercolor"
  },
  "pattern": {
    "scale": "medium",
    "repeat_type": "half-drop",
    "density": "moderate"
  },
  "mood": {
    "primary": "romantic",
    "secondary": ["elegant", "soft"]
  },
  "keywords": {
    "search_tags": ["floral", "vintage", "botanical", "rose", "garden"],
    "description": "Romantic vintage floral pattern with soft watercolor roses"
  },
  "usage_suggestion": {
    "products": ["dress", "curtain", "bedding"],
    "season": ["spring", "summer"],
    "target_market": ["women", "home decor"]
  }
}
```

## 데이터베이스 스키마

`models.py`에 정의된 `AnalysisResult` ORM 모델 (`analysis_results` 테이블):

| 컬럼                    | 타입         | 설명                                |
| ----------------------- | ------------ | ----------------------------------- |
| `id`                    | Integer (PK) | 자동 증가                           |
| `filename`              | String(500)  | 파일명 (인덱스)                     |
| `image_hash`            | String(64)   | SHA-256 해시 - 중복 체크용 (인덱스) |
| `image_url`             | String(1000) | S3 URL                              |
| `model`                 | String(100)  | 사용된 LLM 모델명 (인덱스)          |
| `resolution`            | String(50)   | 이미지 해상도                       |
| `success`               | Boolean      | 분석 성공 여부                      |
| `metadata`              | JSON         | 추출된 메타데이터                   |
| `cost_usd` / `cost_krw` | Float        | API 비용                            |
| `elapsed_time`          | Float        | 처리 시간 (초)                      |
| `error`                 | Text         | 에러 메시지                         |
| `created_at`            | DateTime     | 생성 시각 (인덱스)                  |

마이그레이션은 Alembic으로 관리:

```bash
# 마이그레이션 실행
alembic upgrade head
```

## 파일 구조

```
image-metadata-mvp/
├── app.py              # Streamlit 메인 앱
├── models.py           # SQLAlchemy ORM 모델 (Alembic 마이그레이션용)
├── app_gradio.py       # Gradio 버전 앱
├── alembic/            # DB 마이그레이션
│   ├── env.py
│   └── versions/       # 마이그레이션 스크립트
├── requirements.txt    # Python 패키지
└── README.md           # 이 파일
```

## CTO 데모 포인트

1. **실시간 분석**: 이미지 업로드 즉시 분석 결과 표시
2. **비용 효율**: 1200개 이미지에 400원 (Gemini 2.5 Flash-Lite)
3. **구조화된 출력**: 검색 및 필터링에 바로 사용 가능한 JSON
4. **영구 저장**: S3 + RDS로 데이터 유실 없는 분석 결과 관리
5. **중복 방지**: 이미지 해시 기반 중복 분석 방지
