# 텍스타일 이미지 메타데이터 추출기 MVP

CTO 데모용 프로토타입 - LLM Vision API를 활용한 텍스타일 디자인 이미지 분류 및 메타데이터 추출

## 비용 요약

| 모델 | 1200개 비용 | 원화 환산 |
|-----|------------|----------|
| **Gemini 2.5 Flash-Lite** | **$0.31** | **약 400원** |
| Gemini 3 Flash | $2.14 | 약 2,800원 |

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

### 2. API 키 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
# GEMINI_API_KEY=your_api_key_here
```

**API 키 발급**: [Google AI Studio](https://aistudio.google.com/) → Get API Key

### 3. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 기능

- **이미지 업로드**: 여러 이미지 동시 업로드 지원
- **메타데이터 추출**:
  - 카테고리 분류 (floral, geometric, ethnic 등)
  - 색상 분석 (Hex 코드, 팔레트 이름)
  - 스타일/무드 분석
  - 검색 키워드 생성
  - 제품 추천
- **비용 추적**: 실시간 API 비용 계산
- **1200개 예상 비용**: 대량 처리 시 예상 비용 표시
- **CSV 다운로드**: 분석 결과 내보내기

## 추출되는 메타데이터

```json
{
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

## 파일 구조

```
image-metadata-mvp/
├── app.py              # Streamlit 메인 앱
├── requirements.txt    # Python 패키지
├── .env.example        # 환경변수 템플릿
└── README.md           # 이 파일
```

## CTO 데모 포인트

1. **실시간 분석**: 이미지 업로드 즉시 분석 결과 표시
2. **비용 효율**: 1200개 이미지에 400원 (Gemini 2.5 Flash-Lite)
3. **구조화된 출력**: 검색 및 필터링에 바로 사용 가능한 JSON
4. **확장 가능**: 배치 처리, DB 저장 등 추가 가능
