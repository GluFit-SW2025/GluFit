# 🍽️ 당뇨 관리 음식 인식 시스템

AI 기반 한국 음식 인식 및 당뇨 위험도 평가 앱

---

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 구조](#시스템-구조)
3. [Firestore 데이터 사용 방법](#firestore-데이터-사용-방법)
4. [모델 정보](#모델-정보)
5. [앱 주요 기능](#앱-주요-기능)
6. [최종 출력 화면](#최종-출력-화면)

---

## 🎯 프로젝트 개요

### 목적
당뇨병 환자 및 고위험군을 위한 음식 영양소 분석 및 위험도 평가 시스템

### 주요 기능
- 📸 **음식 사진 인식**: 150종 한국 음식 자동 분류
- 📊 **영양소 정보 제공**: 칼로리, 탄수화물, 당류, 나트륨 등
- ⚠️ **당뇨 위험도 평가**: 사용자 당뇨 단계별 맞춤 경고
- 📝 **식단 히스토리**: 오늘/과거 섭취 음식 기록 및 조회
- 📈 **영양소 누적 통계**: 일일 섭취량 추적

---

## 🏗️ 시스템 구조

```
사용자
  ↓
📱 플러터 앱
  ↓
┌─────────────────────┐
│ 1. 이미지 촬영/선택  │
└─────────────────────┘
  ↓
┌─────────────────────┐
│ 2. PyTorch 모델     │
│    음식 분류 실행   │
│    결과: "김치찌개"  │
└─────────────────────┘
  ↓
┌─────────────────────┐
│ 3. Firestore 조회   │
│    영양소 데이터    │
└─────────────────────┘
  ↓
┌─────────────────────┐
│ 4. 당뇨 위험도 계산 │
│    (사용자 레벨별)  │
└─────────────────────┘
  ↓
┌─────────────────────┐
│ 5. 결과 화면 표시   │
│    + DB 히스토리 저장│
└─────────────────────┘
```

---

## 🔥 Firestore 데이터 사용 방법

### Firestore 데이터 구조

```
Collection: nutrition_db
├─ Document: "김치찌개"
│  ├─ 식품명: "김치찌개"
│  ├─ 칼로리: 150
│  ├─ 탄수화물: 12.5
│  ├─ 총당류: 3.2
│  ├─ 단백질: 8.5
│  ├─ 지방: 7.2
│  ├─ 식이섬유: 2.1
│  ├─ 나트륨: 850
│  ├─ 콜레스테롤: 25
│  ├─ 1회제공량: "250g"
│  └─ 식품대분류: "음식"
│
├─ Document: "비빔밥"
│  └─ (동일 구조)
│
└─ ... (150개 음식)

Collection: users
└─ Document: {userId}
   ├─ 이름: "홍길동"
   ├─ 당뇨레벨: 3
   └─ 가입일: "2024-11-19"

Collection: user_history
└─ Document: {userId}
   └─ subcollection: meals
      ├─ Document: {timestamp}
      │  ├─ 음식명: "김치찌개"
      │  ├─ 칼로리: 150
      │  ├─ 탄수화물: 12.5
      │  ├─ 총당류: 3.2
      │  ├─ 위험여부: true
      │  ├─ 날짜: "2024-11-19"
      │  └─ 시간: "12:30"
      └─ ...
```

---

### 1. 영양소 데이터 조회

음식 이름으로 영양소 정보 가져오기:

```dart
import 'package:cloud_firestore/cloud_firestore.dart';

Future<Map<String, dynamic>?> getNutritionData(String foodName) async {
  try {
    // Firestore에서 음식명으로 조회
    DocumentSnapshot doc = await FirebaseFirestore.instance
        .collection('nutrition_db')
        .doc(foodName)  // 음식명이 Document ID
        .get();
    
    if (doc.exists) {
      return doc.data() as Map<String, dynamic>;
    } else {
      print('음식 정보를 찾을 수 없습니다: $foodName');
      return null;
    }
  } catch (e) {
    print('데이터 조회 오류: $e');
    return null;
  }
}

// 사용 예시
void example() async {
  var nutrition = await getNutritionData('김치찌개');
  
  if (nutrition != null) {
    print('칼로리: ${nutrition['칼로리']} kcal');
    print('탄수화물: ${nutrition['탄수화물']}g');
    print('총당류: ${nutrition['총당류']}g');
    print('단백질: ${nutrition['단백질']}g');
    print('지방: ${nutrition['지방']}g');
    print('나트륨: ${nutrition['나트륨']}mg');
  }
}
```

---

### 2. 식단 히스토리 저장

사용자가 분석한 음식을 히스토리에 저장:

```dart
Future<void> saveMealHistory(
  String userId, 
  String foodName, 
  Map<String, dynamic> nutrition,
  bool isDangerous
) async {
  try {
    // 현재 시간
    DateTime now = DateTime.now();
    String timestamp = now.millisecondsSinceEpoch.toString();
    
    // 저장할 데이터
    Map<String, dynamic> mealData = {
      '음식명': foodName,
      '칼로리': nutrition['칼로리'],
      '탄수화물': nutrition['탄수화물'],
      '총당류': nutrition['총당류'],
      '단백질': nutrition['단백질'],
      '지방': nutrition['지방'],
      '나트륨': nutrition['나트륨'],
      '위험여부': isDangerous,
      '날짜': '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}',
      '시간': '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}',
      'timestamp': now,
    };
    
    // Firestore에 저장
    await FirebaseFirestore.instance
        .collection('user_history')
        .doc(userId)
        .collection('meals')
        .doc(timestamp)
        .set(mealData);
    
    print('식단 히스토리 저장 완료');
  } catch (e) {
    print('히스토리 저장 오류: $e');
  }
}
```

---

### 3. 오늘의 식단 히스토리 조회

오늘 먹은 음식 목록 가져오기:

```dart
Future<List<Map<String, dynamic>>> getTodayMeals(String userId) async {
  try {
    DateTime now = DateTime.now();
    String today = '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
    
    // 오늘 날짜로 필터링
    QuerySnapshot snapshot = await FirebaseFirestore.instance
        .collection('user_history')
        .doc(userId)
        .collection('meals')
        .where('날짜', isEqualTo: today)
        .orderBy('timestamp', descending: true)  // 최신순 정렬
        .get();
    
    List<Map<String, dynamic>> meals = [];
    for (var doc in snapshot.docs) {
      meals.add(doc.data() as Map<String, dynamic>);
    }
    
    return meals;
  } catch (e) {
    print('히스토리 조회 오류: $e');
    return [];
  }
}

// 사용 예시
void showTodayMeals() async {
  String userId = 'user123';  // 로그인한 사용자 ID
  List<Map<String, dynamic>> meals = await getTodayMeals(userId);
  
  print('오늘 먹은 음식 ${meals.length}개');
  for (var meal in meals) {
    print('${meal['시간']} - ${meal['음식명']} (${meal['칼로리']}kcal)');
  }
}
```

---

### 4. 날짜별 식단 히스토리 조회

특정 날짜의 식단 조회:

```dart
Future<List<Map<String, dynamic>>> getMealsByDate(String userId, String date) async {
  try {
    QuerySnapshot snapshot = await FirebaseFirestore.instance
        .collection('user_history')
        .doc(userId)
        .collection('meals')
        .where('날짜', isEqualTo: date)  // 예: '2024-11-19'
        .orderBy('timestamp', descending: true)
        .get();
    
    List<Map<String, dynamic>> meals = [];
    for (var doc in snapshot.docs) {
      meals.add(doc.data() as Map<String, dynamic>);
    }
    
    return meals;
  } catch (e) {
    print('히스토리 조회 오류: $e');
    return [];
  }
}
```

---

### 5. 일일 영양소 누적 계산

오늘 섭취한 총 영양소 계산:

```dart
Future<Map<String, double>> calculateDailyNutrition(String userId) async {
  List<Map<String, dynamic>> meals = await getTodayMeals(userId);
  
  Map<String, double> total = {
    '칼로리': 0,
    '탄수화물': 0,
    '총당류': 0,
    '단백질': 0,
    '지방': 0,
    '나트륨': 0,
  };
  
  for (var meal in meals) {
    total['칼로리'] = (total['칼로리'] ?? 0) + (meal['칼로리'] ?? 0);
    total['탄수화물'] = (total['탄수화물'] ?? 0) + (meal['탄수화물'] ?? 0);
    total['총당류'] = (total['총당류'] ?? 0) + (meal['총당류'] ?? 0);
    total['단백질'] = (total['단백질'] ?? 0) + (meal['단백질'] ?? 0);
    total['지방'] = (total['지방'] ?? 0) + (meal['지방'] ?? 0);
    total['나트륨'] = (total['나트륨'] ?? 0) + (meal['나트륨'] ?? 0);
  }
  
  return total;
}

// 사용 예시
void showDailySummary() async {
  String userId = 'user123';
  Map<String, double> total = await calculateDailyNutrition(userId);
  
  print('오늘 총 섭취량:');
  print('칼로리: ${total['칼로리']} kcal');
  print('탄수화물: ${total['탄수화물']}g');
  print('총당류: ${total['총당류']}g');
}
```

---

## 🤖 모델 정보

### 모델 스펙
- **아키텍처**: MobileNetV2 (PyTorch)
- **클래스 수**: 150개 한국 음식
- **입력 크기**: 224x224 RGB
- **출력**: 150차원 확률 벡터
- **파일**: `model_mobile.ptl` (~15MB)

### 음식 분류 예시 코드

```dart
import 'package:pytorch_mobile/pytorch_mobile.dart';

class FoodClassifier {
  late Model _model;
  
  // 150개 음식 클래스 리스트
  final List<String> foodClasses = [
    '가지볶음', '간장게장', '갈비구이', '갈비찜', '갈비탕',
    '갈치구이', '갈치조림', '감자전', '감자조림', '감자채볶음',
    // ... (150개 전체 리스트)
  ];
  
  // 모델 로드
  Future<void> loadModel() async {
    _model = await PyTorchMobile.loadModel('assets/model_mobile.ptl');
  }
  
  // 음식 분류
  Future<Map<String, dynamic>> classifyFood(String imagePath) async {
    // 이미지 예측
    var result = await _model.getImagePrediction(
      imagePath,
      224, 224,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    );
    
    // 최고 확률 인덱스 찾기
    double maxProb = result.reduce((a, b) => a > b ? a : b);
    int predictedIndex = result.indexOf(maxProb);
    String foodName = foodClasses[predictedIndex];
    
    return {
      '음식명': foodName,
      '신뢰도': maxProb,
      '인덱스': predictedIndex,
    };
  }
}
```

---

## 📱 앱 주요 기능

### 1. 회원가입 및 당뇨 위험도 설정

사용자가 회원가입 시 당뇨 위험 단계 선택:

```
레벨 1: 정상 (당뇨 없음)
레벨 2: 경계성 (공복혈당장애)
레벨 3: 당뇨병 환자
레벨 4: 당뇨 합병증 위험군
```

---

### 2. 당뇨 위험도 판단 기준

각 레벨별 영양소 1회 제공량 기준:

| 레벨 | 당류(g) | 탄수화물(g) | 나트륨(mg) | 칼로리(kcal) |
|------|---------|-------------|-----------|--------------|
| 1    | ≤ 20    | ≤ 60        | ≤ 2000    | ≤ 700        |
| 2    | ≤ 15    | ≤ 50        | ≤ 1800    | ≤ 600        |
| 3    | ≤ 10    | ≤ 40        | ≤ 1500    | ≤ 500        |
| 4    | ≤ 5     | ≤ 30        | ≤ 1200    | ≤ 400        |

---

### 3. 위험도 판단 로직

```dart
class DiabetesRiskChecker {
  // 위험도 기준
  static const Map<int, Map<String, double>> limits = {
    1: {'당류': 20, '탄수화물': 60, '나트륨': 2000, '칼로리': 700},
    2: {'당류': 15, '탄수화물': 50, '나트륨': 1800, '칼로리': 600},
    3: {'당류': 10, '탄수화물': 40, '나트륨': 1500, '칼로리': 500},
    4: {'당류': 5,  '탄수화물': 30, '나트륨': 1200, '칼로리': 400},
  };
  
  // 위험 여부 판단
  static bool isDangerous(int userLevel, Map<String, dynamic> nutrition) {
    var limit = limits[userLevel]!;
    
    if ((nutrition['총당류'] ?? 0) > limit['당류']!) return true;
    if ((nutrition['탄수화물'] ?? 0) > limit['탄수화물']!) return true;
    if ((nutrition['나트륨'] ?? 0) > limit['나트륨']!) return true;
    if ((nutrition['칼로리'] ?? 0) > limit['칼로리']!) return true;
    
    return false;
  }
  
  // 경고 메시지 생성
  static List<String> getWarnings(int userLevel, Map<String, dynamic> nutrition) {
    var limit = limits[userLevel]!;
    List<String> warnings = [];
    
    if ((nutrition['총당류'] ?? 0) > limit['당류']!) {
      warnings.add('⚠️ 당류 초과: ${nutrition['총당류']}g (기준: ${limit['당류']}g)');
    }
    if ((nutrition['탄수화물'] ?? 0) > limit['탄수화물']!) {
      warnings.add('⚠️ 탄수화물 초과: ${nutrition['탄수화물']}g (기준: ${limit['탄수화물']}g)');
    }
    if ((nutrition['나트륨'] ?? 0) > limit['나트륨']!) {
      warnings.add('⚠️ 나트륨 초과: ${nutrition['나트륨']}mg (기준: ${limit['나트륨']}mg)');
    }
    if ((nutrition['칼로리'] ?? 0) > limit['칼로리']!) {
      warnings.add('⚠️ 칼로리 초과: ${nutrition['칼로리']}kcal (기준: ${limit['칼로리']}kcal)');
    }
    
    return warnings;
  }
}
```

---

## 🖼️ 최종 출력 화면

### 화면 1: 음식 분석 결과 (위험 음식)

```
┌─────────────────────────────────┐
│                                 │
│      [음식 이미지 표시]          │
│                                 │
└─────────────────────────────────┘

🍲 김치찌개
신뢰도: 95.2%

─────────────────────────────────

📊 영양 정보 (1회 제공량: 250g)

칼로리        150 kcal
탄수화물      12.5 g
  └ 당류      3.2 g
단백질        8.5 g
지방          7.2 g
식이섬유      2.1 g
나트륨        850 mg
콜레스테롤    25 mg

─────────────────────────────────

⚠️ 당뇨 위험도 평가 (레벨 3)

❌ 주의가 필요한 음식입니다!

경고 사항:
• 나트륨 초과: 850mg (기준: 500mg 이하)

[히스토리에 저장] [다시 촬영]
```

---

### 화면 2: 안전한 음식 (레벨 3 기준)

```
┌─────────────────────────────────┐
│                                 │
│      [음식 이미지 표시]          │
│                                 │
└─────────────────────────────────┘

🥗 시금치나물
신뢰도: 92.7%

─────────────────────────────────

📊 영양 정보 (1회 제공량: 100g)

칼로리        35 kcal
탄수화물      6.2 g
  └ 당류      1.1 g
단백질        3.5 g
지방          0.8 g
식이섬유      2.9 g
나트륨        320 mg
콜레스테롤    0 mg

─────────────────────────────────

✅ 당뇨 위험도 평가 (레벨 3)

안전한 음식입니다!

[히스토리에 저장] [다시 촬영]
```

---

### 화면 3: 오늘의 식단 히스토리

```
📅 오늘의 식단 (2024-11-19)

─────────────────────────────────

총 섭취량:
칼로리: 1,250 kcal
탄수화물: 185g
총당류: 28g
단백질: 65g
지방: 32g
나트륨: 3,200mg ⚠️

─────────────────────────────────

🍽️ 섭취 내역

08:30  아침
  🍚 김밥 (350 kcal) ✅

12:45  점심
  🍲 김치찌개 (150 kcal) ⚠️
  🍚 잡곡밥 (300 kcal) ✅

19:20  저녁
  🥩 불고기 (450 kcal) ⚠️

─────────────────────────────────

⚠️ 오늘의 건강 알림

• 나트륨을 너무 많이 섭취했어요!
  (기준: 1500mg, 섭취: 3200mg)
• 저녁 식사 후 가벼운 산책을 추천해요

[자세히 보기] [통계 보기]
```

---

## 📦 필요한 Flutter 패키지

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # Firebase
  firebase_core: ^2.24.0
  cloud_firestore: ^4.13.0
  firebase_auth: ^4.15.0
  
  # ML 모델
  pytorch_mobile: ^0.2.2
  
  # 이미지 처리
  image_picker: ^1.0.5
  image: ^4.1.3
  
  # UI
  intl: ^0.18.1
  fl_chart: ^0.65.0  # 통계 차트
```

---

## 🚀 개발 시작하기

### 1. 프로젝트 클론
```bash
git clone [repository_url]
cd diabetes-food-app
```

### 2. Flutter 패키지 설치
```bash
flutter pub get
```

### 3. Firebase 설정 파일 추가
- `google-services.json` (Android)
- `GoogleService-Info.plist` (iOS)

### 4. 모델 파일 추가
- `assets/model_mobile.ptl` 추가
- `pubspec.yaml`에 등록

### 5. 실행
```bash
flutter run
```

---

## 📝 주요 코드 흐름

```dart
// 1. 이미지 선택
final picker = ImagePicker();
final image = await picker.pickImage(source: ImageSource.camera);

// 2. 음식 분류
FoodClassifier classifier = FoodClassifier();
var result = await classifier.classifyFood(image.path);
String foodName = result['음식명'];

// 3. 영양소 조회
var nutrition = await getNutritionData(foodName);

// 4. 위험도 평가
int userLevel = 3;  // 사용자의 당뇨 레벨
bool isDangerous = DiabetesRiskChecker.isDangerous(userLevel, nutrition);
var warnings = DiabetesRiskChecker.getWarnings(userLevel, nutrition);

// 5. 히스토리 저장
await saveMealHistory(userId, foodName, nutrition, isDangerous);

// 6. 결과 화면 표시
showResultScreen(foodName, nutrition, isDangerous, warnings);
```
