"""
Football Match Outcome Prediction
مشروع Machine Learning (Classification) لتوقع نتيجة المباراة

البيانات: مصممة خصيصًا (20,000 مباراة)، 60% منطق و40% تشويش عشوائي واقعي
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# 1. تحميل البيانات
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/AHMEDMAHMOUD207/file2/main/file.csv"
df = pd.read_csv(url)

print("عدد الصفوف والأعمدة:", df.shape)


# ---------------------------------------------------------
# 2. تجهيز المدخلات (X) والهدف (y)
# ---------------------------------------------------------
X = df[["team_rating", "opponent_rating", "is_home", "recent_form_points", "key_players_injured"]]
y = df["won_match"]


# ---------------------------------------------------------
# 3. تجربة أولى: تقسيمة واحدة (train_test_split)
# ---------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

print("\nدقة تقسيمة واحدة (train_test_split):", accuracy)


# ---------------------------------------------------------
# 4. التحقق الموثوق: Cross-Validation
# ---------------------------------------------------------
model_cv = LogisticRegression()
scores = cross_val_score(model_cv, X, y, cv=10)

print("\nنتائج الـ 10 تقسيمات (Cross-Validation):")
print(scores)
print("متوسط الدقة الموثوقة:", scores.mean())


# ---------------------------------------------------------
# الاستنتاجات (Insights)
# ---------------------------------------------------------
"""
1. دقة الموديل الحقيقية والموثوقة حوالي 63% (بعد Cross-Validation)، مقارنة
   بخط أساس حوالي 50% لو تم التخمين العشوائي - أي أن الموديل يتعلم نمطًا
   حقيقيًا وليس مجرد تخمين.

2. حجم البيانات أثّر بشكل كبير على استقرار النتائج: على 600 صف، تراوحت
   نتائج Cross-Validation بين 56.7% و66.7% (فرق ~10%). على 20,000 صف،
   تراوحت بين 61.3% و64.35% فقط (فرق ~3%) - ما يؤكد أن البيانات الأكبر
   تعطي تقييمًا أكثر ثباتًا وموثوقية.

3. نتيجة واحدة من train_test_split (65.83%) كانت مضللة قليلاً مقارنة
   بمتوسط Cross-Validation الحقيقي (~60-63%) - درس مهم في أن الاعتماد على
   تقسيمة واحدة فقط قد يعطي انطباعًا غير دقيق عن أداء الموديل الفعلي.

4. 63% دقة تعتبر نتيجة منطقية وصحية لهذا النوع من البيانات، لأن 40% من
   النتيجة عشوائية بتصميم البيانات نفسها (تمامًا كما في كرة القدم الواقعية)
   فلا يمكن لأي موديل تجاوز حد معين من الدقة مهما كان قويًا.
"""
