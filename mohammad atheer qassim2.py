import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# قراءة البيانات من الملف
df = pd.read_csv("C:/Users/Mas/Desktop/New folder (2)/perfume_dataset.csv")


# عرض البيانات الأولية للتأكد من القراءة الصحيحة
print(df.head())

# حذف القيم الفارغة في العمود "title" بدلاً من "Ingredients"
df = df.dropna(subset=["title"])

# حذف التكرارات
df = df.drop_duplicates()

# إعادة تعيين الفهرس
df.reset_index(drop=True, inplace=True)

# استخدام TfidfVectorizer لتحليل نصوص العمود "title"
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["title"])

# حساب مصفوفة التشابه باستخدام cosine_similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# واجهة Streamlit
st.title("Perfume Recommendations System")
st.write("اختر عطرًا للحصول على توصيات مشابهة بناءً على الوصف")

# قائمة منسدلة لاختيار العطر
selected_perfume = st.selectbox("اختر عطرًا:", df["title"])

# عرض التوصيات بناءً على التشابه
if selected_perfume:
    selected_index = df[df["title"] == selected_perfume].index[0]
    similarity_scores = list(enumerate(similarity_matrix[selected_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    st.write("**توصيات مشابهة:**")
    for i, (index, score) in enumerate(sorted_scores[1:6]):
        st.write(f"{i+1}. {df['title'][index]} - Similarity Score: {score:.2f}")
