import joblib
import streamlit as st

try:
    svm_pipeline_model = joblib.load("Main/svm_pineline_model.jb")
except FileNotFoundError:
    st.error(
        "Không tìm thấy file mô hình 'svm_pipeline_model.jb'. Hãy chắc chắn bạn đã copy file này vào cùng thư mục!"
    )

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

new_input = st.text_area("News Articles:", "")

if st.button("Check news"):
    if new_input.strip():
        predict = svm_pipeline_model.predict([new_input])

        if predict[0] == 1:
            st.success("The news is Real!")
        else:
            st.error("The news is Fake!")
    else:
        st.warning("Please enter some text to analyze.")
