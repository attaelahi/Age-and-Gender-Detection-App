import streamlit as st
from PIL import Image
from transformers import pipeline

# Load the age detection model
age_pipe = pipeline("image-classification", model="ares1123/photo_age_detection")

# Load the gender detection model
gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification")

# Define CSS styles
STYLE = """
<style>
.upload-button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}
.result-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
.result {
    font-size: 18px;
    font-weight: bold;
    color: #333333;
}
</style>
"""

# Define the Streamlit app
def main():
    st.set_page_config(page_title="Age and Gender Detection App", layout="centered", initial_sidebar_state="auto")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown("<h1>Age and Gender Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<p>Upload a photo to determine the age and gender of the person.</p>", unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload_image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Analyze button
        if st.button("Analyze"):
            # Analyze the image and predict age and gender
            predicted_age, predicted_gender = detect_age_and_gender(image)

            # Display results
            with st.container():
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown(f'<p class="result">Predicted Age: {predicted_age}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result">Predicted Gender: {predicted_gender}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Function to detect age and gender using the provided models
def detect_age_and_gender(image):
    # Analyze the image and get the predicted age
    age_result = age_pipe(image)
    predicted_age = age_result[0]['label'].split('_')[-1]

    # Analyze the image and get the predicted gender
    gender_result = gender_pipe(image)
    predicted_gender = gender_result[0]['label'].split('_')[-1]

    return predicted_age, predicted_gender

if __name__ == "__main__":
    main()
