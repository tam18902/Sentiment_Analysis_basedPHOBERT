import streamlit as st
from fine_tune_model import *

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
def home():
    import streamlit as st
    from PIL import Image
    st.sidebar.success("You are at home now 💒.")
    image = Image.open('./image/emoji.png')
    st.image(image)
    st.write("# Welcome to Sentiment Analysis! 👋")
    st.write("""Đây là sản phẩm demo cho mô hình phân loại dữ liệu kiểu chữ viết - TEXT CLASSIFICATION\n
    Ứng dụng cho phép dự đoán riêng lẻ các câu văn thông thường, hoặc dữ liệu text lưu trữ dạng tập tin.\n
    Vì được huấn luyện trên tập các đánh giá của sinh viên về một khóa học, dữ liệu nên có nội dung tương tự để có kết quả tốt nhất\n
    Để tìm hiểu thêm về ứng dụng hãy đến với SideBar để chuyển hướng.!
    """)

def chatbox_feedback():
    import streamlit as st
    from PIL import Image
    import time
    import numpy as np
    #user_image = Image.open("./image/girl.png")
    #user_image = user_image.resize((8, 8), Image.LANCZOS)
    #bot_image = Image.open("./image/robot.png")
    #bot_image = bot_image.resize((8, 8), Image.LANCZOS)
    st.sidebar.success("Start predict with our dummy bot 👨‍💻.")
    neg_icon = Image.open("./image/negative.png")
    neu_icon = Image.open("./image/neutral.png")
    pos_icon = Image.open("./image/positive.png")
    sentiment_image = [neg_icon,neu_icon,pos_icon]
    reponse_sentiment = [
        ["Có vẻ bạn không được hài lòng về khóa học này!!😥", 
         "Thật đáng tiếc khi khóa học không mang lại niềm vui cho bạn!!",
         "Tôi mong rằng bạn sẽ có trải nghiệm tốt hơn ở những khóa học khác !!"],
        ["Tôi không rõ ràng về cảm xúc của bạn!!",
         "Không rõ bạn đang vui hay buồn vì khóa học này!!",
         "Cảm xúc của bạn ổn định quá đi mất :3"],
        ["Tôi cũng vui vì bạn thấy vui về khóa học này ❤️",
         "Có lẻ khóa học đã mang cho bạn những trải nghiệm tuyệt vời! 😘😘",
         "Mong là khóa học nào bạn cũng thấy hạnh phúc như vậy! 🥰"]
    ]
    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your feedback? Tell me 🧐"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        prob = predict([prompt])[0]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            label = np.argmax(prob)
            message_placeholder.image(sentiment_image[label])
            time.sleep(0.75)
            full_response = ""
            assistant_response = f"[Negative: {prob[0]:.2f}, Neutral: {prob[1]:.2f}, Positive: {prob[2]:.2f}].  \
            {np.random.choice(reponse_sentiment[label])}"
            
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def file_feedback():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import base64
    st.sidebar.success("Single-line text file is the best. 😊")
    text_show = ["Negative", "Neutral", "Positive"]
    bytes_data = b""
    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    uploaded_files = st.file_uploader("Choose a TXT(s) file", type=['txt'], accept_multiple_files=True)
    start = st.button("Starting predict.")
    if start:
        count = 0
        for uploaded_file in uploaded_files:
            count +=1
            bytes_data += uploaded_file.read()
            st.write("Filename:", uploaded_file.name)
        if count == 0:
            st.error('No file has choosen', icon="🚨")
        else:
            st.success(f'{count} file(s)', icon="✅")
            text = bytes_data.replace(b'\r',b'').decode("utf-8").split('\n')
            if len(text) > 100:
                st.error('Too much line in all files. Limited 100 lines', icon="🚨")
            else:
                status_pred = st.empty()
                with open("./image/pleasewait.gif", "rb") as gif:
                    data_url = base64.b64encode(gif.read()).decode("utf-8")
                    status_pred.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="wait gif">',
                                        unsafe_allow_html=True,)
                
                pred_ = predict(text)
                result_ = [text_show[np.argmax(pred)] for pred in pred_]
                status_pred.text("Predict Success!")
                df = pd.DataFrame({'Sentence':text, 'Sentiment':result_})
                st.write(df)

                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                "Press to Download",
                csv,
                "dataframe.csv",
                "text/csv",
                key='download-csv'
                )

def about_us():
    import streamlit as st
    st.sidebar.success("Our pleasure to meet you ✌️.")
    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.markdown(f"## Author")
    st.text(
        """
        VNUHCM - University of Science\n
        Nguyễn Thiên An 👩‍🎓- 📧 20120030@student.hcmus.edu.vn\n
        Nguyễn Minh Tâm 👨‍🎓- 📧 20120368@student.hcmus.edu.vn
    """
    )
    st.markdown("## Architech")
    st.markdown("### PhoBERT: Pre-trained language models for Vietnamese")
    st.markdown("Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese")
    st.text("""
            @inproceedings{phobert,\n\t
            title     = {{PhoBERT: Pre-trained language models for Vietnamese}},\n\t
            author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},\n\t
            booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},\n\t
            year      = {2020},\n\t
            pages     = {1037--1042}\n\t
            }""")
    st.markdown("## Dataset")
    st.markdown("### Title:")
    st.markdown("UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis")
    st.markdown("### Author's Name:")
    st.markdown("Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy")
    st.markdown("### Year:")
    st.markdown("2018")
    st.markdown("### Link:")
    st.markdown("https://nlp.uit.edu.vn/datasets/")
    st.markdown("### Description")          
    st.markdown("Students’ feedback is a vital resource for the interdisciplinary research involving the combining of two different research fields between sentiment analysis and education. Vietnamese Students’ Feedback Corpus (UIT-VSFC) is the resource consists of over 16,000 sentences which are human-annotated with two different tasks: sentiment-based and topic-based classifications.")          

page_names_to_funcs = {
    "Home": home,
    "Chatbox Feedback": chatbox_feedback,
    "File Feedback": file_feedback,
    "About us": about_us
}

demo_name = st.sidebar.selectbox("Where you want to come?", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()