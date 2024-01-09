import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import tempfile
from PIL import Image
import json
import plotly.express as px
import pandas as pd
import os


st.set_page_config(page_title='Computer Vision Dashboard', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')

hide_menu_style = """
    <style>
        MainMenu {visibility: hidden;}
        
        
         div[data-testid="stHorizontalBlock"]> div:nth-child(1)
        {  
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            color:black;
            background:dodgerblue;
            font-weight:bold;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(2)
        {   
            border : 2px solid #doe0db;
            background:dodgerblue;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
    </style>
    """

main_title = """
            <div>
                <h1 style="color:white;
                text-align:center; font-size:35px;
                margin-top:-95px;">
                Computer Vision Analytics</h1>
            </div>
            """
    
sub_title = """
            <div>
                <h6 style="color:dodgerblue;
                text-align:center;
                margin-top:-40px;">
                Detection Dasboard </h6>
            </div>
            """
    
st.markdown(hide_menu_style, 
            unsafe_allow_html=True)

st.markdown(main_title,
            unsafe_allow_html=True)

st.markdown(sub_title,
            unsafe_allow_html=True)

screen = st.empty()
tfile = tempfile.NamedTemporaryFile(delete=False)
st.sidebar.subheader('Settings')
app_mode = st.sidebar.selectbox('App Mode', ['About App', 'Run Object Detection - Video','Run Object Detection - Image'])



if app_mode == 'About App':

    st.markdown(
        """This an application that uses **YOLOv8** model for object detection and tracking, 
        while using **Streamlit** as a GUI for user inputs and computer vision analytics."""
    )
    st.video('Computer Vision Analytics - Made with Clipchamp.mp4') #DEMO VIDEO
    st.write('# About Me')
    st.image('avatar.png', width=200)
    st.markdown("""
                I'm Saleem Adebayo, an AI/ML Developr and Computer Vision Engineer. I have proficent skills in Data Science [FULL STACK]\n
                Data Analysis, Pyhton Automation, Web Scrapping and Data Labeling/Annotation.
                <p>Connect and Follow me on LinkedIn: <a href="https://www.linkedin.com/in/saleem-adebayo-82b512138/" target="_blank">LinkedIn</a></p>
                <p>Follow me on Threads: <a href="https://www.threads.net/@bayoleems" target="_blank">Threads</a></p>
                """, unsafe_allow_html=True)

if app_mode == 'Run Object Detection - Video':

    with open(r'classes.txt') as f:
        classes_file = f.read()
        classes_file = classes_file.split("\n")
   
    class_list = st.sidebar.multiselect(label='Class List', placeholder='Select classes to detect', options=classes_file, default=['person'])
    confidence = st.sidebar.slider('Choose Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    video_file = st.sidebar.file_uploader('Upload Video Here', type= ['mp4','webm','avi','mov'])
    run = st.sidebar.button('Run', type='primary')
    if run:
        tfile.write(video_file.read())
        clip = cv2.VideoCapture(tfile.name)
        model = YOLO('best.onnx', task='detect')
        track_data_vid = {}
        while True:
            ret, frame = clip.read()
            if not ret:
                number_of_classes=[]
                for value in track_data_vid.values():
                    number_of_classes.append(value)
                data = pd.DataFrame({'NUMBER OF DETECTIONS':number_of_classes})
                df = data['NUMBER OF DETECTIONS'].value_counts().reset_index()
                df.columns = ['Detection(s)', 'Number of detections']
                fig = px.bar(df,x='Detection(s)',y='Number of detections',color='Detection(s)')
                st.plotly_chart(fig)
                st.download_button('Download Data', data=df.to_csv(index=False).encode('utf-8'), file_name='data.csv', mime='text/csv')
                break
            results = model.track(frame, persist=True, show=False)
            result = results[0]
            ids = np.array(result.boxes.id)
            conf = np.array(result.boxes.conf)
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")

            for bbox, cls, conf_cent, id in zip(bboxes, classes, conf, ids):
                if (classes_file[cls] in class_list) and (conf_cent >= confidence):
                    id = int(id)
                    track_data_vid[id]= classes_file[cls]
                    with open('track_data_vid.json', 'w+') as fp:
                        json.dump(track_data_vid, fp)
                    x, y, x2, y2 = bbox
                    x = int(x)
                    y = int(y)
                    x2 = int(x2)
                    y2 = int(y2)
                    text_size = cv2.getTextSize(f'{classes_file[cls]}', 0, fontScale=1, thickness=2)[0]
                    c2 = x + text_size[0], y - text_size[1] - 3
                    cv2.rectangle(frame, (x, y), (x2, y2), (255, 144, 50), 2)
                    cv2.rectangle(frame, (x, y), c2, (255, 144, 50), -1, cv2.LINE_AA)
                    cv2.putText(frame, f'{classes_file[cls]}', (x, y - 10), 0, 0.5, (0, 0, 0), 1, lineType = cv2.LINE_AA)
                    screen.image(frame,use_column_width=True, channels='BGR')       
    if st.sidebar.button('Stop Inference', type='secondary'):
        with open('track_data_vid.json', 'r+') as fp:
                track_info = json.load(fp)
        number_of_classes=[]
        for value in track_info.values():
            number_of_classes.append(value)
        data = pd.DataFrame({'NUMBER OF DETECTIONS':number_of_classes})
        df = data['NUMBER OF DETECTIONS'].value_counts().reset_index()
        df.columns = ['Detection(s)', 'Number of detections']
        fig = px.bar(df,x='Detection(s)',y='Number of detections',color='Detection(s)')
        st.plotly_chart(fig)
        st.download_button('Download Data', data=df.to_csv(index=False).encode('utf-8'), file_name='data.csv', mime='text/csv')
        os.remove(r'track_data_vid.json')
        with open('track_data_vid.json', 'w') as fp:
            json.dump({}, fp)

if app_mode == 'Run Object Detection - Image':

    with open(r'classes.txt') as f:
        classes_file = f.read()
        classes_file = classes_file.split("\n")
    st.session_state.track_data_img = {}
    class_list = st.sidebar.multiselect(label='Class List', placeholder='Select classes to detect', options=classes_file, default=['person'])
    confidence = st.sidebar.slider('Choose Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    image_file = st.sidebar.file_uploader('Upload Image Here', type= ['jpg','jpeg','png'])

    if st.sidebar.button('Run', type='primary'):

        image = cv2.imdecode(np.frombuffer(image_file.read(), 'u1'), 1)
        img = np.array(Image.open(image_file))
        model = YOLO('best.onnx', task='detect')
        results = model.track(img, persist=True, show=False)
        result = results[0]
        ids = np.array(result.boxes.id)
        conf = np.array(result.boxes.conf)
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        for bbox, cls, conf_cent, id in zip(bboxes, classes, conf, ids):
            if (classes_file[cls] in class_list) and (conf_cent >= confidence):
                id = int(id)
                st.session_state.track_data_img[id]= classes_file[cls]
                x, y, x2, y2 = bbox
                x = int(x)
                y = int(y)
                x2 = int(x2)
                y2 = int(y2)
                text_size = cv2.getTextSize(f'{classes_file[cls]}', 0, fontScale=1, thickness=2)[0]
                c2 = x + text_size[0], y - text_size[1] - 3
                cv2.rectangle(img, (x, y), (x2, y2), (255, 144, 50), 2)
                cv2.rectangle(img, (x, y), c2, (255, 144, 50), -1, cv2.LINE_AA)
                cv2.putText(img, f'{classes_file[cls]}', (x, y - 10), 0, 0.5, (0, 0, 0), 2, lineType = cv2.LINE_AA)
        st.subheader('Inference')
        st.image(img, use_column_width=True, channels='RGB')
        number_of_classes_img=[]
        for value in st.session_state.track_data_img.values():
            number_of_classes_img.append(value)
        data = pd.DataFrame({'NUMBER OF DETECTIONS':number_of_classes_img})
        df = data['NUMBER OF DETECTIONS'].value_counts().reset_index()
        df.columns = ['Detection(s)', 'Number of detections']
        fig = px.bar(df,x='Detection(s)',y='Number of detections',color='Detection(s)')
        st.plotly_chart(fig)
        st.download_button('Download Data', data=df.to_csv(index=False).encode('utf-8'), file_name='data.csv', mime='text/csv')

