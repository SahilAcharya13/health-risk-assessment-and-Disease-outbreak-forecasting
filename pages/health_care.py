import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
""", unsafe_allow_html=True)

box_content = {
    'Data Analysis': {
        'image': 'https://cdn-icons-png.flaticon.com/512/910/910316.png',
        'text': 'Data Analysis',
        'link': 'data_analysis'
    },
    'Data Base': {
        'image': 'https://cdn-icons-png.freepik.com/512/586/586279.png',
        'text': 'Data Base',
        'link': 'data_base'
    },
    'Hotel': {
        'image': 'https://cdn-icons-png.freepik.com/512/3780/3780959.png',
        'text': 'Hotel',
        'link': 'hotel'
    },
    'Amazon': {
        'image': 'https://cdn.icon-icons.com/icons2/2351/PNG/512/logo_amazon_icon_143189.png',
        'text': 'Amazon',
        'link': 'amazon'
    },
    'Machine Learning': {
        'image': 'https://cdn-icons-png.freepik.com/512/2172/2172891.png',
        'text': 'Machine Learning',
        'link': 'machine_learning'
    },
    'Diabetes Prediction': {
        'image': 'https://cdn-icons-png.flaticon.com/512/7350/7350822.png',
        'text': 'Diabetes Prediction',
        'link': 'diabetis'
    },
    'Brain Tumor': {
        'image': 'https://cdn2.iconfinder.com/data/icons/cancer-survivors-color-line/74/Untitled-3-08-512.png',
        'text': 'Brain Tumor',
        'link': 'brain_tumor'
    },
    'Diabetes Analysis': {
        'image': 'https://static-www.elastic.co/v3/assets/bltefdd0b53724fa2ce/blta401f2e7dad39503/620d844d9d54947c7f131b0a/illustration-industry-health.png',
        'text': 'Diabetes Analysis',
        'link': 'analysis_diabetes'
    },
    'Bone Fracture Detection': {
        'image': 'https://cdn-icons-png.flaticon.com/512/1823/1823760.png',
        'text': 'Bone Fracture Detection',
        'link': 'bone_fracture'
    },
    'Covid Prediction': {
        'image': 'https://cdn.iconscout.com/icon/free/png-256/free-covid19-2221473-1848028.png',
        'text': 'Covid Prediction',
        'link': 'covid19'
    },
    'Heart Attack Data Cleaning': {
        'image': 'https://cdn-icons-png.freepik.com/512/6603/6603844.png',
        'text': 'Heart Attack Data Cleaning',
        'link': 'heart_attack'
    },
    'Diabetes data visualization': {
        'image': 'https://cdn-icons-png.flaticon.com/512/9032/9032016.png',
        'text': 'Diabetes data visualization',
        'link': 'diabetes_data_visualization'
    },
    'Disease analysis': {
        'image': 'https://cdn-icons-png.flaticon.com/512/6534/6534049.png',
        'text': 'Disease analysis',
        'link': 'disease_analysis'
    },
    'Disease Data Visualization': {
        'image': 'https://cdn-icons-png.flaticon.com/512/8638/8638122.png',
        'text': 'Disease Data Visualization',
        'link': 'disease_data_visualization'
    },
    'Stroke Analysis': {
        'image': 'https://cdn-icons-png.flaticon.com/128/2660/2660210.png',
        'text': 'Stroke Analysis',
        'link': 'stroke_analysis'
    },
    'Stroke Visualization': {
        'image': 'https://cdn-icons-png.flaticon.com/256/1006/1006633.png',
        'text': 'Stroke Visualization',
        'link': 'stroke_visualization'
    },
    'Heart attack prediction': {
        'image': 'https://cdn-icons-png.freepik.com/512/2167/2167095.png',
        'text': 'Heart attack prediction',
        'link': 'ht_prediction'
    },
    'Life expectancy Visualization': {
        'image': 'https://cdn-icons-png.flaticon.com/512/8430/8430898.png',
        'text': 'Life expectancy Visualization',
        'link': 'life_expectancy_visualization'
    },
    'Life expectancy Cleaning': {
        'image': 'https://cdn.iconscout.com/icon/premium/png-256-thumb/risinglife-expectancy-1921044-1620528.png',
        'text': 'Life expectancy Cleaning',
        'link': 'life_expectancy_cleaning'
    },
    'Stroke Data Cleaning': {
        'image': 'https://cdn-icons-png.flaticon.com/512/11604/11604233.png',
        'text': 'Stroke Data Cleaning',
        'link': 'stroke_data_cleaning'
    }
}

# Display the boxes
box_titel=f"""<h1>Healthcare Dataset Analysis</h1>"""
st.write(box_titel,unsafe_allow_html=True)
col1,col2=st.columns(2)
with col1:
      for box_name in ['Diabetes Analysis','Stroke Analysis']:
        box_content_info = box_content[box_name]
        box_html = f"""
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)
with col2:
      for box_name in ['Disease analysis']:
        box_content_info = box_content[box_name]
        box_html = f"""
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)

box_titel=f"""<h1>Healthcare Dataset Visualization</h1>"""
st.write(box_titel,unsafe_allow_html=True)
col1 ,col2=st.columns(2)
with col1:
      for box_name in ['Diabetes data visualization','Stroke Visualization']:
        box_content_info = box_content[box_name]
        box_html = f"""
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)
with col2:
    for box_name in ['Disease Data Visualization','Life expectancy Visualization']:
        box_content_info = box_content[box_name]
        box_html = f"""
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)


box_titel=f"""<h1>Healthcare Dataset Cleaning</h1>"""
st.write(box_titel,unsafe_allow_html=True)
col1 ,col2=st.columns(2)
with col1:
      for box_name in ['Heart Attack Data Cleaning','Stroke Data Cleaning']:
        box_content_info = box_content[box_name]
        box_html = f"""
        
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)
with col2:
    for box_name in ['Life expectancy Cleaning']:
        box_content_info = box_content[box_name]
        box_html = f"""

        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)
st.title("Healthcare Prediction")

col1, col2= st.columns(2)
with col1:
    for box_name in ['Diabetes Prediction','Brain Tumor','Heart attack prediction']:
        box_content_info = box_content[box_name]
        box_html = f"""
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)
with col2:
    for box_name in ['Bone Fracture Detection','Covid Prediction']:
        box_content_info = box_content[box_name]
        box_html = f"""
        <a href="{box_content_info['link']}" style="text-decoration: none; color: black;">
            <div style="padding: 20px;margin: 10px; border: 1px solid black; border-radius: 10px; text-align: center; background:#B9CDD2">
                <img src="{box_content_info['image']}" style="width: 100px; height: 100px; object-fit: cover; background-color: transparent;">
                <div colour:white>{box_content_info['text']} </div>
            </div>
        </a>
        """
        st.write(box_html, unsafe_allow_html=True)







