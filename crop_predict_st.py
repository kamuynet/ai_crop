import pandas as pd
from pycaret.regression import *

# データの読み込み
df = pd.read_csv("crop_yield_data.csv")

# モデルの初期化
reg = setup(data=df, target='yield', normalize=True, silent=True)

# モデルのトレーニング
lr = create_model('lr')

# Webアプリケーションのタイトル
from pycaret.utils import version
import streamlit as st
st.set_page_config(page_title='作物産出予想アプリ PyCaret',
                   page_icon='🌾',
                   layout='wide')

# Webアプリケーションのタイトル
st.title('作物産出予想アプリ')

# スライダーの設定
temperature = st.slider('平均気温', min_value=-10.0, max_value=50.0, value=7.0, step=0.1)
rainfall = st.slider('年間降水量', min_value=0.0, max_value=5000.0, value=1000.0, step=0.1)
humidity = st.slider('平均湿度', min_value=0.0, max_value=100.0, value=76.0, step=0.1)
area = st.slider('作付面積(倍)', min_value=0.0, max_value=3.0, value=1.0, step=0.1)


# 予測の実行
input_data = {'temperature': temperature, 'rainfall': rainfall, 'humidity': humidity}
input_data = pd.DataFrame(input_data, index=[0])
prediction = predict_model(lr, data=input_data)
yield_prediction = prediction.Label[0]

# 結果の表示
st.write('推定作物収量:', round(yield_prediction*area,2), 'トン/ヘクタール')
