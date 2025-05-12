import streamlit as st
import pandas as pd
import base64
import textwrap  # 상단 import 필요
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import io
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import joblib
import plotly.graph_objs as go
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler



# 한글 폰트 설정 (맑은 고딕)
matplotlib.rc("font", family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Maple Character Search", layout="wide")

# 이미지 base64 인코딩 함수
def get_base64_from_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 이미지 불러오기
bg_image = get_base64_from_file("bk.png")
logo_image = get_base64_from_file("logo2.png")

@st.cache_resource
def load_model():
    return joblib.load("rsf_model.pkl")

@st.cache_data
def load_data():
    X = pd.read_pickle("X_for_streamlit.pkl")
    df = pd.read_csv("streamlit_user_data_with_image.csv", low_memory=False)
    df2 = pd.read_csv("time_varying_with_all_changes.csv", low_memory=False)
    df2["log_exp_change"] = np.log1p(np.abs(df2["exp_change"]))
    importance_df = pd.read_csv("rsf_feature_importance_lifelines.csv", low_memory=False)
    importance_df["feature"] = importance_df["feature"].str.strip()
    return X, df, df2,importance_df

rsf = load_model()
X, df, df2,importance_df = load_data()


# 세션 상태 초기화
if "search_done" not in st.session_state:
    st.session_state["search_done"] = False
if "char_name" not in st.session_state:
    st.session_state["char_name"] = ""

# 💄 공통 CSS 스타일
st.markdown(f"""
<style>
html, body, .stApp {{
    background-color: #f0e3c4;
    overflow: hidden !important;
    margin: 0; !important;
    padding: 0; !important;
    height: 100vh !important;
    overscroll-behavior: none;
   
}}
header {{ background-color: #e5d4b1 !important; }}
#floating-logo {{
    position: fixed;
    bottom: 24px;
    right: 50px;
    z-index: 9999;
    height: 130px;
    max-width: 260px;
}}
.main {{ padding-top: 0rem !important; }}
.stApp {{
    border: 40px solid #e4d4b4;
    border-radius: 10px;
    box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.15), inset 0 0 12px rgba(180, 160, 120, 0.6);
    padding-bottom: 40px;
}}
.block-container {{
    max-width: 1400px !important;
    margin: 0 auto;
    padding: 0 !important;
    background-color: #f0e3c4 !important;
    display: flex;
    justify-content: center;
}}

.image-wrapper {{
    width: 100%;
    overflow: hidden;
    height: calc(96.5vh - 90px); /* 40px * 2 테두리 감안 */
    
}}
 .footer {{
        position: fixed;
        bottom: 10px;
        right: 45px;
        font-size: 12.5px;
        color: #3a2e1b;
        line-height: 1.6;
        text-align: right;
        font-family: 'sans-serif';
    }}
.image-wrapper img {{
    
    width: 100%;
    height: 90vh
    object-fit: cover;  /* 비율 유지하면서 꽉 채움 */
    display: block;
}}

.logo {{
    position: fixed;
    top: 45%;
    left: 49.5%;
    transform: translate(-50%, -80%);
    width: 500px;  /* ✅ 크기 키움 (기존: 250px) */
    z-index: 2;
}}

.logo-top-left {{
    position: fixed;
    top: 30px;
    left: 50px;
    width: 180px;
    z-index: 1;
}}
div[data-testid="stTextInput"] {{
    position: fixed;
    top: 52%;
    left: 50.5%;  /* ✅ 50% → 51%로 오른쪽 이동 */
    transform: translate(-50%, -80%);
    width: 380px;  /* ✅ 280px → 320px (더 넓게) */
    z-index: 3;
    background-color: 	#f4dbc3;
    border: 4px solid #a17035;  /* ✅ 테두리 두께 up */
    border-radius: 16px;
    box-shadow: 0 0 8px rgba(161, 112, 53, 0.25);
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}}

div[data-testid="stTextInput"] input {{
    background-color: #f5e8d4;
    color: black;
    border: none;
    outline: none;
    font-size: 16px;
    width: 100%;
    height: 42px;
    padding: 0 12px;
    text-align: center;
    font-weight: 500;
}}


div[data-testid="stButton"] {{ display: none !important; }}
</style>
<div class="footer">
        © 2025 데이터와 함께 춤을. All rights reserved.<br>
        This site is not affiliated with NEXON Korea.<br>
        Data sourced from NEXON OpenAPI for non-commercial analysis.
    </div>
""", unsafe_allow_html=True)

# 로그인 화면
if not st.session_state["search_done"]:
    st.markdown(f"""
        <div class="image-wrapper">
            <img src="data:image/png;base64,{bg_image}" />
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<img src="data:image/png;base64,{logo_image}" class="logo"/>', unsafe_allow_html=True)
  

    char_input = st.text_input("", placeholder="캐릭터명을 입력해주세요", key="real_input")
    if char_input:
        st.session_state["char_name"] = char_input.strip()
        st.session_state["search_done"] = True
        st.rerun()


# 캐릭터 정보 탭
else:
    st.markdown(f'<img src="data:image/png;base64,{logo_image}" class="logo-top-left"/>', unsafe_allow_html=True)

    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Gugi&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        #back-button {
            position: fixed;
            top: 70px;
            right: 53px;
            background-color: transparent;
            color: #a85a32;
            border: 2px solid #a85a32;
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: bold;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            text-align: center;
        }
        #back-button:hover {
            background-color: #d07040;
            color: white;
        }
 
        .stTabs [data-baseweb="tab"] {
            font-family: 'Gugi', sans-serif !important;
            font-size: 30px !important;
            font-weight: 800 !important;
            color: #8B0000 !important;  /* 기본 탭 텍스트 색상 */
            padding: 10px 10px !important;
            border-radius: 8px 8px 0 0;
            margin-right: 10px;
            margin-left: 210px;   /* 로고 오른쪽 여백 확보 */
            margin-top: 10px;    /* 상단 여백 줄이기 (필요 시) */
            z-index: 5;
        }
        .stTabs [aria-selected="true"] {
            color: #d23f00 !important;  /* 선택된 탭 색상 */
            background-color: #f4dbc3 !important;
            border-bottom: 3px solid #d23f00 !important;
        }
        </style>
        <a id="back-button" href="?reset=true">← 돌아가기</a>
    """, unsafe_allow_html=True)
   
    tab1, tab2, tab3= st.tabs(["📋 요약 리포트", "🔎 중요 변수 해석", "🎯이탈률 시뮬레이터"])
    # 진단 설명 함수

    def generate_diagnosis(risk_level, info, df):
        negatives, positives = [], []
    
        # 위험 요인
        if "저성장" in info.get("segment", ""):
            negatives.append("어센틱 성장이 낮고")
        if info.get("guild_flag", 1) == 0:
            negatives.append("길드에 가입되어 있지 않으며")
        if info.get("liberation_flag", 1) == 0:
            negatives.append("해방 퀘스트가 완료되지 않았고")
        if info.get("popularity", 0) < df["popularity"].mean():
            negatives.append("인기도가 평균보다 낮습니다")
    
        # 보호 요인
        if info.get("dojang_best_floor", 0) >= 60:
            positives.append("무릉 기록은 안정적이고")
        if info.get("guild_flag", 1) == 1:
            positives.append("길드에 소속되어 있으며")
        if info.get("liberation_flag", 1) == 1:
            positives.append("해방 퀘스트를 완료한 상태입니다")
    
        negative_text = " ".join(negatives)
        positive_text = " ".join(positives)
    
        # 리스크 수준별 문장 구성
        if risk_level == "낮음":
            if negatives and positives:
                return f"{negative_text} 하지만 {positive_text} 전반적으로 이탈 위험은 낮은 편으로 예측되었습니다."
            elif positives:
                return f"{positive_text} 전반적으로 이탈 위험은 낮은 편으로 예측되었습니다."
            else:
                return "전반적으로 안정적인 활동 요인이 많아 이탈 위험은 낮은 것으로 예측되었습니다."
    
        elif risk_level == "중간":
            if positives:
                return f"{negative_text} 반면, {positive_text} 해당 요소들이 이탈 위험을 일정 부분 완화한 것으로 보입니다."
            else:
                return f"{negative_text} 이로 인해 이탈 위험이 다소 높아진 것으로 분석되었습니다."
    
        elif risk_level == "높음":
            if positives:
                return f"{negative_text} 일부 긍정적인 요소도 있었으나, {positive_text} 전반적으로 이탈 위험이 높은 것으로 예측되었습니다."
            else:
                return f"{negative_text} 이로 인해 이탈 위험이 높은 것으로 예측되었습니다."
    
        return "이탈 위험 분석 결과를 해석할 수 없습니다."


  #  def get_top_risk_reasons(importance_df, X_all, user_row, top_n=3):
    #    risk_reasons = []
     #  for feature in importance_df["feature"]:
      #      user_value = user_row[feature]
       #     avg_value = X_all[feature].mean()
        #    if pd.api.types.is_numeric_dtype(X_all[feature]):
         #       if user_value < avg_value:
          #          risk_reasons.append(f"🔻 {feature} 값이 평균보다 낮음")
          #  if len(risk_reasons) >= top_n:
           #     break
   #     return risk_reasons

    
    # 고급 생존 곡선 시각화 함수
    def plot_survival_curve(surv_func, weeks=10):
        x = surv_func.x[:weeks]
        y_survival = surv_func.y[:weeks]
        y_churn = [1 - v for v in y_survival]  # 🔁 이탈 확률
    
        # 세그먼트 평균 이탈확률 계산
        X_with_ocid = X.reset_index(drop=True)
        df_aligned = pd.merge(X_with_ocid[["ocid"]], df, on="ocid", how="left")
        X_clean = X_with_ocid.drop(columns=["ocid"])
    
        user_segment = info["level_segment"]
        df_segment = df[df["level_segment"] == user_segment]
        ocids_segment = df_segment["ocid"]
        X_segment = X[X["ocid"].isin(ocids_segment)].drop(columns=["ocid"])
        func_segment = rsf.predict_survival_function(X_segment)
        segment_churn_avg = np.mean([1 - f.y[:weeks] for f in func_segment], axis=0)
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=x,
            y=y_churn,
            mode="lines+markers+text",
            name="해당 캐릭터",
            line=dict(color="#cc3300", width=4),
            marker=dict(size=9, symbol="circle", color="#cc3300"),
            text=[f"{v:.1%}" for v in y_churn],
            textposition="top center",
            textfont=dict(size=12, color="#cc3300")
        ))
    
        fig.add_trace(go.Scatter(
            x=x,
            y=segment_churn_avg,
            mode="lines",
            name=f"{user_segment} 평균",
            line=dict(color="#3366cc", width=3, dash="dot")
        ))
    
        fig.update_layout(
            title=dict(
                text=f"📉 이탈 곡선: 캐릭터 vs {user_segment} 평균",
                font=dict(size=22, family="Arial", color="#3c2f1c"),
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="주차",
                tickmode="linear",
                dtick=1,
                gridcolor="#e4d3b4"
            ),
            yaxis=dict(
                title="이탈 확률",
                range=[0, max(max(y_churn), max(segment_churn_avg)) + 0.03],
                gridcolor="#e4d3b4"
            ),
            plot_bgcolor="#fff9ec",
            paper_bgcolor="#fff9ec",
            font=dict(family="Segoe UI", size=14, color="#3c2f1c"),
            margin=dict(t=60, b=40, l=40, r=40),
            showlegend=True,
            height=600
        )
    
        return fig


    
    with tab1:
        
        char = df[df["character_name"].fillna("").str.strip() == st.session_state["char_name"]]
        df2_char = df2[df2["character_name"].fillna("").str.strip() == st.session_state["char_name"]]
    
        if char.empty or df2_char.empty:
            st.error("❌ 캐릭터가 존재하지 않습니다.")
        else:
            info = char.iloc[0]
            selected_ocid = info["ocid"]
            X_row = X[X["ocid"] == selected_ocid].drop(columns=["ocid"])
            # ✅ 여기에 반드시 선언되어 있어야 함
            surv_func = rsf.predict_survival_function(X_row)[0]
            diagnosis = generate_diagnosis(info["risk_level"], info, df)
            img = info.get("character_image", "")
            name = info.get("character_name", "이름 없음")
            level = info.get("character_level", "??")
            job = info.get("character_class", "??")
            world = info.get("world_name", "??")
            guild = info.get("character_guild_name", "없음")
            pop = info.get("popularity", "정보 없음")
            union = info.get("union_level", "정보 없음")
            floor = info.get("dojang_best_floor", "정보 없음")
            auth_sum = info.get("authentic_sum", "정보 없음")
            arcane_sum = info.get("arcane_sum", "정보 없음")
            segment = info.get("segment", "미분류")
            auth_line = f"{auth_sum} ({segment})"
            
            # 🔓 해방 여부 처리
            liberation = info.get("liberation_flag", None)
            if liberation == 1:
                liberation_status = "완료"
            elif liberation == 0:
                liberation_status = "미완료"
            else:
                liberation_status = "정보 없음"
    
            # 예측 결과 추가
            survival_prob = info.get("survival_prob_9w", 0)
            risk_score = info.get("risk_score", 0)
            risk_level = info.get("risk_level", "정보 없음")
            median_time = info.get("median_time", "정보 없음")
    
    
            # ✅ 레이아웃: 왼쪽(캐릭터 카드)
            col_card, col_graphs = st.columns([0.75, 1.25])
    
            with col_card:
                html_code = f"""
                <style>
                .character-card {{
                        background: linear-gradient(to bottom, #fffaf0, #f6e6c9);
                        border-radius: 25px;
                        padding: 40px 36px 50px 36px;
                        width: 460px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                        font-family: 'Segoe UI', sans-serif;
                        color: #3c2f1c;
                        margin-left: 80px;
                        text-align: center;
                    }}
                    .character-card img {{
                        width: 220px;
                        height: 220px;
                        border-radius: 50%;
                        border: 5px solid #cfa86e;
                        margin: 0 auto 20px;
                    }}
                    .char-name {{ font-size: 32px; font-weight: 900; margin-bottom: 12px; }}
                    .char-class, .char-guild {{ font-size: 18px; color: #7e5d3c; margin-bottom: 12px; }}
                    .section-title {{
                        font-size: 17px;
                        font-weight: 700;
                        color: #3c2f1c;
                        margin: 20px 0 8px 0;
                        text-align: left;
                        border-bottom: 1px dashed #d6b76e;
                        padding-bottom: 4px;
                    }}
                    .info-block {{
                        display: flex; flex-wrap: wrap; justify-content: space-around;
                        gap: 12px 10px;
                        font-size: 16px;
                        margin-bottom: 16px;
                    }}
                    .info-item .info-label {{ font-weight: bold; color: #5a402a; }}
                    .danger-zone {{ background-color: #fff4e5; border: 1px solid #f0d2a2; border-radius: 10px; padding: 12px; }}
                    .diagnosis {{
                        margin-top: 20px;
                        font-size: 15px;
                        line-height: 1.6;
                        background-color: #fff2d5;
                        padding: 14px;
                        border-radius: 12px;
                        border: 1px solid #e0c287;
                    }}
                    </style>
                    
                    <div class="character-card">
                        <img src="{info['character_image']}" />
                        <div class="char-name">{info['character_name']}</div>
                        <div class="char-class">Lv.{info['character_level']} · {info['character_class']}</div>
                        <div class="char-guild">🌍 {info['world_name']} | 🏠 {info['character_guild_name']}</div>
                    
                        <div class="section">
                            <div class="section-title">📌 기본 상태</div>
                            <div class="info-block">
                                <div class="info-item"><div class="info-label">💖 인기도</div><div>{info['popularity']}</div></div>
                                <div class="info-item"><div class="info-label">💰 유니온</div><div>{info['union_level']}</div></div>
                                <div class="info-item"><div class="info-label">⛩ 무릉</div><div>{info['dojang_best_floor']}층</div></div>
                                <div class="info-item"><div class="info-label">🔓 해방</div><div>{liberation_status}</div></div>
                            </div>
                        </div>
                    
                        <div class="section">
                            <div class="section-title">🔠 심볼 상태</div>
                            <div class="info-block">
                                <div class="info-item"><div class="info-label">🧪 어센틱</div><div>{auth_line}</div></div>
                                <div class="info-item"><div class="info-label">🔹 아케인</div><div>{info['arcane_sum']}</div></div>
                            </div>
                        </div>
                    
                        <div class="section">
                            <div class="section-title">🔎 위험 분석</div>
                            <div class="info-block danger-zone">
                                <div class="info-item"><div class="info-label">📈 이탈확률</div><div>{(1 - info['survival_prob_9w'])*100:.1f}%</div></div>
                                <div class="info-item"><div class="info-label">⚠️ 위험점수</div><div>{info['risk_score']:.3f}</div></div>
                                <div class="info-item"><div class="info-label">🔥 위험등급</div><div>{info['risk_level']}</div></div>
                            </div>
                        </div>
                    
                        <div class="diagnosis">📄 <strong>진단 결과:</strong> {diagnosis}</div>
                    </div>
                    """
                html(html_code, height=950)
                
           # X_all = X.drop(columns=["ocid"])
            #X_user = X[X["ocid"] == info["ocid"]].drop(columns=["ocid"]).iloc[0]
            #risk_reasons = get_top_risk_reasons(importance_df, X_all, X_user)
                
            with col_graphs:
                st.subheader("📉 이탈 곡선")
                fig = plot_survival_curve(surv_func)
                st.plotly_chart(fig, use_container_width=True)
                # 🧮 1. 이탈 시점 예측값 가져오기
                try:
                    expected_week = int(round(float(info["threshold75_time"])))  # threshold75_time 기준
                except:
                    expected_week = None
                
                # 🔢 2. 이탈 확률 계산
                churn_prob = 1 - survival_prob
                
                # 🎯 3. 레벨대 기준 위험 순위 계산
                segment_users = df[df["level_segment"] == info["level_segment"]]
                rank_percentile = ((1 - segment_users["survival_prob_9w"]) > churn_prob).mean() * 100
                rank_text = f"상위 {rank_percentile:.1f}%" if rank_percentile >= 50 else f"하위 {100 - rank_percentile:.1f}%"
                
                # 💬 4. 이탈선 문장 분기
                if expected_week is None:
                    reach_text = "이 캐릭터의 이탈선 도달 시점은 예측할 수 없습니다."
                elif expected_week >= 10:
                    reach_text = "이 캐릭터는 <u>10주차 이내</u>에 이탈선(25%)에 도달하지 않을 것으로 예측됩니다."
                else:
                    reach_text = f"이 캐릭터는 <u>{expected_week}주차</u>에 누적 이탈확률이 25%를 넘을 것으로 예측됩니다."
                
                # 📦 5. 메시지 박스 출력
                message = f"""
                <div style="
                    background-color: #fff5e1;
                    border-left: 6px solid #ff9d00;
                    padding: 25px 28px;                  /* 여백 확대 */
                    margin-top: 30px;
                    border-radius: 12px;
                    font-size: 25px;                   /* ✅ 글씨 키움 */
                    font-weight: 500;
                    color: #3b2a16;
                    line-height: 1.8;                  /* ✅ 줄간 간격 넉넉하게 */
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.05); /* 살짝 입체감 */
                    max-width: 880px;
                ">
                    <strong>⏳ 이탈선 도달 여부:</strong> 이 캐릭터는 <u>{expected_week}주차 이내</u>에 이탈선(25%)에 도달하지 않을 것으로 예측됩니다.<br><br>
                    <strong>📊 이탈 위험 순위:</strong> 동일 레벨대 유저 중 <u>{rank_text} 위험군</u>에 해당합니다.
                </div>
                """
                
                tooltip = """
                <div style="
                    background-color: #fff5e1;
                    border-left: 6px solid #ff9d00;
                    padding: 25px 28px;                  /* 여백 확대 */
                    margin-top: 30px;
                    border-radius: 12px;
                    font-size: 25px;                   /* ✅ 글씨 키움 */
                    font-weight: 500;
                    color: #3b2a16;
                    line-height: 1.8;                  /* ✅ 줄간 간격 넉넉하게 */
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.05); /* 살짝 입체감 */
                    max-width: 880px;
                ">
                    ℹ️ <strong>해석 가이드:</strong> 위 그래프는 <u>생존곡선 S(t)</u>을 기반으로<br>
                    <strong>1 - S(t)</strong> 값으로 계산된 <u>누적 이탈 확률</u>을 보여줍니다.<br>
                    예를 들어 5주차가 3%이면, 5주 이내 누적 이탈률은 3%입니다.
                </div>
                """


                st.markdown(tooltip, unsafe_allow_html=True)
                st.markdown(message, unsafe_allow_html=True)




    with tab2:
        st.subheader("📌 중요 변수 해석")
    
        char_name = st.session_state["char_name"]
        user_row = df[df["character_name"].fillna("").str.strip() == char_name]
    
        if user_row.empty:
            st.warning("해당 캐릭터 정보를 찾을 수 없습니다.")
        else:
            # 1. 중요도 상위 7개
            top7_df = importance_df.sort_values("importance_mean", ascending=False).head(7)
            top7_features = top7_df["feature"].tolist()
            
            # 2. 피라미드 차트
            pyramid_fig = go.Figure(go.Bar(
                x=top7_df["importance_mean"],
                y=top7_df["feature"],
                orientation='h',
                marker=dict(
                    color=['#f9c74f', '#f9844a', '#f8961e', '#f3722c', '#f94144', '#f48fb1', '#ce93d8'],
                    line=dict(color='black', width=1)
                ),
                text=[f"{v:.3f}" for v in top7_df["importance_mean"]],
                textposition='auto'
            ))
            pyramid_fig.update_layout(
                title="🔺 중요 변수 피라미드(top7)",
                yaxis=dict(autorange="reversed"),
                height=600,
                plot_bgcolor="#fffaf0",
                paper_bgcolor="#fffaf0",
                margin=dict(l=50, r=50, t=60, b=60)
            )
    
            # 3. Z-score 레이더 차트
            radar_features = ["character_level", "authentic_sum", "character_age_days", "popularity_zscore"]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[radar_features])
            df_z = pd.DataFrame(scaled, columns=radar_features)
            user_z = df_z.loc[user_row.index].values.flatten()
            avg_z = df_z.mean().values.flatten()
    
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=user_z,
                theta=radar_features,
                fill='toself',
                name='해당 캐릭터',
                line=dict(color='rgba(255,99,71,0.8)', width=3),
                text=[f"{f}: {v:.1f}" for f, v in zip(radar_features, user_row[radar_features].values.flatten())],
                hoverinfo="text"
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=avg_z,
                theta=radar_features,
                fill='toself',
                name='전체 평균',
                line=dict(color='rgba(100,149,237,0.6)', width=3)
            ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                showlegend=True,
                title="🕸 Z-score 기반 캐릭터 vs 평균 비교",
                height=600,
                margin=dict(l=50, r=50, t=60, b=60),
                plot_bgcolor="#fffaf0",
                paper_bgcolor="#fffaf0"
            )
    
            # 🔹 상단 시각화 배치
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(pyramid_fig, use_container_width=True)
            with col2:
                st.plotly_chart(radar_fig, use_container_width=True)
    
            # 🔹 하단 해석/표 병렬 배치
            col3, col4 = st.columns(2)
    
            with col3:
                st.markdown("### 🧾 상태 진단")
            
                # ✅ 상태 카드 내부에 여백 추가
                def display_binary_status(title, value, true_label="해당", false_label="비해당"):
                    icon = "✅" if value else "❌"
                    label = true_label if value else false_label
                    color = "#ccffcc" if value else "#ffe6e6"
                    st.markdown(f"""
                        <div style='
                            background-color: {color};
                            padding: 10px 14px;
                            border-radius: 10px;
                            font-size: 16px;
                            font-weight: bold;
                            margin: 7px 0;
                            min-height: 44px;  /* ✅ 높이 확보 */
                            display: flex;
                            align-items: center;
                        '>
                            {icon} <strong>{title}:</strong> {label}
                        </div>
                    """, unsafe_allow_html=True)
            
                is_safe_combo = user_row["set_combo_group"].values[0] != "세트효과 부족/혼합"
                display_binary_status("🧩 세트효과 부족/혼합", is_safe_combo,
                                      true_label="정상 구성", false_label="부족/혼합 상태")
                display_binary_status("🔓 해방 퀘스트", int(user_row["liberation_flag"]),
                                      true_label="완료", false_label="미완료")
                display_binary_status("👥 길드 소속", int(user_row["guild_flag"]),
                                      true_label="소속", false_label="미소속")
            
                # 📦 아래 빈 블록 추가로 시각 균형 맞추기 (옵션)
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    
            with col4:
                st.markdown("### 📊 실제 수치 (해당 캐릭터)")
            
                metrics_df = pd.DataFrame({
                    "변수명": radar_features,
                    "해당 캐릭터 값": user_row[radar_features].values.flatten(),
                    "전체 평균": df[radar_features].mean().values
                })
                metrics_df["차이"] = metrics_df["해당 캐릭터 값"] - metrics_df["전체 평균"]
            
                def highlight_diff(val):
                    if isinstance(val, (float, int)):
                        return 'color: red' if val > 0 else 'color: blue' if val < 0 else ''
                    return ''
            
                styled_df = metrics_df.style.format({
                    "해당 캐릭터 값": "{:.1f}",
                    "전체 평균": "{:.1f}",
                    "차이": "{:+.1f}"
                }).applymap(highlight_diff, subset=["차이"]
                ).set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold")]},
                    {"selector": "td", "props": [("font-size", "15px")]}
                ])
                with st.container():
                        st.dataframe(styled_df, use_container_width=True)
                        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)            
            


        
        
        
  
    with tab3:
        st.markdown("""
        <style>
        div[data-testid="stButton"] {
            display: block !important;
        }
        </style>
        """, unsafe_allow_html=True)        
        st.markdown("""
        <div style='font-size: 24px; font-weight: bold; color: #3b2a16; background-color: #f9e5c9;
                    padding: 10px 16px; border-radius: 12px; border-left: 5px solid #e88f2a; margin-bottom: 20px;'>
            🧪 이탈률 시뮬레이션
        </div>
        """, unsafe_allow_html=True)
    
        info = df[df["character_name"] == st.session_state["char_name"]].iloc[0]
        X_user = X[X["ocid"] == info["ocid"]].drop(columns=["ocid"]).copy()
    
        top7_features = importance_df.sort_values("importance_mean", ascending=False)["feature"].head(7).tolist()
        st.write("\n")
    
        if "sim_values" not in st.session_state:
            st.session_state.sim_values = X_user.copy()
    
        col_left, col_right = st.columns([1.2, 1.8])
    
        with col_left:
            st.markdown("#### 🔧 변수 조정")
            X_sim = st.session_state.sim_values.copy()
            updated = {}
    
            for feature in top7_features:
                if feature not in X.columns:
                    st.warning(f"⚠️ `{feature}`는 X에 존재하지 않아 스킵됩니다.")
                    continue
    
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{feature}**")
                with col2:
                    if pd.api.types.is_numeric_dtype(X[feature]) and not set(X[feature].dropna().unique()).issubset({0, 1}):
                        min_val = float(X[feature].quantile(0.05))
                        max_val = float(X[feature].quantile(0.95))
                        val = float(X_sim[feature].values[0])
                        step_val = 1.0 if "level" in feature or "days" in feature or "sum" in feature else 0.1
                        new_val = st.slider("", min_val, max_val, val, step=step_val, key=feature)
                        X_sim[feature] = new_val
                        updated[feature] = (X_user[feature].values[0], new_val)
    
                    elif set(X[feature].dropna().unique()).issubset({0, 1}):
                        val = int(X_sim[feature].values[0])
                        new_val = st.radio("", ["✅ 해당", "❌ 비해당"],
                                           index=0 if val == 1 else 1,
                                           horizontal=True, key=feature)
                        new_val_bin = 1 if "✅" in new_val else 0
                        X_sim[feature] = new_val_bin
                        updated[feature] = (X_user[feature].values[0], new_val_bin)
    
                st.markdown("<hr style='margin: 4px 0; border-color: #eee;'>", unsafe_allow_html=True)
    
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📤 시뮬레이션 적용 및 결과 보기"):
                    st.session_state.sim_values = X_sim.copy()
            with col_btn2:
                if st.button("↩️ 되돌리기 (초기값)"):
                    st.session_state.sim_values = X_user.copy()
    
        with col_right:
            st.markdown("#### 📈 예측 결과 변화")
        
            # ✅ 예측 실행 - 변경된 값과 원래 값 비교
            X_input_new = st.session_state.sim_values.copy()
            X_input_old = X_user.copy()
        
            # ✅ 생존 함수 계산
            surv_func_new = rsf.predict_survival_function(X_input_new)[0]
            surv_func_old = rsf.predict_survival_function(X_input_old)[0]
        
            # ✅ 타겟 주차 기준 예측값 추출
            target_week = 9
            # 안전하게 도메인 범위 내에서 계산
            target_week = min(target_week, surv_func_new.x.max())
        
            idx_new = (np.abs(surv_func_new.x - target_week)).argmin()
            idx_old = (np.abs(surv_func_old.x - target_week)).argmin()
        
            prob_new = surv_func_new.y[idx_new]
            prob_old = surv_func_old.y[idx_old]
        
            # ✅ 생존 확률을 이탈확률로 변환
            churn_new = 1 - prob_new
            churn_old = 1 - prob_old
        
            # ✅ 중위 생존 기간 예측
            try:
                median_new = float(rsf.predict_median(X_input_new)[0])
            except:
                median_new = None
        
            try:
                median_old = float(rsf.predict_median(X_input_old)[0])
            except:
                median_old = None
        
            # ✅ 위험 점수 예측
            risk_score_new = float(rsf.predict(X_input_new)[0])
            risk_score_old = float(rsf.predict(X_input_old)[0])
        
            # ✅ 출력
            st.metric("📉 이탈 확률 (9주차)", f"{churn_new*100:.1f}%", delta=f"{(churn_new - churn_old)*100:+.1f}%")
            st.metric("🧭 위험 점수 (risk_score)", f"{risk_score_new:.3f}", delta=f"{risk_score_new - risk_score_old:+.3f}")
            
            if median_old is not None and median_new is not None:
                st.metric("⏱ 예측 생존 기간 (주)", f"{median_new:.1f}주", delta=f"{median_new - median_old:+.1f}주")
            else:
                st.markdown("⚠️ 중위 생존 기간 예측 불가")
        
            # ✅ 변경된 변수 요약
            st.markdown("#### 🔍 조정한 변수 요약")
            for k, (before, after) in updated.items():
                if before != after:
                    st.markdown(f"- `{k}`: {before} → **{after}**")
        
            # ✅ 디버깅용 출력 (필요 시 주석 해제)
            # st.write("✅ X_user 원본:", X_input_old)
            # st.write("✅ 변경된 값:", X_input_new)
            # st.write("✅ surv_func_new.x:", surv_func_new.x)
            # st.write("✅ surv_func_new.y[:10]:", surv_func_new.y[:10])



