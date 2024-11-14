import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

from huggingface_hub import HfApi


# Hugging Face API로 파일 다운로드
api = HfApi()
file_info = api.dataset_download_files("HCIlab-SMWU/Pet_Food_Dataset", use_auth_token="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")

# 파일 경로 설정 및 데이터 불러오기
file_path = file_info["files"]["feeddata_unique2.xlsx"]["local_path"]
df = pd.read_excel(file_path)

# 필터링 함수
def filter_feed(age_input, allergies_input, health_concerns_input, pet_type_input, sort_option):
    data = df.copy()
    error_message = ""  # 에러 메시지 초기화

    # 반려동물 종류 필터링
    if pet_type_input:
        data = data[data['종'] == pet_type_input]

    # 연령 필터링
    if age_input == "1살 미만":
        age_filter = ["퍼피", "키튼", "전연령"]
    elif age_input == "1살 이상, 7살 이하":
        age_filter = ["어덜트", "전연령"]
    else:
        age_filter = ["시니어", "전연령"]

    data = data[data['급여대상'].str.contains('|'.join(age_filter), na=False)]

    # 알러지 필터링 (선택한 알러지가 포함되지 않는 데이터)
    if allergies_input:
        allergy_pattern = '|'.join(allergies_input)
        data = data[~data['주원료'].str.contains(allergy_pattern, na=False)]

    # 주원료가 빈 값인 행 제외
    data = data[data['주원료'].notna() & (data['주원료'] != "")]

    # 건강 고민 필터링
    health_mapping = {
        "치아/구강": ["치석제거", "구강관리"],
        "뼈/관절": ["관절강화"],
        "피부/모질": ["피모관리"],
        "알러지": ["저알러지"],
        "비만": ["다이어트/중성화", "체중유지"],
        "비뇨기": ["유리너리(비뇨계)", "결석예방", "신장/요로", "음수량증진"],
        "눈": ["눈건강"],
        "소화기": ["소화개선"],
        "행동": ["분리불안해소", "스트레스완화"],
        "심장": ["심장건강"],
        "호흡기": ["호흡기관리"],
        "노화": ["항산화"],
        "헤어볼": ["헤어볼"]
    }

    health_patterns = []

    # 건강 고민이 3개를 초과할 경우
    if len(health_concerns_input) > 3:
        error_message = "건강 고민은 최대 3개까지 선택 가능합니다."
        return pd.DataFrame(), error_message  # 빈 데이터프레임과 에러 메시지 반환

    for concern in health_concerns_input:
        health_patterns.append(health_mapping.get(concern, []))

    # 모든 건강 고민 조건을 만족하는 데이터 필터링
    filtered_data = data.copy()

    for patterns in health_patterns:
        if patterns:  # 빈 패턴이 아닐 경우
            filtered_data = filtered_data[filtered_data['기능'].str.contains('|'.join(patterns), na=False)]

    # 결과에 종, 급여대상, 기능, 주원료 추가
    results = filtered_data[['Cleaned_Product_Name', '종', '급여대상', '기능', '주원료']]

    # KMeans를 적용하여 클러스터링
    if len(results) > 10:
        # 원-핫 인코딩 수행
        one_hot_ingredients = pd.get_dummies(results['주원료'])
        one_hot_features = pd.get_dummies(results['기능'])
        
        # 원-핫 인코딩된 데이터와 기존 데이터를 결합
        features = pd.concat([one_hot_ingredients, one_hot_features], axis=1)
        
        # KMeans 모델 생성 및 학습
        kmeans = KMeans(n_clusters=10, random_state=0)
        kmeans.fit(features)
        
        # 각 클러스터의 중심점에 대한 거리 계산
        distances = kmeans.transform(features)  # features와 각 중심점 간의 거리 행렬
        
        # 각 데이터 포인트가 속한 클러스터와 그 중심점과의 거리 추가
        results['cluster'] = kmeans.labels_
        results['distance_to_centroid'] = [distances[i][label] for i, label in enumerate(kmeans.labels_)]
        
        # 각 클러스터에서 중심점에 가장 가까운 데이터만 선택
        closest_to_centroid = results.loc[results.groupby('cluster')['distance_to_centroid'].idxmin()]
        
        # 정렬 옵션에 따라 정렬 수행
        if sort_option == "가나다순":
            closest_to_centroid = closest_to_centroid.sort_values(by=['Cleaned_Product_Name']).reset_index(drop=True)

        results = closest_to_centroid[['Cleaned_Product_Name', '종', '급여대상', '기능', '주원료']]
        
    # 에러 메시지 처리
    if results.empty:
        error_message = "알러지 음식이 포함되지 않으면서 선택한 건강 고민을 포함한 사료가 없습니다."

    # 최종 결과 반환
    return results, error_message  # 결과와 에러 메시지를 함께 반환

# Gradio 인터페이스 설정
with gr.Blocks() as demo:
    gr.Markdown("# 반려동물 사료 추천")
    
    age_input = gr.Dropdown(["1살 미만", "1살 이상, 7살 이하", "7살 이상"], label="연령대")
    allergies_input = gr.CheckboxGroup(
        ["소", "돼지", "닭", "오리", "양", "칠면조", "생선/해산물", "사슴", "연어", "치즈/유지방", "참치", "밀", "쌀"],
        label="주원료 알러지"
    )
    health_concerns_input = gr.CheckboxGroup(
        ["치아/구강", "뼈/관절", "피부/모질", "알러지", "비만", "비뇨기", "눈", "소화기", "행동", "심장", "호흡기", "노화", "헤어볼"],
        label="건강 고민 (최대 3개 선택 가능)"
    )
    pet_type_input = gr.Dropdown(["강아지", "고양이"], label="반려동물 종류")
    sort_option = gr.Radio(["추천순", "가나다순"], label="정렬 방식", value="추천순")  # 추천순과 가나다순 선택
    # 추천 사료 보기 버튼 설정
    submit_button = gr.Button("추천 사료 보기")
    output = gr.Dataframe()
    error_output = gr.Textbox(label="에러 메시지", interactive=False)
    
    # 정렬 방식이 변경될 때마다 자동으로 필터링 함수 호출
    sort_option.change(
        fn=filter_feed, 
        inputs=[age_input, allergies_input, health_concerns_input, pet_type_input, sort_option], 
        outputs=[output, error_output]
    )
    
    
    submit_button.click(
        fn=filter_feed, 
        inputs=[age_input, allergies_input, health_concerns_input, pet_type_input, sort_option], 
        outputs=[output, error_output]
    )

    def check_health_concerns(health_concerns_input):
        if len(health_concerns_input) > 3:
            return "건강 고민은 최대 3개까지 선택 가능합니다."
        return ""

    health_concerns_input.change(fn=check_health_concerns, inputs=health_concerns_input, outputs=error_output)

# 애플리케이션 실행
demo.launch(share=True)
