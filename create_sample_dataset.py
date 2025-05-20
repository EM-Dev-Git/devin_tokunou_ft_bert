import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

job_descriptions = [
    "Seeking a data scientist with experience in machine learning, statistical analysis, and data visualization. Must have skills in Python, R, and SQL. Responsibilities include developing predictive models, analyzing large datasets, and communicating insights to stakeholders.",
    "Data scientist position available. The ideal candidate will have a strong background in statistics, machine learning, and programming. Experience with deep learning frameworks and big data technologies is a plus.",
    "Looking for a data scientist to join our analytics team. Must have expertise in data mining, machine learning algorithms, and statistical modeling. PhD in a quantitative field preferred.",
    "Data scientist needed to develop and implement machine learning models. Responsibilities include data preprocessing, feature engineering, model selection, and deployment. Proficiency in Python and data visualization required.",
    "Data scientist role to extract insights from complex datasets. Will build predictive models and develop data-driven solutions. Strong background in statistics and machine learning required.",
    
    "Machine learning engineer position to develop and deploy ML models at scale. Experience with TensorFlow, PyTorch, and cloud computing platforms required. Will work closely with data scientists and software engineers.",
    "Seeking a machine learning engineer to build and maintain ML pipelines. Must have experience with model deployment, MLOps, and performance optimization. Strong programming skills required.",
    "Machine learning engineer needed to implement algorithms and build scalable ML systems. Experience with distributed computing and model serving frameworks is essential.",
    "Looking for a machine learning engineer to develop ML infrastructure and workflows. Will optimize ML models for production and ensure reliable deployment. Experience with containerization and cloud services required.",
    "Machine learning engineer position available. Will be responsible for developing efficient implementations of machine learning algorithms and deploying them in production environments.",
    
    "Software engineer needed to develop and maintain web applications. Proficiency in JavaScript, React, and Node.js required. Will work in an agile team environment developing user-facing features.",
    "Looking for a backend software engineer with experience in Java, Spring, and microservices architecture. Will design and implement RESTful APIs and data storage solutions.",
    "Software engineer position to develop mobile applications. Experience with Swift, Kotlin, and cross-platform frameworks required. Will collaborate with designers and product managers.",
    "Full-stack software engineer needed for our product team. Must have experience with modern web technologies and cloud platforms. Will build scalable and maintainable software systems.",
    "Software engineer role available. Will develop and test code, troubleshoot issues, and implement new features. Strong programming skills and CS fundamentals required.",
    
    "Management consultant position available. Will advise clients on business strategy, operational improvements, and digital transformation. MBA and consulting experience required.",
    "Seeking a technology consultant to help clients leverage IT for business growth. Will assess requirements, recommend solutions, and manage implementation projects.",
    "Consultant needed to provide expertise in supply chain optimization. Will analyze client operations, identify inefficiencies, and recommend improvements.",
    "Financial consultant position to advise clients on investment strategies, risk management, and financial planning. CFA certification and finance background required.",
    "Business consultant role available. Will work with clients to solve complex business problems, develop strategies, and implement solutions for sustainable growth."
]

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]

def create_sample_dataset():
    """
    サンプルデータセットを作成し、CSVファイルに保存する関数
    
    機能:
    1. 求人情報のテキストとそれに対応するラベルからデータフレームを作成
    2. データを訓練用とテスト用に分割（訓練:70%, テスト:30%）
    3. 分割したデータをCSVファイルとして保存
    
    引数:
        なし
        
    戻り値:
        なし（CSVファイルとして保存される）
        
    副作用:
        - ~/bert_finetuning_project/data/train.csv ファイルを作成
        - ~/bert_finetuning_project/data/test.csv ファイルを作成
        - 処理結果をコンソールに出力
    """
    
    df = pd.DataFrame({
        'text': job_descriptions,  # テキスト列
        'label': labels            # ラベル列
    })
    
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    
    data_dir = os.path.join(os.path.expanduser("~"), "bert_finetuning_project", "data")
    
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    print(f"訓練データ: {len(train_df)}件, テストデータ: {len(test_df)}件")
    print(f"データを {data_dir} に保存しました")

if __name__ == "__main__":
    create_sample_dataset()
