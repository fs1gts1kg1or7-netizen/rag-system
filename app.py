import streamlit as st

import os
from dotenv import load_dotenv

# LlamaIndex関連
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# .envを読み込む
load_dotenv()

# タイトル
st.title("🧠 健康RAGアプリ(運動&食事&睡眠)")

# 説明
st.write("質問を入力すると、健康に関する情報をもとに回答します。")

# 質問入力
question = st.text_input("質問を入力しEnterを押してください🔍")

if question:
        # ==============================
        # ① LLM設定・embedding設定
        # ==============================

        llm = OpenAI(model="gpt-4o-mini")

        embedding_model = OpenAIEmbedding(
            model = "text-embedding-3-small"
        )


        # ==============================
        # ② データ読み込み
        # ==============================

        documents = SimpleDirectoryReader(
            input_dir = "data",
            required_exts = [".txt"],  # .txtファイルのみ
        ).load_data()

        st.write("読み込んだドキュメント数:", len(documents))

        # ==============================
        # ③ チャンク分割
        # ==============================

        parser = SimpleNodeParser.from_defaults(
            chunk_size=300,
            chunk_overlap=50
        )

        nodes = parser.get_nodes_from_documents(documents)

        st.write("作成されたチャンク数:", len(nodes))

        # ==============================
        # ④ ベクトルインデックス作成
        # ==============================

        index = VectorStoreIndex(
            nodes,
            embedding_model = embedding_model
            )

        # ==============================
        # ⑤ クエリエンジン作成
        # ==============================

        query_engine = index.as_query_engine(
            similarity_top_k=5,
            llm=llm
        )


        # ==============================
        # ⑧ 検索
        # ==============================

        response = query_engine.query(question)
        
        st.divider()  # 区切り線

        # ==============================
        # ⑦ 回答表示
        # ==============================

        # ==============================
        # ⑨ 「答えられない」判定
        # ==============================

        # スコアが低い場合は回答しない
        threshold = 0.85

        valid_nodes = [
            node for node in response.source_nodes
            if node.score >= threshold
        ]

        if len(valid_nodes) == 0:
            st.subheader("\n===== 🎓回答🎓 =====")
            st.error("関連性の高い情報が見つからなかったため、回答できませんでした。")

        elif len(valid_nodes) < 2:
            st.subheader("\n===== 🎓回答🎓 =====")
            st.warning("十分な根拠となる情報が揃わなかったため、回答を控えました。")

        else:
            st.subheader("\n===== 🎓回答🎓 =====")
            st.success(response.response)

        # ==============================
        # ⑩ 検索結果表示
        # ==============================

        st.subheader("\n===== 💡参考情報💡（検索されたチャンク） =====")

        for i, node in enumerate(response.source_nodes):
            
            with st.expander(f"\n--- 参考 {i+1} ---"):

                # スコア
                st.write("類似度スコア:", round(node.score, 3))

                # 出典
                source = node.node.metadata.get("file_name", "不明")
                st.write("出典ファイル:", source)

                # 改行だけ消す
                text = node.node.text.replace("\n", " ")

                st.write("関連文章:")
                st.write(text[:200]) 