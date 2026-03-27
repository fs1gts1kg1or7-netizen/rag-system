import os
from dotenv import load_dotenv

# LlamaIndex関連
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# .envを読み込む
load_dotenv()

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

print("読み込んだドキュメント数:", len(documents))

# ==============================
# ③ チャンク分割
# ==============================

parser = SimpleNodeParser.from_defaults(
    chunk_size=300,
    chunk_overlap=50
)

nodes = parser.get_nodes_from_documents(documents)

print("作成されたチャンク数:", len(nodes))

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
# ⑥ 質問
# ==============================

question = input("\n質問を入力してください: ")

# ==============================
# ⑧ 検索
# ==============================

response = query_engine.query(question)

# ==============================
# ⑦ 回答表示
# ==============================

print("\n===== 質問 =====")
print(question)


# ==============================
# ⑨ 「答えられない」判定（重要）
# ==============================

# スコアが低い場合は回答しない
threshold = 0.85

valid_nodes = [
    node for node in response.source_nodes
    if node.score >= threshold
]

if len(valid_nodes) == 0:
    print("\n===== 回答 =====")
    print("関連性の高い情報が見つからなかったため、回答できませんでした。")

elif len(valid_nodes) < 2:
    print("\n===== 回答 =====")
    print("十分な根拠となる情報が揃わなかったため、回答を控えました。")

else:
    print("\n===== 回答 =====")
    print(response)

# ==============================
# ⑩ 検索結果表示
# ==============================

print("\n===== 参考情報（検索されたチャンク） =====")
print("取得チャンク数:", len(response.source_nodes))

for i, node in enumerate(response.source_nodes):

    print(f"\n--- 参考 {i+1} ---")

    # スコア
    print("類似度スコア:", round(node.score, 3))

    # 出典
    source = node.node.metadata.get("file_name", "不明")
    print("出典ファイル:", source)

    # 本文
    print("関連文章:")
    print(node.node.text[:300])