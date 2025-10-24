import os
import sys
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import torch.nn as nn

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def evaluate_mlp_on_sentence(test_sentence, mlp):
    # tokenise + embed (same mean_pooling as training)
    encoded = tokenizer([test_sentence], padding=True, truncation=True, return_tensors='pt').to(device)  # device = "cpu" or "cuda"
    with torch.no_grad():
        embed_out = embed_model(**encoded)                   # embed_out[0] is token embeddings
        embedding = mean_pooling(embed_out, encoded['attention_mask'])  # shape (1, D)
        embedding = F.normalize(embedding, p=2, dim=1)       # same normalization as training

    # convert to tensor of shape (1, D) and feed MLP
    with torch.no_grad():
        mlp_input = embedding.float()   # already a tensor; ensure dtype
        pred = mlp(mlp_input).cpu().numpy().item()  # scalar

    # clamp to [0,1] since labels are in this range
    pred = float(np.clip(pred, 0.0, 1.0))
    
    return pred
  

def create_text_block(df: pd.DataFrame, track_id: int) -> str:
    if track_id not in df['id'].values:
        return ""
    df = df[df['id'] == track_id].iloc[0]
    app_name = df['name'] if pd.notna(df['name']) else ""
    subtitle = df['subtitle'] if pd.notna(df['subtitle']) else ""
    description = df['description'] if pd.notna(df['description']) else ""
    update_note = df['update_notes'] if pd.notna(df['update_notes']) else ""
    
    text_block = f"""App: {app_name}
Subtitle: {subtitle}
Description: {description}
Updates: {update_note}"""
    return text_block


class MLPRegressorTorch(nn.Module):
    def __init__(self, in_dim, hidden=(512,128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # for h in hidden:
        #     layers.append(nn.Linear(prev, h))
        #     layers.append(nn.ReLU())
        #     if dropout and dropout > 0.0:
        #         layers.append(nn.Dropout(dropout))
        #     prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)




in_dim = 1024

if in_dim == 1024:
    # Overkill maybe
    embed_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
elif in_dim == 768:
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
    embed_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
else:
    raise ValueError("Unsupported in_dim")


device = torch.device("cpu")
embed_model.to(device)
embed_model.eval()


mlp = MLPRegressorTorch(in_dim=in_dim, hidden=(768,128), dropout=0.15)
mlp.load_state_dict(torch.load("./mlp_regressor_torch.pth", map_location="cpu"))
mlp.eval()


def test_on_trained_data():
    print("Testing on trained data...")
    data = pd.read_csv("./app-unique.csv")
    # labels = pd.read_csv("./app-labels.csv")
    
    track_id_match_f = lambda x: re.search(r'id(\d+)', x).group(1) 
    data["id"] = data["app_link"].apply(track_id_match_f).astype(int)
    test_ids =  data['id'].tolist().copy()
    test_ids = data['id'].sample(10, random_state=21).tolist()
    for track_id in test_ids:
        text_block = create_text_block(data, track_id)
        pred = evaluate_mlp_on_sentence(text_block, mlp)
        print(text_block)
        print(f"ID {track_id}: Predicted: {pred:.4f}", '\n')


def test_specific():
    print("\nTesting on specific examples...")
    app_name = "この素晴らしい世界に祝福を！このファン 冒険者ブック"
    name = "Problematic app"
    text = """
App: この素晴らしい世界に祝福を！このファン 冒険者ブック
Subtitle: 9+
Description: 『この素晴らしい世界に祝福を！ファンタスティックデイズ 』は2025年1月30日をもちまして、サービスを終了しました。 現在はオフライン版としてお楽しみいただけます。＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿ 冒険者ブックでは、『この素晴らしい世界に祝福を！ファンタスティックデイズ 』の下記機能をお楽しみいただけます。
Updates: 1. オフライン版へ更新しました
2. ・不具合の修正
3. ・メニュー画面の一部レイアウトを変更しました
4. ・アプリアイコンを変更しました
5. ・アプリアイコンを変更しました ・不具合の修正
6. ・ホーム画面に「バトル衣装」表示機能を追加しました ・ホーム画面に設定出来るメンバー数が5人から10人まで設定できるようになりました
7. ・アプリアイコンを変更しました ・襲来イベント開催時に「メイン・名声クエスト」から直接どちらかに遷移できるようになりました ・不具合の修正
8. ・アプリアイコンを変更しました ・イベントに「チャレンジバトル」を追加しました ※「チャレンジバトル」は2/27(火)15:00～開催のイベントにてプレイ可能です ・新機能「キャラランク」を追加しました ・不具合の修正
9. ・アプリアイコンを変更しました ・フレンドリスト画面およびアリーナランキング画面に「称号」の表示を追加しました ・不具合の修正
"""
    pred = evaluate_mlp_on_sentence(text, mlp)
    print(f"{name}: {pred:.4f} (expected < 0.3)", '\n')
    
def test_on_unseen():
    print("\nTesting on unseen examples...")
    # Example 1: Active game with regular updates
    example_1 = """
    App: パズル&ドラゴンズ
    Subtitle: 12+
    Description: 大人気のパズルRPG！モンスターを集めて、育てて、進化させよう！3つ以上の同じ色のドロップを揃えて消すだけの簡単操作で、誰でも楽しめます。
    Updates: 1. 新キャラクター追加
    2. イベントダンジョン開催中
    3. バランス調整とバグ修正
    4. 新機能「超覚醒」実装
    5. コラボイベント開始
    """
    expected_1 = "> 0.3"

    # Example 2: Service termination notice
    example_2 = """
    App: ファイナルファンタジー レコードキーパー
    Subtitle: 4+
    Description: 『ファイナルファンタジー レコードキーパー』は2022年9月29日をもちまして、サービスを終了いたしました。長らくのご愛顧、誠にありがとうございました。
    Updates: 1. サービス終了のお知らせ
    2. データ引き継ぎ終了
    3. 最終アップデート
    4. サーバー停止しました
    """
    expected_2 = "< 0.2"
    # Example 3: Migration announcement
    example_3 = """
    App: モンスターストライク
    Subtitle: 9+
    Description: 引っ張って放つだけの簡単操作！爽快なバトルが楽しめる協力プレイRPG。2025年3月より新バージョンへ移行予定です。現バージョンは順次サポート終了となります。
    Updates: 1. 新バージョン移行のお知らせ
    2. データ移行ツール提供開始
    3. 現バージョンサポート終了予定の案内
    4. 移行期間中の注意事項
    5. 最終プレイ期限のご案内
    """
    expected_3 = "< 0.3"
    # Example 4: Extended maintenance
    example_4 = """
    App: グランブルーファンタジー
    Subtitle: 12+
    Description: 現在、技術的な問題により長期メンテナンス中です。復旧時期については改めてお知らせいたします。ご迷惑をおかけして申し訳ございません。
    Updates: 1. 緊急メンテナンス開始
    2. 長期メンテナンス移行のお知らせ
    3. 復旧見込み時期未定
    4. 補償内容検討中
    """
    expected_4 = "< 0.2"
    # Example 5: Normal active app
    example_5 = """
    App: LINE：ディズニー ツムツム
    Subtitle: 4+
    Description: ディズニーのかわいいキャラクターたちが登場するパズルゲーム！ツムを3つ以上つなげて消していく簡単ルールで、誰でも楽しめます。
    Updates: 1. 新ツム追加
    2. 期間限定イベント開催
    3. ログインボーナス実施中
    4. 不具合の修正
    5. パフォーマンス改善
    6. 新機能追加
    """
    expected_5 = "> 0.3"

    example_6 = """
    App: StreetPhotographer Pro
    Subtitle: Photo editor & offline gallery
    Description: StreetPhotographer Pro offers pro-grade filters, in-app purchases for premium packs, and a social feed. For privacy we also support an offline-only gallery mode that stores photos locally if the user turns it on.
    Updates: 1. Added premium filter pack (IAP)
    2. Improved social feed and friend tagging
    3. Added offline-only gallery toggle (for privacy-conscious users)
    """
    expected_6 = "> 0.3"

    example_7 = """
    App: Sunset Notes
    Subtitle: Your retired journal, safely stored
    Description: Sunset Notes moves all your completed notes into a “retired” section, keeping them hidden from the main dashboard. You can still access and edit them anytime. Premium users unlock full tagging, search, and cloud sync. The app continues to function normally despite the “retired” label.
    Updates:
    1. Introduced “retired” folder for old notes
    2. Minor bug fixes and UI polish
    3. Cloud sync improvements for premium users

    """
    expected_7 = "> 0.3"
    tests = [
    ("Active game with regular updates", example_1, expected_1),
    ("Service termination notice", example_2, expected_2),
    ("Migration announcement", example_3, expected_3),
    ("Extended maintenance", example_4, expected_4),
    ("Normal active app", example_5, expected_5),
    ("Photo editing app", example_6, expected_6),
    ("Workout tracking app", example_7, expected_7),
    ]

    for name, text, expected in tests:
        pred = evaluate_mlp_on_sentence(text, mlp)
        print(f"{name}: {pred:.4f} (expected {expected})", '\n')

test_on_trained_data()
test_specific()
test_on_unseen()