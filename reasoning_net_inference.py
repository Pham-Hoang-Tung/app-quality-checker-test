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
        s_pred, r_pred = mlp(mlp_input)
        s_pred = s_pred.cpu().numpy().item()  
        # r_pred = r_pred.cpu().numpy().item()  
        # scalar

    # clamp to [0,1] since labels are in this range
    s_pred = float(np.clip(s_pred, 0.0, 1.0))
    # r_pred = float(np.clip(r_pred, 0.0, 1.0))

    return s_pred


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



class ReasonAndScoreNet(nn.Module):
    def __init__(self, in_dim, reason_dim=384, hidden=(512, 128), dropout=0.1):
        super().__init__()
        
        # Shared encoder MLP
        enc_layers = []
        prev = in_dim
        for h in hidden:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Reason head: forces causal explanation
        self.reason_head = nn.Sequential(
            nn.Linear(prev, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(320, reason_dim)
        )

        self.score_head = nn.Sequential(
            nn.Linear(reason_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.encoder(x)                 # shared representation
        r_pred = self.reason_head(h)        # reasoning embedding
        s_pred = self.score_head(r_pred)    # score MUST flow through reason
        return s_pred.squeeze(-1), r_pred


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


mlp = ReasonAndScoreNet(in_dim=1024, reason_dim=1024, hidden=(640, 64), dropout=0.3)
mlp.load_state_dict(torch.load("mlp_regressor_reasoning_torch.pth", map_location="cpu"))
mlp.eval()
    
def test_on_trained_data():
    passed = 0
    print("Testing on trained data...")
    data = pd.read_csv("./app-unique.csv")
    label = pd.read_csv("./app-labels.csv")
    # data = pd.read_csv("/home/tungf/App-anormalies-detection/data/noise-data-with-labels-and-id.csv")
    total = 413
    track_id_match_f = lambda x: re.search(r'id(\d+)', x).group(1) 
    data["id"] = data["app_link"].apply(track_id_match_f).astype(int)
    test_ids = label['id'].tolist().copy()
    test_ids = label['id'].sample(total, random_state=21).tolist()
    for track_id in test_ids:
        text_block = create_text_block(data, track_id)
        score = label[label['id'] == track_id]['score'].values[0]
        score = float(score)
        if score <0.3:
            expected_condition = "low"
        elif score >=0.3 and score<0.6:
            expected_condition = "medium"
        elif score >=0.6:
            expected_condition = "high"
        # pred = evaluate_mlp_on_sentence(text_block, mlp)
        if run_test(track_id, text_block, expected_condition):
            passed += 1
        # print(text_block)
        # print(f"ID {track_id}: Predicted: {pred:.4f}, Actual: {score:.4f}", '\n')

     # Print summary
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed ({passed*100//total}%)")
    print("="*70 + "\n")
    
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

# Test runner with pass/fail tracking
def run_test(name, text, expected_condition):
    """Run a single test and return if it passed."""
    pred = evaluate_mlp_on_sentence(text, mlp)
    
    if expected_condition == "high":
        passed = pred >= 0.3
        expected_str = "> 0.3"
    elif expected_condition == "medium":
        passed = 0.3 <= pred < 0.6
        expected_str = "0.3 - 0.6"
    elif expected_condition == "low":
        passed = pred < 0.3
        expected_str = "< 0.3"
    else:
        passed = False
        expected_str = expected_condition
    
    status = "✓" if passed else "✗"
    print(f"{status} {name}: {pred:.4f} (expected {expected_str})")
    return passed


def run_all_tests():
    """Run all test cases and show summary."""
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70 + "\n")
    passed = 0
    total = 0
    
    # Test 1: Temporary maintenance
    total += 1
    text = """App: パズドラ メンテナンス中
Subtitle: 12+
Description: 現在メンテナンス中です。2時間後の14:00に再開予定です。ご不便をおかけしますが、しばらくお待ちください。
Updates: 1. 定期メンテナンス実施中
2. 14:00再開予定
3. メンテナンス後に新機能追加"""
    if run_test("Temporary Maintenance", text, "high"):
        passed += 1
    
    # Test 2: Archive mode offline only
    total += 1
    text = """App: レトロゲームコレクション
Subtitle: 4+
Description: アーカイブ版として引き続きプレイ可能です。全ての機能をオフラインで楽しめます。今後も遊び続けられます。
Updates: 1. アーカイブ版に移行
2. オフラインモード完全対応
3. 全コンテンツアクセス可能"""
    if run_test("Archive Mode Offline Only", text, "low"):
        passed += 1
    
    # Test 3: Retired branding but active
    total += 1
    text = """App: 写真編集プロ リタイアド
Subtitle: 4+
Description: 「リタイアド」エディションとして新しくリリース。すべての機能が使えます。プレミアムフィルター追加、クラウド同期対応。
Updates: 1. リタイアドエディション発売
2. 新フィルターパック追加
3. パフォーマンス向上
4. バグ修正"""
    if run_test("Retired Branding But Active", text, "high"):
        passed += 1
    
    # Test 4: Service ended offline only
    total += 1
    text = """App: RPGクエスト オフライン
Subtitle: 9+
Description: オンラインサービスは終了しましたが、オフラインモードで全ストーリー、全キャラクター、全機能をお楽しみいただけます。
Updates: 1. オフライン完全版に更新
2. 全DLC同梱
3. データ保存機能追加
4. UI改善"""
    if run_test("Service Ended Offline Only", text, "low"):
        passed += 1
    
    # Test 5: Migration with continued service
    total += 1
    text = """App: ソーシャルネットワーク Pro
Subtitle: 12+
Description: 新バージョンへの移行期間中です。現在のアプリは引き続き使用できます。移行後もすべての機能が利用可能です。
Updates: 1. 新バージョン移行開始
2. 現バージョンも継続サポート
3. データ自動移行対応
4. 新機能プレビュー公開中"""
    if run_test("Migration But Service Continues", text, "high"):
        passed += 1
    
    # Test 6: Sunset named but fully active
    total += 1
    text = """App: Sunset Photo Editor Pro
Subtitle: 4+
Description: プロフェッショナルな写真編集アプリ。毎月新しいフィルターとツールを追加中。プレミアム機能でクラウド同期も対応。
Updates: 1. 新フィルター20種追加
2. AI自動補正機能
3. クラウドストレージ統合
4. パフォーマンス最適化
5. コミュニティ機能追加"""
    if run_test("Sunset Named But Active", text, "high"):
        passed += 1
    
    # Test 7: Legacy version still supported
    total += 1
    text = """App: ゲームエンジン レガシー
Subtitle: 4+
Description: レガシー版として引き続きサポート中。すべての機能が使用可能で、定期的なセキュリティアップデートを提供しています。
Updates: 1. セキュリティパッチ適用
2. 互換性向上
3. バグ修正
4. ドキュメント更新"""
    if run_test("Legacy Version Still Supported", text, "high"):
        passed += 1
    
    # Test 8: Discontinued standalone
    total += 1
    text = """App: カメラフィルター スタンドアロン
Subtitle: 4+
Description: サーバー接続を終了しましたが、アプリはスタンドアロンとして完全動作します。すべてのフィルターと機能が使えます。
Updates: 1. スタンドアロン版にアップデート
2. 全フィルター永久アンロック
3. ローカル保存機能強化
4. 動作安定性向上"""
    if run_test("Discontinued Server Standalone", text, "low"):
        passed += 1
    
    # Test 9: Final version complete package
    total += 1
    text = """App: アドベンチャー ファイナルエディション
Subtitle: 9+
Description: ファイナルエディションとして完全版を配信。全章、全DLC、全キャラクター収録。追加コンテンツ不要の完全版です。
Updates: 1. ファイナルエディション公開
2. 全コンテンツ同梱
3. 最終調整とバグ修正完了
4. グラフィック向上"""
    if run_test("Final Version Complete Package", text, "high"):
        passed += 1
    
    # Test 10: Adversarial - Sunset Archive Pro
    total += 1
    text = """App: Sunset Archive Pro
Subtitle: Enterprise document management
Description: Despite the 'sunset' and 'archive' branding (marketing realignment for legacy product line), this build receives full monthly security maintenance. All premium collaboration features remain active with guaranteed SLAs. Cloud sync, real-time editing, and API integrations continue uninterrupted. The 'archive' terminology reflects portfolio categorization, not service discontinuation.
Updates: 1. Monthly security patch cycle maintained
2. Premium collaboration latency optimization
3. Cloud sync encryption hardening
4. API rate limit improvements
5. Branding update includes 'sunset archive' labels for catalog consistency"""
    if run_test("Adversarial: Sunset Archive Pro", text, "high"):
        passed += 1
    
    # Test 11: Adversarial - Legacy Scheduler
    total += 1
    text = """App: Legacy Scheduler - Extended Maintenance Edition
Subtitle: Calendar and task management
Description: Labeled 'legacy', 'deprecated', and 'end-of-life' for regulatory compliance documentation. However, active development continues with recurring event engine upgrades, cross-platform sync improvements, and enterprise SSO integration. The 'extended maintenance' designation ensures predictable support windows with no service interruption planned.
Updates: 1. Extended maintenance window confirmed (indefinite support continuation)
2. Recurring event rule engine optimization
3. Enterprise SSO authentication added
4. Cross-device sync stability enhancement
5. Compliance labeling: 'deprecated legacy end-of-life' (documentation only, no functional impact)"""
    if run_test("Adversarial: Legacy Scheduler", text, "high"):
        passed += 1
    
    # Test 12: Adversarial - Final Edition Complete
    total += 1
    text = """App: Final Edition Complete Productivity Suite
Subtitle: All-in-one workspace
Description: Branded 'final', 'complete', 'no further expansion' to indicate feature maturity and stability baseline. Both offline and online modes fully supported. Subscription tier continues receiving performance telemetry tuning, vulnerability patches, and cloud infrastructure upgrades. 'Final' denotes completion, not abandonment.
Updates: 1. Performance telemetry-driven optimization deployed
2. Offline cache integrity verification enhanced
3. Premium export pipeline 40% speed improvement
4. Wording clarification: 'final complete stable end-of-feature-expansion' (NOT end of support)
5. Cloud infrastructure migrated to improved hosting"""
    if run_test("Adversarial: Final Edition Complete", text, "high"):
        passed += 1
    
    # Test 13: Adversarial - Discontinued Server Enhanced
    total += 1
    text = """App: Discontinued Server Mirror - Enhanced Standalone
Subtitle: Analytics and reporting
Description: Server components 'discontinued', 'decommissioned', 'sunset' as planned obsolescence strategy. Standalone client gained expanded local processing engine, refurbished analytics pipeline, and continuous quarterly security patches. All primary workflows intact. Premium tier analytics upgraded with offline ML inference.
Updates: 1. Local analytics engine (replaced remote aggregation, performance improved)
2. Quarterly security patch cycle active
3. Client-side rendering latency reduction
4. Server decommission clarification: 'offline continuity ensured, no loss of functionality'
5. Premium ML anomaly detection upgrade (offline mode)"""
    if run_test("Adversarial: Discontinued Server", text, "high"):
        passed += 1
    
    # Test 14: Adversarial - Retired Vault
    total += 1
    text = """App: Retired Notes Vault - Knowledge Archive
Subtitle: Structured note management
Description: Marketing terms 'retired', 'archival', 'sunset', 'hibernated' used for product repositioning. Editing, tagging, encrypted sync, and premium AI summarization all actively maintained. New feature: graph-based note traversal. 'Retired' indicates stability maturity, not neglect.
Updates: 1. AI summarization model refresh (latest LLM integration)
2. Encrypted sync throughput 3x increase
3. Graph-based tag traversal added
4. Marketing copy revision: 'retired sunset archival' terminology cluster
5. Export format expansion (Markdown, JSON, XML)"""
    if run_test("Adversarial: Retired Vault", text, "high"):
        passed += 1
    
    # Test 15: Adversarial - Hibernation Tasks
    total += 1
    text = """App: Hibernation Tasks - Long-Term Stable Release
Subtitle: Enterprise task automation
Description: 'Long-term support (LTS)', 'hibernation', 'legacy', 'final cycle' phrasing clarifies predictable maintenance cadence. Active SLA guarantees for push notification sync, premium automation scripting, and error monitoring dashboard. Stability maturity does not equal termination.
Updates: 1. LTS cadence reconfirmed (monthly patch schedule continues)
2. Automation script sandbox security improvements
3. Push notification retry logic hardening
4. Documentation update: 'hibernation legacy final' labeling (alignment only)
5. Premium tier: workflow analytics dashboard added"""
    if run_test("Adversarial: Hibernation Tasks LTS", text, "high"):
        passed += 1
    
    # Test 16: test_on_unseen - Active game
    total += 1
    text = """App: パズル&ドラゴンズ
Subtitle: 12+
Description: 大人気のパズルRPG！モンスターを集めて、育てて、進化させよう！3つ以上の同じ色のドロップを揃えて消すだけの簡単操作で、誰でも楽しめます。
Updates: 1. 新キャラクター追加
2. イベントダンジョン開催中
3. バランス調整とバグ修正
4. 新機能「超覚醒」実装
5. コラボイベント開始"""
    if run_test("Active Game Regular Updates", text, "high"):
        passed += 1
    
    # Test 17: Service termination
    total += 1
    text = """App: ファイナルファンタジー レコードキーパー
Subtitle: 4+
Description: 『ファイナルファンタジー レコードキーパー』は2022年9月29日をもちまして、サービスを終了いたしました。長らくのご愛顧、誠にありがとうございました。
Updates: 1. サービス終了のお知らせ
2. データ引き継ぎ終了
3. 最終アップデート
4. サーバー停止しました"""
    if run_test("Service Termination Notice", text, "low"):
        passed += 1
    
    # Test 18: Migration announcement
    total += 1
    text = """App: モンスターストライク
Subtitle: 9+
Description: 引っ張って放つだけの簡単操作！爽快なバトルが楽しめる協力プレイRPG。2025年3月より新バージョンへ移行予定です。現バージョンは順次サポート終了となります。
Updates: 1. 新バージョン移行のお知らせ
2. データ移行ツール提供開始
3. 現バージョンサポート終了予定の案内
4. 移行期間中の注意事項
5. 最終プレイ期限のご案内"""
    if run_test("Migration Announcement", text, "low"):
        passed += 1
    
    # Test 19: Extended maintenance
    total += 1
    text = """App: グランブルーファンタジー
Subtitle: 12+
Description: 現在、技術的な問題により長期メンテナンス中です。復旧時期については改めてお知らせいたします。ご迷惑をおかけして申し訳ございません。
Updates: 1. 緊急メンテナンス開始
2. 長期メンテナンス移行のお知らせ
3. 復旧見込み時期未定
4. 補償内容検討中"""
    if run_test("Extended Maintenance", text, "low"):
        passed += 1
    
    # Test 20: Normal active app
    total += 1
    text = """App: LINE：ディズニー ツムツム
Subtitle: 4+
Description: ディズニーのかわいいキャラクターたちが登場するパズルゲーム！ツムを3つ以上つなげて消していく簡単ルールで、誰でも楽しめます。
Updates: 1. 新ツム追加
2. 期間限定イベント開催
3. ログインボーナス実施中
4. 不具合の修正
5. パフォーマンス改善
6. 新機能追加"""
    if run_test("Normal Active App", text, "high"):
        passed += 1
    
    # Test 21: Photo editing app with IAP
    total += 1
    text = """App: StreetPhotographer Pro
Subtitle: Photo editor & offline gallery
Description: StreetPhotographer Pro offers pro-grade filters, in-app purchases for premium packs, and a social feed. For privacy we also support an offline-only gallery mode that stores photos locally if the user turns it on.
Updates: 1. Added premium filter pack (IAP)
2. Improved social feed and friend tagging
3. Added offline-only gallery toggle (for privacy-conscious users)"""
    if run_test("Photo Editing App with IAP", text, "high"):
        passed += 1
    
    # Test 22: Sunset Notes app
    total += 1
    text = """App: Sunset Notes
Subtitle: Your retired journal, safely stored
Description: Sunset Notes moves all your completed notes into a "retired" section, keeping them hidden from the main dashboard. You can still access and edit them anytime. Premium users unlock full tagging, search, and cloud sync. The app continues to function normally despite the "retired" label.
Updates: 1. Introduced "retired" folder for old notes
2. Minor bug fixes and UI polish
3. Cloud sync improvements for premium users"""
    if run_test("Sunset Notes Active App", text, "high"):
        passed += 1
    
    # Print summary
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed ({passed*100//total}%)")
    print("="*70 + "\n")
    
    return passed, total


# Run all tests
if __name__ == "__main__":
    test_on_trained_data()
    test_specific()
    test_on_unseen()
    run_all_tests()
