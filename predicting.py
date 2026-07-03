"""
移籍予測パイプライン - 数理危険度×ステージ2LightGBMハイブリッド（LINE完全統合・テストモード版）
==================================================================================
1. データロード: Kaggle(davidcariboo)の最新データをネットから直接ストリーミング
2. ステージ1: 数理モデルによる「移籍危険度スコア（直近移籍ペナルティ内蔵）」の動的算出（正解ラベル完全撤廃）
3. ステージ2: どのクラブに移籍するかを総当たりで引力シミュレーション（LightGBM）
4. LINE通知 : テスト送信モードで上位選手の予測結果を安全にLINE公式アカウントからプッシュ通知
"""

import os
import json
import logging
import warnings
from datetime import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 毎日、その日の日付を勝手に取得して更新するロジック
TODAY = pd.Timestamp(datetime.now().date())

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TOP_CANDIDATES_COUNT = 30
TOP_DESTINATIONS_COUNT = 3
NEGATIVE_SAMPLES_PER_PLAYER = 3
RANDOM_SEED = 42
STAGE2_ESTIMATORS = 200
SCORE_TEMPERATURE = 5
TEST_SIZE = 0.2 

# ==========================================
# 1. データ読み込み関数（kagglehub公式ロード版）
# ==========================================
import kagglehub

def load_data():
    logger.info("📡 kagglehubを使って、Kaggle公式から最新データを直接ロード中...")
    dataset_dir = kagglehub.dataset_download("davidcariboo/player-scores")
    logger.info(f"💾 データセットのダウンロード先: {dataset_dir}")
    
    data = {
        "players": pd.read_csv(f"{dataset_dir}/players.csv"),
        "transfers": pd.read_csv(f"{dataset_dir}/transfers.csv"),
        "clubs": pd.read_csv(f"{dataset_dir}/clubs.csv"),
        "appearances": pd.read_csv(f"{dataset_dir}/appearances.csv"),
        "player_valuations": pd.read_csv(f"{dataset_dir}/player_valuations.csv"),
    }
    
    for key, df in data.items():
        logger.info(f"✅ {key}をロード完了: {len(df)} 行")

    data["transfers"]["transfer_date"] = pd.to_datetime(data["transfers"]["transfer_date"])
    data["players"]["date_of_birth"] = pd.to_datetime(data["players"]["date_of_birth"])
    data["player_valuations"]["date"] = pd.to_datetime(data["player_valuations"]["date"])
    return data

# ==========================================
# 2. ユーティリティ・計算関数
# ==========================================
def calculate_age(birth_dates, ref_date=TODAY):
    ref_year = ref_date.dt.year if isinstance(ref_date, pd.Series) else ref_date.year
    ref_month = ref_date.dt.month if isinstance(ref_date, pd.Series) else ref_date.month
    ref_day = ref_date.dt.day if isinstance(ref_date, pd.Series) else ref_date.day
    
    return (ref_year - birth_dates.dt.year) - (
        (ref_month < birth_dates.dt.month)
        | ((ref_month == birth_dates.dt.month) & (ref_day < birth_dates.dt.day))
    )

def safe_divide(numerator, denominator, fill_value=1.0):
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        return np.where(denominator > 0, numerator / denominator, fill_value)
    return numerator / denominator if denominator > 0 else fill_value

def create_feature_dicts(df_club_pos_counts, df_club_connection, df_agent_pipeline, df_club_nat_affinity, df_clubs, df_club_market_profile):
    logger.info("🔧 高速推論のためのルックアップ辞書を構築中...")
    pos_count_dict = {(r.current_club_id, r.position): r.position_squad_size for r in df_club_pos_counts.itertuples()}
    conn_dict = {(r.from_club_id, r.to_club_id): r.club_connection_score for r in df_club_connection.itertuples()}
    agent_dict = {(r.from_league_id, r.to_club_id): r.agent_pipeline_score for r in df_agent_pipeline.itertuples()}
    affinity_dict = {(r.to_club_id, r.country_of_citizenship): r.club_nationality_affinity for r in df_club_nat_affinity.itertuples()}
    squad_size_dict = df_clubs.set_index('club_id')['squad_size'].to_dict()
    club_avg_val_dict = df_club_market_profile.set_index('current_club_id')['avg_club_value'].to_dict()
    club_name_map = df_clubs.set_index('club_id')['name'].to_dict()
    league_to_clubs = df_clubs.groupby('domestic_competition_id')['club_id'].apply(list).to_dict()

    return {
        'pos_count_dict': pos_count_dict, 'conn_dict': conn_dict, 'agent_dict': agent_dict,
        'affinity_dict': affinity_dict, 'squad_size_dict': squad_size_dict,
        'club_avg_val_dict': club_avg_val_dict, 'club_name_map': club_name_map, 'league_to_clubs': league_to_clubs
    }

# ==========================================
# 3. 新・ステージ1: 数理危険度スコア算出関数（リアル力学完全版）
# ==========================================
def calculate_transfer_vulnerability(players_df, appearances_df, transfers_df):
    logger.info("🔮 契約満了年・個人昇格力学を内蔵した数理モデルで、移籍危険度を算出中...")
    
    # 全選手の出場時間を集計
    player_stats = appearances_df.groupby('player_id').agg(
        total_minutes=('minutes_played', 'sum'),
        total_games=('appearance_id', 'count')
    ).reset_index()
    
    # クラブごとの平均市場価値を算出（個人昇格の判定用）
    club_avg_value = players_df.groupby('current_club_id')['market_value_in_eur'].mean().reset_index(name='club_avg_player_value')
    
    # データの結合
    df = pd.merge(players_df, player_stats, on='player_id', how='left').fillna(0)
    df = pd.merge(df, club_avg_value, on='current_club_id', how='left').fillna(1)
    
    df['age'] = calculate_age(df['date_of_birth'])
    
    # --- 📊 各種力学コンポーネントの計算 ---
    
    # 1. 市場価値のベース引力（対数でなめらかに）
    market_attraction = np.log1p(df['market_value_in_eur'])
    
    # 2. 出場時間による不満度（極端な跳ね上がりを抑える対数仕様）
    playing_dissatisfaction = np.log1p(10000 / (df['total_minutes'] + 100))
    
    # 3. 年齢適性フィルタ（22〜26歳をピークにする）
    age_factor = np.exp(-((df['age'] - 24) ** 2) / 50)
    
    # 4. 🌟【追加】契約残年数による爆発力 (2026年基準)
    # 契約満了年が入っていない場合はデフォルトで3年残りと仮定
    df['contract_expiration_year'] = pd.to_numeric(df['contract_expiration_year'], errors='coerce').fillna(2029)
    df['years_left'] = df['contract_expiration_year'] - 2026
    
    # 残り0年以下（今すぐ満了）は3.0倍、残り1年は2.0倍、2年は1.2倍、それ以上はペナルティ
    contract_factor = np.where(df['years_left'] <= 0, 3.0,
                       np.where(df['years_left'] == 1, 2.0,
                       np.where(df['years_left'] == 2, 1.2, 0.5)))
    
    # 5. 🌟【追加】個人昇格のポテンシャル (チームの平均よりどれだけ突出しているか)
    # チーム平均の何倍の価値があるか（上限5倍で頭打ち）
    df['value_ratio_to_club'] = np.minimum(safe_divide(df['market_value_in_eur'], df['club_avg_player_value']), 5.0)
    # 際立っている選手ほどスコアを乗算（1.0倍〜2.0倍のブースト）
    individual_promotion_factor = 1.0 + (df['value_ratio_to_club'] / 5.0)

    # --- 🚀 総合危険度スコアの結合 ---
    df['p_transfer_score'] = market_attraction * playing_dissatisfaction * age_factor * contract_factor * individual_promotion_factor
    
    # 6. 直近移籍ペナルティ
    recent_moved_players = transfers_df[transfers_df['transfer_date'] >= '2025-07-01']['player_id'].unique()
    df['p_transfer_score'] = np.where(df['player_id'].isin(recent_moved_players), df['p_transfer_score'] * 0.1, df['p_transfer_score'])
    
    # 0〜1の範囲に綺麗に正規化
    min_s, max_s = df['p_transfer_score'].min(), df['p_transfer_score'].max()
    if max_s - min_s > 0:
        df['p_transfer'] = (df['p_transfer_score'] - min_s) / (max_s - min_s)
    else:
        df['p_transfer'] = 0.0
        
    return df

# ==========================================
# 4. ステージ2: データ準備 ＆ モデル訓練（移籍先予測）
# ==========================================
def prepare_and_train_stage2(transfers_df, players_df, clubs_df):
    logger.info("📊 【ステージ2】移籍先クラブシミュレーション用のデータ構築中...")

    club_market_profile = players_df.groupby('current_club_id').agg(
        total_club_value=('market_value_in_eur', 'sum'),
        avg_club_value=('market_value_in_eur', 'mean')
    ).reset_index()

    club_pos_counts = players_df.groupby(['current_club_id', 'position']).size().reset_index(name='position_squad_size')
    club_connection = transfers_df.groupby(['from_club_id', 'to_club_id']).size().reset_index(name='club_connection_score')

    transfers_with_src_league = pd.merge(
        transfers_df, clubs_df[['club_id', 'domestic_competition_id']],
        left_on='from_club_id', right_on='club_id', how='left'
    ).rename(columns={'domestic_competition_id': 'from_league_id'})

    agent_pipeline = transfers_with_src_league.groupby(['from_league_id', 'to_club_id']).size().reset_index(name='agent_pipeline_score')

    transfers_with_nat = pd.merge(transfers_df, players_df[['player_id', 'country_of_citizenship']], on='player_id', how='inner')
    club_nat_affinity = transfers_with_nat.groupby(['to_club_id', 'country_of_citizenship']).size().reset_index(name='club_nationality_affinity')

    merged_df = pd.merge(transfers_df, players_df, on='player_id', how='inner')
    merged_df['age'] = calculate_age(merged_df['date_of_birth'], merged_df['transfer_date'])

    predict_base = merged_df[[
        'player_id', 'name', 'position', 'age', 'market_value_in_eur_x', 'from_club_id',
        'to_club_id', 'transfer_fee', 'country_of_citizenship', 'transfer_date'
    ]].rename(columns={'market_value_in_eur_x': 'market_value_in_eur'})

    predict_base = pd.merge(
        predict_base, clubs_df[['club_id', 'domestic_competition_id']],
        left_on='from_club_id', right_on='club_id', how='left'
    ).rename(columns={'domestic_competition_id': 'from_league_id'}).drop('club_id', axis=1)

    final_df = pd.merge(predict_base, clubs_df[['club_id', 'name', 'domestic_competition_id', 'squad_size']], left_on='to_club_id', right_on='club_id', how='left')
    final_df = pd.merge(final_df, club_connection, on=['from_club_id', 'to_club_id'], how='left')
    final_df = pd.merge(final_df, agent_pipeline, on=['from_league_id', 'to_club_id'], how='left')
    final_df = pd.merge(final_df, club_nat_affinity, on=['to_club_id', 'country_of_citizenship'], how='left')

    final_df = pd.merge(final_df, club_market_profile.rename(columns={'current_club_id': 'from_club_id', 'avg_club_value': 'from_avg_value'}), on='from_club_id', how='left')
    final_df = pd.merge(final_df, club_market_profile.rename(columns={'current_club_id': 'to_club_id', 'avg_club_value': 'to_avg_value'}), on='to_club_id', how='left')

    final_df['from_avg_value'] = final_df['from_avg_value'].fillna(1)
    final_df['to_avg_value'] = final_df['to_avg_value'].fillna(1)
    final_df['club_status_gap'] = safe_divide(final_df['to_avg_value'], final_df['from_avg_value'])

    for col in ['total_minutes', 'total_goals', 'total_assists', 'total_games']:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df[['club_connection_score', 'agent_pipeline_score', 'club_nationality_affinity']] = final_df[['club_connection_score', 'agent_pipeline_score', 'club_nationality_affinity']].fillna(0)
    final_df['total_pipeline_power'] = final_df['club_connection_score'] * 2 + final_df['agent_pipeline_score']
    final_df['is_transfer'] = np.where(final_df['transfer_fee'].notna(), 1, 0)

    positives = final_df[final_df['is_transfer'] == 1].copy()
    positives = pd.merge(positives, club_pos_counts, left_on=['to_club_id', 'position'], right_on=['current_club_id', 'position'], how='left').drop('current_club_id', axis=1)

    rng = np.random.default_rng(RANDOM_SEED)
    league_to_clubs_dict = clubs_df.groupby('domestic_competition_id')['club_id'].apply(list).to_dict()
    pos_count_dict = {(r.current_club_id, r.position): r.position_squad_size for r in club_pos_counts.itertuples()}

    negatives_list = []
    for row in positives.itertuples(index=False):
        candidates = [c for c in league_to_clubs_dict.get(row.domestic_competition_id, []) if c != row.to_club_id]
        if len(candidates) >= NEGATIVE_SAMPLES_PER_PLAYER:
            for sampled_club_id in rng.choice(candidates, size=NEGATIVE_SAMPLES_PER_PLAYER, replace=False):
                from_v = club_market_profile[club_market_profile['current_club_id'] == row.from_club_id]['avg_club_value'].values
                to_v = club_market_profile[club_market_profile['current_club_id'] == sampled_club_id]['avg_club_value'].values
                from_v = from_v[0] if len(from_v) > 0 else 1
                to_v = to_v[0] if len(to_v) > 0 else 1

                negatives_list.append({
                    'age': row.age, 'market_value_in_eur': row.market_value_in_eur, 'position': row.position,
                    'domestic_competition_id': row.domestic_competition_id, 'squad_size': row.squad_size,
                    'position_squad_size': pos_count_dict.get((sampled_club_id, row.position), 0),
                    'country_of_citizenship': row.country_of_citizenship, 'total_pipeline_power': 0,
                    'club_nationality_affinity': 0, 'club_status_gap': safe_divide(to_v, from_v),
                    'total_minutes': row.total_minutes, 'total_goals': row.total_goals, 'total_assists': row.total_assists,
                    'total_games': row.total_games, 'is_transfer': 0
                })

    negatives = pd.DataFrame(negatives_list) if negatives_list else pd.DataFrame()

    stage2_features = ['age', 'market_value_in_eur', 'position', 'domestic_competition_id', 'squad_size', 'position_squad_size', 'country_of_citizenship', 'total_pipeline_power', 'club_nationality_affinity', 'club_status_gap', 'total_minutes', 'total_goals', 'total_assists', 'total_games']
    stage2_cat_cols = ['position', 'domestic_competition_id', 'country_of_citizenship']

    balanced_df = pd.concat([positives[stage2_features + ['is_transfer']], negatives[stage2_features + ['is_transfer']]], axis=0).reset_index(drop=True)
    balanced_df['position_squad_size'] = balanced_df['position_squad_size'].fillna(0)

    X2 = balanced_df[stage2_features].copy()
    for col in stage2_cat_cols:
        X2[col] = X2[col].astype('category')
    y2 = balanced_df['is_transfer']

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y2)

    logger.info("🤖 【ステージ2】モデルのフィッティングを実行中...")
    model = lgb.LGBMClassifier(n_estimators=STAGE2_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    model.fit(X2_train, y2_train)
    logger.info(f"✅ ステージ2訓練完了。検証セットAUC: {roc_auc_score(y2_test, model.predict_proba(X2_test)[:, 1]):.4f}")

    return model, stage2_features, stage2_cat_cols, club_market_profile, club_pos_counts, club_connection, agent_pipeline, club_nat_affinity

# ==========================================
# 5. 二段階の統合推論エンジン
# ==========================================
def run_inference(infer_base_with_p, clubs_df, model_stage2, stage2_features, stage2_cat_cols, feature_dicts):
    logger.info("🌍 グローバル移籍シミュレーションの実行中...")

    top_candidates = infer_base_with_p.sort_values('p_transfer', ascending=False).head(TOP_CANDIDATES_COUNT)

    all_club_ids = clubs_df['club_id'].unique()
    club_to_league_dict = clubs_df.set_index('club_id')['domestic_competition_id'].to_dict()

    results = []

    for idx, row in enumerate(top_candidates.itertuples(index=False), 1):
        src_league = row.player_club_domestic_competition_id
        target_clubs = [c for c in all_club_ids if c != row.current_club_id]

        rows = []
        for club_id in target_clubs:
            dst_league = club_to_league_dict.get(club_id, 'Unknown')
            pipeline_power = feature_dicts['conn_dict'].get((row.current_club_id, club_id), 0) * 2 + feature_dicts['agent_dict'].get((src_league, club_id), 0)
            nat_affinity = feature_dicts['affinity_dict'].get((club_id, row.country_of_citizenship), 0)

            # リーグが異なる、かつ過去のコネクションも国籍アフィニティも皆無な無謀な移籍は足切り
            if dst_league != src_league and pipeline_power == 0 and nat_affinity == 0:
                continue

            from_v = feature_dicts['club_avg_val_dict'].get(row.current_club_id, 1)
            to_v = feature_dicts['club_avg_val_dict'].get(club_id, 1)
            status_gap = safe_divide(to_v, from_v)
            
            # 🌟 ヨーロッパメガクラブから世界の裏側のような、格差ギャップが極端すぎる移籍を足切り(0.15倍〜6.0倍に限定)
            if status_gap < 0.15 or status_gap > 6.0:
                continue

            rows.append({
                'age': row.age, 'market_value_in_eur': row.recent_market_value, 'position': row.position,
                'domestic_competition_id': dst_league, 'squad_size': feature_dicts['squad_size_dict'].get(club_id, np.nan),
                'position_squad_size': feature_dicts['pos_count_dict'].get((club_id, row.position), 0),
                'country_of_citizenship': row.country_of_citizenship, 'total_pipeline_power': pipeline_power,
                'club_nationality_affinity': nat_affinity, 'club_status_gap': status_gap,
                'total_minutes': row.total_minutes, 'total_goals': row.total_goals, 'total_assists': row.total_assists,
                'total_games': row.total_games, 'to_club_id': club_id
            })

        if not rows:
            continue

        cand_df = pd.DataFrame(rows)
        X2_infer = cand_df[stage2_features].copy()
        for col in stage2_cat_cols:
            X2_infer[col] = X2_infer[col].astype('category')

        # LightGBMの生の予測確率を取得
        raw_scores = model_stage2.predict_proba(X2_infer)[:, 1]
        
        # 理想的な移籍は「status_gap = 1.0 (同格)」または「1.5〜3.0倍 (ステップアップ)」です。
        # 格差が離れるほど（0.2倍の格下や、5倍以上の高すぎる壁）、指数関数的に引力を減衰させます。
        gaps = cand_df['club_status_gap'].values
        
        # 同格(1.0)〜少し上のクラブ(1.5)あたりをピークにした引力補正係数
        # 格差ギャップが離れるほどペナルティが強くなります
        gap_penalty = np.exp(-((gaps - 1.2) ** 2) / 0.5) 
        
        # 生スコアにペナルティを掛け算して、現実的な引力に補正
        corrected_scores = raw_scores * gap_penalty
        
        # ソフトマックス関数で確率に変換
        exp_scores = np.exp(corrected_scores * SCORE_TEMPERATURE)
        p_destination = exp_scores / exp_scores.sum()

        cand_df['p_destination'] = p_destination
        cand_df['p_transfer'] = row.p_transfer
        cand_df['combined_score'] = cand_df['p_transfer'] * cand_df['p_destination']

        top3 = cand_df.sort_values('combined_score', ascending=False).head(TOP_DESTINATIONS_COUNT)

        for dest in top3.itertuples(index=False):
            results.append({
                'player_id': row.player_id, 'player_name': row.name, 'p_transfer': round(row.p_transfer, 3),
                'to_club_id': dest.to_club_id, 'p_destination_given_transfer': round(dest.p_destination, 3),
                'combined_score': round(dest.combined_score, 4)
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['to_club_name'] = results_df['to_club_id'].map(feature_dicts['club_name_map'])
        return results_df[['player_name', 'p_transfer', 'to_club_name', 'p_destination_given_transfer', 'combined_score']]
    else:
        return pd.DataFrame(columns=['player_name', 'p_transfer', 'to_club_name', 'p_destination_given_transfer', 'combined_score'])

# ==========================================
# 6. LINE 定時スカウティングレポートシステム
# ==========================================
def run_realtime_alert_system(current_results_df, previous_results_path):
    logger.info("📡 LINEへの定時スカウティングレポート配信処理を開始...")

    if current_results_df.empty:
        logger.warning("⚠️ 予測結果が空のため、通知をスキップします。")
        return

    # GitHub Actions の Secrets から安全に呼び出し
    line_access_token = os.environ.get("LINE_ACCESS_TOKEN")
    line_user_id = os.environ.get("LINE_USER_ID")

    if not line_access_token or not line_user_id:
        logger.warning("⚠️ LINEの環境変数が検出できません。GitHubのSecrets設定を確認してください。")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {line_access_token}",
    }

    # 🌟 【ここがポイント】変な差分条件は一切挟まず、純粋に総合スコアが「今、最も高い上位3名」を抽出
    reports = current_results_df.sort_values('combined_score', ascending=False).head(3)

    # 3名分のデータを1つのメッセージに綺麗にパッキングして、LINEのプッシュ上限（1回）に収める
    message_text = f"📊【AI移籍市場・定時観測レポート】📊\n"
    message_text += f"計算基準日: {TODAY.strftime('%Y-%m-%d')}\n"
    message_text += f"----------------------------------------\n\n"

    for rank, alert in enumerate(reports.itertuples(), 1):
        message_text += (
            f"👑 【第{rank}位】\n"
            f"👤 選手名: {alert.player_name}\n"
            f"➡️ 移籍有力先: {alert.to_club_name}\n"
            f"📈 移籍危険度: {alert.p_transfer * 100:.1f}%\n"
            f"🎯 行き先シンクロ率: {alert.p_destination_given_transfer * 100:.1f}%\n"
            f"🔥 総合引力スコア: {alert.combined_score:.4f}\n"
            f"----------------------------------------\n"
        )
        
    message_text += f"\n💡 考察: 本数値は出場時間・市場価値・契約残年数・個人昇格力学を数理統合した長期トレンド予測です。"

    payload = {
        "to": line_user_id,
        "messages": [{"type": "text", "text": message_text}],
    }

    res = requests.post(url, headers=headers, data=json.dumps(payload))
    if res.status_code == 200:
        logger.info("✅ 本日の正確なスカウティングレポートをLINEに送信しました。")
    else:
        logger.error(f"❌ LINE送信エラー: {res.status_code} - {res.text}")

    # 履歴保存（今後の時系列分析用）
    current_results_df.to_csv(previous_results_path, index=False)
    
# ==========================================
# 7. メインパイプラインエントリーポイント
# ==========================================
def main():
    try:
        data = load_data()

        # ステージ2の学習（先行して関係性モデルを構築）
        logger.info("\n" + "=" * 80)
        logger.info(" 【ステージ2】移籍先クラブ予測モデルの訓練")
        logger.info("=" * 80)
        (
            model_stage2, stage2_features, stage2_cat_cols, club_market_profile,
            club_pos_counts, club_connection, agent_pipeline, club_nat_affinity
        ) = prepare_and_train_stage2(data['transfers'], data['players'], data['clubs'])

        # ルックアップテーブル辞書の構築
        feature_dicts = create_feature_dicts(
            club_pos_counts, club_connection, agent_pipeline, club_nat_affinity, data['clubs'], club_market_profile
        )

        # 🔮 新ステージ1: 数理危険度モデルの実行
        logger.info("\n" + "=" * 80)
        logger.info(" 【新ステージ1】数理モデルによる移籍危険度の算出")
        logger.info("=" * 80)
        
        current_players = data['players'][data['players']['last_season'] >= 2025].copy()
        current_players = pd.merge(current_players, data['clubs'][['club_id', 'domestic_competition_id']], left_on='current_club_id', right_on='club_id', how='left')
        
        # 数理ベースの危険度割り当て
        scored_players = calculate_transfer_vulnerability(current_players, data['appearances'], data['transfers'])
        
        # 必要な変数マッピングを推論用に統合
        latest_valuation = data['player_valuations'].sort_values('date').groupby('player_id').tail(1)[['player_id', 'market_value_in_eur']].rename(columns={'market_value_in_eur': 'recent_market_value'})
        player_stats = data['appearances'].groupby('player_id').agg(
            total_minutes=('minutes_played', 'sum'), total_goals=('goals', 'sum'),
            total_assists=('assists', 'sum'), total_games=('appearance_id', 'count')
        ).reset_index()
        
        infer_base = pd.merge(scored_players[['player_id', 'name', 'position', 'country_of_citizenship', 'date_of_birth', 'current_club_id', 'domestic_competition_id', 'p_transfer']], player_stats, on='player_id', how='left')
        infer_base = pd.merge(infer_base, latest_valuation, on='player_id', how='inner')
        infer_base[['total_minutes', 'total_goals', 'total_assists', 'total_games']] = infer_base[['total_minutes', 'total_goals', 'total_assists', 'total_games']].fillna(0)
        infer_base['age'] = calculate_age(infer_base['date_of_birth'])
        infer_base = infer_base.rename(columns={'domestic_competition_id': 'player_club_domestic_competition_id'})

        # 推論エンジンのキック
        logger.info("\n" + "=" * 80)
        logger.info(" 【推論】グローバル移籍シミュレーションの実行")
        logger.info("=" * 80)
        results_df = run_inference(
            infer_base, data['clubs'], model_stage2, stage2_features, stage2_cat_cols, feature_dicts
        )

        # ローカルに最新結果をCSV保存
        output_file = 'transfer_predictions.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"✅ 最新の予測マトリクスを {output_file} に保存しました。")

        # 🚀 リアルタイムLINEアラートシステムを始動（テストモード送信）
        run_realtime_alert_system(results_df, "prev_predictions.csv")

        # コンソールログに最終結果を表示
        print("\n" + "=" * 100)
        print(" 👑 【数理不満度×クラブ格の壁モデル】統合移籍予測シミュレーション結果 👑")
        print("=" * 100)
        if not results_df.empty:
            print(results_df.head(30).to_string(index=False))
        else:
            print("予測条件を満たす現実的な移籍候補が検出されませんでした。")
        print("=" * 100)

    except Exception as e:
        logger.error(f"❌ パイプライン実行に致命的なエラーが発生しました: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
