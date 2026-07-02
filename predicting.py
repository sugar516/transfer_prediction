"""
移籍予測パイプライン - 二段階分類器（LINE Messaging API 完全統合版）
==================================================================
1. データロード: Kaggle(davidcariboo)の最新データをネットから直接ストリーミング
2. ステージ1: 選手が移籍するかどうかをLightGBMでガチ学習＆推論
3. ステージ2: どのクラブに移籍するかを総当たりで引力シミュレーション
4. LINE通知 : 前日比のスコア変動を検知し、安全にLINE公式アカウントからプッシュ通知
"""

import os
import json
import logging
import warnings
from datetime import datetime # 👈 これが import されているのを確認
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 🌟 毎日、その日の日付を勝手に取得して更新するロジックに変更！
TODAY = pd.Timestamp(datetime.now().date())

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# 定数・設定パラメータ
# ==========================================
# 1. BASE_URL はリポジトリの直下までに変更します
BASE_URL = "https://raw.githubusercontent.com/davidcariboo/player-scores/master"

# ==========================================
# 1. データ読み込み関数（正しいURLマッピング版）
# ==========================================
def load_data(base_url=BASE_URL):
    logger.info("📡 インターネット上の最新データベース(davidcariboo/player-scores)から正確なパスでロード中...")
    
    # 🌟 各CSVファイルが置かれている実際のフォルダ階層を正しくマッピング
    data_files = {
        "players": f"{base_url}/data/players/players.csv",
        "transfers": f"{base_url}/data/transfers/transfers.csv",
        "clubs": f"{base_url}/data/clubs/clubs.csv",
        "appearances": f"{base_url}/data/appearances/appearances.csv",
        "player_valuations": f"{base_url}/data/player_valuations/player_valuations.csv",
    }
    
    data = {}
    for key, url in data_files.items():
        try:
            # 🌟 timeout引数を、PandasがHTTP通信時に理解できる storage_options に修正します
            data[key] = pd.read_csv(url, storage_options={"timeout": 30})
            logger.info(f"✅ {key}をロード完了: {len(data[key])} 行")
        except Exception as e:
            logger.error(f"❌ {key}のダウンロードに失敗 (URL: {url}): {e}")
            raise

    # 日付型の変換
    data["transfers"]["transfer_date"] = pd.to_datetime(data["transfers"]["transfer_date"])
    data["players"]["date_of_birth"] = pd.to_datetime(data["players"]["date_of_birth"])
    data["player_valuations"]["date"] = pd.to_datetime(data["player_valuations"]["date"])
    return data

# ==========================================
# 2. ユーティリティ・計算関数
# ==========================================
def calculate_age(birth_dates, ref_date=TODAY):
    return (ref_date.year - birth_dates.dt.year) - (
        (ref_date.month < birth_dates.dt.month)
        | ((ref_date.month == birth_dates.dt.month) & (ref_date.day < birth_dates.dt.day))
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
# 3. ステージ1: モデル訓練（移籍確率予測）
# ==========================================
def train_stage1_model(players_df, transfers_df, appearances_df):
    logger.info("🤖 【ステージ1】移籍確率予測モデル(LightGBM)の学習を開始...")
    
    # 蓄積スタッツの計算
    player_stats = appearances_df.groupby('player_id').agg(
        total_minutes=('minutes_played', 'sum'),
        total_goals=('goals', 'sum'),
        total_assists=('assists', 'sum'),
        total_games=('appearance_id', 'count')
    ).reset_index()

    df1 = pd.merge(players_df, player_stats, on='player_id', how='left')
    df1[['total_minutes', 'total_goals', 'total_assists', 'total_games']] = df1[['total_minutes', 'total_goals', 'total_assists', 'total_games']].fillna(0)
    df1['age'] = calculate_age(df1['date_of_birth'])
    
    # 正解ラベル（過去に移籍実績があるか）
    df1['is_transfer'] = np.where(df1['player_id'].isin(transfers_df['player_id']), 1, 0)

    features = ['age', 'market_value_in_eur', 'position', 'country_of_citizenship', 'total_minutes', 'total_goals', 'total_assists', 'total_games']
    cat_cols = ['position', 'country_of_citizenship']

    X = df1[features].copy()
    for col in cat_cols:
        X[col] = X[col].astype('category')
    y = df1['is_transfer']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    model = lgb.LGBMClassifier(n_estimators=STAGE1_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)

    logger.info(f"✅ ステージ1訓練完了。検証セットAUC: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.4f}")
    return model, features, cat_cols

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

    # 必須スタッツの次元補完
    for col in ['total_minutes', 'total_goals', 'total_assists', 'total_games']:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df[['club_connection_score', 'agent_pipeline_score', 'club_nationality_affinity']] = final_df[['club_connection_score', 'agent_pipeline_score', 'club_nationality_affinity']].fillna(0)
    final_df['total_pipeline_power'] = final_df['club_connection_score'] * 2 + final_df['agent_pipeline_score']
    final_df['is_transfer'] = np.where(final_df['transfer_fee'].notna(), 1, 0)

    positives = final_df[final_df['is_transfer'] == 1].copy()
    positives = pd.merge(positives, club_pos_counts, left_on=['to_club_id', 'position'], right_on=['current_club_id', 'position'], how='left').drop('current_club_id', axis=1)

    # ネガティブサンプリング
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
def run_inference(players_df, clubs_df, appearances_df, player_valuations_df, model_stage1, model_stage2, stage1_features, stage1_cat_cols, stage2_features, stage2_cat_cols, feature_dicts):
    logger.info("🌍 2026年現在の現役選手データを生成し、移籍予測を開始します...")

    current_players = players_df[players_df['last_season'] >= 2025].copy()
    current_players = pd.merge(current_players, clubs_df[['club_id', 'domestic_competition_id']], left_on='current_club_id', right_on='club_id', how='left')

    # 正確な出場スタッツの動的マージ（バグ回避）
    player_stats = appearances_df.groupby('player_id').agg(
        total_minutes=('minutes_played', 'sum'), total_goals=('goals', 'sum'),
        total_assists=('assists', 'sum'), total_games=('appearance_id', 'count')
    ).reset_index()

    latest_valuation = player_valuations_df.sort_values('date').groupby('player_id').tail(1)[['player_id', 'market_value_in_eur']].rename(columns={'market_value_in_eur': 'recent_market_value'})

    infer_base = pd.merge(current_players[['player_id', 'name', 'position', 'country_of_citizenship', 'date_of_birth', 'current_club_id', 'domestic_competition_id']], player_stats, on='player_id', how='left')
    infer_base = pd.merge(infer_base, latest_valuation, on='player_id', how='inner')
    infer_base[['total_minutes', 'total_goals', 'total_assists', 'total_games']] = infer_base[['total_minutes', 'total_goals', 'total_assists', 'total_games']].fillna(0)

    infer_base['market_value_in_eur'] = infer_base['recent_market_value']
    infer_base['age'] = calculate_age(infer_base['date_of_birth'])
    infer_base = infer_base.rename(columns={'domestic_competition_id': 'player_club_domestic_competition_id'})

    logger.info("🎯 [Stage 1/2] 全員の移籍確率(P_transfer)を算出中...")
    X1_infer = infer_base[stage1_features].copy()
    for col in stage1_cat_cols:
        X1_infer[col] = X1_infer[col].astype('category')

    infer_base['p_transfer'] = model_stage1.predict_proba(X1_infer)[:, 1]
    top_candidates = infer_base.sort_values('p_transfer', ascending=False).head(TOP_CANDIDATES_COUNT)

    all_club_ids = clubs_df['club_id'].unique()
    club_to_league_dict = clubs_df.set_index('club_id')['domestic_competition_id'].to_dict()

    logger.info("🎯 [Stage 2/2] 上位有力選手の移籍先クラブ総当たりシミュレーション中...")
    results = []

    for idx, row in enumerate(top_candidates.itertuples(index=False), 1):
        src_league = row.player_club_domestic_competition_id
        target_clubs = [c for c in all_club_ids if c != row.current_club_id]

        rows = []
        for club_id in target_clubs:
            dst_league = club_to_league_dict.get(club_id, 'Unknown')
            pipeline_power = feature_dicts['conn_dict'].get((row.current_club_id, club_id), 0) * 2 + feature_dicts['agent_dict'].get((src_league, club_id), 0)
            nat_affinity = feature_dicts['affinity_dict'].get((club_id, row.country_of_citizenship), 0)

            if dst_league != src_league and pipeline_power == 0 and nat_affinity == 0:
                continue

            from_v = feature_dicts['club_avg_val_dict'].get(row.current_club_id, 1)
            to_v = feature_dicts['club_avg_val_dict'].get(club_id, 1)
            status_gap = safe_divide(to_v, from_v)

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

        raw_scores = model_stage2.predict_proba(X2_infer)[:, 1]
        exp_scores = np.exp(raw_scores * SCORE_TEMPERATURE)
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
    results_df['to_club_name'] = results_df['to_club_id'].map(feature_dicts['club_name_map'])
    return results_df[['player_name', 'p_transfer', 'to_club_name', 'p_destination_given_transfer', 'combined_score']]

# ==========================================
# 6. LINE 差分検知＆プッシュ通知システム
# ==========================================
def run_realtime_alert_system(current_results_df, previous_results_path):
    logger.info("⚡ 前日比のスコア変動速度(Velocity)の解析を開始...")

    if os.path.exists(previous_results_path):
        prev_df = pd.read_csv(previous_results_path)
    else:
        # 初回実行時は前日比差分を擬似的に発生させる
        prev_df = current_results_df.copy()
        prev_df['combined_score'] = prev_df['combined_score'] * 0.7

    comparison = pd.merge(current_results_df, prev_df, on=['player_name', 'to_club_name'], suffixes=('_today', '_prev'))
    comparison['score_velocity'] = comparison['combined_score_today'] - comparison['combined_score_prev']
    
    # 確率急上昇の検知のしきい値
    alerts = comparison[comparison['score_velocity'] > 0.005].sort_values('score_velocity', ascending=False)

    if alerts.empty:
        logger.info("✅ 本日、異常な移籍引力の急上昇をみせた選手は存在しませんでした。")
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

    # 上位3名の異常変動をLINEにプッシュ
    for alert in alerts.head(3).itertuples():
        message_text = (
            f"🚨【AI移籍予兆アラート】🚨\n\n"
            f"👤 選手名: {alert.player_name}\n"
            f"➡️ 移籍有力先: {alert.to_club_name}\n\n"
            f"📈 移籍発生確率: {alert.p_transfer_today * 100:.1f}%\n"
            f"🎯 行き先シンクロ率: {alert.p_destination_given_transfer_today * 100:.1f}%\n"
            f"🔥 総合スコア: {alert.combined_score_today:.4f} (▲ +{alert.score_velocity:.4f})\n\n"
            f"💡 考察: メディカルチェックや代理人間の裏交渉が活発化している可能性があります。ロマーノ氏の発表前に先読み成功か！？"
        )

        payload = {
            "to": line_user_id,
            "messages": [{"type": "text", "text": message_text}],
        }

        res = requests.post(url, headers=headers, data=json.dumps(payload))
        if res.status_code == 200:
            logger.info(f"✅ {alert.player_name} の爆速アラートをLINEに送信しました。")
        else:
            logger.error(f"❌ LINE送信エラー: {res.status_code} - {res.text}")

    # 今日分のデータを明日の比較用に保存
    current_results_df.to_csv(previous_results_path, index=False)

# ==========================================
# 7. メインパイプラインエントリーポイント
# ==========================================
def main():
    try:
        data = load_data()

        # ステージ1の学習
        model_stage1, stage1_features, stage1_cat_cols = train_stage1_model(
            data['players'], data['transfers'], data['appearances']
        )

        # ステージ2の学習
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

        # 推論エンジンのキック
        logger.info("\n" + "=" * 80)
        logger.info(" 【推論】グローバル移籍シミュレーションの実行")
        logger.info("=" * 80)
        results_df = run_inference(
            data['players'], data['clubs'], data['appearances'], data['player_valuations'],
            model_stage1, model_stage2, stage1_features, stage1_cat_cols, stage2_features, stage2_cat_cols, feature_dicts
        )

        # ローカルに最新結果をCSV保存
        output_file = 'transfer_predictions.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"✅ 最新の予測マトリクスを {output_file} に保存しました。")

        # 🚀 リアルタイムLINEアラートシステムを始動
        run_realtime_alert_system(results_df, "prev_predictions.csv")

        # コンソールログに最終結果を表示
        print("\n" + "=" * 100)
        print(" 👑 【世界開放×クラブ格の壁モデル】統合移籍予測シミュレーション結果 👑")
        print("=" * 100)
        print(results_df.head(30).to_string(index=False))
        print("=" * 100)

    except Exception as e:
        logger.error(f"❌ パイプライン実行に致命的なエラーが発生しました: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
