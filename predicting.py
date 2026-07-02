# ==========================================
# 0. データ読み込み（🌐davidcariboo版・直結ロード）
# ==========================================
print("📡 Kaggleの最新データベース(davidcariboo/player-scores)から直接ロード中...")

# davidcariboo氏が毎日自動更新している大元リポジトリのURL
BASE_URL = "https://raw.githubusercontent.com/davidcariboo/player-scores/master/data"

players = pd.read_csv(f"{BASE_URL}/players.csv")
transfers = pd.read_csv(f"{BASE_URL}/transfers.csv")
clubs = pd.read_csv(f"{BASE_URL}/clubs.csv")
appearances = pd.read_csv(f"{BASE_URL}/appearances.csv")
player_valuations = pd.read_csv(f"{BASE_URL}/player_valuations.csv")

print("✅ データのロードが完了しました！")

# ==========================================
# 3. 統合推論：【世界完全開放版】現役選手について世界中のクラブへの移籍をシミュレーション
# ==========================================
current_players = players[players['last_season'] >= 2025].copy()
current_players = pd.merge(current_players, clubs[['club_id', 'domestic_competition_id']],
                            left_on='current_club_id', right_on='club_id', how='left')

# 各選手の最新スナップショット(=現在の特徴量)を取得
latest_snap_idx = snapshots.sort_values('snapshot_date').groupby('player_id').tail(1)
infer_base = pd.merge(current_players[['player_id', 'name', 'position', 'country_of_citizenship',
                                        'date_of_birth', 'current_club_id', 'domestic_competition_id']],
                       latest_snap_idx[['player_id', 'recent_market_value', 'valuation_growth_rate',
                                        'total_minutes', 'total_goals', 'total_assists', 'total_games']],
                       on='player_id', how='inner')

infer_base['age'] = (TODAY.year - infer_base['date_of_birth'].dt.year) - \
                    ((TODAY.month < infer_base['date_of_birth'].dt.month) |
                     ((TODAY.month == infer_base['date_of_birth'].dt.month) &
                      (TODAY.day < infer_base['date_of_birth'].dt.day)))
infer_base = infer_base.rename(columns={'domestic_competition_id': 'player_club_domestic_competition_id'})

# Stage1: 移籍発生確率の予測
X1_infer = infer_base[stage1_features].copy()
for c in stage1_cat_cols:
    X1_infer[c] = X1_infer[c].astype('category')
infer_base['p_transfer'] = model_stage1.predict_proba(X1_infer)[:, 1]

# Stage1スコア上位30名（最も動く可能性が高い選手）を抽出
top_candidates = infer_base.sort_values('p_transfer', ascending=False).head(30)

# 🌟【世界開放ロジック】全クラブのリストと、国籍アフィニティのあるクラブを高速探索するための準備
all_club_ids = clubs['club_id'].unique()
club_to_league_dict = clubs.set_index('club_id')['domestic_competition_id'].to_dict()

results = []
print("🌍 世界中のクラブを対象に、海外移籍の総当たりシミュレーションを実行中...")

for row in top_candidates.itertuples(index=False):
    src_league = row.player_club_domestic_competition_id

    # 🌟【フィルターの破壊】国内リーグ限定をやめ、「世界中のすべてのクラブ」をターゲットにする
    # ただし、自分が今いるクラブ（current_club_id）は除外
    target_clubs = [c for c in all_club_ids if c != row.current_club_id]

    rows = []
    for c in target_clubs:
        dst_league = club_to_league_dict.get(c, 'Unknown')

        # 過去の取引実績（クラブ間コネ + リーグ間エージェントルート）
        # ★国境を越えてセルティックやシント＝トロイデンのような海外ルートがここで牙をむく！
        pipeline_power = conn_dict.get((row.current_club_id, c), 0) * 2 + agent_dict.get((src_league, c), 0)
        nat_affinity = affinity_dict.get((c, row.country_of_citizenship), 0)

        # 🚀 高速化のための枝切り：海外移籍において「過去に一度もその国籍の獲得実績がなく、
        #    かつクラブ・リーグ間のコネも完全に0」のクラブは、計算をスキップしてメモリを節約
        if dst_league != src_league and pipeline_power == 0 and nat_affinity == 0:
            continue

        rows.append({
            'age': row.age, 'market_value_in_eur': row.recent_market_value, 'position': row.position,
            'domestic_competition_id': dst_league, # 移籍先クラブの属するリーグID
            'squad_size': squad_size_dict.get(c, np.nan),
            'position_squad_size': pos_count_dict.get((c, row.position), 0),
            'country_of_citizenship': row.country_of_citizenship,
            'total_pipeline_power': pipeline_power,
            'club_nationality_affinity': nat_affinity,
            'total_minutes': row.total_minutes, 'total_goals': row.total_goals,
            'total_assists': row.total_assists, 'total_games': row.total_games,
            'recent_market_value': row.recent_market_value, 'valuation_growth_rate': row.valuation_growth_rate,
            'to_club_id': c
        })

    if not rows:
        continue

    cand_df = pd.DataFrame(rows)
    X2_infer = cand_df[stage2_features].copy()
    for c_ in stage2_cat_cols:
        X2_infer[c_] = X2_infer[c_].astype('category')

# ==========================================
# 2. Stage2モデルの学習：「行き先クラブはどこか」（🌟格の壁を実装🌟）
# ==========================================
# 各クラブの「総市場価値」と「平均市場価値（格の指標）」を計算
club_market_profile = players.groupby('current_club_id').agg(
    total_club_value=('market_value_in_eur', 'sum'),
    avg_club_value=('market_value_in_eur', 'mean')
).reset_index()

club_pos_counts = players.groupby(['current_club_id', 'position']).size().reset_index(name='position_squad_size')
club_connection = transfers.groupby(['from_club_id', 'to_club_id']).size().reset_index(name='club_connection_score')
transfers_with_src_league = pd.merge(transfers, clubs[['club_id', 'domestic_competition_id']],
                                      left_on='from_club_id', right_on='club_id', how='left') \
    .rename(columns={'domestic_competition_id': 'from_league_id'})
agent_pipeline = transfers_with_src_league.groupby(['from_league_id', 'to_club_id']).size().reset_index(name='agent_pipeline_score')
transfers_with_nat = pd.merge(transfers, players[['player_id', 'country_of_citizenship']], on='player_id', how='inner')
club_nat_affinity = transfers_with_nat.groupby(['to_club_id', 'country_of_citizenship']).size().reset_index(name='club_nationality_affinity')

merged_df = pd.merge(transfers, players, on="player_id", how="inner")
merged_df['age'] = (merged_df['transfer_date'].dt.year - merged_df['date_of_birth'].dt.year) - \
                   ((merged_df['transfer_date'].dt.month < merged_df['date_of_birth'].dt.month) |
                    ((merged_df['transfer_date'].dt.month == merged_df['date_of_birth'].dt.month) &
                     (merged_df['transfer_date'].dt.day < merged_df['date_of_birth'].dt.day)))
predict_base = merged_df[['player_id', 'name', 'position', 'age', 'market_value_in_eur_x', 'from_club_id',
                           'to_club_id', 'transfer_fee', 'country_of_citizenship', 'transfer_date']] \
    .rename(columns={'market_value_in_eur_x': 'market_value_in_eur'})
predict_base = pd.merge(predict_base, clubs[['club_id', 'domestic_competition_id']],
                         left_on='from_club_id', right_on='club_id', how='left') \
    .rename(columns={'domestic_competition_id': 'from_league_id'}).drop('club_id', axis=1)
final_df = pd.merge(predict_base, clubs[['club_id', 'name', 'domestic_competition_id', 'squad_size']],
                     left_on='to_club_id', right_on='club_id', how='left')
final_df = attach_career_stats(final_df, 'transfer_date')
final_df = attach_valuation_momentum(final_df, 'transfer_date')
final_df = pd.merge(final_df, club_connection, on=['from_club_id', 'to_club_id'], how='left')
final_df = pd.merge(final_df, agent_pipeline, on=['from_league_id', 'to_club_id'], how='left')
final_df = pd.merge(final_df, club_nat_affinity, on=['to_club_id', 'country_of_citizenship'], how='left')

# 移籍元と移籍先のクラブの平均市場価値を紐付け
final_df = pd.merge(final_df, club_market_profile.rename(columns={'current_club_id': 'from_club_id', 'avg_club_value': 'from_avg_value'}), on='from_club_id', how='left')
final_df = pd.merge(final_df, club_market_profile.rename(columns={'current_club_id': 'to_club_id', 'avg_club_value': 'to_avg_value'}), on='to_club_id', how='left')
final_df['from_avg_value'] = final_df['from_avg_value'].fillna(1)
final_df['to_avg_value'] = final_df['to_avg_value'].fillna(1)

# 🌟 新特徴量：クラブの格のギャップ（移籍先 / 移籍元）
final_df['club_status_gap'] = final_df['to_avg_value'] / final_df['from_avg_value']

fill_cols2 = ['total_minutes', 'total_goals', 'total_assists', 'total_games', 'club_connection_score',
              'agent_pipeline_score', 'club_nationality_affinity', 'recent_market_value', 'valuation_growth_rate']
final_df[fill_cols2] = final_df[fill_cols2].fillna(0)
final_df['total_pipeline_power'] = final_df['club_connection_score'] * 2 + final_df['agent_pipeline_score']
final_df['is_transfer'] = np.where(final_df['transfer_fee'].notna(), 1, 0)

positives = final_df[final_df['is_transfer'] == 1].copy()
positives = pd.merge(positives, club_pos_counts, left_on=['to_club_id', 'position'],
                      right_on=['current_club_id', 'position'], how='left').drop('current_club_id', axis=1)

league_to_clubs = clubs.groupby('domestic_competition_id')['club_id'].apply(list).to_dict()
pos_count_dict = {(r.current_club_id, r.position): r.position_squad_size for r in club_pos_counts.itertuples()}
conn_dict = {(r.from_club_id, r.to_club_id): r.club_connection_score for r in club_connection.itertuples()}
agent_dict = {(r.from_league_id, r.to_club_id): r.agent_pipeline_score for r in agent_pipeline.itertuples()}
affinity_dict = {(r.to_club_id, r.country_of_citizenship): r.club_nationality_affinity for r in club_nat_affinity.itertuples()}
squad_size_dict = clubs.set_index('club_id')['squad_size'].to_dict()
club_avg_val_dict = club_market_profile.set_index('current_club_id')['avg_club_value'].to_dict()

rng = np.random.default_rng(42)
negatives_list = []
for row in positives.itertuples(index=False):
    candidates = [c for c in league_to_clubs.get(row.domestic_competition_id, []) if c != row.to_club_id]
    if len(candidates) >= 3:
        for s_club_id in rng.choice(candidates, size=3, replace=False):
            from_v = club_avg_val_dict.get(row.from_club_id, 1)
            to_v = club_avg_val_dict.get(s_club_id, 1)

            negatives_list.append({
                'age': row.age, 'market_value_in_eur': row.market_value_in_eur, 'position': row.position,
                'domestic_competition_id': row.domestic_competition_id,
                'squad_size': squad_size_dict.get(s_club_id, np.nan),
                'position_squad_size': pos_count_dict.get((s_club_id, row.position), 0),
                'country_of_citizenship': row.country_of_citizenship,
                'total_pipeline_power': conn_dict.get((row.from_club_id, s_club_id), 0) * 2
                                        + agent_dict.get((row.from_league_id, s_club_id), 0),
                'club_nationality_affinity': affinity_dict.get((s_club_id, row.country_of_citizenship), 0),
                'club_status_gap': to_v / from_v if from_v > 0 else 1, # 🌟 偽物にも格のギャップを仕込む
                'total_minutes': row.total_minutes, 'total_goals': row.total_goals,
                'total_assists': row.total_assists, 'total_games': row.total_games,
                'recent_market_value': row.recent_market_value,
                'valuation_growth_rate': row.valuation_growth_rate, 'is_transfer': 0
            })
negatives = pd.DataFrame(negatives_list)

stage2_features = ['age', 'market_value_in_eur', 'position', 'domestic_competition_id', 'squad_size',
                    'position_squad_size', 'country_of_citizenship', 'total_pipeline_power',
                    'club_nationality_affinity', 'club_status_gap', 'total_minutes', 'total_goals', 'total_assists',
                    'total_games', 'recent_market_value', 'valuation_growth_rate']
balanced_df = pd.concat([positives[stage2_features + ['is_transfer']],
                          negatives[stage2_features + ['is_transfer']]], axis=0).reset_index(drop=True)
balanced_df['position_squad_size'] = balanced_df['position_squad_size'].fillna(0)

stage2_cat_cols = ['position', 'domestic_competition_id', 'country_of_citizenship']
X2 = balanced_df[stage2_features].copy()
for c in stage2_cat_cols:
    X2[c] = X2[c].astype('category')
y2 = balanced_df['is_transfer']

X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

model_stage2 = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
model_stage2.fit(X2_tr, y2_tr)
print(f"Stage2 Test AUC (With Status Gap): {roc_auc_score(y2_te, model_stage2.predict_proba(X2_te)[:, 1]):.4f}")


# ==========================================
# 3. 統合推論：【世界完全開放＋格の壁モデル】
# ==========================================
current_players = players[players['last_season'] >= 2025].copy()
current_players = pd.merge(current_players, clubs[['club_id', 'domestic_competition_id']],
                            left_on='current_club_id', right_on='club_id', how='left')

latest_snap_idx = snapshots.sort_values('snapshot_date').groupby('player_id').tail(1)
infer_base = pd.merge(current_players[['player_id', 'name', 'position', 'country_of_citizenship',
                                        'date_of_birth', 'current_club_id', 'domestic_competition_id']],
                       latest_snap_idx[['player_id', 'recent_market_value', 'valuation_growth_rate',
                                        'total_minutes', 'total_goals', 'total_assists', 'total_games']],
                       on='player_id', how='inner')

infer_base['age'] = (TODAY.year - infer_base['date_of_birth'].dt.year) - \
                    ((TODAY.month < infer_base['date_of_birth'].dt.month) |
                     ((TODAY.month == infer_base['date_of_birth'].dt.month) &
                      (TODAY.day < infer_base['date_of_birth'].dt.day)))
infer_base = infer_base.rename(columns={'domestic_competition_id': 'player_club_domestic_competition_id'})

X1_infer = infer_base[stage1_features].copy()
for c in stage1_cat_cols:
    X1_infer[c] = X1_infer[c].astype('category')
infer_base['p_transfer'] = model_stage1.predict_proba(X1_infer)[:, 1]

top_candidates = infer_base.sort_values('p_transfer', ascending=False).head(30)

all_club_ids = clubs['club_id'].unique()
club_to_league_dict = clubs.set_index('club_id')['domestic_competition_id'].to_dict()
club_name_map = clubs.set_index('club_id')['name'].to_dict()

results = []

for row in top_candidates.itertuples(index=False):
    src_league = row.player_club_domestic_competition_id
    target_clubs = [c for c in all_club_ids if c != row.current_club_id]

    rows = []
    for c in target_clubs:
        dst_league = club_to_league_dict.get(c, 'Unknown')
        pipeline_power = conn_dict.get((row.current_club_id, c), 0) * 2 + agent_dict.get((src_league, c), 0)
        nat_affinity = affinity_dict.get((c, row.country_of_citizenship), 0)

        # 🌟 現実的な計算のための微細な枝切りは維持
        if dst_league != src_league and pipeline_power == 0 and nat_affinity == 0:
            continue

        from_v = club_avg_val_dict.get(row.current_club_id, 1)
        to_v = club_avg_val_dict.get(c, 1)
        status_gap = to_v / from_v if from_v > 0 else 1

        rows.append({
            'age': row.age, 'market_value_in_eur': row.recent_market_value, 'position': row.position,
            'domestic_competition_id': dst_league,
            'squad_size': squad_size_dict.get(c, np.nan),
            'position_squad_size': pos_count_dict.get((c, row.position), 0),
            'country_of_citizenship': row.country_of_citizenship,
            'total_pipeline_power': pipeline_power,
            'club_nationality_affinity': nat_affinity,
            'club_status_gap': status_gap, # 🌟 予測時にも格のギャップを入力
            'total_minutes': row.total_minutes, 'total_goals': row.total_goals,
            'total_assists': row.total_assists, 'total_games': row.total_games,
            'recent_market_value': row.recent_market_value, 'valuation_growth_rate': row.valuation_growth_rate,
            'to_club_id': c
        })

    if not rows:
        continue

    cand_df = pd.DataFrame(rows)
    X2_infer = cand_df[stage2_features].copy()
    for c_ in stage2_cat_cols:
        X2_infer[c_] = X2_infer[c_].astype('category')

    raw_scores = model_stage2.predict_proba(X2_infer)[:, 1]
    exp_scores = np.exp(raw_scores * 5)
    p_destination = exp_scores / exp_scores.sum()

    cand_df['p_destination'] = p_destination
    cand_df['p_transfer'] = row.p_transfer
    cand_df['combined_score'] = cand_df['p_transfer'] * cand_df['p_destination']

    top3 = cand_df.sort_values('combined_score', ascending=False).head(3)
    for t in top3.itertuples(index=False):
        results.append({
            'player_id': row.player_id, 'player_name': row.name, 'p_transfer': round(row.p_transfer, 3),
            'to_club_id': t.to_club_id, 'p_destination_given_transfer': round(t.p_destination, 3),
            'combined_score': round(t.combined_score, 4)
        })

results_df = pd.DataFrame(results)
results_df['to_club_name'] = results_df['to_club_id'].map(club_name_map)
results_df = results_df[['player_name', 'p_transfer', 'to_club_name', 'p_destination_given_transfer', 'combined_score']]
results_df.to_csv('/content/transfer_predictions.csv', index=False)

print("\n" + "="*80)
print(" 👑 【世界開放×格の壁モデル】統合移籍予測シミュレーション結果 👑")
print("="*80)
print(results_df.head(30).to_string(index=False))
