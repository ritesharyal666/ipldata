import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

st.set_page_config(layout="wide", page_title="IPL Match Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("deliveries.csv.gz", compression='gzip')
    return df

df = load_data()

# IPL matches per season with playoff structure (corrected 2008 playoffs)
matches_per_season = {
    2008: {"total_matches": 58, "playoffs": ["Final","Semifinal 2","Semifinal 1"]},
    2009: {"total_matches": 57, "playoffs": ["Final", "Semifinal 2", "Semifinal 1"]},
    2010: {"total_matches": 59, "playoffs": ["Final", "Eliminator", "Qualifier 2 ", "Qualifier 1"]},
    2011: {"total_matches": 72, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2012: {"total_matches": 76, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2013: {"total_matches": 76, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2014: {"total_matches": 60, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2015: {"total_matches": 59, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2016: {"total_matches": 60, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2017: {"total_matches": 59, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2018: {"total_matches": 60, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2019: {"total_matches": 60, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2020: {"total_matches": 60, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2021: {"total_matches": 60, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2022: {"total_matches": 74, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2023: {"total_matches": 74, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]},
    2024: {"total_matches": 71, "playoffs": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1"]}
}
# Map matches to seasons and match types
season_match_info = {}
match_ids = sorted(df['match_id'].unique())
idx = 0

for year, info in matches_per_season.items():
    total_matches = info["total_matches"]
    playoffs = info["playoffs"]
    
    season_match_info[year] = {}
    
    # Assign league matches first
    for i in range(total_matches - len(playoffs)):
        if idx >= len(match_ids):
            break
        mid = match_ids[idx]
        season_match_info[year][mid] = "League"
        idx += 1
    
    # Assign playoff matches in reverse order (so last match is Final)
    for match_type in reversed(playoffs):
        if idx >= len(match_ids):
            break
        mid = match_ids[idx]
        season_match_info[year][mid] = match_type
        idx += 1

    if idx >= len(match_ids):
        break

def get_match_display_names(df, season_match_info):
    display_names = {}
    for season, matches in season_match_info.items():
        for mid, mtype in matches.items():
            sub = df[df['match_id'] == mid]
            teams = list(sub['batting_team'].dropna().unique())
            teams_sorted = sorted(teams)
            if len(teams_sorted) >= 2:
                display_name = f"{season} | {mtype} | {teams_sorted[0]} vs {teams_sorted[1]}"
            elif len(teams_sorted) == 1:
                display_name = f"{season} | {mtype} | {teams_sorted[0]} vs Unknown"
            else:
                display_name = f"{season} | {mtype} | Unknown vs Unknown"
            display_names[mid] = display_name
    return display_names

match_display_names = get_match_display_names(df, season_match_info)

def get_season_for_match(mid):
    for season, matches in season_match_info.items():
        if mid in matches:
            return season
    return None

def batting_order(inning_df):
    valid_balls = inning_df[~inning_df['extras_type'].isin(['wides', 'noballs'])]
    order = (
        valid_balls.groupby('batter')[['over', 'ball']]
        .min()
        .reset_index()
        .sort_values(by=['over', 'ball'])
    )
    return order['batter'].tolist()

def batting_figures(inning_df):
    valid_balls = inning_df[~inning_df['extras_type'].isin(['wides', 'noballs'])]
    
    batsman_stats = valid_balls.groupby('batter').agg(
        Runs=('batsman_runs', 'sum'),
        Balls=('ball', 'count'),
        Fours=('batsman_runs', lambda x: (x==4).sum()),
        Sixes=('batsman_runs', lambda x: (x==6).sum())
    ).reset_index()

    batsman_stats['SR'] = (batsman_stats['Runs'] / batsman_stats['Balls'] * 100).round(2)
    
    dismissals = inning_df.dropna(subset=['player_dismissed']).copy()
    dismissals['Dismissal'] = dismissals.apply(lambda x: 
        f"b {x['bowler']}" if x['dismissal_kind'] == 'bowled' else
        f"c {x['fielder']} b {x['bowler']}" if x['dismissal_kind'] in ['caught','caught and bowled'] else
        f"lbw b {x['bowler']}" if x['dismissal_kind'] == 'lbw' else
        f"st {x['fielder']} b {x['bowler']}" if x['dismissal_kind'] == 'stumped' else
        f"run out ({x['fielder']})" if x['dismissal_kind'] == 'run out' else
        x['dismissal_kind'], axis=1)
    
    batsman_stats = batsman_stats.merge(
        dismissals[['player_dismissed', 'Dismissal']].rename(columns={'player_dismissed':'batter'}),
        on='batter', how='left')
    batsman_stats['Dismissal'].fillna('not out', inplace=True)

    order = batting_order(inning_df)
    batsman_stats = batsman_stats.set_index('batter').loc[order].reset_index()
    return batsman_stats[['batter', 'Runs', 'Balls', 'Fours', 'Sixes', 'SR', 'Dismissal']]

def bowling_figures(inning_df):
    valid_balls = inning_df[~inning_df['extras_type'].isin(['wides'])]
    
    bowling_stats = valid_balls.groupby('bowler').agg(
        Balls=('ball', 'count'),
        Runs=('total_runs', 'sum'),
        Maidens=('over', lambda x: x.nunique() if (valid_balls[valid_balls['bowler']==x.name]
                                               .groupby('over')['total_runs'].sum() == 0).any() else 0)
    ).reset_index()
    
    wicket_types = ['bowled','caught','caught and bowled','lbw','hit wicket','stumped']
    wickets = inning_df[
        (inning_df['is_wicket'] == 1) & 
        (inning_df['dismissal_kind'].isin(wicket_types))
    ].groupby('bowler').size().reset_index(name='Wickets')
    
    bowling_stats = bowling_stats.merge(wickets, on='bowler', how='left')
    bowling_stats['Wickets'] = bowling_stats['Wickets'].fillna(0).astype(int)
    
    bowling_stats['Overs'] = bowling_stats['Balls'].apply(
        lambda x: f"{x//6}.{x%6}" if x%6 !=0 else f"{x//6}.0")
    bowling_stats['Econ'] = (bowling_stats['Runs'] / (bowling_stats['Balls']/6)).round(2)
    
    return bowling_stats[['bowler', 'Overs', 'Maidens', 'Runs', 'Wickets', 'Econ']].sort_values(
        ['Wickets', 'Econ'], ascending=[False, True])

def plot_over_comparison(inning1_df, inning2_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    team1 = inning1_df['batting_team'].iloc[0]
    team2 = inning2_df['batting_team'].iloc[0]
    
    # Inning 1
    runs1 = inning1_df.groupby('over')['total_runs'].sum().reindex(range(1,21), fill_value=0)
    ax.bar(runs1.index-0.2, runs1.values, width=0.4, color='#1f77b4', label=team1)
    
    # Inning 2
    runs2 = inning2_df.groupby('over')['total_runs'].sum().reindex(range(1,21), fill_value=0)
    ax.bar(runs2.index+0.2, runs2.values, width=0.4, color='#ff7f0e', label=team2)
    
    # Powerplay shading
    ax.axvspan(0.5, 6.5, alpha=0.1, color='green', label='Powerplay')
    
    # Add run values on bars
    for i, val in enumerate(runs1):
        ax.text(i+1-0.2, val+0.5, str(val), ha='center')
    for i, val in enumerate(runs2):
        ax.text(i+1+0.2, val+0.5, str(val), ha='center')
    
    ax.set_xlabel('Over')
    ax.set_ylabel('Runs')
    ax.set_title(f'Over-by-Over Comparison: {team1} vs {team2}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    st.pyplot(fig)

def match_summary(match_df):
    innings = match_df['inning'].unique()
    innings.sort()
    
    team1 = match_df[match_df['inning'] == 1]['batting_team'].iloc[0]
    team1_runs = match_df[match_df['inning'] == 1]['total_runs'].sum()
    team1_wickets = match_df[(match_df['inning'] == 1) & 
                           (match_df['player_dismissed'].notna())].shape[0]
    
    if 2 in innings:
        team2 = match_df[match_df['inning'] == 2]['batting_team'].iloc[0]
        team2_runs = match_df[match_df['inning'] == 2]['total_runs'].sum()
        team2_wickets = match_df[(match_df['inning'] == 2) & 
                               (match_df['player_dismissed'].notna())].shape[0]
        target = team1_runs + 1
    else:
        team2 = None
        team2_runs = None
        team2_wickets = None
        target = None

    balls_bowled = len(match_df[(match_df['inning']==2)])
    overs_bowled = balls_bowled // 6 + (balls_bowled % 6)/6 if balls_bowled > 0 else 0
    crr = (team2_runs / overs_bowled) if overs_bowled > 0 else 0

    balls_remaining = 120 - balls_bowled
    runs_remaining = target - team2_runs if target is not None else 0
    rrr = (runs_remaining / (balls_remaining / 6)) if balls_remaining > 0 else 0

    return {
        'team1': team1,
        'team1_runs': team1_runs,
        'team1_wickets': team1_wickets,
        'team2': team2,
        'team2_runs': team2_runs,
        'team2_wickets': team2_wickets,
        'target': target,
        'crr': crr,
        'rrr': rrr
    }

def season_stats(season_df):
    # Most runs
    batsman_runs = season_df.groupby('batter')['batsman_runs'].sum()
    top_batsman = batsman_runs.idxmax()
    top_runs = batsman_runs.max()
    
    # Strike rate
    valid_balls = season_df[~season_df['extras_type'].isin(['wides', 'noballs'])]
    balls_faced = valid_balls[valid_balls['batter'] == top_batsman].shape[0]
    strike_rate = round((top_runs / balls_faced) * 100, 2) if balls_faced > 0 else 0
    
    # Sixes/Fours
    sixes = season_df[(season_df['batter'] == top_batsman) & (season_df['batsman_runs'] == 6)].shape[0]
    fours = season_df[(season_df['batter'] == top_batsman) & (season_df['batsman_runs'] == 4)].shape[0]
    
    # Most wickets
    wicket_types = ['bowled', 'caught', 'caught and bowled', 'hit wicket', 'lbw', 'stumped']
    wickets = (
        season_df[
            (season_df['is_wicket'] == 1) & 
            (season_df['dismissal_kind'].isin(wicket_types))
        ]
        .groupby('bowler')
        .size()
    )
    top_bowler = wickets.idxmax()
    top_wickets = wickets.max()
    
    # Economy
    valid_balls = season_df[~season_df['extras_type'].isin(['wides'])]
    balls_bowled = valid_balls[valid_balls['bowler'] == top_bowler].shape[0]
    runs_given = season_df[season_df['bowler'] == top_bowler]['total_runs'].sum()
    economy = round(runs_given / (balls_bowled / 6), 2) if balls_bowled > 0 else 0
    
    # Fielding stats
    catches = season_df[
        (season_df['is_wicket'] == 1) & 
        (season_df['dismissal_kind'].isin(['caught', 'caught and bowled']))
    ].groupby('fielder').size()
    top_catcher = catches.idxmax() if not catches.empty else "N/A"
    top_catches = catches.max() if not catches.empty else 0
    
    stumpings = season_df[
        (season_df['is_wicket'] == 1) & 
        (season_df['dismissal_kind'] == 'stumped')
    ].groupby('fielder').size()
    top_stumper = stumpings.idxmax() if not stumpings.empty else "N/A"
    top_stumpings = stumpings.max() if not stumpings.empty else 0
    
    # Most sixes/fours
    sixes_all = season_df[season_df['batsman_runs'] == 6].groupby('batter').size()
    top_sixer = sixes_all.idxmax()
    top_sixes = sixes_all.max()
    
    fours_all = season_df[season_df['batsman_runs'] == 4].groupby('batter').size()
    top_fourer = fours_all.idxmax()
    top_fours = fours_all.max()
    
    return {
        'top_batsman': top_batsman,
        'top_runs': top_runs,
        'strike_rate': strike_rate,
        'sixes': sixes,
        'fours': fours,
        'top_bowler': top_bowler,
        'top_wickets': top_wickets,
        'economy': economy,
        'top_catcher': top_catcher,
        'top_catches': top_catches,
        'top_stumper': top_stumper,
        'top_stumpings': top_stumpings,
        'top_sixer': top_sixer,
        'top_sixes': top_sixes,
        'top_fourer': top_fourer,
        'top_fours': top_fours
    }

# Streamlit UI
st.title("📊 IPL Match Dashboard")

selected_display_name = st.selectbox(
    "Select Match (Season | Type | Teams)",
    options=[match_display_names[mid] for mid in match_display_names],
    index=0
)

# Get selected match_id
selected_match_id = None
for mid, name in match_display_names.items():
    if name == selected_display_name:
        selected_match_id = mid
        break

if selected_match_id is None:
    st.error("Invalid match selected")
    st.stop()

season = get_season_for_match(selected_match_id)
season_df = df[df['match_id'].isin(season_match_info[season].keys())]
match_df = df[df['match_id'] == selected_match_id]

match_type = season_match_info[season][selected_match_id]
st.subheader(f"🏏 {season} | {match_type}")

summary = match_summary(match_df)
team1 = summary['team1']
team2 = summary['team2']

# Display scores at the top
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**{team1}**  \n{summary['team1_runs']}/{summary['team1_wickets']}")
with col2:
    if team2:
        st.markdown(f"**{team2}**  \n{summary['team2_runs']}/{summary['team2_wickets']}")

if summary['target'] is not None:
    st.markdown(f"🎯 **Target:** {summary['target']} ")

# Display innings data
innings = sorted(match_df['inning'].unique())
if len(innings) == 2:
    plot_over_comparison(
        match_df[match_df['inning']==1], 
        match_df[match_df['inning']==2]
    )

for i in innings:
    inning_df = match_df[match_df['inning'] == i]
    team = inning_df['batting_team'].iloc[0]
    
    st.markdown(f"### {team} Batting")
    st.dataframe(
        batting_figures(inning_df).style.format({'SR': '{:.2f}'}),
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown(f"### {team} Bowling")
    st.dataframe(
        bowling_figures(inning_df),
        hide_index=True,
        use_container_width=True
    )

# Season stats
st.markdown("---")
st.header(f"🌟 {season} Season Highlights")

stats = season_stats(season_df)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Top Run Scorer:** {stats['top_batsman']} ({stats['top_runs']} runs, SR: {stats['strike_rate']}%)")
    st.markdown(f"**Most Sixes:** {stats['top_sixer']} ({stats['top_sixes']} sixes)")
    st.markdown(f"**Most Fours:** {stats['top_fourer']} ({stats['top_fours']} fours)")

with col2:
    st.markdown(f"**Top Wicket Taker:** {stats['top_bowler']} ({stats['top_wickets']} wickets, Economy: {stats['economy']})")
    st.markdown(f"**Most Catches:** {stats['top_catcher']} ({stats['top_catches']} catches)")
    st.markdown(f"**Most Stumpings:** {stats['top_stumper']} ({stats['top_stumpings']} stumpings)")
