"""
Crypto AI Dashboard v2 — Professional Analytics Platform
Tabs: Live Analytics | 📊 Backtesting
New: News panel, news sentiment chart, last-updated timestamp
"""

import streamlit as st
import pymongo
import pandas as pd
from datetime import datetime, timedelta
import time

from config import settings
from learning_engine import get_indicator_stats, get_current_weights, run_learning_cycle

st.set_page_config(page_title="Crypto AI Platform", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.main {padding: 0.5rem 1.5rem;}
h1 {text-align: center;}
.regime-bull {background: linear-gradient(90deg, #1b5e20, #388e3c); color: white; padding: 12px 20px;
              border-radius: 8px; text-align: center; font-size: 18px; margin-bottom: 16px;}
.regime-bear {background: linear-gradient(90deg, #b71c1c, #d32f2f); color: white; padding: 12px 20px;
              border-radius: 8px; text-align: center; font-size: 18px; margin-bottom: 16px;}
.regime-sideways {background: linear-gradient(90deg, #e65100, #ff9800); color: white; padding: 12px 20px;
                  border-radius: 8px; text-align: center; font-size: 18px; margin-bottom: 16px;}
.top-card {background: #1a1a2e; border: 1px solid #16213e; border-radius: 10px;
           padding: 16px; color: white; text-align: center;}
.metric-card {background: #0d1b2a; border: 1px solid #1b263b; border-radius: 10px;
              padding: 20px; color: white; text-align: center; margin-bottom: 8px;}
.metric-value {font-size: 28px; font-weight: bold; margin: 4px 0;}
.metric-label {font-size: 13px; color: #aaa;}
.win {color: #4caf50;} .loss {color: #f44336;} .neutral {color: #ffc107;}
.news-card {background: #12172b; border-left: 3px solid #4caf50; border-radius: 6px;
            padding: 10px 14px; margin: 6px 0; color: #ddd;}
.news-card-neg {background: #12172b; border-left: 3px solid #f44336; border-radius: 6px;
                padding: 10px 14px; margin: 6px 0; color: #ddd;}
.news-card-neu {background: #12172b; border-left: 3px solid #ffc107; border-radius: 6px;
                padding: 10px 14px; margin: 6px 0; color: #ddd;}
.ts-badge {background: #1e2a40; border-radius: 4px; padding: 4px 10px;
           font-size: 12px; color: #90caf9; display: inline-block;}
/* Ranking cards */
.rank-gold   {background: linear-gradient(135deg,#2a1f00,#3d2c00); border: 2px solid #ffd700;
              border-radius: 12px; padding: 18px; color: white; text-align: center; margin-bottom: 8px;}
.rank-silver {background: linear-gradient(135deg,#1a1a1a,#2d2d2d); border: 2px solid #c0c0c0;
              border-radius: 12px; padding: 18px; color: white; text-align: center; margin-bottom: 8px;}
.rank-bronze {background: linear-gradient(135deg,#1a0d00,#2a1500); border: 2px solid #cd7f32;
              border-radius: 12px; padding: 18px; color: white; text-align: center; margin-bottom: 8px;}
.rank-normal {background: #12172b; border: 1px solid #283048; border-radius: 12px;
              padding: 18px; color: white; text-align: center; margin-bottom: 8px;}
.conf-bar-wrap {background:#1e2a40; border-radius:4px; height:10px; width:100%; margin:4px 0;}
.conf-bar      {background:linear-gradient(90deg,#4caf50,#00e676); border-radius:4px; height:10px;}
.conf-bar-med  {background:linear-gradient(90deg,#ff9800,#ffcc02); border-radius:4px; height:10px;}
.conf-bar-low  {background:linear-gradient(90deg,#f44336,#ff5722); border-radius:4px; height:10px;}
</style>
""", unsafe_allow_html=True)


def get_db_connection():
    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    return client, client[settings.DATABASE_NAME]


@st.cache_data(ttl=60)
def load_signals():
    """Load latest signal per symbol and merge ranked_opportunities for top_rank field"""
    try:
        client, db = get_db_connection()
        # idx_signal_lookup index on (symbol, timestamp DESC) handles sort with no RAM limit
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$symbol',
                'doc': {'$first': '$$ROOT'}
            }},
            {'$replaceRoot': {'newRoot': '$doc'}},
            {'$sort': {'final_score': -1}}
        ]
        signals = list(db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))

        # Merge top_rank from ranked_opportunities (stored separately by ranking engine)
        try:
            ranked = list(db['ranked_opportunities'].find(
                {}, {'symbol': 1, 'top_rank': 1, 'rank_score': 1, 'opportunity_score': 1, '_id': 0}
            ))
            rank_map = {r['symbol']: r for r in ranked}
            for s in signals:
                sym = s.get('symbol', '')
                if sym in rank_map:
                    s['top_rank'] = rank_map[sym].get('top_rank')
                    s['rank_score'] = rank_map[sym].get('rank_score', s.get('rank_score', 0))
                    s['opportunity_score'] = rank_map[sym].get('opportunity_score', s.get('opportunity_score', 0))
        except Exception:
            pass  # ranking merge is optional — signals still show without it

        client.close()
        return pd.DataFrame(signals) if signals else pd.DataFrame()
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_backtest_result():
    try:
        client, db = get_db_connection()
        doc = db['backtest_results'].find_one({}, sort=[('timestamp', pymongo.DESCENDING)])
        client.close()
        return doc
    except:
        return None


@st.cache_data(ttl=120)
def load_news_for_symbol(symbol: str, limit: int = 8):
    """Load recent news headlines from news_data collection"""
    try:
        currency = symbol.replace('USDT', '').replace('USD', '')
        client, db = get_db_connection()
        docs = list(db[settings.COLLECTION_NEWS_DATA].find(
            {'symbol': currency}
        ).sort('published_at', pymongo.DESCENDING).limit(limit))
        client.close()
        return docs
    except:
        return []


@st.cache_data(ttl=120)
def load_news_sentiment_history(symbols: list, hours: int = 72):
    """Load news sentiment scores over time for charting"""
    try:
        client, db = get_db_connection()
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        docs = list(db[settings.COLLECTION_NEWS_DATA].find(
            {'symbol': {'$in': symbols}, 'published_at': {'$gte': cutoff}}
        ).sort('published_at', pymongo.ASCENDING))
        client.close()
        return docs
    except:
        return []


@st.cache_data(ttl=30)
def load_ranked_opportunities(limit: int = 20):
    """Load latest ranked opportunities from MongoDB (ranked_opportunities collection)"""
    try:
        client, db = get_db_connection()
        docs = list(
            db['ranked_opportunities']
            .find({}, {'_id': 0})
            .sort('rank_score', pymongo.DESCENDING)
            .limit(limit)
        )
        client.close()
        return pd.DataFrame(docs) if docs else pd.DataFrame()
    except Exception as e:
        st.error(f"Ranking DB Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_signal_db_stats():
    """Get signal count and latest scan timestamp from DB.
    Uses ranked_opportunities batch_ts as the primary 'Last Signal' time
    because ranked_opportunities is updated every 5-min scan cycle.
    Falls back to ai_signals timestamp if unavailable.
    """
    try:
        client, db = get_db_connection()
        sig_count  = db[settings.COLLECTION_AI_SIGNALS].count_documents({})
        news_count = db[settings.COLLECTION_NEWS_DATA].count_documents({})

        # Primary: ranked_opportunities batch_ts = true last-scan time
        ranked_latest = db['ranked_opportunities'].find_one(
            {}, {'batch_ts': 1}, sort=[('batch_ts', pymongo.DESCENDING)]
        )
        # Fallback: ai_signals timestamp
        sig_latest = db[settings.COLLECTION_AI_SIGNALS].find_one(
            {}, {'timestamp': 1}, sort=[('timestamp', pymongo.DESCENDING)]
        )

        ts_ranked = ranked_latest.get('batch_ts') if ranked_latest else None
        ts_sig    = sig_latest.get('timestamp')   if sig_latest    else None

        # Use the most recent of the two
        def _clean(ts):
            if ts and hasattr(ts, 'tzinfo') and ts.tzinfo:
                return ts.replace(tzinfo=None)
            return ts

        ts_ranked_c = _clean(ts_ranked)
        ts_sig_c    = _clean(ts_sig)

        if ts_ranked_c and ts_sig_c:
            latest_ts = max(ts_ranked_c, ts_sig_c)
        else:
            latest_ts = ts_ranked_c or ts_sig_c

        client.close()
        return {'signal_count': sig_count, 'latest_ts': latest_ts, 'news_count': news_count}
    except Exception:
        return {'signal_count': 0, 'latest_ts': None, 'news_count': 0}


@st.cache_data(ttl=60)
def load_learning_data():
    """Load latest learning engine data: weights + indicator stats + weight history."""
    try:
        client, db = get_db_connection()
        # Current weights document
        weights_doc = db['model_weights'].find_one(
            {'type': 'current'},
            sort=[('cycle_ts', -1)],
            projection={'_id': 0}
        )
        # Indicator stats sorted by win rate
        stats = list(db['indicator_stats'].find({}, {'_id': 0}).sort('win_rate', -1))
        # Last 10 backup snapshots for the history table
        history = list(
            db['model_weights']
            .find({'type': 'backup'}, {'_id': 0, 'weights': 1, 'cycle_ts': 1})
            .sort('cycle_ts', -1)
            .limit(10)
        )
        client.close()
        return {
            'weights_doc':     weights_doc,
            'indicator_stats': stats,
            'weight_history':  history,
        }
    except Exception:
        return {'weights_doc': None, 'indicator_stats': [], 'weight_history': []}


@st.cache_data(ttl=120)
def load_regime_history(days: int = 7):
    """Load regime history for the last N days for chart display."""
    try:
        client, db = get_db_connection()
        since = datetime.utcnow() - timedelta(days=days)
        docs = list(
            db['regime_history']
            .find({'detected_at': {'$gte': since}}, {'_id': 0})
            .sort('detected_at', 1)
        )
        client.close()
        return docs
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_strategy_config():
    """Load current active strategy config and recent config history."""
    try:
        client, db = get_db_connection()
        active = db['strategy_configs'].find_one(
            {'active': True}, {'_id': 0}, sort=[('created_at', pymongo.DESCENDING)]
        )
        recent = list(
            db['strategy_configs']
            .find({}, {'_id': 0})
            .sort('created_at', pymongo.DESCENDING)
            .limit(5)
        )
        opt_log = list(
            db['optimization_log']
            .find({}, {'_id': 0})
            .sort('run_at', pymongo.DESCENDING)
            .limit(3)
        )
        regime_count = db['regime_history'].count_documents({})
        client.close()
        return {
            'active': active,
            'recent': recent,
            'opt_log': opt_log,
            'regime_count': regime_count,
        }
    except Exception:
        return {'active': None, 'recent': [], 'opt_log': [], 'regime_count': 0}



@st.cache_data(ttl=60)
def load_rl_data():
    """Load RL optimizer state: current params + performance history."""
    try:
        client, db = get_db_connection()
        params = db['rl_parameters'].find_one({}, {'_id': 0},
                                               sort=[('last_updated', pymongo.DESCENDING)])
        perf_hist = list(
            db['rl_performance_history']
            .find({}, {'_id': 0})
            .sort('run_at', pymongo.DESCENDING)
            .limit(10)
        )
        param_hist = list(
            db['rl_parameter_history']
            .find({}, {'_id': 0, 'rl_weight_adjustment': 1,
                       'reward_score': 1, 'episode': 1, 'snapshot_at': 1})
            .sort('snapshot_at', pymongo.DESCENDING)
            .limit(20)
        )
        client.close()
        return {
            'params':     params,
            'perf_hist':  perf_hist,
            'param_hist': param_hist,
        }
    except Exception:
        return {'params': None, 'perf_hist': [], 'param_hist': []}


@st.cache_data(ttl=60)
def load_whale_data(limit: int = 20):
    """Load latest whale data per symbol for the dashboard panel."""
    try:
        client, db = get_db_connection()
        # Aggregate: latest whale doc per symbol
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$symbol',
                'whale_score': {'$first': '$whale_score'},
                'whale_signal': {'$first': '$whale_signal'},
                'whale_score_norm': {'$first': '$whale_score_norm'},
                'timestamp': {'$first': '$timestamp'},
                'metrics': {'$first': '$metrics'},
            }},
            {'$sort': {'whale_score': -1}},
            {'$limit': limit},
        ]
        docs = list(db['whale_data'].aggregate(pipeline, allowDiskUse=True))
        client.close()
        return docs
    except Exception:
        return []


@st.cache_data(ttl=120)
def load_whale_history(symbol: str, hours: int = 6):
    """Load whale score history for a single symbol."""
    try:
        from datetime import timedelta
        client, db = get_db_connection()
        since = datetime.utcnow() - timedelta(hours=hours)
        docs = list(
            db['whale_data']
            .find({'symbol': symbol, 'timestamp': {'$gte': since}},
                  {'_id': 0, 'timestamp': 1, 'whale_score': 1, 'whale_signal': 1})
            .sort('timestamp', 1)
        )
        client.close()
        return docs
    except Exception:
        return []

def main():
    st.title("🚀 Crypto AI Analytics Platform")

    # ═══════ TAB NAVIGATION ═══════
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📡 Live Analytics",
        "🏆 Top Opportunities",
        "📊 Backtesting",
        "🧠 AI Learning",
        "🎯 Strategy Intelligence",
        "🐋 Whale Intelligence",
        "📋 Paper Trading",
        "⏰ Best Scan Hours",
    ])

    with tab1:
        render_live_analytics()

    with tab2:
        render_top_opportunities()

    with tab3:
        render_backtesting()

    with tab4:
        render_ai_learning()

    with tab5:
        render_strategy_intelligence()

    with tab6:
        render_whale_panel()

    with tab7:
        render_paper_trading_panel()

    with tab8:
        render_scan_time_panel()


# ══════════════════════════════════════════════════════════════
# TAB 1: LIVE ANALYTICS
# ══════════════════════════════════════════════════════════════

def render_live_analytics():
    df = load_signals()
    stats = get_signal_db_stats()

    if df.empty:
        st.warning("⚠️ No data yet. Run `python scheduler.py` first.")
        return

    # ─── LAST UPDATED TIMESTAMP ───
    latest_ts = stats.get('latest_ts')
    sig_count = stats.get('signal_count', 0)
    news_count = stats.get('news_count', 0)
    if latest_ts:
        if isinstance(latest_ts, datetime):
            from datetime import timedelta
            ts_ist = latest_ts + timedelta(hours=5, minutes=30)
            ts_str = ts_ist.strftime('%Y-%m-%d %H:%M IST')
        else:
            ts_str = str(latest_ts)
        st.markdown(
            f'<span class="ts-badge">🕐 Last Signal: {ts_str} &nbsp;|&nbsp; '
            f'📊 {sig_count:,} signals stored &nbsp;|&nbsp; '
            f'📰 {news_count:,} news items</span>',
            unsafe_allow_html=True
        )

    st.markdown("")  # spacer

    # ─── MARKET REGIME BANNER ───
    regime = df['market_regime'].iloc[0] if 'market_regime' in df.columns else 'UNKNOWN'
    regime_class = {'BULL': 'regime-bull', 'BEAR': 'regime-bear'}.get(regime, 'regime-sideways')
    regime_emoji = {'BULL': '🟢 BULL MARKET', 'BEAR': '🔴 BEAR MARKET'}.get(regime, '🟡 SIDEWAYS')
    st.markdown(f'<div class="{regime_class}">{regime_emoji}</div>', unsafe_allow_html=True)

    # ─── TOP 3 OPPORTUNITIES ───
    st.subheader("🔥 Top 3 Opportunities")
    top3 = df[df['top_rank'].notna()].sort_values('top_rank').head(3) if 'top_rank' in df.columns else pd.DataFrame()

    if not top3.empty:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                signal_color = {'BUY': '🟢', 'HOLD': '🟡', 'SELL': '🔴'}.get(row.get('final_signal', ''), '⚪')
                whale_icon = {'ACCUMULATION': '🐋⬆', 'DISTRIBUTION': '🐋⬇'}.get(row.get('whale_signal', ''), '—')
                news_sent = row.get('news_sentiment', 'NEUTRAL')
                news_icon = {'BULLISH': '📰🟢', 'BEARISH': '📰🔴'}.get(news_sent, '📰🟡')
                st.markdown(f"""
                <div class="top-card">
                    <h3>#{int(row['top_rank'])} {row['symbol']}</h3>
                    <p style="font-size:22px; color:#4caf50;">${row.get('price', 0):,.4f}</p>
                    <p>{signal_color} <b>{row.get('final_signal', 'N/A')}</b> | Score: <b>{row.get('final_score', 0)}</b></p>
                    <p>Probability: <b>{row.get('probability_up', 0)}%</b> | Whale: {whale_icon}</p>
                    <p>News: {news_icon} | Opportunity: <b>{row.get('opportunity_score', 0)}</b></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Ranking data not available yet")

    st.markdown("---")

    # ─── SIDEBAR FILTERS ───
    st.sidebar.header("⚙️ Filters")
    min_score = st.sidebar.slider("Min Profit Score", 0, 100, 0, 5)
    min_conf = st.sidebar.slider("Min Confidence", 0, 100, 0, 5)

    risk_opts = df['risk_level'].unique().tolist() if 'risk_level' in df.columns else []
    risk_filter = st.sidebar.multiselect("Risk Level", risk_opts, default=risk_opts)

    signal_opts = df['final_signal'].unique().tolist() if 'final_signal' in df.columns else []
    signal_filter = st.sidebar.multiselect("Signal", signal_opts, default=signal_opts)

    search = st.sidebar.text_input("Search Symbol", "").upper()
    top_n = st.sidebar.slider("Show Top N", 5, 90, 30, 5)

    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    auto = st.sidebar.checkbox("Auto-refresh (60s)")
    st.sidebar.markdown("---")
    st.sidebar.info(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    # ─── NEWS PANEL SELECTOR ───
    st.sidebar.markdown("---")
    st.sidebar.header("📰 News Panel")
    available_symbols = df['symbol'].tolist() if 'symbol' in df.columns else []
    selected_coin = st.sidebar.selectbox("View news for:", ["(Select coin)"] + available_symbols[:30])

    # ─── APPLY FILTERS ───
    fdf = df.copy()
    if 'profit_score' in fdf.columns:
        fdf = fdf[fdf['profit_score'] >= min_score]
    if 'confidence' in fdf.columns:
        fdf = fdf[fdf['confidence'] >= min_conf]
    if risk_filter and 'risk_level' in fdf.columns:
        fdf = fdf[fdf['risk_level'].isin(risk_filter)]
    if signal_filter and 'final_signal' in fdf.columns:
        fdf = fdf[fdf['final_signal'].isin(signal_filter)]
    if search and 'symbol' in fdf.columns:
        fdf = fdf[fdf['symbol'].str.contains(search)]
    if 'final_score' in fdf.columns:
        fdf = fdf.sort_values('final_score', ascending=False).head(top_n)

    # ─── STATS ROW ───
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Coins", len(df))
    c2.metric("🟢 BUY", len(df[df['final_signal'] == 'BUY']) if 'final_signal' in df.columns else 0)
    c3.metric("🟡 HOLD", len(df[df['final_signal'] == 'HOLD']) if 'final_signal' in df.columns else 0)
    c4.metric("🔴 SELL", len(df[df['final_signal'] == 'SELL']) if 'final_signal' in df.columns else 0)
    c5.metric("Avg Prob ↑", f"{df['probability_up'].mean():.0f}%" if 'probability_up' in df.columns else "N/A")

    st.markdown("---")

    # ─── MAIN TABLE ───
    st.subheader(f"📊 All Coins ({len(fdf)})")

    if fdf.empty:
        st.warning("No coins match filters")
    else:
        display_cols = [
                'symbol', 'price', 'final_signal', 'risk_adjusted_score',
                'final_score', 'technical_score', 'risk_score', 'risk_level',
                'news_sentiment', 'news_score', 'profit_score', 'confidence',
                'probability_up', 'whale_signal', 'volatility_penalty',
                'liquidity_score', 'rsi', 'volume_spike', 'volatility', 'headline_count']
        display_cols = [c for c in display_cols if c in fdf.columns]
        display = fdf[display_cols].copy()

        rename = {
            'symbol': 'Symbol', 'price': 'Price', 'final_signal': 'Signal',
            'risk_adjusted_score': '🛡 RA Score', 'final_score': 'AI Score',
            'technical_score': 'Tech Score', 'risk_score': 'Risk Score',
            'risk_level': 'Risk Level',
            'news_sentiment': 'News', 'news_score': 'News Score',
            'profit_score': 'Profit Score', 'confidence': 'Confidence',
            'probability_up': 'Prob ↑ %', 'whale_signal': 'Whale',
            'volatility_penalty': 'Vol Pen', 'liquidity_score': 'Liquidity',
            'rsi': 'RSI', 'volume_spike': 'Vol Spike', 'volatility': 'Volatility',
            'headline_count': 'Headlines',
        }
        display = display.rename(columns=rename)

        if 'Price' in display.columns:
            display['Price'] = display['Price'].apply(lambda x: f"${x:,.4f}" if x else "N/A")

        def color_signal(val):
            c = {'BUY': '#28a745', 'HOLD': '#ffc107', 'SELL': '#dc3545'}
            return f'background-color: {c.get(val, "#6c757d")}; color: white; font-weight: bold'

        def color_score(val):
            try:
                s = float(val)
                if s >= 75: return 'background-color: #28a745; color: white'
                elif s >= 45: return 'background-color: #ffc107; color: black'
                else: return 'background-color: #dc3545; color: white'
            except:
                return ''

        def color_news(val):
            c = {'BULLISH': '#1b5e20', 'BEARISH': '#b71c1c', 'NEUTRAL': '#4a4a00'}
            return f'background-color: {c.get(val, "#333")}; color: white; font-weight: bold'

        def color_risk_level(val):
            """Color for LOW / MEDIUM / HIGH risk level badges."""
            c = {
                'LOW':    'background-color: #1b5e20; color: white; font-weight: bold',
                'MEDIUM': 'background-color: #e65100; color: white; font-weight: bold',
                'HIGH':   'background-color: #b71c1c; color: white; font-weight: bold',
                # legacy values from profit_score (capitalized)
                'Low':    'background-color: #1b5e20; color: white',
                'Medium': 'background-color: #e65100; color: white',
                'High':   'background-color: #b71c1c; color: white',
            }
            return c.get(str(val), '')

        def color_whale(val):
            c = {'ACCUMULATION': '#1565c0', 'DISTRIBUTION': '#d32f2f', 'NONE': '#555'}
            return f'background-color: {c.get(val, "#555")}; color: white'

        styled = display.style
        if 'Signal'     in display.columns: styled = styled.map(color_signal,     subset=['Signal'])
        if 'AI Score'   in display.columns: styled = styled.map(color_score,      subset=['AI Score'])
        if '🛡 RA Score' in display.columns: styled = styled.map(color_score,      subset=['🛡 RA Score'])
        if 'Risk Score' in display.columns: styled = styled.map(color_score,      subset=['Risk Score'])
        if 'News'       in display.columns: styled = styled.map(color_news,       subset=['News'])
        if 'Risk Level' in display.columns: styled = styled.map(color_risk_level, subset=['Risk Level'])
        if 'Whale'      in display.columns: styled = styled.map(color_whale,      subset=['Whale'])

        st.dataframe(styled, use_container_width=True, height=600)

    # ─── CHARTS ROW ───
    if not fdf.empty and 'final_score' in fdf.columns:
        st.markdown("---")
        ch1, ch2, ch3, ch4 = st.columns(4)
        with ch1:
            st.subheader("📊 AI Score (Top 10)")
            if 'symbol' in fdf.columns:
                st.bar_chart(fdf.head(10).set_index('symbol')['final_score'])
        with ch2:
            st.subheader("🛡 Risk-Adjusted (Top 10)")
            if 'risk_adjusted_score' in fdf.columns and 'symbol' in fdf.columns:
                st.bar_chart(fdf.head(10).set_index('symbol')['risk_adjusted_score'])
            elif 'symbol' in fdf.columns:
                st.bar_chart(fdf.head(10).set_index('symbol')['final_score'])
        with ch3:
            st.subheader("🎯 Probability (Top 10)")
            if 'probability_up' in fdf.columns:
                st.bar_chart(fdf.head(10).set_index('symbol')['probability_up'])
        with ch4:
            st.subheader("📈 Signals Distribution")
            if 'final_signal' in fdf.columns:
                st.bar_chart(fdf['final_signal'].value_counts())

    # ─── RISK OVERVIEW ───
    if not fdf.empty and 'risk_level' in fdf.columns:
        st.markdown("---")
        st.subheader("🛡 Risk Overview")
        r1, r2, r3, r4, r5 = st.columns(5)
        low_n    = len(fdf[fdf['risk_level'].isin(['LOW', 'Low'])])
        med_n    = len(fdf[fdf['risk_level'].isin(['MEDIUM', 'Medium'])])
        high_n   = len(fdf[fdf['risk_level'].isin(['HIGH', 'High'])])
        avg_ra   = fdf['risk_adjusted_score'].mean() if 'risk_adjusted_score' in fdf.columns else 0
        avg_vol  = fdf['volatility_penalty'].mean()  if 'volatility_penalty'  in fdf.columns else 0
        r1.metric("🟢 LOW Risk",    low_n)
        r2.metric("🟡 MEDIUM Risk", med_n)
        r3.metric("🔴 HIGH Risk",   high_n)
        r4.metric("🛡 Avg RA Score", f"{avg_ra:.1f}")
        r5.metric("⚠️ Avg Vol Pen",  f"{avg_vol:.1f}")

    # ─── NEWS SENTIMENT CHART ───
    st.markdown("---")
    _render_news_sentiment_chart(fdf)

    # ─── NEWS PANEL FOR SELECTED COIN ───
    if selected_coin and selected_coin != "(Select coin)":
        st.markdown("---")
        _render_news_panel(selected_coin)

    # ─── FOOTER ───
    st.markdown("---")
    st.caption("AI Scoring v3: Tech(70%) + News(30%) | Risk-Adjusted: VolPen + DDPen + LiqBonus + ProbWeight + RRBonus | Probability: Trend(25%)+Momentum(20%)+Vol(5%)+News(25%)+Whale(15%)+Regime(10%)")
    st.caption(f"Dashboard: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if auto:
        time.sleep(60)
        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2: TOP OPPORTUNITIES (Ranking Engine)
# ══════════════════════════════════════════════════════════════

def render_top_opportunities():
    """Render the Best Coin Ranking AI dashboard section."""
    st.subheader("🏆 Best Coin Ranking AI — Top Opportunities")
    st.caption("Rank = Tech(35%) + Prob(30%) + News(15%) + Volume(10%) + Trend(10%) · auto-updated every 5 min")

    rdf = load_ranked_opportunities(limit=30)

    if rdf.empty:
        st.warning("⚠️ No ranked data yet. Run `python scheduler.py` (ranking runs after every scan) "
                   "or `python ranking_engine.py` to rank existing signals.")
        st.code("python scheduler.py", language="bash")
        return

    # ── Data freshness check — use batch_ts (true scan time) ────────────────
    # batch_ts = when scheduler last ran a full scan cycle
    # created_at gets renewed on every upsert even with stale field values
    last_ts      = rdf['batch_ts'].max()  if 'batch_ts'   in rdf.columns else (
                   rdf['created_at'].max() if 'created_at' in rdf.columns else None)
    data_age_min = None
    if last_ts is not None:
        try:
            ts_clean = last_ts.replace(tzinfo=None) if hasattr(last_ts, 'replace') else last_ts
            data_age_min = round((datetime.now(datetime.UTC).replace(tzinfo=None) - ts_clean).total_seconds() / 60, 0)
        except Exception:
            pass
    if data_age_min is not None:
        if data_age_min > 10:
            st.error(
                f"⚠️ **Stale Data ({int(data_age_min)} min old)** — "
                f"Keep `python scheduler.py` running for live updates."
            )
        else:
            st.success(f"✅ Data is fresh — scanned {int(data_age_min)} min ago", icon="✅")

    # ── Pre-fetch live prices from CoinGecko (cached, no geo-restrictions) ────
    top_syms = rdf['symbol'].head(10).tolist() if 'symbol' in rdf.columns else []
    live_prices = {}
    try:
        import requests as _req
        # Use CoinGecko simple/price — works from any server (no Binance geo-block)
        _ids_map = {}
        for _s in top_syms:
            _coin = _s.replace('USDT','').lower()
            # common CoinGecko ID overrides
            _overrides = {'btc':'bitcoin','eth':'ethereum','sol':'solana',
                         'xrp':'ripple','bnb':'binancecoin','ada':'cardano',
                         'doge':'dogecoin','dot':'polkadot','matic':'matic-network',
                         'link':'chainlink','ltc':'litecoin','avax':'avalanche-2',
                         'atom':'cosmos','uni':'uniswap','trx':'tron'}
            _ids_map[_s] = _overrides.get(_coin, _coin)
        _ids_str = ','.join(set(_ids_map.values()))
        _resp = _req.get(
            'https://api.coingecko.com/api/v3/simple/price',
            params={'ids': _ids_str, 'vs_currencies': 'usd'},
            timeout=8
        )
        if _resp.status_code == 200:
            _cg = _resp.json()
            for _s, _cid in _ids_map.items():
                if _cid in _cg and 'usd' in _cg[_cid]:
                    live_prices[_s] = float(_cg[_cid]['usd'])
    except Exception:
        pass  # fall back to stored prices gracefully

    # ─────────────────────────────────────────────────────────────────────────

    # ── Stats row ──
    total = len(rdf)
    buy_c  = len(rdf[rdf['signal'] == 'BUY'])  if 'signal' in rdf.columns else 0
    hold_c = len(rdf[rdf['signal'] == 'HOLD']) if 'signal' in rdf.columns else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🎯 Ranked Coins",    total)
    s2.metric("🟢 BUY Signals",     buy_c)
    s3.metric("🟡 HOLD Signals",    hold_c)
    def _fmt_ist(dt):
        from zoneinfo import ZoneInfo as _ZI
        import datetime as _dt
        ist_dt = dt.replace(tzinfo=_dt.timezone.utc).astimezone(_ZI('Asia/Kolkata'))
        h = ist_dt.strftime('%I').lstrip('0') or '12'
        return ist_dt.strftime(f'{h}:%M %p IST')
    s4.metric("🕐 Last Scan",
              f"{int(data_age_min)} min ago" if data_age_min is not None else
              (_fmt_ist(last_ts) if isinstance(last_ts, datetime) else str(last_ts)[:16]
               if last_ts is not None else "N/A"))

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 🎯 HIGH CONVICTION BOARD — v3 Multi-Timeframe (SWING / POSITION / TREND)
    # ══════════════════════════════════════════════════════════════════════════
    # Shows coins that appeared consistently in top-10 across the last 2 hours
    # of scans. Needs >= 5 scans to populate (starts filling after ~1.5 hours).
    # The conviction_picks collection is updated every 30 minutes by the scheduler.
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("🎯 High Conviction Board")
    st.caption(
        "Coins that stayed in top-10 across ≥5 of last 8 scans (~2h). "
        "**Far more reliable than a single-scan snapshot.** "
        "Updates every 30 min · Needs ~1.5h of scan data to populate."
    )

    _conviction_picks = None
    try:
        import pymongo as _pym
        _cv_uri    = __import__('config', fromlist=['settings']).settings.MONGO_URI
        _cv_dbname = __import__('config', fromlist=['settings']).settings.DATABASE_NAME
        _cv_client = _pym.MongoClient(_cv_uri, serverSelectionTimeoutMS=4000)
        _cv_doc    = _cv_client[_cv_dbname]['conviction_picks'].find_one({}, {'_id': 0})
        _cv_client.close()
        if _cv_doc:
            _conviction_picks = _cv_doc
    except Exception:
        pass

    if _conviction_picks and any(
        _conviction_picks.get(k) for k in ['swing', 'position', 'trend']
    ):
        _meta = _conviction_picks.get('metadata', {})
        _n_scans = _meta.get('scans_analysed', 0)
        _gen_at  = _conviction_picks.get('generated_at')
        _age_str = ''
        if _gen_at:
            try:
                _age_m = int((datetime.utcnow() - _gen_at.replace(tzinfo=None)).total_seconds() / 60)
                _age_str = f" · Updated {_age_m} min ago"
            except Exception:
                pass
        st.info(f"📊 Analysed **{_n_scans} scans** in last 2h{_age_str}", icon="📊")

        _swing_picks    = _conviction_picks.get('swing', [])
        _position_picks = _conviction_picks.get('position', [])
        _trend_picks    = _conviction_picks.get('trend', [])

        # ── Fix 4: BUY NOW alert — highlight best actionable pick ──────────
        # Criteria: Streak ≥ 5 + R:R ≥ 1.5x + position_score > 70
        # Scans all 3 columns, picks the highest-scored coin that qualifies.
        _buy_candidates = []
        for _p in (_swing_picks + _position_picks + _trend_picks):
            _p_rr  = _p.get('risk_reward_ratio') or 0
            # Safety: recalculate R:R from entry/tp/sl if stored value is bad
            if float(_p_rr) <= 0:
                _pre = float(_p.get('entry_price') or 0)
                _ptp = float(_p.get('take_profit') or 0)
                _psl = float(_p.get('stop_loss') or 0)
                if _pre > 0 and _pre > _psl and _ptp > _pre:
                    _p_rr = round((_ptp - _pre) / (_pre - _psl), 2)
            _p_str = _p.get('streak', 0)
            _p_pos = _p.get('position_score', 0) or 0
            if float(_p_str) >= 5 and float(_p_rr) >= 1.5 and float(_p_pos) > 70:
                _buy_candidates.append(_p)
        if _buy_candidates:
            _buy_candidates.sort(key=lambda x: x.get('avg_rank_score', 0), reverse=True)
            _best = _buy_candidates[0]
            _best_sym  = _best.get('symbol', '?')
            _best_rr   = float(_best.get('risk_reward_ratio') or 0)
            _best_str  = _best.get('streak', 0)
            _best_sc   = float(_best.get('avg_rank_score', 0) or 0)
            _best_tp   = _best.get('take_profit')
            _best_en   = _best.get('entry_price')
            _pct_to_tp = ''
            if _best_tp and _best_en and float(_best_en) > 0:
                _pct_to_tp = f" · +{(float(_best_tp)/float(_best_en)-1)*100:.1f}% to TP"
            _best_type = _best.get('trade_type', 'SWING')
            # Determine column from which list it came
            for _blist, _blabel in [(_swing_picks,'SWING'),(_position_picks,'POSITION'),(_trend_picks,'TREND')]:
                if any(x.get('symbol') == _best_sym for x in _blist):
                    _best_type = _blabel
                    break
            st.markdown(
                f"""
                <div style='background:linear-gradient(90deg,#e65100,#ff6f00);
                border-radius:10px;padding:14px 18px;margin-bottom:12px;
                border-left:5px solid #fff3e0;'>
                <span style='color:#fff3e0;font-size:0.8em;font-weight:600;
                letter-spacing:1px;'>🚨 BUY NOW ALERT</span><br>
                <span style='color:#ffffff;font-size:1.15em;font-weight:700;'>
                {_best_sym}</span>
                <span style='color:#ffe0b2;'>&nbsp;·&nbsp;{_best_sc:.1f} pts
                &nbsp;·&nbsp;Streak {_best_str}
                &nbsp;·&nbsp;R:R {_best_rr:.1f}x{_pct_to_tp}
                &nbsp;·&nbsp;{_best_type}</span>
                </div>""",
                unsafe_allow_html=True
            )

        _c_sw, _c_po, _c_tr = st.columns(3)

        # ── Helper: render one conviction pick card ────────────────────────
        def _render_conviction_card(col, picks, label, emoji, hold_time, gain_range, color):
            with col:
                st.markdown(
                    f"<div style='background:{color};border-radius:10px;padding:12px 14px 8px;"
                    f"margin-bottom:8px;'>"
                    f"<h4 style='margin:0;color:#fff;'>{emoji} {label}</h4>"
                    f"<small style='color:rgba(255,255,255,0.75);'>"
                    f"Hold: {hold_time} · Target: {gain_range}</small></div>",
                    unsafe_allow_html=True
                )
                if not picks:
                    st.caption("⏳ Not enough data yet — keep scheduler running")
                    return
                for p in picks[:3]:
                    sym      = p.get('symbol', '?')
                    score    = p.get('avg_rank_score') or p.get('avg_score', 0)
                    streak   = p.get('streak', 0)
                    appear   = p.get('appearances', 0)
                    n_scans  = p.get('n_scans', 1)
                    rr       = p.get('risk_reward_ratio')
                    entry    = p.get('entry_price')
                    tp       = p.get('take_profit')
                    sl       = p.get('stop_loss')
                    # Safety: recalculate R:R if stored value is bad
                    if (rr is None or float(rr) <= 0) and entry and tp and sl:
                        _e = float(entry); _t = float(tp); _s = float(sl)
                        if _e > 0 and _e > _s and _t > _e:
                            rr = round((_t - _e) / (_e - _s), 2)
                    gain24   = p.get('price_change_24h_pct', 0) or 0
                    daily    = p.get('daily_trend', '')
                    pos_sc   = p.get('position_score', 0) or 0

                    streak_bar = '🟢' * min(int(streak), 5) + '⬜' * max(0, 5 - min(int(streak), 5))
                    consist    = f"{appear}/{n_scans} scans"
                    rr_str     = f"R:R {float(rr):.1f}x" if rr is not None else ""
                    trend_tag  = f" · {daily}" if daily and daily not in ('', 'SIDEWAYS') else ""

                    # Fix 3: compute +% remaining to TP
                    tp_pct_str = ''
                    if entry and tp and float(entry) > 0:
                        tp_pct = (float(tp) / float(entry) - 1) * 100
                        tp_pct_str = f"+{tp_pct:.1f}% to TP"

                    coin_md = (
                        f"**{sym}** &nbsp; `{float(score):.1f} pts`  \n"
                        f"{streak_bar} Streak: **{streak}** &nbsp;|&nbsp; {consist}  \n"
                        f"24h: {float(gain24):+.1f}%{trend_tag}"
                        + (f" &nbsp;|&nbsp; {rr_str}" if rr_str else "")
                    )
                    st.markdown(coin_md)
                    if entry and tp and sl:
                        st.markdown(
                            f"<small>Entry: `{entry:.4g}` &nbsp; "
                            + (f"**{tp_pct_str}** &nbsp; " if tp_pct_str else "")
                            + f"TP: `{tp:.4g}` &nbsp; SL: `{sl:.4g}` "
                            f"&nbsp; pos={pos_sc:.0f}</small>",
                            unsafe_allow_html=True
                        )
                    st.markdown("---")


        _render_conviction_card(
            _c_sw, _swing_picks,    "SWING",    "⚡", "4–12h",    "+3–8%",  "#1565C0"
        )
        _render_conviction_card(
            _c_po, _position_picks, "POSITION", "📈", "1–3 days", "+8–15%", "#2E7D32"
        )
        _render_conviction_card(
            _c_tr, _trend_picks,    "TREND",    "🚀", "1–2 weeks","+15–40%","#6A1B9A"
        )

    else:
        # Not enough scans yet — show informational placeholder
        _c_sw, _c_po, _c_tr = st.columns(3)
        for _col, _lbl, _em, _hold, _gain, _clr in [
            (_c_sw, "SWING",    "⚡", "4–12h",    "+3–8%",   "#1565C0"),
            (_c_po, "POSITION", "📈", "1–3 days", "+8–15%",  "#2E7D32"),
            (_c_tr, "TREND",    "🚀", "1–2 weeks","+15–40%", "#6A1B9A"),
        ]:
            with _col:
                st.markdown(
                    f"<div style='background:{_clr};border-radius:10px;padding:12px 14px 8px;"
                    f"margin-bottom:8px;'>"
                    f"<h4 style='margin:0;color:#fff;'>{_em} {_lbl}</h4>"
                    f"<small style='color:rgba(255,255,255,0.75);'>Hold: {_hold} · "
                    f"Target: {_gain}</small></div>",
                    unsafe_allow_html=True
                )
                st.caption("⏳ Populates after ~1.5h of scanning")

    st.markdown("---")

    # ── Live Scan Snapshot (Top 3 from most recent scan) ─────────────────────
    # This shows the top 3 coins from the LAST scan only.
    # Unlike the High Conviction Board above, this changes every scan.
    # Use it to see what's hot RIGHT NOW — confirm with the board above before buying.
    # ──────────────────────────────────────────────────────────────────────────

    # ── Top 3 Cards ──
    st.subheader("🥇 Top 3 Opportunities")
    top3 = rdf.head(3)
    card_classes = ['rank-gold', 'rank-silver', 'rank-bronze']
    medals       = ['🥇 #1', '🥈 #2', '🥉 #3']

    cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        if idx >= 3:
            break
        sym = row['symbol']

        # ── Use live price (always preferred for entry), stored for indicators ──
        stored_price = float(row.get('price', 0) or row.get('entry_price', 0) or 0)
        live_price   = live_prices.get(sym, None)
        use_price    = live_price if live_price else stored_price

        # ── AI-Driven Entry/SL/TP via compute_smart_levels() ──────────────────
        # Strategy:
        #   • price → always live Binance (freshest possible)
        #   • ema20/ema50/atr/rsi → use stored values if non-null (set by scan)
        #                            fall back to price-based estimates if null
        #   • whale/news/regime/mtf → always stored (change slowly)
        try:
            from ai.smart_levels import compute_smart_levels as _csl

            # Helper: use stored value if it's a real number, else estimate
            def _fresh(field, estimate):
                v = row.get(field)
                try:
                    fv = float(v)
                    return fv if fv > 0 else estimate
                except (TypeError, ValueError):
                    return estimate

            _coin_ctx = {
                # Price — live Binance beats everything
                'price':              use_price,
                # Technical indicators — use stored real scan values when available
                'ema20':              _fresh('ema20',  use_price),
                'ema50':              _fresh('ema50',  use_price * 0.97),
                'atr':                _fresh('atr',    use_price * 0.025),
                'rsi':                _fresh('rsi',    50.0),
                'rsi_4h':             _fresh('rsi_4h', 50.0),
                'volatility':         _fresh('volatility', 0.025),
                'nearest_support':    row.get('nearest_support'),
                'nearest_resistance': row.get('nearest_resistance'),
                'mtf_confirmed':      bool(row.get('mtf_confirmed', False)),
                # AI signals — stored from scan (change slowly, remain valid)
                'probability_up':  float(row.get('probability_up', 50) or 50),
                'whale_signal':    str(row.get('whale_signal', 'NONE') or 'NONE'),
                'news_score':      float(row.get('news_score', 0) or 0),
                'market_regime':   str(row.get('market_regime', 'NEUTRAL') or 'NEUTRAL'),
            }
            _lvls = _csl(_coin_ctx)
            ep    = _lvls['entry_price']
            sl    = _lvls['stop_loss']
            tp    = _lvls['take_profit']
            rr    = _lvls['risk_reward_ratio']
            tq    = _lvls.get('trade_quality_score', 0)
            e_why = _lvls.get('entry_logic', '')
            s_why = _lvls.get('sl_logic', '')
            t_why = _lvls.get('tp_logic', '')
        except Exception as _ex:
            ep = use_price
            sl = round(use_price * 0.95, 8)
            tp = round(use_price * 1.12, 8)
            rr = 2.4
            tq = 0
            e_why = s_why = t_why = f'fallback: {_ex}'
        # ─────────────────────────────────────────────────────────────────────


        price_tag = "● Live" if live_price else (f"⚠ {int(data_age_min)}m old" if (data_age_min and data_age_min > 10) else "")
        price_color = "#4caf50" if live_price else "#f44336"
        price_label = (
            f"${ep:,.5g} <span style='color:{price_color};font-size:10px;'>{price_tag}</span>"
            if price_tag else f"${ep:,.5g}"
        )

        with cols[idx]:
            sig   = row.get('signal', 'HOLD')
            score = float(row.get('rank_score', 0))
            prob  = float(row.get('probability_up', 0))
            tech  = float(row.get('technical_score', 0))
            news  = str(row.get('news_sentiment', 'NEUTRAL'))
            whale = str(row.get('whale_signal', 'NONE'))

            sig_color   = {'BUY': '#4caf50', 'STRONG_BUY': '#00e676', 'HOLD': '#ff9800'}.get(sig, '#aaa')
            news_icon   = {'BULLISH': '📰🟢', 'BEARISH': '📰🔴', 'NEUTRAL': '📰🟡'}.get(news, '📰🟡')
            whale_icon  = {'ACCUMULATION': '🐋⬆', 'DISTRIBUTION': '🐋⬇'}.get(whale, '—')
            bar_class   = 'conf-bar' if score >= 65 else ('conf-bar-med' if score >= 45 else 'conf-bar-low')
            bar_w       = int(score)

            st.markdown(f"""
            <div class="{card_classes[idx]}">
                <h3 style="margin:0 0 6px 0;">{medals[idx]} {sym}</h3>
                <div style="font-size:26px; color:{sig_color}; font-weight:bold; margin-bottom:4px;">
                    {score:.1f} pts
                </div>
                <div style="font-size:13px; color:{sig_color}; margin-bottom:8px;">{sig}</div>
                <!-- Confidence bar -->
                <div style="text-align:left; font-size:11px; color:#aaa;">Rank Score</div>
                <div class="conf-bar-wrap">
                    <div class="{bar_class}" style="width:{bar_w}%;"></div>
                </div>
                <!-- Metrics -->
                <table style="width:100%; font-size:12px; margin-top:10px; text-align:left;">
                <tr><td style="color:#aaa;">Prob Up</td>
                    <td style="color:#4caf50;"><b>{prob:.0f}%</b></td></tr>
                <tr><td style="color:#aaa;">Tech Score</td>
                    <td><b>{tech:.0f}</b></td></tr>
                <tr><td style="color:#aaa;">News</td>
                    <td>{news_icon}</td></tr>
                <tr><td style="color:#aaa;">Whale</td>
                    <td>{whale_icon}</td></tr>
                <tr style="border-top:1px solid #333;"><td style="color:#aaa;">Entry</td>
                    <td>{price_label}</td></tr>
                <tr><td style="color:#f44336;">Stop Loss</td>
                    <td>${sl:,.5g}</td></tr>
                <tr><td style="color:#4caf50;">Take Profit</td>
                    <td>${tp:,.5g}</td></tr>
                <tr><td style="color:#90caf9;">R:R</td>
                    <td><b>{rr:.1f}x</b></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Top 10 Table ──
    st.subheader("📋 Full Ranked List")
    n_show = st.slider("Show top N coins", 5, min(30, total), min(10, total), 5)
    display_cols = ['rank', 'symbol', 'signal', 'rank_score', 'technical_score',
                    'probability_up', 'news_sentiment', 'volume_strength', 'trend_strength',
                    'entry_price', 'stop_loss', 'take_profit', 'risk_reward_ratio']
    display_cols = [c for c in display_cols if c in rdf.columns]
    tdf = rdf[display_cols].head(n_show).copy()

    rename = {
        'rank': '#', 'symbol': 'Symbol', 'signal': 'Signal',
        'rank_score': 'Rank Score', 'technical_score': 'Tech',
        'probability_up': 'Prob ↑%', 'news_sentiment': 'News',
        'volume_strength': 'Vol Str', 'trend_strength': 'Trend Str',
        'entry_price': 'Entry', 'stop_loss': 'SL', 'take_profit': 'TP',
        'risk_reward_ratio': 'R:R',
    }
    tdf = tdf.rename(columns=rename)

    def _c_signal(v):
        return {'BUY': 'background-color:#28a745;color:white;font-weight:bold',
                'STRONG_BUY': 'background-color:#00c853;color:white;font-weight:bold',
                'HOLD': 'background-color:#ff9800;color:white;font-weight:bold'}.get(str(v), '')

    def _c_score(v):
        try:
            s = float(v)
            if s >= 65: return 'background-color:#1b5e20;color:white'
            elif s >= 45: return 'background-color:#f57f17;color:black'
            return 'background-color:#b71c1c;color:white'
        except: return ''

    def _c_news(v):
        return {'BULLISH':'background-color:#1b5e20;color:white',
                'BEARISH':'background-color:#b71c1c;color:white',
                'NEUTRAL':'background-color:#37474f;color:white'}.get(str(v), '')

    styled = tdf.style
    if 'Signal'     in tdf.columns: styled = styled.map(_c_signal, subset=['Signal'])
    if 'Rank Score' in tdf.columns: styled = styled.map(_c_score,  subset=['Rank Score'])
    if 'News'       in tdf.columns: styled = styled.map(_c_news,   subset=['News'])
    st.dataframe(styled, use_container_width=True, height=420)

    st.markdown("---")

    # ── Charts ──
    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("📊 Rank Score (Top 10)")
        if 'rank_score' in rdf.columns and 'symbol' in rdf.columns:
            chart_data = rdf.head(10).set_index('symbol')[['rank_score']]
            st.bar_chart(chart_data)
    with ch2:
        st.subheader("🎯 Probability Up (Top 10)")
        if 'probability_up' in rdf.columns and 'symbol' in rdf.columns:
            chart_data = rdf.head(10).set_index('symbol')[['probability_up']]
            st.bar_chart(chart_data)

    # ── Probability bars per coin ──
    st.markdown("---")
    st.subheader("📈 Profit Probability Bars")
    for _, row in rdf.head(10).iterrows():
        prob = float(row.get('probability_up', 0))
        sym  = row.get('symbol', '')
        sig  = row.get('signal', '')
        bar_class = 'conf-bar' if prob >= 55 else ('conf-bar-med' if prob >= 40 else 'conf-bar-low')
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">'
            f'<span style="width:110px;font-size:13px;color:#ddd;">{sym}</span>'
            f'<div class="conf-bar-wrap" style="flex:1;">'
            f'  <div class="{bar_class}" style="width:{int(prob)}%;"></div>'
            f'</div>'
            f'<span style="width:55px;font-size:13px;color:#90caf9;">{prob:.0f}% ↑</span>'
            f'<span style="font-size:11px;color:#aaa;">{sig}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )



def _render_news_sentiment_chart(fdf: pd.DataFrame):
    """News sentiment scores for top coins as bar chart"""
    st.subheader("📰 News Sentiment — Top Coins")

    if fdf.empty or 'news_score' not in fdf.columns:
        st.info("No news data available")
        return

    chart_df = fdf[['symbol', 'news_score']].copy() if 'symbol' in fdf.columns else pd.DataFrame()
    if chart_df.empty:
        return

    chart_df = chart_df.dropna(subset=['news_score'])
    chart_df = chart_df.sort_values('news_score', ascending=False).head(20)
    chart_df = chart_df.set_index('symbol')

    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(chart_df['news_score'])
    with col2:
        bullish = len(chart_df[chart_df['news_score'] >= 0.2])
        bearish = len(chart_df[chart_df['news_score'] <= -0.2])
        neutral = len(chart_df) - bullish - bearish
        st.metric("🟢 Bullish", bullish)
        st.metric("🟡 Neutral", neutral)
        st.metric("🔴 Bearish", bearish)


def _render_news_panel(symbol: str):
    """Display latest news headlines for a selected coin"""
    st.subheader(f"📰 Latest News — {symbol}")
    items = load_news_for_symbol(symbol)

    if not items:
        st.info(f"No news stored yet for {symbol}. News is collected during market scans.")
        return

    for item in items:
        score = item.get('sentiment_score', 0)
        label = item.get('sentiment_label', 'NEUTRAL')
        title = item.get('title', '')
        source = item.get('source', '')
        pub_at = item.get('published_at', '')
        impact = item.get('impact_score', 0)

        card_class = 'news-card' if score >= 0.1 else ('news-card-neg' if score <= -0.1 else 'news-card-neu')
        label_badge = {'POSITIVE': '🟢', 'NEGATIVE': '🔴', 'NEUTRAL': '🟡'}.get(label, '🟡')

        pub_str = ''
        if isinstance(pub_at, datetime):
            pub_str = pub_at.strftime('%b %d %H:%M')
        elif pub_at:
            pub_str = str(pub_at)[:16]

        url = item.get('url', '')
        title_display = f'<a href="{url}" target="_blank" style="color:#90caf9; text-decoration:none;">{title}</a>' if url else title

        st.markdown(f"""
        <div class="{card_class}">
            {label_badge} <b>{title_display}</b><br>
            <small style="color:#888;">📡 {source} &nbsp;·&nbsp; 🕐 {pub_str} &nbsp;·&nbsp;
            Score: {score:+.2f} &nbsp;·&nbsp; Impact: {impact:.2f}</small>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: BACKTESTING
# ══════════════════════════════════════════════════════════════

def render_backtesting():
    st.subheader("📊 Backtesting Results")

    # ─── RUN BACKTEST UI ───
    with st.expander("⚙️ Run New Backtest", expanded=False):
        st.caption("Configure parameters and run a fresh backtest against your historical data.")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bt_days    = st.slider("Lookback Days",    min_value=7,  max_value=180, value=90,  step=7)
            bt_balance = st.number_input("Starting Balance ($)", min_value=100, max_value=100000,
                                         value=1000, step=100)
        with bc2:
            bt_tp      = st.slider("Take Profit %",   min_value=2,  max_value=20,  value=8,   step=1)
            bt_sl      = st.slider("Stop Loss %",     min_value=1,  max_value=15,  value=4,   step=1)
        with bc3:
            bt_score   = st.slider("Min AI Score",    min_value=20, max_value=80,  value=50,  step=5)
            bt_prob    = st.slider("Min Probability %", min_value=20, max_value=70, value=40, step=5)

        if st.button("▶ Run Backtest Now", type="primary"):
            with st.spinner(f"Running backtest ({bt_days} days) — please wait..."):
                try:
                    from backtesting.backtester import Backtester
                    import pymongo
                    bt = Backtester(settings_override={
                        'initial_balance':    bt_balance,
                        'take_profit':        bt_tp   / 100,
                        'stop_loss':          bt_sl   / 100,
                        'min_score':          bt_score,
                        'min_probability':    bt_prob,
                        'allow_hold_entry':   False,
                        'risk_per_trade':     0.05,
                        'max_open_positions': 5,
                        'fee_rate':           0.00075,
                    })
                    # CRITICAL: must connect before running
                    if not bt.connect():
                        st.error("❌ Could not connect to database. Check MongoDB connection.")
                    else:
                        res = bt.run_backtest(days=bt_days)
                        bt.close()
                        m   = res.get('metrics', {})
                        wr  = m.get('win_rate', 0)
                        ret = m.get('return_percent', 0)
                        pf  = m.get('profit_factor', 0)
                        total = m.get('total_trades', 0)

                        # Save result to MongoDB so dashboard picks it up
                        try:
                            _c = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
                            _db = _c[settings.DATABASE_NAME]
                            _db['backtest_results'].insert_one(res)
                            _c.close()
                        except Exception as save_err:
                            st.warning(f"Result computed but could not save to DB: {save_err}")

                        if total == 0:
                            st.warning("⚠️ Backtest ran but found 0 trades. "
                                       "This usually means no AI signals exist in the selected period "
                                       "with Score≥50 / Prob≥40. Try lowering the thresholds or "
                                       "increasing Lookback Days.")
                        elif ret > 0:
                            st.success(
                                f"✅ Backtest complete! Return: {ret:+.2f}% | "
                                f"Win Rate: {wr:.1f}% | Profit Factor: {pf:.2f} | "
                                f"Trades: {total}"
                            )
                        else:
                            st.warning(
                                f"⚠️ Backtest done. Return: {ret:+.2f}% | "
                                f"Win Rate: {wr:.1f}% | Profit Factor: {pf:.2f} | "
                                f"Trades: {total}"
                            )
                        st.cache_data.clear()
                        st.rerun()
                except Exception as ex:
                    st.error(f"Error running backtest: {ex}")

    st.markdown("---")

    result = load_backtest_result()

    if not result:
        st.info("⚠️ No backtest results yet. Use the **Run New Backtest** panel above to generate results.")
        stats = get_signal_db_stats()
        if stats['signal_count'] > 90:
            st.success(f"✅ {stats['signal_count']:,} historical signals stored — ready for backtesting!")
        else:
            st.warning(f"Only {stats['signal_count']} signals stored. "
                       f"Let scheduler run for 24+ hours to accumulate more history.")
        return

    metrics = result.get('metrics', {})
    period = result.get('period', {})
    trades = result.get('trade_history', [])
    equity = result.get('equity_curve', [])

    # ─── PERIOD INFO ───
    st.markdown(f"**Period:** {period.get('start', 'N/A')[:10]} → {period.get('end', 'N/A')[:10]} "
                f"({period.get('days', 0)} days)")

    # ─── METRICS CARDS ───
    st.markdown("### Performance Overview")

    ret_pct = metrics.get('return_percent', 0)
    ret_cls = "win" if ret_pct > 0 else ("loss" if ret_pct < 0 else "neutral")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {ret_cls}">{ret_pct:+.2f}%</div>
            <div class="metric-label">${metrics.get('balance_initial', 0):,.0f} → ${metrics.get('balance_final', 0):,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        wr = metrics.get('win_rate', 0)
        wr_cls = "win" if wr >= 55 else ("loss" if wr < 40 else "neutral")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value {wr_cls}">{wr:.1f}%</div>
            <div class="metric-label">{metrics.get('winning_trades', 0)}W / {metrics.get('losing_trades', 0)}L ({metrics.get('total_trades', 0)} total)</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        dd = metrics.get('max_drawdown', 0)
        dd_cls = "win" if dd < 5 else ("loss" if dd > 15 else "neutral")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value {dd_cls}">{dd:.2f}%</div>
            <div class="metric-label">Peak to trough</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        pf = metrics.get('profit_factor', 0)
        pf_cls = "win" if pf > 1.5 else ("loss" if pf < 1.0 else "neutral")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value {pf_cls}">{pf:.2f}</div>
            <div class="metric-label">Gross Profit / Gross Loss</div>
        </div>""", unsafe_allow_html=True)

    # ─── ROW 2: Additional metrics ───
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Trade</div>
            <div class="metric-value">{metrics.get('average_trade', 0):+.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with m6:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Best Trade</div>
            <div class="metric-value win">{metrics.get('best_trade', 0):+.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with m7:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Worst Trade</div>
            <div class="metric-value loss">{metrics.get('worst_trade', 0):+.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with m8:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">TP / SL Hits</div>
            <div class="metric-value">{metrics.get('tp_hits', 0)} / {metrics.get('sl_hits', 0)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ─── EQUITY CURVE ───
    if equity:
        st.subheader("📈 Equity Curve")
        eq_df = pd.DataFrame(equity)
        if 'timestamp' in eq_df.columns and 'equity' in eq_df.columns:
            eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
            eq_df = eq_df.set_index('timestamp')
            if len(eq_df) > 200:
                eq_df = eq_df.iloc[::max(1, len(eq_df)//200)]
            st.line_chart(eq_df['equity'], use_container_width=True)

    st.markdown("---")

    # ─── TRADE TABLE ───
    if trades:
        st.subheader(f"📋 Trade History ({len(trades)} trades)")

        tdf = pd.DataFrame(trades)
        display_cols = ['symbol', 'entry_price', 'exit_price', 'pnl_percent',
                        'profit_usdt', 'exit_reason', 'duration_minutes']
        display_cols = [c for c in display_cols if c in tdf.columns]
        tdf_display = tdf[display_cols].copy()

        rename = {
            'symbol': 'Symbol', 'entry_price': 'Entry', 'exit_price': 'Exit',
            'pnl_percent': 'PnL %', 'profit_usdt': 'Profit ($)',
            'exit_reason': 'Exit Reason', 'duration_minutes': 'Duration (min)',
        }
        tdf_display = tdf_display.rename(columns=rename)

        def color_pnl(val):
            try:
                v = float(val)
                if v > 0: return 'background-color: #1b5e20; color: white'
                elif v < 0: return 'background-color: #b71c1c; color: white'
                else: return ''
            except:
                return ''

        styled = tdf_display.style
        if 'PnL %' in tdf_display.columns:
            styled = styled.map(color_pnl, subset=['PnL %'])
        if 'Profit ($)' in tdf_display.columns:
            styled = styled.map(color_pnl, subset=['Profit ($)'])

        st.dataframe(styled, use_container_width=True, height=400)

        if 'exit_reason' in tdf.columns:
            st.subheader("📊 Exit Reasons")
            st.bar_chart(tdf['exit_reason'].value_counts())
    else:
        st.info("No trades in this backtest period")

    # ─── SETTINGS USED ───
    bt_settings = result.get('settings', {})
    if bt_settings:
        with st.expander("⚙️ Backtest Settings"):
            s1, s2, s3, s4 = st.columns(4)
            s1.write(f"**Balance:** ${bt_settings.get('initial_balance', 1000):,.0f}")
            s2.write(f"**TP:** {bt_settings.get('take_profit', 0.08)*100:.0f}% | "
                     f"**SL:** {bt_settings.get('stop_loss', 0.04)*100:.0f}%")
            s3.write(f"**Min Score:** {bt_settings.get('min_score', 70)}")
            s4.write(f"**Min Prob:** {bt_settings.get('min_probability', 60)}%")


# ══════════════════════════════════════════════════════════════
# TAB 4: AI LEARNING INSIGHTS
# ══════════════════════════════════════════════════════════════

def render_ai_learning():
    st.subheader("🧠 AI Self-Learning Insights")
    st.caption("The system analyses backtesting outcomes weekly to adjust indicator weights. "
               "Max ±10% change per cycle. Rollback backups are always kept.")

    data = load_learning_data()
    weights_doc   = data.get('weights_doc')
    stats_list    = data.get('indicator_stats', [])
    weight_history = data.get('weight_history', [])

    # ── Manual trigger ──────────────────────────────────────────
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("▶ Run Learning Cycle Now", type="primary"):
            with st.spinner("Running learning cycle (30-day lookback)..."):
                result = run_learning_cycle(lookback_days=30)
            if result['trade_count'] > 0:
                st.success(
                    f"✅ Learning complete — {result['trade_count']} trades analysed. "
                    f"Reload the page to see updated stats."
                )
                for msg in result['improvements']:
                    st.info(f"• {msg}")
            else:
                st.warning("⚠️ Not enough resolved trades yet — weights unchanged. "
                           "Needs more historical market data.")
            load_learning_data.clear()
    with col_info:
        if weights_doc:
            cycle_ts = weights_doc.get('cycle_ts')
            ts_str   = cycle_ts.strftime('%Y-%m-%d %H:%M UTC') if isinstance(cycle_ts, datetime) else 'N/A'
            st.caption(f"Last cycle: **{ts_str}** | Next: weekly via scheduler")
        else:
            st.caption("No learning cycle run yet — click Run or wait for weekly scheduler.")

    st.markdown("---")

    # ── Current model weights ────────────────────────────────────
    st.subheader("⚖️ Current Model Weights")
    if weights_doc and weights_doc.get('weights'):
        weights  = weights_doc['weights']
        wr_map   = weights_doc.get('indicator_wr', {})
        thresh   = weights_doc.get('thresholds', {})

        # Weights table
        wdf_rows = []
        for k, v in weights.items():
            wr   = wr_map.get(k)
            wr_s = f"{wr*100:.1f}%" if wr else 'N/A'
            wdf_rows.append({'Indicator': k.title(), 'Weight': round(v, 4),
                             'Weight %': f"{v*100:.1f}%", 'Win Rate': wr_s})
        wdf = pd.DataFrame(wdf_rows).sort_values('Weight', ascending=False)
        st.dataframe(wdf, use_container_width=True, hide_index=True)

        # Weight bar chart
        wdf_chart = pd.DataFrame({'Weight': weights}, index=weights.keys())
        st.bar_chart(wdf_chart)

        # Optimal thresholds
        if thresh:
            tc1, tc2 = st.columns(2)
            tc1.metric("Optimal Score Threshold",
                       thresh.get('optimal_score_threshold', 'N/A'),
                       f"{thresh.get('score_threshold_wr', 0)*100:.1f}% win rate")
            tc2.metric("Optimal Prob Threshold",
                       thresh.get('optimal_prob_threshold', 'N/A'),
                       f"{thresh.get('prob_threshold_wr', 0)*100:.1f}% win rate")
    else:
        st.info("No weights saved yet. Run a learning cycle first.")

    st.markdown("---")

    # ── Indicator win rates ──────────────────────────────────────
    st.subheader("📊 Indicator Win Rates")
    if stats_list:
        for stat in stats_list:
            key       = stat.get('indicator', '')
            label     = stat.get('label', key.title())
            wr        = stat.get('win_rate', 0)
            trades    = stat.get('trades', 0)
            wins      = stat.get('wins', 0)
            losses    = stat.get('losses', 0)
            contr     = stat.get('contribution', 0)
            confidence = stat.get('confidence', 'LOW')

            wr_pct   = wr * 100
            bar_col  = '#4caf50' if wr >= 0.60 else '#f44336' if wr < 0.40 else '#ffc107'
            badge_bg = '#1b5e20' if wr >= 0.60 else '#b71c1c' if wr < 0.40 else '#e65100'
            trend    = '↑ BULLISH INDICATOR' if contr > 0.05 else '↓ WEAK INDICATOR' if contr < -0.05 else '→ NEUTRAL'
            contr_str = f"{contr:+.1%}"

            st.markdown(
                f'<div style="background:#1a1a2e;border-radius:8px;padding:12px 16px;'
                f'margin-bottom:8px;border-left:4px solid {bar_col};">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-weight:bold;color:#ddd;">{label}</span>'
                f'<span style="background:{badge_bg};color:white;border-radius:4px;'
                f'padding:2px 10px;font-size:13px;">{wr_pct:.1f}% win rate</span>'
                f'</div>'
                f'<div style="margin:6px 0 4px 0;background:#333;border-radius:4px;height:8px;">'
                f'<div style="background:{bar_col};width:{int(wr_pct)}%;height:8px;border-radius:4px;"></div>'
                f'</div>'
                f'<span style="color:#aaa;font-size:12px;">{trades} trades &nbsp;•&nbsp; '
                f'{wins}W / {losses}L &nbsp;•&nbsp; contribution: {contr_str} &nbsp;•&nbsp; '
                f'confidence: {confidence} &nbsp;•&nbsp; {trend}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No indicator stats yet — run a learning cycle first.")

    st.markdown("---")

    # ── Weight history ───────────────────────────────────────────
    st.subheader("📈 Weight Change History")
    if weight_history:
        rows = []
        for doc in weight_history:
            row = {'Cycle': doc.get('cycle_ts', '').strftime('%Y-%m-%d') if isinstance(doc.get('cycle_ts'), datetime) else 'N/A'}
            row.update({k.title(): round(v, 4) for k, v in (doc.get('weights') or {}).items()})
            rows.append(row)
        hist_df = pd.DataFrame(rows)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No weight history yet — run at least one learning cycle.")

    # ── Strategy summary ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Strategy Improvements")
    if weights_doc and weights_doc.get('weights'):
        w = weights_doc['weights']
        wr_map = weights_doc.get('indicator_wr', {})
        items = sorted(wr_map.items(), key=lambda x: x[1], reverse=True)
        if items:
            best_ind, best_wr = items[0]
            worst_ind, worst_wr = items[-1]
            st.success(f"🏆 **Best indicator:** {best_ind.title()} — {best_wr*100:.1f}% win rate | Weight: {w.get(best_ind,0)*100:.1f}%")
            if worst_wr < 0.50:
                st.warning(f"⚠️ **Weakest indicator:** {worst_ind.title()} — {worst_wr*100:.1f}% win rate | Weight reduced to {w.get(worst_ind,0)*100:.1f}%")
            st.info("ℹ️ Weights auto-adjust weekly. Max ±10% change per cycle. Rollback backups always saved.")
        else:
            st.info("Run a learning cycle to see strategy improvements.")
    else:
        st.info("No learning data yet. Click **▶ Run Learning Cycle Now** above.")



# ══════════════════════════════════════════════════════════════
# RL OPTIMIZER PANEL (embedded in AI Learning tab)
# ══════════════════════════════════════════════════════════════

def render_rl_panel():
    """RL Optimizer panel — shown inside the AI Learning tab."""
    st.divider()
    st.markdown("## RL Optimizer")

    rl = load_rl_data()
    params    = rl.get('params') or {}
    perf_hist = rl.get('perf_hist', [])
    param_hist= rl.get('param_hist', [])

    # ── Current RL State ──
    st.markdown("### Current RL Parameters")

    rl_w     = float(params.get('rl_weight_adjustment', 1.0))
    ep       = int(params.get('episode', 0))
    reward   = float(params.get('reward_score', 0.0))
    ent_thr  = float(params.get('entry_threshold', 45.0))
    prob_thr = float(params.get('prob_threshold', 35.0))
    last_upd = params.get('last_updated')

    c1, c2, c3, c4, c5 = st.columns(5)
    delta_w = f"{(rl_w - 1.0)*100:+.1f}%" if rl_w != 1.0 else "neutral"
    with c1:
        st.metric("RL Weight", f"{rl_w:.4f}", delta=delta_w)
    with c2:
        st.metric("Episode", str(ep))
    with c3:
        st.metric("Cum. Reward", f"{reward:+.1f}")
    with c4:
        st.metric("Entry Threshold", f"{ent_thr:.1f}")
    with c5:
        st.metric("Prob Threshold", f"{prob_thr:.1f}")

    if last_upd:
        st.caption(f"Last updated: {last_upd.strftime('%Y-%m-%d %H:%M UTC') if hasattr(last_upd,'strftime') else last_upd}")

    # Safety clamp visual
    st.markdown("**RL Weight Safety Range: 0.80 - 1.20**")
    pct = (rl_w - 0.80) / 0.40 * 100
    pct = max(0, min(100, pct))
    color = "#4caf50" if 0.95 <= rl_w <= 1.10 else "#ff9800" if 0.85 <= rl_w <= 1.15 else "#f44336"
    st.markdown(
        f"<div style='background:#1e2a40;border-radius:4px;height:12px;width:100%;'>"
        f"<div style='width:{pct:.0f}%;background:{color};border-radius:4px;height:12px;'></div></div>",
        unsafe_allow_html=True
    )
    st.caption(f"Position: {rl_w:.4f} ({pct:.0f}% of safe range)")

    # Indicator weights
    if params.get('indicator_weights'):
        st.markdown("**Indicator Weight Multipliers**")
        iw = params['indicator_weights']
        ic1, ic2, ic3, ic4 = st.columns(4)
        cols = [ic1, ic2, ic3, ic4]
        for idx, (k, v) in enumerate(iw.items()):
            delta = f"{(v-1.0)*100:+.1f}%"
            cols[idx % 4].metric(k.upper(), f"{v:.3f}", delta=delta)

    # ── Performance History ──
    if perf_hist:
        st.divider()
        st.markdown("### Before vs After RL Comparison")

        latest = perf_hist[0]
        b = latest.get('before', {})
        a = latest.get('after', {})
        imp = latest.get('improvement_pct', 0)

        bvsc1, bvsc2, bvsc3 = st.columns(3)
        with bvsc1:
            st.metric("Return %",
                      f"{a.get('return_pct',0):+.1f}%",
                      delta=f"{a.get('return_pct',0)-b.get('return_pct',0):+.1f}pp")
        with bvsc2:
            st.metric("Win Rate",
                      f"{a.get('win_rate',0):.1f}%",
                      delta=f"{a.get('win_rate',0)-b.get('win_rate',0):+.1f}pp")
        with bvsc3:
            if imp > 0:
                st.success(f"Improvement: +{imp:.1f}%")
            elif imp < -1:
                st.warning(f"Change: {imp:.1f}%")
            else:
                st.info("Neutral change")

        # Chart: return before vs after across all episodes
        if len(perf_hist) > 1:
            try:
                import plotly.graph_objects as go
                episodes = [h.get('episode', i) for i, h in enumerate(reversed(perf_hist))]
                b_rets   = [h.get('before', {}).get('return_pct', 0) for h in reversed(perf_hist)]
                a_rets   = [h.get('after', {}).get('return_pct', 0) for h in reversed(perf_hist)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=episodes, y=b_rets, name='Before RL',
                                         line=dict(color='#f44336', width=2)))
                fig.add_trace(go.Scatter(x=episodes, y=a_rets, name='After RL',
                                         line=dict(color='#4caf50', width=2)))
                fig.update_layout(
                    height=220,
                    xaxis=dict(title='Episode', gridcolor='#1e2a40'),
                    yaxis=dict(title='Return %', gridcolor='#1e2a40', ticksuffix='%'),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        # Table
        rows = []
        for h in perf_hist[:5]:
            run_at = h.get('run_at')
            rows.append({
                'Run At':    run_at.strftime('%Y-%m-%d %H:%M') if hasattr(run_at,'strftime') else '?',
                'Episode':   h.get('episode', '?'),
                'RL Weight': f"{h.get('rl_weight', 1.0):.4f}",
                'Before %':  f"{h.get('before',{}).get('return_pct',0):+.1f}%",
                'After %':   f"{h.get('after',{}).get('return_pct',0):+.1f}%",
                'Improvement': f"{h.get('improvement_pct', 0):+.1f}%",
                'Applied':   'YES' if h.get('applied') else 'NO',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No RL performance history yet. Run the comparison below.")

    # ── Manual Controls ──
    st.divider()
    st.markdown("### Manual RL Controls")
    btn1, btn2 = st.columns(2)

    with btn1:
        if st.button("Run RL Learning Now", type="primary"):
            with st.spinner("Running RL learning cycle..."):
                try:
                    from ai.rl_optimizer import run_rl_learning
                    result = run_rl_learning(lookback_days=60, force=True)
                    if result.get('applied'):
                        p = result.get('params', {})
                        st.success(
                            f"Episode {result['episode']} complete! "
                            f"RL weight: {p.get('rl_weight_adjustment',1.0):.4f} | "
                            f"Reward: {result['reward']:+.2f}"
                        )
                    else:
                        st.warning(f"Not applied: {result.get('error','unknown')}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error: {ex}")

    with btn2:
        if st.button("Run Before/After Comparison"):
            with st.spinner("Running backtest comparison (may take 1-2 min)..."):
                try:
                    from rl_backtest_compare import run_comparison
                    result = run_comparison(days=60)
                    imp = result.get('improvement_pct', 0)
                    if imp > 0:
                        st.success(f"Improvement: +{imp:.1f}% | RL weight: {result['rl_weight']:.4f}")
                    else:
                        st.info(f"Change: {imp:.1f}% | RL weight: {result['rl_weight']:.4f}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error: {ex}")


# ══════════════════════════════════════════════════════════════
# TAB 5: STRATEGY INTELLIGENCE
# ══════════════════════════════════════════════════════════════

def render_strategy_intelligence():
    """Strategy Intelligence tab — shows active strategy config, RL params, and optimization log."""
    st.markdown("## 🎯 Strategy Intelligence")
    st.caption("Current active strategy configuration, RL-tuned parameters, and optimisation history.")

    cfg = load_strategy_config()
    active   = cfg.get('active') or {}
    recent   = cfg.get('recent', [])
    opt_log  = cfg.get('opt_log', [])

    # ── Active config ──
    if active:
        st.markdown("### Active Strategy Config")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Entry Threshold", active.get('entry_threshold', '?'))
        with c2:
            st.metric("Stop Loss %", f"{active.get('stop_loss_pct', 0):.1f}%")
        with c3:
            st.metric("Take Profit %", f"{active.get('take_profit_pct', 0):.1f}%")

        # Extra fields
        extra = {k: v for k, v in active.items()
                 if k not in ('entry_threshold', 'stop_loss_pct', 'take_profit_pct',
                              '_id', 'created_at', 'updated_at', 'is_active')}
        if extra:
            st.json(extra)
    else:
        st.info("No active strategy config found. Scheduler will create one on first optimisation cycle.")

    # ── RL Parameters ──
    st.divider()
    st.markdown("### RL Optimizer Parameters")
    rl = load_rl_data()
    params = rl.get('params') or {}
    if params:
        r1, r2, r3 = st.columns(3)
        with r1:
            rl_w = float(params.get('rl_weight_adjustment', 1.0))
            st.metric("RL Weight", f"{rl_w:.4f}",
                      delta=f"{(rl_w - 1.0)*100:+.1f}%")
        with r2:
            ep = rl.get('episode', 0)
            st.metric("Episode", str(ep))
        with r3:
            cr = rl.get('cumulative_reward', 0)
            st.metric("Cumulative Reward", f"{cr:+.1f}")
    else:
        st.info("No RL parameters yet — runs daily.")

    # ── Recent configs ──
    if recent:
        st.divider()
        st.markdown("### Recent Strategy Versions")
        rows = []
        for cfg_item in recent[:5]:
            ts = cfg_item.get('created_at') or cfg_item.get('updated_at')
            rows.append({
                'Date':           ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else '?',
                'Entry Thresh':   cfg_item.get('entry_threshold', '?'),
                'Stop Loss %':    f"{cfg_item.get('stop_loss_pct', 0):.1f}%",
                'Take Profit %':  f"{cfg_item.get('take_profit_pct', 0):.1f}%",
                'Active':         '✅' if cfg_item.get('is_active') else '—',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Optimisation log ──
    if opt_log:
        st.divider()
        st.markdown("### Optimisation History")
        log_rows = []
        for entry in opt_log[:10]:
            ts = entry.get('run_at')
            log_rows.append({
                'Run At': ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else '?',
                'Best Score':    f"{entry.get('best_score', 0):.2f}",
                'Win Rate':      f"{entry.get('win_rate', 0):.1f}%",
                'Return %':      f"{entry.get('return_pct', 0):+.1f}%",
                'Profit Factor': f"{entry.get('profit_factor', 0):.2f}",
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)
    else:
        st.info("No optimisation runs yet — runs once per week.")

    # ── Manual trigger ──
    st.divider()
    if st.button("Run Strategy Optimisation Now", type="primary"):
        with st.spinner("Running strategy optimisation (may take 1-2 min)..."):
            try:
                from optimization.auto_optimizer import run_optimization
                result = run_optimization(lookback_days=90)
                if result.get('applied'):
                    bc = result.get('best_config', {})
                    bp = result.get('best_perf', {})
                    st.success(
                        f"✅ New config applied! "
                        f"TP={bc.get('take_profit',0)*100:.0f}% "
                        f"SL={bc.get('stop_loss',0)*100:.0f}% "
                        f"Score≥{bc.get('min_score',0)} | "
                        f"Win Rate: {bp.get('win_rate',0):.1f}% | "
                        f"Return: {bp.get('return_pct',0):+.2f}%"
                    )
                else:
                    st.info(f"No improvement found over current config. Keeping existing settings.")
                st.cache_data.clear()
                st.rerun()
            except Exception as ex:
                st.error(f"Error: {ex}")



# ══════════════════════════════════════════════════════════════
# PAPER TRADING P&L PANEL
# ══════════════════════════════════════════════════════════════

def render_paper_trading_panel():
    """Auto paper trade tracking panel — no manual input needed."""
    st.markdown("## 📋 Paper Trading P&L (Auto-Tracked)")
    st.caption("Every conviction pick is automatically logged and tracked. WIN = price hit TP. LOSS = price hit SL. EXPIRED = 72h timeout.")

    try:
        import pymongo as _pm
        from services.paper_trading import get_paper_trade_summary
        client = _pm.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
        db = client[settings.DATABASE_NAME]
        summary = get_paper_trade_summary(db, days=30)
        client.close()
    except Exception as ex:
        st.error(f"Paper trading data unavailable: {ex}")
        return

    if not summary or summary.get('total_trades', 0) == 0:
        st.info("No paper trades yet. Starts automatically when conviction picks are generated (next conviction job run).")
        return

    # ── Summary metrics ──
    wr = summary.get('win_rate_pct', 0)
    total_pnl = summary.get('total_pnl_pct', 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Trades", summary.get('total_trades', 0))
    with m2:
        st.metric("Open", summary.get('open_count', 0))
    with m3:
        wr_color = "normal" if wr >= 50 else "inverse"
        st.metric("Win Rate", f"{wr:.1f}%", delta=f"{'Above' if wr>=50 else 'Below'} 50%")
    with m4:
        st.metric("Wins / Losses", f"{summary.get('win_count',0)}W / {summary.get('loss_count',0)}L")
    with m5:
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"{total_pnl:+.1f}%", delta=f"Avg win: {summary.get('avg_win_pct',0):+.1f}%")

    # ── Overall verdict ──
    if wr >= 60:
        st.success(f"🎯 Strong performance! Win rate {wr:.1f}% — signals are reliable")
    elif wr >= 50:
        st.info(f"📊 Decent performance. Win rate {wr:.1f}%")
    else:
        st.warning(f"⚠️ Win rate {wr:.1f}% — signals need recalibration")

    st.divider()

    # ── Open trades ──
    open_t = summary.get('open_trades', [])
    if open_t:
        st.markdown(f"### 🟡 Open Trades ({len(open_t)})")
        rows = []
        for t in open_t:
            # Safety: recalculate R:R if stored value is bad
            _rr = t.get('risk_reward')
            if not _rr or float(_rr) <= 0:
                _e2 = float(t.get('entry_price', 0) or 0)
                _t2 = float(t.get('take_profit', 0) or 0)
                _s2 = float(t.get('stop_loss', 0) or 0)
                if _e2 > 0 and _e2 > _s2 and _t2 > _e2:
                    _rr = round((_t2 - _e2) / (_e2 - _s2), 2)
            rows.append({
                'Symbol': t.get('symbol', '?'),
                'Type': t.get('trade_type', '?'),
                'Entry': f"{t.get('entry_price', 0):.6g}",
                'TP': f"{t.get('take_profit', 0):.6g}",
                'SL': f"{t.get('stop_loss', 0):.6g}",
                'Expected +%': f"+{t.get('expected_tp_pct', 0):.1f}%",
                'R:R': f"{float(_rr):.1f}" if _rr else '-',
                'Opened (IST)': t.get('opened_ist', '?'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Recent closed trades ──
    closed = summary.get('closed_trades', [])
    if closed:
        st.markdown("### 📚 Recent Closed Trades")
        rows2 = []
        for t in closed:
            outcome = t.get('outcome', '?')
            pnl = t.get('actual_pnl_pct', 0) or 0
            rows2.append({
                'Symbol': t.get('symbol', '?'),
                'Type': t.get('trade_type', '?'),
                'Outcome': outcome,
                'P&L': f"{pnl:+.2f}%",
                'Held (h)': f"{t.get('hold_hours', 0):.0f}h",
                'Closed (IST)': t.get('closed_ist', '?'),
            })
        df2 = pd.DataFrame(rows2)
        def highlight_outcome(row):
            color = '#1a4d2e' if row['Outcome'] == 'WIN' else '#4d1a1a' if row['Outcome'] == 'LOSS' else '#333'
            return [f'background-color: {color}'] * len(row)
        st.dataframe(df2.style.apply(highlight_outcome, axis=1), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# BEST SCAN TIME ANALYSIS (IST)
# ══════════════════════════════════════════════════════════════

def render_scan_time_panel():
    """Best IST trading hours based on historical signal win rates."""
    st.markdown("## ⏰ Best Scan Hours (IST)")
    st.caption("Which hours of the day (IST) historically produce BUY signals that go on to gain 4%+ within 24h?")

    try:
        import pymongo as _pm
        from services.scan_time_analysis import get_best_scan_hours
        import plotly.graph_objects as go
        client = _pm.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
        db = client[settings.DATABASE_NAME]
        data = get_best_scan_hours(db, lookback_days=2)
        client.close()
    except Exception as ex:
        st.error(f"Scan time analysis unavailable: {ex}")
        return

    if not data or not data.get('chart_data'):
        st.info("Insufficient data for scan time analysis (need 2 days of BUY signals with outcomes).")
        return

    chart = data['chart_data']
    best  = data.get('best_hours_ist', [])
    worst = data.get('worst_hours_ist', [])

    # Best hours highlight
    if best:
        best_labels = [f"{h:02d}:00 IST" for h in best[:3]]
        st.success(f"🏆 **Best hours to scan:** {', '.join(best_labels)}")
    if worst:
        worst_labels = [f"{h:02d}:00 IST" for h in worst]
        st.warning(f"⚠️ **Avoid these hours:** {', '.join(worst_labels)} (lowest win rate)")

    # Bar chart
    hours  = [c['label_ist'] for c in chart]
    rates  = [c['win_rate'] for c in chart]
    counts = [c['count'] for c in chart]
    colors = [
        'rgba(76,175,80,0.85)' if c['hour_ist'] in best else
        'rgba(244,67,54,0.70)' if c['hour_ist'] in worst else
        'rgba(66,133,244,0.65)' for c in chart
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours, y=rates, marker_color=colors,
        text=[f"{r:.0f}%<br>({n})" for r, n in zip(rates, counts)],
        textposition='outside', name='Win Rate %'
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='#ffc107',
                  annotation_text='50% baseline')
    fig.update_layout(
        height=320,
        yaxis=dict(title='Win Rate %', range=[0, 100], gridcolor='#1e2a40'),
        xaxis=dict(title='Hour (IST)', gridcolor='#1e2a40'),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='white'), margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Analysis based on {data.get('signals_analysed', 0)} BUY signals over last 2 days.")


# ══════════════════════════════════════════════════════════════
# WHALE INTELLIGENCE PANEL
# ══════════════════════════════════════════════════════════════

def render_whale_panel():
    """Whale Intelligence tab content."""
    st.markdown("## 🐋 Whale Intelligence")
    st.caption("Real-time large capital movement detection from aggTrades, order book, exchange flow, and depth analysis.")


    whale_docs = load_whale_data(limit=30)

    if not whale_docs:
        st.info("No whale data yet. Scheduler runs whale scan every 10 minutes. Check back soon!")
        return

    # ── Summary metrics ──
    accum = sum(1 for d in whale_docs if d.get('whale_signal') == 'ACCUMULATION')
    dist  = sum(1 for d in whale_docs if d.get('whale_signal') == 'DISTRIBUTION')
    neutral = len(whale_docs) - accum - dist
    all_scores = [d.get('whale_score', 50) for d in whale_docs]
    avg_score = round(sum(all_scores) / max(len(all_scores), 1), 1)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("🟢 Accumulation", str(accum))
    with sc2:
        st.metric("🔴 Distribution", str(dist))
    with sc3:
        st.metric("⚪ Neutral", str(neutral))
    with sc4:
        st.metric("Avg Whale Score", f"{avg_score:.1f}/100")

    # Overall sentiment
    if accum > dist * 1.5:
        st.success("🐋 Market Whale Sentiment: **BULLISH** — Net accumulation detected across top coins")
    elif dist > accum * 1.5:
        st.error("🐳 Market Whale Sentiment: **BEARISH** — Net distribution detected across top coins")
    else:
        st.info("🌊 Market Whale Sentiment: **NEUTRAL** — Mixed signals")

    # ── Symbol table ──
    st.divider()
    st.markdown("### Whale Score — Top Coins")

    rows = []
    for d in whale_docs:
        sym = d.get('_id', '?')
        score = d.get('whale_score', 50)
        sig   = d.get('whale_signal', 'NONE')
        norm  = d.get('whale_score_norm', 0.0)
        ts    = d.get('timestamp')
        m     = d.get('metrics', {})

        sig_emoji = '🟢' if sig == 'ACCUMULATION' else '🔴' if sig == 'DISTRIBUTION' else '⚪'
        rows.append({
            'Symbol':        sym,
            'Signal':        f"{sig_emoji} {sig}",
            'Whale Score':   f"{score:.1f}/100",
            'Buy Pressure':  f"{m.get('whale_buy_pressure', 0.5)*100:.0f}%",
            'Sell Pressure': f"{m.get('whale_sell_pressure', 0.5)*100:.0f}%",
            'Large Trades':  str(m.get('large_trade_count', 0)),
            'Flow Bias':     f"{m.get('exchange_flow_bias', 0.0):+.3f}",
            'OB Imbalance':  f"{m.get('order_book_imbalance', 0.0):+.3f}",
            'Updated':       ts.strftime('%H:%M') if hasattr(ts, 'strftime') else '?',
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

    # ── Buy vs Sell pressure bars (top 10) ──
    st.divider()
    st.markdown("### Buy vs Sell Pressure — Top 10")
    try:
        import plotly.graph_objects as go
        top10 = whale_docs[:10]
        syms  = [d.get('_id', '?') for d in top10]
        m_list= [d.get('metrics', {}) for d in top10]
        buys  = [m.get('whale_buy_pressure', 0.5) * 100 for m in m_list]
        sells = [m.get('whale_sell_pressure', 0.5) * 100 for m in m_list]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Buy Pressure', x=syms, y=buys,
                             marker_color='rgba(76, 175, 80, 0.8)'))
        fig.add_trace(go.Bar(name='Sell Pressure', x=syms, y=sells,
                             marker_color='rgba(244, 67, 54, 0.8)'))
        fig.update_layout(
            barmode='group', height=280,
            yaxis=dict(title='Pressure %', ticksuffix='%', gridcolor='#1e2a40'),
            xaxis=dict(gridcolor='#1e2a40'),
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white'), margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # ── Whale score bar chart ──
    st.divider()
    st.markdown("### Whale Score Distribution")
    try:
        scores_sorted = sorted(whale_docs, key=lambda d: d.get('whale_score', 50), reverse=True)[:20]
        syms2  = [d.get('_id', '?') for d in scores_sorted]
        scores2= [d.get('whale_score', 50) for d in scores_sorted]
        colors = ['rgba(76,175,80,0.8)' if d.get('whale_signal') == 'ACCUMULATION'
                  else 'rgba(244,67,54,0.8)' if d.get('whale_signal') == 'DISTRIBUTION'
                  else 'rgba(100,100,100,0.6)' for d in scores_sorted]

        fig2 = go.Figure(go.Bar(x=syms2, y=scores2, marker_color=colors))
        fig2.add_hline(y=70, line_dash='dash', line_color='#4caf50', annotation_text='Accumulation zone')
        fig2.add_hline(y=30, line_dash='dash', line_color='#f44336', annotation_text='Distribution zone')
        fig2.update_layout(
            height=260,
            yaxis=dict(title='Whale Score (0-100)', range=[0, 100], gridcolor='#1e2a40'),
            xaxis=dict(gridcolor='#1e2a40'),
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white'), margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass

    # ── Manual scan trigger ──
    st.divider()
    st.markdown("### Manual Controls")
    if st.button("Run Whale Scan Now", type="primary"):
        with st.spinner(f"Scanning top {settings.WHALE_SCAN_TOP_N} symbols for whale activity..."):
            try:
                import pymongo as _pm
                from ai.whale_tracker import run_whale_scan
                client, db = get_db_connection()
                from services.binance_scanner import get_top_symbols
                symbols = get_top_symbols(30)
                results = run_whale_scan(symbols, db=db)
                client.close()
                acc = sum(1 for r in results if r.get('whale_signal') == 'ACCUMULATION')
                dist_ = sum(1 for r in results if r.get('whale_signal') == 'DISTRIBUTION')
                st.success(f"Whale scan complete: {len(results)} coins | {acc} buying, {dist_} selling")
                st.cache_data.clear()
                st.rerun()
            except Exception as ex:
                st.error(f"Error: {ex}")

if __name__ == "__main__":
    main()
