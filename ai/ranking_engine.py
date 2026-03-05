"""
Ranking Engine — selects Top 3 opportunities by composite opportunity_score
"""

import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def rank_coins(results: List[Dict]) -> List[Dict]:
    """
    Rank all scanned coins and select Top 3 opportunities.

    Ranking factors (opportunity_score 0-100):
        - Final AI score:   30%
        - Probability:      25%
        - Volume strength:  15%
        - Risk level:       10% (inverse — Low risk = high score)
        - Whale signals:    10%
        - News sentiment:   10%

    Adds to each result:
        top_rank: int (1, 2, 3) or None
        opportunity_score: float
    """
    if not results:
        return results

    for r in results:
        try:
            ai = r.get('final_score', 50)
            prob = r.get('probability_up', 50)

            # Volume: volume_spike mapped to 0-100
            vs = r.get('volume_spike', 1.0)
            vol_score = min(100, vs * 40)

            # Risk: Low=90, Medium=50, High=20
            risk = r.get('risk_level', 'Medium')
            risk_map = {'Low': 90, 'Medium': 50, 'High': 20}
            risk_score = risk_map.get(risk, 50)

            # Whale
            ws = r.get('whale_score', 0)
            whale_signal = r.get('whale_signal', 'NONE')
            whale_adj = ws if whale_signal == 'ACCUMULATION' else (50 - ws if whale_signal == 'DISTRIBUTION' else 30)

            # News
            ns = r.get('news_score', 0)  # -1 to +1
            news_adj = (ns + 1) * 50  # 0-100

            # Composite
            opp = (
                ai * 0.30 +
                prob * 0.25 +
                vol_score * 0.15 +
                risk_score * 0.10 +
                whale_adj * 0.10 +
                news_adj * 0.10
            )

            r['opportunity_score'] = round(min(100, max(0, opp)), 2)

        except Exception as e:
            r['opportunity_score'] = 0
            logger.error(f"Ranking error for {r.get('symbol')}: {e}")

    # Sort descending by opportunity_score
    results.sort(key=lambda x: x.get('opportunity_score', 0), reverse=True)

    # Assign top_rank to top 3
    for i, r in enumerate(results):
        r['top_rank'] = (i + 1) if i < 3 else None

    top3 = results[:3]
    top3_str = ", ".join(r["symbol"] + "(" + str(r["opportunity_score"]) + ")" for r in top3)
    logger.info(f"Top 3: {top3_str}")

    return results


if __name__ == "__main__":
    test = [
        {'symbol': 'BTC', 'final_score': 80, 'probability_up': 75, 'volume_spike': 2.5,
         'risk_level': 'Low', 'whale_score': 60, 'whale_signal': 'ACCUMULATION', 'news_score': 0.5},
        {'symbol': 'ETH', 'final_score': 65, 'probability_up': 60, 'volume_spike': 1.2,
         'risk_level': 'Medium', 'whale_score': 30, 'whale_signal': 'NONE', 'news_score': 0},
        {'symbol': 'SOL', 'final_score': 72, 'probability_up': 70, 'volume_spike': 3.0,
         'risk_level': 'Low', 'whale_score': 70, 'whale_signal': 'ACCUMULATION', 'news_score': 0.8},
    ]
    ranked = rank_coins(test)
    for r in ranked:
        print(f"#{r['top_rank']} {r['symbol']}: Opportunity={r['opportunity_score']}")
