# Sports Research Lab v33

This build separates Basketball (NBA/WNBA) from NFL while keeping Type a Bet, Bet Slips, Edge Alerts, My Bets, and Data Admin shared.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Bundled data

`seed_data/` contains a compact local foundation built from the files supplied for this project:

- NBA player-game history: 2023-24, 2024-25, 2025-26
- WNBA player-game history: 2024, 2025, 2026 seed snapshot
- NFL player weekly stats: 2024-2026
- NFL snap counts: 2024-2026
- NFL weekly rosters: 2024-2026
- NFL schedules/games: 2024-2026
- NFL play-by-play Parquet: 2024-2026

The local files are fallbacks/seed history. Live feeds are still checked for current schedules, rosters, injuries, game logs, markets, and current-season updates.

## Main v33 changes

- Separate `Basketball` and `NFL` workspaces.
- Basketball player reports read local historical player-game data when a live endpoint fails.
- NFL player reports read nflverse player weeks + snap counts and show football-specific workload/production.
- NFL game reports use recent pass/rush EPA, defensive EPA allowed, rest, environment, roster/QB context, and PBP success-rate fields when Parquet support is available.
- NFL player workload is snap share, not basketball minutes.
- Type a Bet and Bet Slips support the local NFL data format.
- Find a Bet can scan the full selected roster with local fallbacks.
- Kalshi Edge Alerts keep model probability, Kalshi price, sportsbook confirmation, expected value, tracked historical accuracy, and conservative risk sizing separate.
- StatMuse is not used as a data dependency.
- Optional targeted current-player web search can check injury/restriction/role news without a visible Ask-AI page when `TAVILY_API_KEY` is configured.

## Optional environment variables

- `TAVILY_API_KEY`: targeted current public-source searches and optional edge corroboration.
- `GROQ_API_KEY`: optional summarization only in the existing edge-corroboration path. Core stats/models do not depend on it.

## Important data behavior

Bundled current-season files are snapshots. The app is intentionally designed so live/current feeds can extend them. A local historical row should never be interpreted as proof that an injury status, lineup, minutes restriction, snap limitation, or sportsbook price is current.

## Sportsbook coverage

The automatic sportsbook feed exposes only the books/providers returned by ESPN for that event. The Edge Alerts page includes manual entry for another sportsbook/app so the app does not pretend it fetched FanDuel, BetMGM, Caesars, etc. when they were not actually returned.
