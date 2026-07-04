# Football Match Outcome Analysis & Prediction

Exploratory data analysis and machine learning project on the 2024/25 European football season, examining home advantage patterns and predicting match outcomes from team form.

## Objective

1. Does home advantage still hold in modern football, and does it vary by competition?
2. Can a team's recent form (last 5 matches) predict the outcome of their next match?

## Dataset

- **Source:** [Football Matches Dataset 2025](https://www.kaggle.com/datasets/tarekmasryo/football-matches-20242025-top-5-leagues) (Kaggle)
- 1,941 matches from the 2024/25 season across 6 competitions: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and the UEFA Champions League
- No missing values; clean, structured match-level data with scores, teams, referees, and derived fields (goal difference, match outcome, points)

## Methodology

1. **EDA** — analyzed overall match outcome distribution and broke it down by competition to check for home advantage patterns
2. **Feature engineering** — built rolling 5-match form features (average points, goals for, goals against) for each team, calculated strictly from *prior* matches to avoid data leakage. Matches with no prior team history (matchday 1) were dropped.
3. **Modeling** — trained and compared two classifiers to predict match outcome (Home Win / Draw / Away Win):
   - Logistic Regression (default and class-balanced)
   - Random Forest (class-balanced)
4. **Evaluation** — compared accuracy and macro F1-score, with particular attention to how well each model handled the minority "Draw" class
5. **SQL analysis** — loaded the dataset into a local SQLite database and re-analyzed it using SQL (SELECT, WHERE, GROUP BY, aggregate functions, and a self-JOIN comparing each team's home vs. away performance)

## Key Findings

- **Home teams won 42.9%** of matches, vs. 33.3% for away teams and 23.9% draws — home advantage is real
- Home advantage is **strongest in the UEFA Champions League** (~51% home win) and **weakest in the Bundesliga** (~39% home win, away teams nearly matching at ~36%)
- A default Logistic Regression reached 50.4% accuracy but **completely failed to predict draws**
- A **class-balanced Logistic Regression** (46.2% accuracy, 0.43 macro F1) traded a small amount of accuracy for meaningfully better, more useful predictions across all three outcomes — chosen as the final model
- All six form features contributed roughly equally to predictions — no single stat dominates match outcomes
- **SQL self-join analysis** revealed that mid-table teams like AS Monaco and Aston Villa had the largest home/away performance gaps (21 points over the season), while top clubs like Real Madrid and Barcelona performed far more consistently regardless of venue

## Tech Stack

- Python, pandas, matplotlib
- scikit-learn (Logistic Regression, Random Forest)
- SQL / SQLite (aggregate queries, self-JOIN)
- Jupyter Notebook

## How to Run

1. Clone this repo
2. Install dependencies: `pip install pandas matplotlib scikit-learn jupyter`
3. Download the dataset from the [Kaggle link above](https://www.kaggle.com/datasets/tarekmasryo/football-matches-20242025-top-5-leagues) and place the CSV in the project folder
4. Open `explore.ipynb` and run all cells

## Limitations & Next Steps

Form-based features alone can't capture injuries, tactical changes, or squad rotation, and results are based on a single season. Future improvements could include head-to-head history, Elo-style strength ratings, gradient boosting models, and a time-based train/test split for more realistic evaluation.