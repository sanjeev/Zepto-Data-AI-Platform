# Module 2 — Analytics Pipeline (`/analytics`)

One continuous pipeline over the Titanic dataset: **load once → profile → clean → explore → model → tune → persist**.

## Files

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Task 1–6: the **single** `sns.load_dataset('titanic')` call, immediate `df.to_csv("titanic.csv", index=False)`, profiling, missing-value rules, univariate / bivariate / multivariate analysis, z-score check |
| `02_modeling.ipynb` | Task 7–15: reads the same `titanic.csv` (never reloads from seaborn), stratified split, `ColumnTransformer` + `Pipeline`, 3 classifiers, metrics, imbalance study, GridSearchCV + OOB, fare regression, comparison table, joblib persistence |
| `cleaning.py` | The shared cleaning rules (threshold rule) imported by both notebooks |
| `titanic.csv` | The committed offline fallback, written straight after the one load (891 rows × 15 cols) |
| `titanic_best_pipeline.joblib` | The saved **complete** fitted pipeline (preprocessing + tuned Random Forest) |
| `predict_demo.py` | Reloads the pipeline with `joblib.load` and predicts on raw new passengers |
| `figures/*.png` | Charts saved by the notebooks (supporting artifacts only; every chart is interpreted in text below and in the notebooks) |

## Run

```bash
pip install -r analytics/requirements.txt
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb       # loads + saves titanic.csv
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb  # ~1 min (GridSearchCV)
python predict_demo.py
```

Run order matters: `01_eda` creates `titanic.csv` and `02_modeling` reads it. If `sns.load_dataset` cannot reach the internet, `01_eda` falls back to the committed `titanic.csv`.
*macOS note:* if the python.org build of Python raises `CERTIFICATE_VERIFY_FAILED` on the seaborn download, run
`export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` first, or run the "Install Certificates.command" that ships with that Python.

**Load-once guarantee:** `sns.load_dataset` appears exactly once in the module (the first code cell of `01_eda.ipynb`). `02_modeling.ipynb` starts from `pd.read_csv("titanic.csv")` and applies the same `cleaning.structural_clean` rules. Only data-independent steps (dropping the 2 rows without `embarked`, `deck → Unknown`) run before the split. Age imputation, encoding and scaling are learned inside the pipeline from the training split only.

---

# Part A — Profiling, cleaning and the data story

## Task 1 — Profile
`df.shape` = **(891, 15)**. `df.info()` and `df.describe()` are printed in `01_eda.ipynb`. Columns with missing values:

| Column | Missing | % missing |
|---|---|---|
| `embarked` | 2 | 0.22% |
| `embark_town` | 2 | 0.22% |
| `age` | 177 | 19.87% |
| `deck` | 688 | 77.22% |

Target balance: **61.62% died / 38.38% survived**.

## Task 2 — Missing-value handling
Measured on the raw 891 rows (table above), four columns have missing values. Rule: **< 5% → drop rows, 5–30% → impute, too high → drop column or encode "missing" as its own category.**

| Column | Missing | Rule triggered | Decision |
|---|---|---|---|
| `embarked` | 2 rows = **0.22%** | < 5% → drop rows | Drop the 2 rows (891 → 889). |
| `embark_town` | 2 rows = **0.22%** | < 5% → drop rows | Same 2 passengers as `embarked` (it is the same information spelled out), so the same drop fixes both. |
| `age` | 177 rows = **19.87%** | 5–30% → impute | Impute with the **median age within each `sex` × `pclass` group**. The median is robust to the right tail of ages, and grouping keeps the strong class/age link (1st-class passengers were older) instead of pulling everyone to one global value. |
| `deck` | 688 rows = **77.22%** | > 30% → imputation unreliable | Imputing would invent a deck for more than 3 of every 4 passengers, so instead **`Unknown` is encoded as its own category**. I keep the column rather than drop it because the missingness carries information: a deck is recorded for 81.0% of 1st-class passengers but only 8.7% of 2nd and 2.4% of 3rd, and passengers with a known deck survived at 67.0% vs 29.9% for `Unknown`. (The models in `02_modeling` still leave `deck` out, since `pclass` and `fare` already carry that signal.) |

These rules live in `cleaning.py` so that `02_modeling.ipynb` applies exactly the same decisions.

## Task 3 — Univariate analysis (`age`, `fare`)

Histograms and box plots: `figures/01_univariate_age_fare.png`.

| Column | Q1 | Q3 | IQR | Lower fence | Upper fence | Outliers |
|---|---|---|---|---|---|---|
| age (cleaned) | 21.50 | 36.00 | 14.50 | −0.25 | 57.75 | **32** (all high) |
| age (raw, non-missing) | 20.12 | 38.00 | 17.88 | −6.69 | 64.81 | 11 |
| fare | 7.90 | 31.00 | 23.10 | −26.76 | 65.66 | **114** (all high) |

**Outliers (IQR rule).** `fare` has **114 outliers**, all above the upper fence of 65.66 (Q1 = 7.90, Q3 = 31.00, IQR = 23.10); none fall below the lower fence. `age` (after cleaning) has **32 outliers**, all above 57.75 (Q1 = 21.5, Q3 = 36.0, IQR = 14.5). On the raw non-missing ages there are only 11 (fence 64.81). The difference comes from the imputation: placing 177 values at group medians packs the middle of the distribution and narrows the IQR from 17.88 to 14.5 years. I keep all of these rows. They are real elderly passengers and real luxury fares, not data-entry errors.

**Skewness of `fare`.** mean = **32.10** > median = **14.45** > mode = **8.05**, so **`fare` is strongly right-skewed** (positively skewed; sample skewness = 4.80). Most passengers paid cheap 3rd-class fares near the mode, while a small number of very expensive 1st-class tickets (up to 512.33) pull the mean far to the right of the median. That long right tail also explains why the box plot shows so many high-side outliers.

## Task 4 — Bivariate analysis

Survival rates computed with boolean masks (`&` / `|`):

| Group | Survival % |
|---|---|
| female | 74.04 |
| male | 18.89 |
| pclass 1 | 62.62 |
| pclass 2 | 47.28 |
| pclass 3 | 24.24 |
| female & pclass 1 | 96.74 |
| female & pclass 2 | 92.11 |
| female & pclass 3 | 50.00 |
| male & pclass 1 | 36.89 |
| male & pclass 2 | 15.74 |
| male & pclass 3 | 13.54 |
| female \| age < 16 | 71.59 |

**Reading the breakdowns.** Sex is the dominant factor: **74.04%** of women survived vs **18.89%** of men. Class matters too, with a monotone gradient of **62.62% → 47.28% → 24.24%** from 1st to 3rd class. Combined, the gap is stark: 1st- and 2nd-class women survived at **96.74%** and **92.11%**, while 3rd-class women survived at only **50.00%**. Men stay low everywhere, with **36.89%** in 1st class and just **15.74%** / **13.54%** in 2nd / 3rd. The `&`/`|` masks agree: "women or children (< 16)" survived at 71.59% vs 16.39% for adult men.

### Correlation heatmap — `survived, pclass, age, sibsp, parch, fare` (`figures/02_correlation_heatmap.png`)

`adult_male` and `alone` are excluded because they are derived flags, not independent measurements.

**The two strongest correlations** are the off-diagonal pairs with the largest |r| (ranked table above):

1. **`pclass`–`fare`: r = −0.548.** Higher class *number* (a cheaper class) goes with a lower fare. This is the ticket-pricing structure: median non-zero fare is 61.98 in 1st class, 15.02 in 2nd and 8.05 in 3rd. The two columns largely encode the same socio-economic signal, which is worth remembering for linear models (collinearity).
2. **`sibsp`–`parch`: r = +0.415.** Passengers travelling with siblings/spouses also tended to travel with parents/children, because both counts rise together for family groups.

(Third, just behind, is `pclass`–`age` at −0.411: older passengers were more often in 1st class.) For the target itself, `survived` correlates most with `pclass` (−0.336) and `fare` (+0.255). Sex is not in this numeric-only matrix, but the masks above show it is the strongest driver of all.

## Task 5 — Multivariate data story: who survived, and why?

Chart file: `figures/03_story_class_sex.png`

**Chart 1 — Class × sex:** Within every class, women's survival bars sit far above men's, and within each sex survival falls as class goes from 1st to 3rd. The "women first" rule was applied almost completely in 1st and 2nd class (96.7% and 92.1%), but only half of 3rd-class women survived. So sex sets the baseline, and class decides how much of that advantage a woman could actually use. This chart sets up the two main axes of the story.

Chart file: `figures/04_story_age_sex_class.png`

**Chart 2 — Age group × sex × class heatmap:** Women in 1st/2nd class survive at 0.85–1.00 at almost every age, while adult men (19–60) in 2nd/3rd class survive at only 0.04–0.13. Age helps mostly for boys: males aged 0–12 survived at 1.00 in 1st and 2nd class and 0.36 in 3rd, versus 0.13 for 3rd-class men aged 19–35. Across everyone, children 0–12 survived at 58.0% vs 35.8% for adults 19–35 and 19.0% for seniors 61+. A few cells hold only one or two passengers (e.g. the 0.00 for 1st-class girls is a single person), so those cells should be read as anecdotes, not rates.

Chart file: `figures/05_story_age_fare_scatter.png`

**Chart 3 — Age vs fare scatter (log fare):** Survivors (green) cluster in the upper band of fares and dying passengers (grey) dominate the dense low-fare strip around 7–15. Survival rises steadily by fare quartile: **19.7% → 30.4% → 45.5% → 57.7%**. Fare acts as a proxy for class and cabin location (closer to the boat deck), so it repeats Chart 1's class effect in continuous form. Across all ages, the low-fare deaths are mostly men (the marker style shows sex).

Chart file: `figures/06_story_family_size.png`

**Chart 4 — Family size:** Survival is non-monotone: passengers travelling alone (n = 535) survived at 30.1%, small families of 2–4 did best (55.3%, 57.8%, 72.4%), and large families of 5+ collapsed (20.0%, 13.6%, 33.3%, then 0% for sizes 8 and 11). Large families were overwhelmingly 3rd class (54 of the 62 passengers in families of 5+) and had to find and move many people at once, while solo travellers were mostly adult men. That explains why both extremes do badly. It also suggests a non-linear family-size effect that a linear model would miss.

Chart file: `figures/07_story_age_class_box.png`

**Chart 5 — Age by class, split by survival:** 1st-class passengers were older overall, which matches the `pclass`–`age` correlation of −0.41. In every class, survivors were younger than non-survivors: median age **35.0 vs 40.0** in 1st, **28.0 vs 30.0** in 2nd, **21.5 vs 25.0** in 3rd. The age effect is modest compared with sex and class, but it points the same way as the "children first" pattern in Chart 2.

### The story in one paragraph
Survival on the Titanic was decided first by **sex** (74% of women vs 19% of men), then by **class/wealth** (63% → 47% → 24% from 1st to 3rd class, and 20% → 58% from the lowest to the highest fare quartile), and then, to a lesser degree, by **age** (children 0–12: 58%) and **group size** (small families did best; solo travellers and large, mostly 3rd-class families did worst). The evacuation followed "women and children first", but access to the boats depended on class. That is why a 3rd-class woman had only a coin-flip chance (50%) while a 1st-class woman almost always survived (97%). These patterns tell the modeling notebook which features to use: `sex`, `pclass`, `fare`, `age`, `sibsp`, `parch` and `embarked`.

## Task 6 — Exploratory z-score check (`figures/08_zscore_before_after.png`)

| | mean | std |
|---|---|---|
| age | 29.0654 | 13.2702 |
| fare | 32.0967 | 49.6975 |
| age_z | 0.0000 | 1.0000 |
| fare_z | 0.0000 | 1.0000 |

**Check passed.** Before scaling, `age` has mean 29.07 / std 13.27 and `fare` has mean 32.10 / std 49.70. After `z = (x − mean) / std`, both `age_z` and `fare_z` have **mean 0.0000 and std 1.0000**. The overlaid histograms show that the *shape* is unchanged: `fare_z` is still right-skewed, with a max of 9.66 standard deviations. A z-score is a linear shift-and-rescale, so it puts features on a common scale for distance- or gradient-based models but does not remove skew or outliers (that would need e.g. a log transform). This check is exploratory only; the modeling pipeline fits its own `StandardScaler` on the training split alone.

---

# Part B — Predictive modeling

## Task 7 — Stratified split

Features: `pclass, age, sibsp, parch, fare, sex, embarked`. Target: `survived`. `alive` is excluded because it is the target written as text (it would leak the answer). `class`, `embark_town`, `who`, `adult_male` and `alone` duplicate or derive from kept columns. `deck` is 77% unknown. `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 711 train and 178 test rows.

**Why stratify.** The target is imbalanced, with **38.25% survived vs 61.75% died** (about 1.6 : 1). A plain random 80/20 split could shift the survivor share of the 178-row test set by several percentage points just by chance. That would change precision, recall and F1 for reasons unrelated to the models, and make model comparisons noisy. With `stratify=y`, both splits keep the same balance (train 38.26%, test 38.20% survived). The split happens **before** any imputation, encoding or scaling, so no preprocessing step ever sees the test rows.

## Task 8 — Preprocessing (train-only fit)

**Preprocessing choices (all fit on `X_train` only).**
- `age` (140 missing in train, 37 in test) and the other numeric features use `SimpleImputer(strategy="median")`. The median (**28.0**) is learned from the training rows only and then *applied* to test rows. This is deliberately simpler than Task 2's sex × pclass group median, because a global median from train is easy to apply to any future passenger and cannot leak test information. The tree-based models can still learn class/sex/age interactions themselves.
- `sex`, `embarked` use `SimpleImputer(most_frequent)` followed by `OneHotEncoder(handle_unknown="ignore", drop="if_binary")`, producing `sex_male` plus three `embarked_*` dummies. `embarked` has no missing values after the structural cleaning, but the imputer is kept so the saved pipeline can handle raw new data with a missing port.
- All numeric features go through `StandardScaler`.

Everything sits inside one `ColumnTransformer` wrapped in a `Pipeline` with the estimator. `Pipeline.fit(X_train, …)` runs `fit_transform` on train, and `predict(X_test)` runs `transform` only, so the structure itself guarantees that no step is ever fit or refit on test data or on the full pre-split dataset.

## Task 9 — Three classifiers + tree plot (`figures/09_decision_tree.png`)

Logistic Regression, Decision Tree (`max_depth=4`) and Random Forest (300 trees) are each wrapped in the same preprocessing pipeline and fit on the identical split.

**Reading the tree.** The root split is `sex_male`, so the tree, like the EDA, finds sex the most informative variable. Females are split next by class (scaled `pclass ≤ 0.21` ⇒ 1st/2nd class) and then by fare. Males are split first by age: scaled `age ≤ −1.99` is about **3.5 years old**, and boys that young are predicted to survive. After that comes class. Depth is capped at 4 (with `min_samples_leaf=5`) so the tree stays readable and does not memorise the 711 training rows.

## Task 10 — Evaluation (`figures/10_confusion_matrices.png`, `figures/11_roc_curves.png`)

**Results (test set, n = 178).**

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Confusion [[TN FP] [FN TP]] |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | **0.8610** | [[98, 12], [21, 47]] |
| Decision Tree | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8510 | [[97, 13], [23, 45]] |
| Random Forest | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8348 | [[97, 13], [23, 45]] |

Logistic Regression is best on every metric at the default 0.5 threshold. The Decision Tree and the untuned Random Forest happen to make identical hard predictions on this test set, but their probability rankings differ (AUC 0.851 vs 0.835). The untuned forest, with unlimited depth, overfits the training rows. All three models miss about a third of survivors (recall 0.66–0.69): false negatives (21–23) outnumber false positives (12–13), which fits the 38/62 class imbalance pushing borderline cases toward "died".

## Task 11 — Imbalance handling

**Imbalance comparison (Logistic Regression, test set).** Train has 439 died / 272 survived (61.7% / 38.3%). SMOTE balanced the *training* fold to 439 / 439, and the test set stayed untouched at 110 / 68.

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| (a) baseline | **0.7966** | 0.6912 | 0.7402 |
| (b) `class_weight='balanced'` | 0.7083 | **0.7500** | 0.7286 |
| (c) SMOTE (train fold only) | 0.7463 | 0.7353 | **0.7407** |

**Conclusion.** **SMOTE worked best overall.** It has the highest F1 (0.7407) and the highest ROC AUC (0.8668), and it recovers survivors (recall +4.4 points over baseline) while giving up only 5 points of precision. `class_weight='balanced'` pushes recall highest (0.75) but over-corrects, dropping precision to 0.71 and giving the lowest F1. Re-weighting moves the decision boundary for every training point, while SMOTE adds synthetic minority points only where survivors actually live in feature space. Because the imbalance is mild (38/62), all three F1 scores are within about 1 point; on 178 test rows, one passenger moves recall by 1.5 points. The practical choice depends on the cost of a missed survivor: use SMOTE (or `balanced`) when recall matters, and the baseline when precision matters.

## Task 12 — Hyperparameter tuning

**GridSearchCV result.** The grid covered 3 × 4 × 3 = 36 combinations × 5 stratified folds on the training split, scored by ROC AUC, with `RandomForestClassifier(oob_score=True, …)` inside the full preprocessing pipeline.
- **Best parameters:** `n_estimators = 400`, `max_depth = 6`, `max_features = None`
- **Best mean CV ROC AUC:** 0.8778
- **OOB score of the refit best forest:** **0.8129** (out-of-bag accuracy on the training split)
- Test set: accuracy 0.8371, precision 0.8679, recall 0.6765, F1 0.7603, ROC AUC 0.8476

Limiting depth to 6 is the key change against the untuned forest (unlimited depth). It stops trees from memorising individual passengers: CV ROC AUC goes 0.8752 → 0.8778, test accuracy 0.7978 → 0.8371, and test F1 0.7143 → 0.7603. With only 9 encoded features, `max_features=None` (every feature considered at every split) beat `sqrt`/`log2`, because there are too few features for random subspaces to add much useful diversity. The OOB accuracy (0.813) is close to the CV accuracy (0.830) and the test accuracy (0.837), so the forest is not overfitting.

## Task 13 — Regression side-task: predict `fare` (`figures/12_regression_residuals.png`)

Multivariate `LinearRegression` on `pclass, age, sibsp, parch, survived, sex, embarked`. It uses the same preprocessing pattern and the same train/test rows as the classifiers. Adjusted R² = 1 − (1 − R²)(n − 1)/(n − p − 1).

**Regression metrics (test set, n = 178, p = 9 predictors after encoding):** MAE = **19.75**, RMSE = **41.27**, R² = **0.3471**, Adjusted R² = **0.3122**.

**Heteroscedasticity: yes.** The residual-vs-fitted plot fans out, with small, tight residuals for cheap predicted fares and a widening spread (mostly large positive residuals) as the predicted fare grows. The numbers confirm it. Mean |residual| rises from 9.71 (sd 7.39) in the lowest third of fitted values to 14.32 (sd 9.26) in the middle and 35.40 (sd 59.20) in the top third, and corr(|residual|, fitted) = 0.32. The spread of errors is therefore not constant, which breaks a key OLS assumption: coefficient standard errors are unreliable, and the model under-predicts the very expensive 1st-class fares. RMSE being twice the MAE points the same way (a few huge misses). Modeling `log(fare)` or using a tree-based regressor would be the natural next step. An R² of 0.35 means class, sex, age, family and port explain only about a third of fare variance, because fare is per ticket (shared by groups) and varies a lot within class.

## Task 14 — Model comparison table

Two separate metric groups. Classification metrics (0–1, higher is better) and regression metrics (MAE/RMSE in fare units, lower is better; R² higher is better) are on different scales and must not be compared with each other. "—" means "not applicable to this model type".

| Model | **Classification:** Accuracy | Precision | Recall | F1 | ROC AUC | **Regression:** MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | 0.8610 | — | — | — | — |
| Decision Tree | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8510 | — | — | — | — |
| Random Forest | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8348 | — | — | — | — |
| Random Forest (tuned, GridSearchCV) | **0.8371** | **0.8679** | 0.6765 | **0.7603** | 0.8476 | — | — | — | — |
| Linear Regression (target: fare) | — | — | — | — | — | 19.7531 | 41.2701 | 0.3471 | 0.3122 |

Model selection by 5-fold stratified CV on the training split:

| Model | CV ROC AUC (mean ± std) | CV F1 | CV accuracy |
|---|---|---|---|
| Random Forest (tuned) | **0.8778 ± 0.0115** | **0.7611** | **0.8298** |
| Random Forest | 0.8752 ± 0.0149 | 0.7574 | 0.8228 |
| Logistic Regression | 0.8443 ± 0.0367 | 0.7100 | 0.7848 |
| Decision Tree | 0.8432 ± 0.0207 | 0.7040 | 0.7989 |

### Final recommendation
**Deploy the tuned Random Forest pipeline** (`n_estimators=400, max_depth=6, max_features=None`), chosen on the training split rather than the test set: it has the highest 5-fold CV ROC AUC (**0.8778 ± 0.0115**, vs 0.8443 ± 0.0367 for Logistic Regression) and the highest CV F1 (0.7611). On the held-out test set it also leads on accuracy (**0.8371**), precision (**0.8679**) and F1 (**0.7603**), and its OOB accuracy of 0.8129 agrees with those numbers. Its test ROC AUC (0.8476) is below Logistic Regression's 0.8610, but on only 178 rows that gap sits inside LR's larger CV variability, so LR remains a strong, more interpretable fallback. Its main weakness is recall (0.6765), so if missing a true survivor is the costly error, lower the decision threshold or retrain with the SMOTE variant from Task 11. The regression metrics (R² 0.347) are on a different scale and describe only how predictable `fare` is, not which classifier to deploy.

## Task 15 — Saved pipeline

`joblib.dump(full_pipeline, "titanic_best_pipeline.joblib")` saves the **whole** fitted `Pipeline([("pre", ColumnTransformer(imputers + one-hot + scaler)), ("clf", RandomForestClassifier(oob_score=True, n_estimators=400, max_depth=6, max_features=None))])`, not the bare estimator. The last cell of `02_modeling.ipynb` and `predict_demo.py` reload it with `joblib.load` and confirm that:

- predictions on the raw test rows are identical to the in-memory pipeline (test accuracy 0.8371), and
- it scores brand-new raw passengers directly, including one with a missing `age` and missing `embarked`:

| pclass | age | sibsp | parch | fare | sex | embarked | predicted | P(survived) |
|---|---|---|---|---|---|---|---|---|
| 1 | 29 | 0 | 0 | 211.34 | female | S | 1 | 0.997 |
| 3 | 24 | 0 | 0 | 7.90 | male | S | 0 | 0.068 |
| 2 | NaN | 1 | 2 | 26.00 | female | None | 1 | 0.837 |
