```python
import pandas as pd
import math
import contractions
from collections import Counter
```


```python
def make_bigram(tokens):
    bigramy = list(zip(tokens[:-1], tokens[1:]))
    return bigramy

def make_trigram(tokens):
    trigram = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    return trigram

def score_laplace_bigram(song):
    if len(song) ==0:
        return None
    score = 0.0
    for pair in song:
        score += math.log(prob_dict.get(pair, 1/V))
    score = math.exp(score/len(song))
    return score

def score_interpolated_trigram(trigram, l1, l2, l3):
    """Calculates the interpolated probability of a single trigram."""
    w1, w2, w3 = trigram
    bigram = (w2, w3)
    unigram = w3

    # Safely look up probabilities. If not found, use the default_prob!
    p3 = prob_triples.get(trigram, default_prob)
    p2 = prob_dict.get(bigram, default_prob)
    p1 = prob_words.get(unigram, default_prob)

    p_interp = (l3 * p3) + (l2 * p2) + (l1 * p1)
    return p_interp
```


```python
def grid_search_lambdas(validation_trigrams):
    """Tests all combinations of lambdas to find the one with the best score."""
    best_lambdas = (0, 0, 0)
    best_score = float('-inf') # Start with negative infinity so any score beats it

    # We step from 0.0 to 1.0 by 0.1
    # Using integers 0-10 avoids Python floating-point rounding errors
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j

            l1 = i / 10.0 # Unigram weight
            l2 = j / 10.0 # Bigram weight
            l3 = k / 10.0 # Trigram weight

            # Score the entire validation set with these specific lambdas
            current_log_total = 0.0
            for trigram in validation_trigrams:
                p = score_interpolated_trigram(trigram, l1, l2, l3)
                current_log_total += math.log(p)

            # Normalize by length to get the average
            avg_score = current_log_total / len(validation_trigrams)

            # If this combination scored higher (closer to 0), save it!
            if avg_score > best_score:
                best_score = avg_score
                best_lambdas = (l1, l2, l3)
                print(f"New Best! L1(Uni):{l1}, L2(Bi):{l2}, L3(Tri):{l3} | Score: {best_score}")

    return best_lambdas
```


```python
def get_predictability_score(song_trigrams, l1, l2, l3):
    """
    Calculates the final predictability score of a song using our optimized lambdas.
    Returns a geometric average probability (closer to 1.0 = highly predictable).
    """
    # If the song is too short to have trigrams, return a baseline
    if len(song_trigrams) == 0:
        return 0.0

    total_log_prob = 0.0

    for trigram in song_trigrams:
        w1, w2, w3 = trigram
        bigram = (w2, w3)
        unigram = w3

        # Safely look up probabilities with the Laplace fallback
        p3 = prob_triples.get(trigram, default_prob)
        p2 = prob_dict.get(bigram, default_prob)
        p1 = prob_words.get(unigram, default_prob)

        # Interpolate using our perfectly tuned weights
        p_interp = (l3 * p3) + (l2 * p2) + (l1 * p1)

        total_log_prob += math.log(p_interp)

    # Calculate the normalized geometric average (Perplexity inverse)
    final_score = math.exp(total_log_prob / len(song_trigrams))

    return final_score
```


```python
data = pd.read_csv("C:/Users/Anna/Desktop/{ANIA}/studia/inżynierka/data/data.csv")
```


```python
data = data[data['views'] >= 2000]
data['lyrics_clean'] = data['lyrics_clean'].str.replace('  ', ' ')
data['lyrics_clean'] = data['lyrics_clean'].str.replace('  ', ' ')
data = data.dropna(subset=['lyrics_clean'])
```


```python
data['lyrics_expanded'] = data['lyrics_clean'].apply(contractions.fix)
data['lyrics_clean'] = data['lyrics_expanded'].str.replace(r'[^\w\s]', '', regex=True)
data['tokens'] = data['lyrics_clean'].apply(lambda x: x.lower().split())
```


```python
data['bigram'] = data['tokens'].apply(make_bigram)
data['trigram'] = data['tokens'].apply(make_trigram)
```


```python
# 1. TRAIN SET: Build vocabulary and probabilities (e.g., 1960 - 1999)
train_data = data[data['year'] <= 1999].copy()

# 2. VALIDATION SET: Tune your Lambdas (e.g., 2000 - 2009)
val_data = data[(data['year'] >= 2000) & (data['year'] <= 2009)].copy()

# 3. TEST SET: Calculate final predictability (e.g., 2010 - 2022)
test_data = data[data['year'] >= 2010].copy()
```


```python
# Flatten tokens, bigrams, and trigrams for the training set
train_words = [word for song in train_data['tokens'] for word in song]
train_bigrams = [pair for song in train_data['bigram'] for pair in song]
train_trigrams = [triple for song in train_data['trigram'] for triple in song]
```


```python
tokens_count = Counter(train_words)
pairs_count = Counter(train_bigrams)
triples_count = Counter(train_trigrams)
```


```python
V = len(tokens_count)
default_prob = 1.0 / V
```


```python
# Unigram Probabilities
prob_words = {word: count/len(train_words) for word, count in tokens_count.items()}
```


```python
# Bigram Probabilities (with Laplace)
prob_dict = {pair: (count + 1) / (tokens_count[pair[0]] + V)
             for pair, count in pairs_count.items()}
```


```python
# Trigram Probabilities (with Laplace)
prob_triples = {triple: (count + 1) / (pairs_count[(triple[0], triple[1])] + V)
                for triple, count in triples_count.items()}
```


```python
# Flatten ALL validation trigrams into one giant list
all_val_trigrams = [trigram for song in val_data['trigram'] for trigram in song]
```


```python
# Run your grid search (make sure your grid_search_lambdas function uses 'all_val_trigrams')
best_weights = grid_search_lambdas(all_val_trigrams)
```

    New Best! L1(Uni):0.0, L2(Bi):0.0, L3(Tri):1.0 | Score: -9.674714389288281
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[23], line 2
          1 # Run your grid search (make sure your grid_search_lambdas function uses 'all_val_trigrams')
    ----> 2 best_weights = grid_search_lambdas(all_val_trigrams)
    

    Cell In[8], line 20, in grid_search_lambdas(validation_trigrams)
         18 for trigram in validation_trigrams:
         19     p = score_interpolated_trigram(trigram, l1, l2, l3)
    ---> 20     current_log_total += math.log(p)
         22 # Normalize by length to get the average
         23 avg_score = current_log_total / len(validation_trigrams)
    

    KeyboardInterrupt: 



```python
# Unpack the winning weights automatically
BEST_L1, BEST_L2, BEST_L3 = best_weights
print(f"Optimal Weights -> L1: {BEST_L1}, L2: {BEST_L2}, L3: {BEST_L3}")
```

    Optimal Weights -> L1: 0.6, L2: 0.4, L3: 0.0
    


```python
BEST_L1, BEST_L2, BEST_L3 = 0.6, 0.4, 0.0
test_data['predictability'] = test_data['trigram'].apply(
    lambda x: get_predictability_score(x, BEST_L1, BEST_L2, BEST_L3))
```


```python
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>tag</th>
      <th>artist</th>
      <th>year</th>
      <th>views</th>
      <th>features</th>
      <th>lyrics_clean</th>
      <th>lyrics_expanded</th>
      <th>tokens</th>
      <th>bigram</th>
      <th>trigram</th>
      <th>predictability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>El Diablo</td>
      <td>pop</td>
      <td>(Elena Tsagrinou)</td>
      <td>2021</td>
      <td>53376</td>
      <td>{"Έλενα Τσαγκρινού (Elena Tsagrinou)"}</td>
      <td>i fell in love i fell in love i gave my heart ...</td>
      <td>i fell in love i fell in love i gave my heart ...</td>
      <td>[i, fell, in, love, i, fell, in, love, i, gave...</td>
      <td>[(i, fell), (fell, in), (in, love), (love, i),...</td>
      <td>[(i, fell, in), (fell, in, love), (in, love, i...</td>
      <td>0.001432</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Afrika</td>
      <td>pop</td>
      <td>(Ivan Dorn)</td>
      <td>2017</td>
      <td>11840</td>
      <td>{"Иван Дорн (Ivan Dorn)"}</td>
      <td>i m going to spread that hot rythm of african ...</td>
      <td>i m going to spread that hot rythm of african ...</td>
      <td>[i, m, going, to, spread, that, hot, rythm, of...</td>
      <td>[(i, m), (m, going), (going, to), (to, spread)...</td>
      <td>[(i, m, going), (m, going, to), (going, to, sp...</td>
      <td>0.001697</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Beverly</td>
      <td>pop</td>
      <td>(Ivan Dorn)</td>
      <td>2017</td>
      <td>6527</td>
      <td>{"Иван Дорн (Ivan Dorn)"}</td>
      <td>beverly storefronts of the beauty yeah picture...</td>
      <td>beverly storefronts of the beauty yeah picture...</td>
      <td>[beverly, storefronts, of, the, beauty, yeah, ...</td>
      <td>[(beverly, storefronts), (storefronts, of), (o...</td>
      <td>[(beverly, storefronts, of), (storefronts, of,...</td>
      <td>0.000839</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Collaba</td>
      <td>pop</td>
      <td>(Ivan Dorn)</td>
      <td>2017</td>
      <td>10245</td>
      <td>{"Иван Дорн (Ivan Dorn)"}</td>
      <td>who wants to have a little bit of fun go back ...</td>
      <td>who wants to have a little bit of fun go back ...</td>
      <td>[who, wants, to, have, a, little, bit, of, fun...</td>
      <td>[(who, wants), (wants, to), (to, have), (have,...</td>
      <td>[(who, wants, to), (wants, to, have), (to, hav...</td>
      <td>0.001378</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Groovy Shit</td>
      <td>pop</td>
      <td>(Ivan Dorn)</td>
      <td>2017</td>
      <td>2960</td>
      <td>{"Иван Дорн (Ivan Dorn)"}</td>
      <td>obscure raw mind want to allure heaven and die...</td>
      <td>obscure raw mind want to allure heaven and die...</td>
      <td>[obscure, raw, mind, want, to, allure, heaven,...</td>
      <td>[(obscure, raw), (raw, mind), (mind, want), (w...</td>
      <td>[(obscure, raw, mind), (raw, mind, want), (min...</td>
      <td>0.001781</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1852810</th>
      <td>Too Late</td>
      <td>rap</td>
      <td>zamir</td>
      <td>2018</td>
      <td>2653</td>
      <td>{​zamir}</td>
      <td>tell me is it too late to change and do not sa...</td>
      <td>tell me is it too late to change and do not sa...</td>
      <td>[tell, me, is, it, too, late, to, change, and,...</td>
      <td>[(tell, me), (me, is), (is, it), (it, too), (t...</td>
      <td>[(tell, me, is), (me, is, it), (is, it, too), ...</td>
      <td>0.001764</td>
    </tr>
    <tr>
      <th>1852811</th>
      <td>bubble tea</td>
      <td>pop</td>
      <td>zamir &amp; marc indigo</td>
      <td>2020</td>
      <td>11269</td>
      <td>{"​zamir &amp; marc indigo"}</td>
      <td>uh you be like bubble tea you are like a summe...</td>
      <td>uh you be like bubble tea you are like a summe...</td>
      <td>[uh, you, be, like, bubble, tea, you, are, lik...</td>
      <td>[(uh, you), (you, be), (be, like), (like, bubb...</td>
      <td>[(uh, you, be), (you, be, like), (be, like, bu...</td>
      <td>0.002668</td>
    </tr>
    <tr>
      <th>1852812</th>
      <td>lemonade</td>
      <td>pop</td>
      <td>zamir &amp; marc indigo</td>
      <td>2020</td>
      <td>4222</td>
      <td>{"​zamir &amp; marc indigo"}</td>
      <td>it feels like every summers getting hotter so ...</td>
      <td>it feels like every summer's getting hotter so...</td>
      <td>[it, feels, like, every, summers, getting, hot...</td>
      <td>[(it, feels), (feels, like), (like, every), (e...</td>
      <td>[(it, feels, like), (feels, like, every), (lik...</td>
      <td>0.002193</td>
    </tr>
    <tr>
      <th>1852813</th>
      <td>on the move</td>
      <td>pop</td>
      <td>zamir &amp; marc indigo</td>
      <td>2019</td>
      <td>17717</td>
      <td>{Chevy,"​zamir &amp; marc indigo"}</td>
      <td>i am on the move cannot stop me if you try coa...</td>
      <td>i am on the move cannot stop me if you try coa...</td>
      <td>[i, am, on, the, move, cannot, stop, me, if, y...</td>
      <td>[(i, am), (am, on), (on, the), (the, move), (m...</td>
      <td>[(i, am, on), (am, on, the), (on, the, move), ...</td>
      <td>0.002650</td>
    </tr>
    <tr>
      <th>1852840</th>
      <td>Its Always Sunny With You</td>
      <td>pop</td>
      <td>{Parentheses}</td>
      <td>2017</td>
      <td>6844</td>
      <td>{}</td>
      <td>ok i am just going to take the photo right now...</td>
      <td>ok i am just going to take the photo right now...</td>
      <td>[ok, i, am, just, going, to, take, the, photo,...</td>
      <td>[(ok, i), (i, am), (am, just), (just, going), ...</td>
      <td>[(ok, i, am), (i, am, just), (am, just, going)...</td>
      <td>0.003367</td>
    </tr>
  </tbody>
</table>
<p>257480 rows × 12 columns</p>
</div>




```python
test_data['predictability'].describe()
```




    count    257480.000000
    mean          0.002289
    std           0.001339
    min           0.000000
    25%           0.001318
    50%           0.001994
    75%           0.002977
    max           0.019773
    Name: predictability, dtype: float64




```python
timeline_results = []

# Define the start years for our training decades
decades = [1960, 1970, 1980, 1990, 2000, 2010]
```


```python
# Custom weights to heavily emphasize trigram predictability
L1 = 0.1
L2 = 0.3
L3 = 0.6
```

# Code


```python

```


```python
for start_year in decades:
    train_start = start_year
    train_end = start_year + 9

    test_start = start_year + 10
    test_end = start_year + 19

    print(f"Training on {train_start}s... Testing on {test_start}s...")

    # 1. Filter the datasets for this specific window
    train_data = data[(data['year'] >= train_start) & (data['year'] <= train_end)]
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()

    # Safety check: if we run out of data, break the loop
    if train_data.empty or test_data.empty:
        break

    # 2. Build the Vocabulary & Counts ONLY on the training decade
    train_words = [word for song in train_data['tokens'] for word in song]
    train_bigrams = [pair for song in train_data['bigram'] for pair in song]
    train_trigrams = [triple for song in train_data['trigram'] for triple in song]

    tokens_count = Counter(train_words)
    pairs_count = Counter(train_bigrams)
    triples_count = Counter(train_trigrams)

    V = len(tokens_count)
    default_prob = 1.0 / V if V > 0 else 0.00001

    # Calculate probabilities (with Laplace smoothing)
    prob_words = {word: count/len(train_words) for word, count in tokens_count.items()}

    prob_dict = {pair: (count + 1) / (tokens_count[pair[0]] + V)
                 for pair, count in pairs_count.items()}

    prob_triples = {triple: (count + 1) / (pairs_count[(triple[0], triple[1])] + V)
                    for triple, count in triples_count.items()}

    # 3. Define the scoring function dynamically for this decade's dictionaries
    def score_decade_song(song_trigrams):
        if len(song_trigrams) == 0:
            return 0.0

        total_log_prob = 0.0
        for trigram in song_trigrams:
            w1, w2, w3 = trigram
            bigram = (w2, w3)
            unigram = w3

            p3 = prob_triples.get(trigram, default_prob)
            p2 = prob_dict.get(bigram, default_prob)
            p1 = prob_words.get(unigram, default_prob)

            p_interp = (L3 * p3) + (L2 * p2) + (L1 * p1)
            total_log_prob += math.log(p_interp)

        return math.exp(total_log_prob / len(song_trigrams))

    # 4. Score the test decade
    test_data['predictability'] = test_data['trigram'].apply(score_decade_song)

    # 5. Save the results
    avg_score = test_data['predictability'].mean()
    median_score = test_data['predictability'].median() # Median is great if a few songs act as weird outliers

    timeline_results.append({
        'Trained_On': f"{train_start}s",
        'Tested_On': f"{test_start}s",
        'Average_Predictability': avg_score,
        'Median_Predictability': median_score,
        'Songs_Analyzed': len(test_data)
    })

# Convert the results into a clean dataframe
results_df = pd.DataFrame(timeline_results)
print("\nDone! Here are your thesis results:")
print(results_df)
```

    Training on 1960s... Testing on 1970s...
    Training on 1970s... Testing on 1980s...
    Training on 1980s... Testing on 1990s...
    Training on 1990s... Testing on 2000s...
    Training on 2000s... Testing on 2010s...
    Training on 2010s... Testing on 2020s...
    
    Done! Here are your thesis results:
      Trained_On Tested_On  Average_Predictability  Median_Predictability  \
    0      1960s     1970s                0.000976               0.000849   
    1      1970s     1980s                0.001022               0.000861   
    2      1980s     1990s                0.000927               0.000736   
    3      1990s     2000s                0.000903               0.000755   
    4      2000s     2010s                0.001105               0.000923   
    5      2010s     2020s                0.001526               0.001295   
    
       Songs_Analyzed  
    0            9274  
    1           11445  
    2           21929  
    3           43523  
    4          212011  
    5           45469  
    


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional style for the thesis
sns.set_theme(style="whitegrid")

# Create a figure with a good size for a document
plt.figure(figsize=(10, 6))

# Plot the Average Predictability (The mean)
sns.lineplot(
    data=results_df,
    x='Tested_On',
    y='Average_Predictability',
    marker='o',
    linewidth=2.5,
    label='Średnia',
    color='royalblue'
)

# Plot the Median Predictability (To check for outliers)
sns.lineplot(
    data=results_df,
    x='Tested_On',
    y='Median_Predictability',
    marker='s', # square markers
    linewidth=2.5,
    linestyle='--', # dashed line to distinguish it
    label='Mediana',
    color='darkorange'
)

# Formatting the chart
plt.title('Ewolucja przewidywalności tekstów piosenek w zależnosci od dekady', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Dekada', fontsize=12, labelpad=10)
plt.ylabel('Wynik przewidywalności', fontsize=12, labelpad=10)

# Make the legend look nice
plt.legend(title='Metric', fontsize=11, title_fontsize=12)

# Ensure the layout is tight so it doesn't get cut off when saving
plt.tight_layout()

# Save the plot directly to your computer so you can drop it into your thesis!
# Make sure to change the path to where you want it saved.
plt.savefig("C:/Users/Anna/Desktop/{ANIA}/studia/inżynierka/predictability_chart.png", dpi=300)

# Display the plot in the notebook
plt.show()
```


    
![png](fhrdfiojeoicdem%2C_files/fhrdfiojeoicdem%2C_26_0.png)
    



```python
for start_year in decades:
    train_start = start_year
    train_end = start_year + 9

    test_start = 2020
    test_end = 2023

    print(f"Training on {train_start}s... Testing on {test_start}s...")

    # 1. Filter the datasets for this specific window
    train_data = data[(data['year'] >= train_start) & (data['year'] <= train_end)]
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()

    # Safety check: if we run out of data, break the loop
    if train_data.empty or test_data.empty:
        break

    # 2. Build the Vocabulary & Counts ONLY on the training decade
    train_words = [word for song in train_data['tokens'] for word in song]
    train_bigrams = [pair for song in train_data['bigram'] for pair in song]
    train_trigrams = [triple for song in train_data['trigram'] for triple in song]

    tokens_count = Counter(train_words)
    pairs_count = Counter(train_bigrams)
    triples_count = Counter(train_trigrams)

    V = len(tokens_count)
    default_prob = 1.0 / V if V > 0 else 0.00001

    # Calculate probabilities (with Laplace smoothing)
    prob_words = {word: count/len(train_words) for word, count in tokens_count.items()}

    prob_dict = {pair: (count + 1) / (tokens_count[pair[0]] + V)
                 for pair, count in pairs_count.items()}

    prob_triples = {triple: (count + 1) / (pairs_count[(triple[0], triple[1])] + V)
                    for triple, count in triples_count.items()}

    # 3. Define the scoring function dynamically for this decade's dictionaries
    def score_decade_song(song_trigrams):
        if len(song_trigrams) == 0:
            return 0.0

        total_log_prob = 0.0
        for trigram in song_trigrams:
            w1, w2, w3 = trigram
            bigram = (w2, w3)
            unigram = w3

            p3 = prob_triples.get(trigram, default_prob)
            p2 = prob_dict.get(bigram, default_prob)
            p1 = prob_words.get(unigram, default_prob)

            p_interp = (L3 * p3) + (L2 * p2) + (L1 * p1)
            total_log_prob += math.log(p_interp)

        return math.exp(total_log_prob / len(song_trigrams))

    # 4. Score the test decade
    test_data['predictability'] = test_data['trigram'].apply(score_decade_song)

    # 5. Save the results
    avg_score = test_data['predictability'].mean()
    median_score = test_data['predictability'].median() # Median is great if a few songs act as weird outliers

    timeline_results.append({
        'Trained_On': f"{train_start}s",
        'Tested_On': f"{test_start}s",
        'Average_Predictability': avg_score,
        'Median_Predictability': median_score,
        'Songs_Analyzed': len(test_data)
    })

# Convert the results into a clean dataframe
results_df_new = pd.DataFrame(timeline_results)
print("\nDone! Here are your thesis results:")
print(results_df_new)
```

    Training on 1960s... Testing on 2020s...
    Training on 1970s... Testing on 2020s...
    Training on 1980s... Testing on 2020s...
    Training on 1990s... Testing on 2020s...
    Training on 2000s... Testing on 2020s...
    Training on 2010s... Testing on 2020s...
    
    Done! Here are your thesis results:
      Trained_On Tested_On  Average_Predictability  Median_Predictability  \
    0      1960s     2020s                0.000839               0.000740   
    1      1970s     2020s                0.000909               0.000783   
    2      1980s     2020s                0.000936               0.000800   
    3      1990s     2020s                0.000933               0.000808   
    4      2000s     2020s                0.001090               0.000930   
    5      2010s     2020s                0.001526               0.001295   
    
       Songs_Analyzed  
    0           45469  
    1           45469  
    2           45469  
    3           45469  
    4           45469  
    5           45469  
    


```python
results_df_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Trained_On</th>
      <th>Tested_On</th>
      <th>Average_Predictability</th>
      <th>Median_Predictability</th>
      <th>Songs_Analyzed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1960s</td>
      <td>2020s</td>
      <td>0.000839</td>
      <td>0.000740</td>
      <td>45469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970s</td>
      <td>2020s</td>
      <td>0.000909</td>
      <td>0.000783</td>
      <td>45469</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980s</td>
      <td>2020s</td>
      <td>0.000936</td>
      <td>0.000800</td>
      <td>45469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990s</td>
      <td>2020s</td>
      <td>0.000933</td>
      <td>0.000808</td>
      <td>45469</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000s</td>
      <td>2020s</td>
      <td>0.001090</td>
      <td>0.000930</td>
      <td>45469</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2010s</td>
      <td>2020s</td>
      <td>0.001526</td>
      <td>0.001295</td>
      <td>45469</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional style
sns.set_theme(style="whitegrid")

# Create the figure
plt.figure(figsize=(10, 6))

# Plot the Average Predictability (Change x to 'Trained_On')
sns.lineplot(
    data=results_df_new,
    x='Trained_On',
    y='Average_Predictability',
    marker='o',
    linewidth=2.5,
    label='Średnia',
    color='royalblue'
)

# Plot the Median Predictability (Change x to 'Trained_On')
sns.lineplot(
    data=results_df_new,
    x='Trained_On',
    y='Median_Predictability',
    marker='s',
    linewidth=2.5,
    linestyle='--',
    label='Mediana',
    color='darkorange'
)

# Update the titles and labels to reflect your new experiment
plt.title('Przewidywalność piosenek z lat 20. XXI wieku na podstawie poprzednich dekad', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Dekada', fontsize=12, labelpad=10)
plt.ylabel('Wynik przewidywalności', fontsize=12, labelpad=10)

# Formatting
plt.legend(title='Metric', fontsize=11, title_fontsize=12)
plt.tight_layout()

plt.savefig("C:/Users/Anna/Desktop/{ANIA}/studia/inżynierka/predictability_chart2.png", dpi=300)

# Display the plot
plt.show()
```


    
![png](fhrdfiojeoicdem%2C_files/fhrdfiojeoicdem%2C_29_0.png)
    



```python
df = all_val_trigrams
```
