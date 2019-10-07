# ActivistsDigest: Helping Austinites scan City Council meeting documentation (http://www.activistsdigest.xyz)

Acitivist's Digest is a tool that lowers the barrier to entry for Austinites to engage with their local government. After a user inputs a topic of interest, Activist's Digest searches through all public comments made a Austin City Council regular meetings for the last 4 years and returns the meeting dates and particular comments most relevant to the user query. This saves a user hours of time otherwise spent finding and opening individual meeting transcripts and skimming the 200-page documents for their topic of interest.

### Under the hood
1. I analyzed every pdf transcript of the biweekly Austin City Council regular meetings that occurred over the last 4 years. This amounted to over 12,000 pages and about 3 million words.
2. The raw text data was processed using regex functions, NLTK, and pandas.
3. I used a word2vec model pre-trained on the Google News corpus to convert the processed text into 300 dimensional vectors. This resulted in a single 300 dimensional vector for each public speaker comment. User input was processed and vectorized in the same way.
4. Using cosine-similarity, user input was compared to each public speaker comment in the dataframe. Comments with the highest similarity score, along with information and links to the full meeting documentation, are returned.
