# Portfolio

## 1. NBA Player Salary Predictions _using RFE + Gradient Boosting Regression_

This project involves analyzing and predicting NBA player salaries. It begins by cleaning and preprocessing player statistics and salary data. Feature selection is performed using Recursive Feature Elimination with Cross-Validation (RFECV), and a Gradient Boosting Regressor is trained on the selected features to predict player salaries. The model's performance is evaluated using the R-squared score, and the actual and predicted salaries are visualized through bar plots.

Keywords: Supervised learning, Regression, Feature Selection, Scikit-learn, RFECV, Gradient Boosting

## 2. Countries Clustering Analysis _using PCA + K-means_

In this project, a dataset containing information about different countries is analyzed using Principal Component Analysis (PCA) and clustering techniques. The dataset is preprocessed by scaling the features using StandardScaler. PCA is applied to reduce the dimensionality of the dataset to three components, and the explained variance ratio is calculated. The optimal number of clusters is determined using the elbow method. K-means clustering is performed on the reduced dataset, and the resulting clusters are visualized in a 3D scatter plot.

Keywords: Unsupervised learning, Clustering, Visualization, Standard scaling, PCA, K-means, elbow method

## 3. Fake News Detection _using Word2Vec + Recurrent Neural Network (LSTM)_

In this project, a fake news classification model is built using a dataset of news articles. The dataset is preprocessed by removing missing values and analyzing the distribution of word counts. The text data is preprocessed by tokenizing, removing stopwords, and converting it into sequences. Word2Vec is used to create word embeddings, and a neural network model is constructed using an embedding layer, LSTM layer, and dense layers. The model is trained on the dataset and evaluated using the accuracy score. The project utilizes various libraries such as TensorFlow, NLTK, Gensim, and Matplotlib for data processing, modeling, and visualization.

Keywords: Classification, Tokenization, Stopwords removal, Word2Vec, LSTM, Neural Network, TensorFlow, NLTK< Gensim, Data visualizatiob

## 4. Cover Letter Generator _using OpenAI Davinci LLM_

This project involves using NLP techniques to generate a cover letter tailored to a specific job vacancy. The project utilizes PDF and website data loaders to extract relevant information about the job applicant from their CV and the job vacancy details. A large language model (OpenAI Davinci) is then used to generate a cover letter by combining the extracted information in one prompt.

Keywords: Text generation, LLM, OpenAI, LangChain, Prompt enginnering

## 5. Twitter Sentiment Analysis _using RoBERTa Fine-tuning_ 

This project focuses on sentiment analysis of Twitter data using a pre-trained language model. The dataset is loaded, preprocessed, and split into training, validation, and testing sets. A sentiment classification model is trained using the Twitter-RoBERTa base model and evaluated using accuracy metrics. The trained model is then used to predict the sentiment of new Twitter data.

Keywords: Sentiment analysis, Transformers, Fine-tuning, HuggingFace

## 6. Propensity for Obesity Detection _using OpenAI Davinci LLM + Random Forest Classification_

This project showcases the integration of LLM for information extraction and prediction. The project goes beyond traditional analysis to extract meaningful insights and generate summaries using OpenAI Davinci LLM based on provided information. This demonstrates the application of natural language processing techniques in the data science pipeline, enhancing the project's complexity and real-world relevance.

Keywords: Classification, Summary Generation, LLM, OpenAI, LangChain, Prompt enginnering, Pipelines.  

## 7. Movie Reccomendation System _using K-Nearest Neighbors Collaborative Filtering_

This project demonstrates the implementation of a movie recommendation system. The dataset used includes information on movies and user ratings. The project involves data wrangling, creating a user-item matrix, and applying the KNN algorithm to find similar movies and make personalized recommendations for users. The code is written in Python, where recommendation system was implemented as a class to efficiently organize and encapsulate the code for the problem. By encapsulating the functionality within a class, the code becomes modular and reusable, making it easier to maintain and extend in the future. The project showcases the ability to analyze and process large datasets, apply collaborative filtering techniques, and provide relevant recommendations based on user preferences.

Keywords: Recommendation System, Collaborative Filtering, Item Similarities, KNN (K-Nearest Neighbors), Personalized Recommendations, Modularity.
