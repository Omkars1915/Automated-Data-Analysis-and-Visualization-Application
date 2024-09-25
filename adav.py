import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Automated Data Analysis & Visualization",
    layout="wide"
)

# Function to load data from CSV file
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to perform data analysis
def perform_analysis(data):
    analysis_result = data.describe()
    return analysis_result

# Function to clean the dataset
def clean_data(data):
    cleaned_data = data.dropna()
    return cleaned_data

# Function to answer FAQ questions
# Function to answer FAQ questions
def answer_faq(question):
    faq = {
        "what is data analysis?": "Data analysis is the process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, informing conclusions, and supporting decision-making.",
        "what is data visualization?": "Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.",
        "why is data analysis important?": "Data analysis helps businesses make better decisions, identify trends and patterns, optimize processes, understand customer behavior, and gain a competitive edge.",
        "what are the different types of data analysis?": "Descriptive, diagnostic, predictive, and prescriptive are the four main types of data analysis.",
        "what are the benefits of data visualization?": "Data visualization makes complex data more understandable, aids in identifying trends and patterns, facilitates decision-making, and helps communicate insights effectively.",
        "what are some popular data visualization tools?": "Tableau, Power BI, QlikView, Google Data Studio, and Matplotlib are some popular data visualization tools.",
        "what is exploratory data analysis (eda)?": "Exploratory data analysis is the process of visually exploring and analyzing data to understand its main characteristics, uncover patterns, and identify relationships between variables.",
        "what is regression analysis?": "Regression analysis is a statistical method used to model the relationship between one dependent variable and one or more independent variables. It is often used for forecasting and prediction.",
        "how does clustering analysis work?": "Clustering analysis is a data mining technique used to group similar data points together based on their characteristics. It helps identify natural groupings or clusters within a dataset.",
        "what is the purpose of time series analysis?": "Time series analysis is used to analyze data collected over time to identify trends, seasonality, and patterns. It is commonly used in forecasting future values based on historical data.",
        "what are outliers in data analysis?": "Outliers are data points that are significantly different from other observations in a dataset. They can distort statistical analyses and should be carefully examined to determine if they are errors or meaningful data.",
        "how can missing data be handled in data analysis?": "Missing data can be handled by imputation (replacing missing values with estimated values), deletion (removing observations with missing values), or using specialized algorithms that can handle missing data.",
        "what is the difference between correlation and causation?": "Correlation refers to a statistical relationship between two variables, whereas causation implies that one variable directly influences the other. Correlation does not imply causation, as a third variable may be responsible for the observed relationship.",
        "what is a histogram?": "A histogram is a graphical representation of the distribution of numerical data. It consists of bars that represent the frequency or relative frequency of data within specific intervals (bins).",
        "how can i visualize categorical data?": "Categorical data can be visualized using bar charts, pie charts, stacked bar charts, and grouped bar charts to show the distribution of categories and their proportions.",
        "what is a box plot (box-and-whisker plot)?": "A box plot is a graphical representation of the distribution of numerical data through quartiles. It displays the median, quartiles, and outliers of a dataset, providing insights into its central tendency and variability.",
        "what is the purpose of a scatter plot?": "A scatter plot is used to visualize the relationship between two numerical variables. Each data point is represented as a dot on the plot, with the horizontal axis representing one variable and the vertical axis representing the other.",
        "how can i visualize geospatial data?": "Geospatial data can be visualized using maps, choropleth maps, heatmaps, and spatial scatter plots to display geographical patterns, distributions, and relationships.",
        "what is a heat map?": "A heat map is a graphical representation of data where values are represented as colors. It is often used to visualize the density or intensity of data points across a two-dimensional grid or map.",
        "what is data mining?": "Data mining is the process of discovering patterns, relationships, and insights from large datasets using statistical, mathematical, and machine learning techniques.",
        "how can i detect trends in time series data?": "Trends in time series data can be detected using methods such as moving averages, exponential smoothing, and regression analysis. Visualization techniques like line charts and decomposition can also help identify trends.",
        "what is a decision tree?": "A decision tree is a tree-like structure used to model decisions and their possible consequences. It consists of nodes representing decision points, branches representing possible outcomes, and leaves representing the final outcomes.",
        "what is the purpose of data preprocessing?": "Data preprocessing involves cleaning, transforming, and preparing raw data for analysis. It aims to improve data quality, remove noise and outliers, and make the data suitable for modeling and visualization.",
        "what is data wrangling?": "Data wrangling, also known as data munging, is the process of transforming and mapping data from raw form into another format with the intent of making it more appropriate and valuable for a variety of downstream purposes, such as analytics.",
        "how can i perform sentiment analysis on text data?": "Sentiment analysis is performed on text data to determine the sentiment or opinion expressed in it. It can be done using natural language processing (NLP) techniques and machine learning algorithms trained on labeled text data.",
        "what is the difference between supervised and unsupervised learning?": "Supervised learning involves training a model on labeled data, where the correct output is provided, while unsupervised learning involves training on unlabeled data, where the model must find patterns and relationships on its own.",
        "what is dimensionality reduction?": "Dimensionality reduction is the process of reducing the number of input variables in a dataset. It is often used to simplify models, speed up computation, and improve visualization and interpretation of data.",
        "what are some common data preprocessing techniques?": "Common data preprocessing techniques include data cleaning (handling missing values, removing duplicates), data transformation (scaling, normalization), and feature engineering (creating new features from existing ones).",
        "how can i assess the performance of a machine learning model?": "The performance of a machine learning model can be assessed using metrics such as accuracy, precision, recall, F1-score, ROC curve, and confusion matrix, depending on the type of problem (classification, regression, etc.).",
        "what is overfitting in machine learning?": "Overfitting occurs when a model learns the training data too well, capturing noise and random fluctuations rather than the underlying patterns. It often leads to poor generalization performance on unseen data.",
        "what is underfitting in machine learning?": "Underfitting occurs when a model is too simple to capture the underlying structure of the data. It results in poor performance on both the training and test datasets.",
        "how can i prevent overfitting in machine learning?": "Overfitting can be prevented by using techniques such as cross-validation, regularization, early stopping, and reducing model complexity.",
        "what is cross-validation?": "Cross-validation is a technique used to assess the performance of a machine learning model by splitting the dataset into multiple subsets, training the model on some subsets, and testing it on others to evaluate its generalization performance.",
        "what is regularization?": "Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function, which penalizes large weights or model complexity.",
        "what are some common algorithms used for classification?": "Common algorithms for classification include logistic regression, decision trees, random forests, support vector machines (SVM), k-nearest neighbors (KNN), and neural networks.",
        "what are some common algorithms used for regression?": "Common algorithms for regression include linear regression, polynomial regression, decision trees, random forests, support vector regression (SVR), and neural networks.",
        "how can i visualize the performance of a classification model?": "The performance of a classification model can be visualized using metrics such as ROC curves, precision-recall curves, confusion matrices, and calibration plots.",
        "how can i visualize the performance of a regression model?": "The performance of a regression model can be visualized using scatter plots of predicted versus actual values, residual plots, and learning curves.",
        "what is the curse of dimensionality?": "The curse of dimensionality refers to the problems that arise when working with high-dimensional data, such as increased computational complexity, sparsity of data, and difficulty in visualization and interpretation.",
        "what is feature selection?": "Feature selection is the process of selecting a subset of relevant features (variables) from a larger set of features to improve model performance, reduce overfitting, and increase interpretability.",
        "what is ensemble learning?": "Ensemble learning is a machine learning technique that combines the predictions of multiple individual models (learners) to improve overall performance. Common ensemble methods include bagging, boosting, and stacking.",
        "what is bagging?": "Bagging (bootstrap aggregating) is an ensemble learning technique where multiple models are trained on different subsets of the training data sampled with replacement. The final prediction is obtained by averaging the predictions of all models.",
        "what is boosting?": "Boosting is an ensemble learning technique that sequentially trains multiple weak learners (models) on the same dataset, with each subsequent learner focusing on the examples that were misclassified by previous learners. The final prediction is a weighted sum of the individual learner predictions.",
        "what is stacking?": "Stacking (stacked generalization) is an ensemble learning technique that combines the predictions of multiple diverse base models using a meta-learner (higher-level model). The base models' predictions serve as input features for the meta-learner, which learns to make the final prediction.",
        "how can i visualize decision boundaries in classification?": "Decision boundaries in classification can be visualized using contour plots, decision trees, or by plotting the predicted class labels on a grid of points in the feature space.",
        "what is the elbow method in clustering?": "The Elbow method is a technique used to determine the optimal number of clusters in a dataset. It involves plotting the within-cluster sum of squares (WCSS) against the number of clusters and identifying the 'elbow' point where the rate of decrease in WCSS slows down.",
        "what is the silhouette score in clustering?": "The Silhouette score is a measure of how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.",
        "what is a dendrogram in hierarchical clustering?": "A dendrogram is a tree-like diagram used to illustrate the arrangement of the clusters produced by hierarchical clustering. It shows the hierarchical relationships between data points and clusters.",
        "how can i visualize the results of hierarchical clustering?": "The results of hierarchical clustering can be visualized using dendrograms, heatmaps, or by plotting the dendrogram alongside the original data points to visualize the clustering structure.",
        "what is the difference between unsupervised and semi-supervised learning?": "Unsupervised learning involves training on unlabeled data, while semi-supervised learning uses a combination of labeled and unlabeled data for training. Semi-supervised learning algorithms typically leverage the unlabeled data to improve model performance and generalization."
    }

    question_lower = question.lower()  # Convert input question to lowercase

    for faq_question in faq.keys():
        if question_lower == faq_question:
            return faq[faq_question]

    return "Sorry, I don't have an answer for that question."


# Main function for Streamlit app
def main():
    st.title('Automated Data Analysis, Visualization, and ChatBot Integration')

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Load data from CSV file
        data = load_data(uploaded_file)

        # Display data analysis results
        st.subheader('Data Analysis Results')
        analysis_result = perform_analysis(data)
        st.write(analysis_result)

        # Button to clean dataset
        clean_button = st.button("Clean Dataset")
        if clean_button:
            cleaned_data = clean_data(data)
            st.success("Dataset cleaned successfully!")
            st.subheader('Cleaned Data Analysis Results')
            cleaned_analysis_result = perform_analysis(cleaned_data)
            st.write(cleaned_analysis_result)

        # Create Pygwalker renderer for interactive visualizations
        renderer = StreamlitRenderer(data)

        # Render Pygwalker explorer for interactive visualization
        st.subheader('Interactive Visualization')
        renderer.explorer()

    # Sidebar for FAQ chatbot
    st.sidebar.title('FAQ Chatbot')
    faq_question = st.sidebar.text_input('Ask a question')
    if faq_question:
        answer = answer_faq(faq_question)
        st.sidebar.write('Answer:', answer)

# Execute main function
if __name__ == "_main_":
    main()