
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import openai


# Set up OpenAI API key
openai.api_key = 'YOUR_API_KEY'

def load_data(file_path):
    data = pd.read_csv(file_path,encoding='latin1')
    return data

def perform_analysis(data):
    analysis_result = data.describe()
    return analysis_result

def clean_data(data):
    cleaned_data = data.dropna()
    return cleaned_data

def visualize_data(data, plot_type):
    st.subheader(f'{plot_type.capitalize()} Plot')

    if plot_type == 'scatter':
        numerical_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if len(numerical_columns) < 2:
            st.warning("Not enough numerical columns available for scatter plot.")
        else:
            selected_columns = st.multiselect("Select two numerical columns for the scatter plot", numerical_columns, default=numerical_columns[:2], key='scatter_columns')
            if len(selected_columns) == 2:
                st.write(f'Scatter Plot for {selected_columns[0]} vs {selected_columns[1]}')
                st.scatter_chart(data[selected_columns])

    elif plot_type in ['line', 'bar', 'histogram', 'box', 'area']:
        st.write(f'{plot_type.capitalize()} Plot')
    if plot_type == 'line':
                    numerical_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                    if numerical_columns:
                        selected_column = st.multiselect("Select a numerical column for the line plot", numerical_columns)
                        st.write(f'Line Plot for {selected_column}')
                        plt.figure(figsize=(8, 6))
                        plt.plot(data[selected_column])
                        st.pyplot()

    elif plot_type == 'bar':
            categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
            if categorical_columns:
                selected_column = st.multiselect("Select a categorical column for the bar plot", categorical_columns)
                st.bar_chart(data[selected_column])
    elif plot_type == 'histogram':
        numerical_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if numerical_columns:
            selected_column = st.multiselect("Select a numerical column for the histogram", numerical_columns, default=numerical_columns[:2])
            st.write(f'Histogram for {selected_column}')
            st.pyplot(plt.hist(data[selected_column], bins=20,  edgecolor='black'))
            st.pyplot(sns.histplot(data[selected_column], kde=True, color='skyblue'))
    elif plot_type == 'box':
        numerical_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if numerical_columns:
            selected_column = st.multiselect("Select a numerical column for the box plot", numerical_columns)
            st.write(f'Box Plot for {selected_column}')
            st.pyplot(sns.boxplot(data=data[selected_column]))
    elif plot_type == 'area':
            st.area_chart(data)

def create_dashboard(selected_charts):
    st.subheader('Editable Dashboard')

    col1, col2 = st.columns(2)

    for i, chart in enumerate(selected_charts):
        delete_button = col2.button(f"Delete {chart}", key=f"delete_{i}")

        with col1:
            st.write(f"#### {chart}")
            st.info("Drag to rearrange")

        if delete_button:
            selected_charts.remove(chart)

    if not selected_charts:
        st.write("No charts selected for the dashboard.")

def chat_with_bot(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def main():
    st.title('Automated Data Analysis and Visualization Application')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        st.subheader('Data Analysis Results')
        analysis_result = perform_analysis(data)
        st.write(analysis_result)

        st.subheader('Clean the Dataset')
        clean_button = st.button("Clean Dataset")
        if clean_button:
            cleaned_data = clean_data(data)
            st.success("Dataset cleaned successfully!")
            st.subheader('Cleaned Data Analysis Results')
            cleaned_analysis_result = perform_analysis(cleaned_data)
            st.write(cleaned_analysis_result)

            # visualize_data(cleaned_data, 'scatter')

            # visualize_data(data, 'scatter')

        st.sidebar.subheader('Chat with the Data Analysis Bot')
        user_input = st.sidebar.text_input("You: ", "")
        if st.sidebar.button("Send"):
            if user_input:
                response = chat_with_bot(user_input)
                st.sidebar.text_area("Bot:", value=response, height=100)

                # Determine whether user input involves chatbot-related commands
                is_chatbot_command = any(command in user_input.lower() for command in ['data analysis', 'describe data', 'clean data', 'scatter plot', 'line plot', 'bar plot', 'histogram', 'box plot', 'area plot'])

                if not is_chatbot_command:
                    selected_charts = []
                    plot_types = ['scatter', 'line', 'bar', 'histogram', 'box', 'area']
                    for plot_type in plot_types:
                        if st.checkbox(f'Select for Dashboard {plot_type.capitalize()} Plot'):
                            visualize_data(data, plot_type)
                            selected_charts.append(f"{plot_type.capitalize()} Plot")

                    if st.button('Create/Edit Dashboard'):
                        create_dashboard(selected_charts)
        selected_charts = []
        plot_types = ['scatter', 'line', 'bar', 'histogram', 'box', 'area']  # Add more plot types as needed
        for plot_type in plot_types:
            if st.checkbox(f'Select for Dashboard {plot_type.capitalize()} Plot'):
                visualize_data(data, plot_type)
                selected_charts.append(f"{plot_type.capitalize()} Plot")

        # Add button to create an editable dashboard
        if st.button('Create/Edit Dashboard'):
            create_dashboard(selected_charts)

if __name__ == "__main__":
    main()